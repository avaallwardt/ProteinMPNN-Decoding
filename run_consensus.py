#!/usr/bin/env python3
"""Marginal-consensus decoding: argmax of per-position marginals averaged over decoding orders.

Motivation
----------
Sequence recovery is a per-position accuracy metric. Under a model
:math:`p(S \\mid X)`, the estimator maximising expected per-position accuracy is
the per-position marginal mode,

.. math::
    \\hat{s}_i = \\arg\\max_a \\; p(s_i = a \\mid X)

not a draw from the joint, and not the joint mode. Standard ProteinMPNN
sampling returns a joint sample under one decoding order, which is a different
estimator optimising a different thing.

Each sample's ``probs`` output gives :math:`p(s_i \\mid s_{\\pi(<i)}, X)` for the
order actually used. Averaging those across many random orders is a Monte Carlo
estimate of the marginal with both the prefix and the order integrated out.
Taking the argmax of that average is the marginal-mode estimator.

This is cheap: the samples are drawn anyway, and the averaging is free.

Arms
----
``baseline``           single sample at T (one joint draw)
``greedy``             single sample at very low T (approximate joint mode)
``consensus_T*``       argmax of marginals averaged over N samples at temperature T
``consensus_then_icm`` consensus followed by conditional ICM sweeps

Usage::

    python experiments/run_consensus.py --set benchmark_set.json \\
        --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \\
        --n-samples 8 --out results/consensus.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ipr import refine  # noqa: E402
from ipr.metrics import sequence_recovery, to_numpy  # noqa: E402


def consensus_sequence(runner, n_samples: int, temperature: float, seed: int):
    """Average per-position marginals over ``n_samples`` random orders, then argmax.

    Returns ``(sequence, mean_probs, agreement)`` where ``agreement`` is the mean
    probability mass the averaged marginal puts on its own argmax -- a measure of
    how much the decoding orders agree with each other.
    """
    total = None
    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        out = runner.sample(temperature=temperature, order=None, rng=rng)
        probs = to_numpy(out["probs"])
        probs = probs[0] if probs.ndim == 3 else probs
        total = probs if total is None else total + probs

    mean_probs = total / n_samples
    # Renormalize: masked positions (if any) would otherwise be all-zero rows.
    row_sums = mean_probs.sum(axis=-1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    mean_probs = mean_probs / safe

    seq = mean_probs.argmax(axis=-1).astype(np.int64)
    agreement = float(mean_probs.max(axis=-1).mean())
    return seq, mean_probs, agreement


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--mpnn", type=Path, default=Path("ProteinMPNN"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=8, help="samples averaged per consensus")
    p.add_argument("--n-samples-list", type=int, nargs="*", default=None,
                   help="sweep the number of averaged samples instead of using --n-samples")
    p.add_argument("--reps", type=int, default=3, help="independent replicates per structure")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--temperatures", type=float, nargs="*", default=[0.1, 0.5, 1.0])
    p.add_argument("--icm-sweeps", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.mpnn.resolve()))
    from ipr.backends import Runner, featurize_pdb, load_model

    records = json.loads(args.set.read_text())
    if args.limit:
        records = records[:args.limit]

    model, info = load_model(args.weights, device=args.device)
    print(f"weights: k_neighbors={info['k_neighbors']}")
    print(f"consensus over {args.n_samples} samples, {args.reps} replicates per structure\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["structure", "length", "arm", "trial", "recovery", "agreement"]
    t0 = time.time()

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()

        for i, rec in enumerate(records):
            try:
                struct = featurize_pdb(rec["path"], device=args.device, designed_chains=[rec["chain"]])
            except Exception as exc:  # noqa: BLE001
                print(f"[{i+1}/{len(records)}] {rec['id']}: featurize failed ({type(exc).__name__})")
                continue

            runner = Runner(model, struct, device=args.device)
            native = runner.native
            mask = runner.designable_mask
            rows = []

            def emit(arm, trial, seq, agreement=""):
                row = {"structure": rec["id"], "length": struct.length, "arm": arm, "trial": trial,
                       "recovery": sequence_recovery(seq, native, mask), "agreement": agreement}
                rows.append(row)
                w.writerow(row)
                return row["recovery"]

            for r in range(args.reps):
                base_seed = args.seed + r * 100

                # One joint draw, the standard estimator.
                emit("baseline", r,
                     refine.single_pass(runner, temperature=0.1,
                                        rng=np.random.default_rng(base_seed)).sequence)

                # Near-greedy: approximate joint mode.
                emit("greedy", r,
                     refine.single_pass(runner, temperature=0.001,
                                        rng=np.random.default_rng(base_seed)).sequence)

                # Marginal consensus at several temperatures and sample counts.
                n_list = args.n_samples_list or [args.n_samples]
                for temp in args.temperatures:
                  for n in n_list:
                    label = f"consensus_T{temp:g}" if len(n_list) == 1 else f"consensus_T{temp:g}_N{n}"
                    seq, mean_probs, agree = consensus_sequence(runner, n, temp, base_seed)
                    emit(label, r, seq, f"{agree:.4f}")

                    # Consensus as initialization for conditional ICM.
                    if abs(temp - 1.0) < 1e-9 and args.icm_sweeps:
                        cur = seq.copy()
                        rng = np.random.default_rng(base_seed)
                        for _ in range(args.icm_sweeps):
                            cur, changed = refine.gibbs_sweep(
                                runner, cur, None, temperature=0.001,
                                mode="sequential", greedy=True, rng=rng)
                            if changed == 0:
                                break
                        emit("consensus_then_icm", r, cur)

            fh.flush()
            base = np.mean([x["recovery"] for x in rows if x["arm"] == "baseline"])
            best_arm = max({a: np.mean([x["recovery"] for x in rows if x["arm"] == a])
                            for a in {x["arm"] for x in rows}}.items(), key=lambda kv: kv[1])
            print(f"[{i+1}/{len(records)}] {rec['id']:>22} L={struct.length:<4} "
                  f"base={base:.1%}  best={best_arm[0]}:{best_arm[1]:.1%}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
