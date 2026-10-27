#!/usr/bin/env python3
"""Compare four estimators of the same thing, under the metric that is actually reported.

Sequence recovery scores each position independently. Under a model p(S | X),
the rule that maximises expected per-position accuracy is the per-position
marginal mode:

    s_i = argmax_a p(s_i = a | X)

This is minimum Bayes risk decoding under Hamming loss. The estimators below
are four different ways of getting at it, or of not getting at it.

``ancestral``
    One autoregressive sample. A draw from the joint under one decoding order.
    This is what ProteinMPNN returns by default and what recovery numbers in
    the literature are usually computed from. It does not target the marginal
    mode at all.

``joint_mode``
    Near-greedy sampling, T=0.001. Approximates the joint mode (the MAP
    sequence) under one decoding order. A different estimator again, and the
    one Viterbi would give in an HMM.

``unconditional``
    ``argmax`` of ProteinMPNN's own ``unconditional_probs``. The function zeroes
    the decoder's attention mask, so every position is predicted from structure
    alone with no sequence context. It is a marginal estimate already present
    in the library, obtained in a single forward pass.

``consensus``
    ``argmax`` of per-position distributions averaged over N sampled decoding
    orders. Each contributing distribution carries the sequence context of its
    own prefix, so this is a Monte Carlo estimate of the marginal that
    integrates over sequence context and over decoding order.

The last two are both marginal estimators. The comparison between them is the
point of this script: ProteinMPNN has shipped a marginal decoder since release,
so the question is whether averaging context-carrying conditionals beats the
context-free one-shot version.

Usage::

    python experiments/run_estimator_comparison.py --set benchmark_set.json \\
        --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \\
        --n-samples 32 --reps 4 --out results/estimators.csv
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
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from ipr import refine  # noqa: E402
from ipr.metrics import sequence_recovery, to_numpy  # noqa: E402


def unconditional_sequence(runner):
    """argmax of ProteinMPNN's context-free per-position marginals.

    One forward pass, fully deterministic, no sampling involved.
    """
    import torch

    s = runner.structure
    with torch.no_grad():
        log_probs = runner.model.unconditional_probs(
            s.X, s.mask, s.residue_idx, s.chain_encoding_all
        )
    lp = to_numpy(log_probs)
    lp = lp[0] if lp.ndim == 3 else lp
    return lp.argmax(axis=-1).astype(np.int64), lp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--mpnn", type=Path, default=Path("ProteinMPNN"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--consensus-temperature", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.mpnn.resolve()))
    from ipr.backends import Runner, featurize_pdb, load_model
    from run_consensus import consensus_sequence

    records = json.loads(args.set.read_text())
    if args.limit:
        records = records[:args.limit]

    model, info = load_model(args.weights, device=args.device)
    print(f"weights: k_neighbors={info['k_neighbors']}")
    print(f"consensus: N={args.n_samples} at T={args.consensus_temperature}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["structure", "length", "arm", "trial", "recovery"]
    t0 = time.time()

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()

        for i, rec in enumerate(records):
            try:
                struct = featurize_pdb(rec["path"], device=args.device,
                                       designed_chains=[rec["chain"]])
            except Exception as exc:  # noqa: BLE001
                print(f"[{i+1}/{len(records)}] {rec['id']}: featurize failed ({type(exc).__name__})")
                continue

            runner = Runner(model, struct, device=args.device)
            native = runner.native
            mask = runner.designable_mask
            got: dict[str, list[float]] = {}

            def emit(arm, trial, seq):
                rv = sequence_recovery(seq, native, mask)
                w.writerow({"structure": rec["id"], "length": struct.length,
                            "arm": arm, "trial": trial, "recovery": rv})
                got.setdefault(arm, []).append(rv)

            # Deterministic, so it only needs computing once per structure.
            uncond_seq, _ = unconditional_sequence(runner)

            for t in range(args.reps):
                s = args.seed + t * 100
                emit("ancestral", t, refine.single_pass(
                    runner, temperature=0.1, rng=np.random.default_rng(s)).sequence)
                emit("joint_mode", t, refine.single_pass(
                    runner, temperature=0.001, rng=np.random.default_rng(s)).sequence)
                emit("unconditional", t, uncond_seq)
                seq, _, _ = consensus_sequence(runner, args.n_samples,
                                               args.consensus_temperature, s)
                emit("consensus", t, seq)

            fh.flush()
            line = "  ".join(f"{k}={np.mean(v):.1%}" for k, v in got.items())
            print(f"[{i+1}/{len(records)}] {rec['id']:>22} L={struct.length:<4} {line}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
