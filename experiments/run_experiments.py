#!/usr/bin/env python3
"""Run the candidate routes to a real recovery improvement, paired across structures.

Arms
----
``baseline``
    Single ProteinMPNN pass, neighbour count read from the checkpoint (48).
``k64``
    Identical, but the model is built with ``k_neighbors=64`` against weights
    trained at 48 -- the configuration used in the original notebook. The
    paired difference against ``baseline`` measures what that mismatch cost.
``temp_*``
    Baseline at other sampling temperatures.
``best_of_N``
    Draw N samples, keep the one the model scores best by NLL. Selection uses
    only the model, never the native sequence, so it is a legitimate method
    rather than selection on the answer.
``entropy_refine`` / ``random_refine``
    Corrected conditional refinement: redesign the least certain residues while
    holding the rest as context. ``random_refine`` is the matched-budget
    control.
``core_first`` / ``random_order``
    Decoding-order comparison at matched temperature.

All arms share seeds per structure, so comparisons are paired and the
trial-to-trial variance from the random decoding order cancels.

Usage::

    python experiments/run_experiments.py --set benchmark_set.json \\
        --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \\
        --trials 8 --out results/experiments.csv
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

from ipr import decoding_orders as do  # noqa: E402
from ipr import refine, uncertainty  # noqa: E402
from ipr.metrics import sequence_recovery, to_numpy  # noqa: E402

SCORE_SEED = 12345
N_SCORE_ORDERS = 3


def score_nll(runner, sequence) -> float:
    """Average NLL of a sequence over a fixed set of decoding orders.

    The orders are fixed by ``SCORE_SEED`` so every arm is scored identically.
    Averaging over several orders reduces the order-dependence of the estimate,
    which matters because a single order gives a noisy read.
    """
    rng = np.random.default_rng(SCORE_SEED)
    vals = []
    for _ in range(N_SCORE_ORDERS):
        order = do.random_order(runner.length, rng)
        vals.append(runner.score(sequence, order=order)["nll"])
    return float(np.mean(vals))


def run_structure(runner_48, runner_64, record, args, writer, native):
    """Run every arm on one structure and write one row per (arm, trial)."""
    sid = record["id"]
    L = runner_48.length
    mask = runner_48.designable_mask
    rows = []

    def emit(arm, trial, seq, extra=None):
        row = {
            "structure": sid,
            "length": L,
            "arm": arm,
            "trial": trial,
            "recovery": sequence_recovery(seq, native, mask),
            "nll": extra.get("nll", "") if extra else "",
            "n_changed": extra.get("n_changed", "") if extra else "",
        }
        rows.append(row)
        writer.writerow(row)
        return row

    # ---- baseline (k=48) and the per-trial pool reused by best_of_N --------
    pool = []
    for t in range(args.trials):
        rng = np.random.default_rng(args.seed + t)
        seq = refine.single_pass(runner_48, temperature=args.temperature, rng=rng).sequence
        nll = score_nll(runner_48, seq)
        pool.append((seq, nll))
        emit("baseline", t, seq, {"nll": nll})

    # ---- best-of-N by model score ----------------------------------------
    # Independent replicates of "draw N, keep the best-scoring one".
    n = args.best_of
    n_reps = max(1, args.trials // n)
    for rep in range(n_reps):
        chunk = pool[rep * n:(rep + 1) * n]
        if len(chunk) < n:
            break
        best_seq, best_nll = min(chunk, key=lambda x: x[1])
        emit(f"best_of_{n}", rep, best_seq, {"nll": best_nll})

    # ---- neighbour-count mismatch ----------------------------------------
    if runner_64 is not None:
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + t)
            seq = refine.single_pass(runner_64, temperature=args.temperature, rng=rng).sequence
            emit("k64_mismatch", t, seq, {"nll": score_nll(runner_48, seq)})

    # ---- temperature sweep -----------------------------------------------
    for temp in args.temperatures:
        if abs(temp - args.temperature) < 1e-12:
            continue
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + t)
            seq = refine.single_pass(runner_48, temperature=temp, rng=rng).sequence
            emit(f"temp_{temp:g}", t, seq, {"nll": score_nll(runner_48, seq)})

    # ---- corrected conditional refinement --------------------------------
    for policy in ("entropy", "random"):
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + t)
            res = refine.uncertainty_guided(
                runner_48,
                temperature_draft=args.temperature,
                temperature_refine=args.refine_temperature,
                policy=policy,
                fraction=args.fraction,
                n_rounds=args.rounds,
                rng=rng,
            )
            emit(f"{policy}_refine", t, res.sequence,
                 {"nll": score_nll(runner_48, res.sequence), "n_changed": res.n_changed})

    # ---- decoding order --------------------------------------------------
    counts = runner_48.structure.neighbor_counts()
    for name, order in (("core_first", do.core_first(counts)),
                        ("surface_first", do.surface_first(counts)),
                        ("n_to_c", do.n_to_c(L))):
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + t)
            seq = refine.single_pass(runner_48, temperature=args.temperature, order=order, rng=rng).sequence
            emit(name, t, seq, {"nll": score_nll(runner_48, seq)})

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", type=Path, required=True, help="benchmark manifest from build_benchmark_set.py")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--mpnn", type=Path, default=Path("ProteinMPNN"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--temperatures", type=float, nargs="*", default=[0.3, 0.03, 0.01])
    p.add_argument("--refine-temperature", type=float, default=0.1)
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--best-of", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only run the first N structures")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.mpnn.resolve()))
    from ipr.backends import Runner, featurize_pdb, load_model

    records = json.loads(args.set.read_text())
    if args.limit:
        records = records[:args.limit]

    model48, info = load_model(args.weights, device=args.device)
    print(f"weights: k_neighbors={info['k_neighbors']} (from checkpoint)")

    # The mismatched model: same weights, wrong graph size.
    from protein_mpnn_utils import ProteinMPNN
    import torch
    ckpt = torch.load(str(args.weights), map_location=args.device, weights_only=False)
    model64 = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                          hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                          augment_eps=0.0, k_neighbors=64)
    model64.load_state_dict(ckpt["model_state_dict"])
    model64.to(args.device).eval()
    print("built comparison model with k_neighbors=64 (notebook configuration)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["structure", "length", "arm", "trial", "recovery", "nll", "n_changed"]

    t0 = time.time()
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(records):
            try:
                struct = featurize_pdb(rec["path"], device=args.device, designed_chains=[rec["chain"]])
            except Exception as exc:  # noqa: BLE001
                print(f"[{i+1}/{len(records)}] {rec['id']}: featurize failed ({type(exc).__name__}), skipping")
                continue

            if struct.length != rec["length"]:
                print(f"[{i+1}/{len(records)}] {rec['id']}: length {struct.length} != manifest "
                      f"{rec['length']}, skipping")
                continue

            runner48 = Runner(model48, struct, device=args.device)
            runner64 = Runner(model64, struct, device=args.device)
            native = runner48.native

            rows = run_structure(runner48, runner64, rec, args, writer, native)
            fh.flush()

            base = np.mean([r["recovery"] for r in rows if r["arm"] == "baseline"])
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(records)}] {rec['id']:>14} L={struct.length:<4} "
                  f"baseline={base:.1%}  ({elapsed/60:.1f} min elapsed)")

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
