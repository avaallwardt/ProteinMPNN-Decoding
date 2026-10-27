#!/usr/bin/env python3
"""Paired benchmark of refinement strategies against the single-pass baseline.

Every arm runs on the same seeds, so trial-to-trial variance from the random
decoding order cancels in the paired comparison. That matters more than it
sounds: in the original notebook the per-trial spread in sequence recovery was
roughly 4%, which is several times any effect being looked for. Unpaired
comparisons at n=5 cannot see through that.

Usage::

    # ProteinMPNN must be importable and weights present
    python experiments/run_benchmark.py \\
        --pdb 1UBQ.pdb \\
        --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \\
        --trials 20 \\
        --out results/benchmark_1ubq.csv

    # Dry run on the toy model, no weights or PyTorch needed
    python experiments/run_benchmark.py --toy --trials 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ipr import refine  # noqa: E402
from ipr.metrics import paired_difference, sequence_recovery, summarize  # noqa: E402


def build_arms(args) -> dict:
    """Map arm name to a callable taking ``(runner, rng)`` and returning a result."""
    t1, t2 = args.temperature_draft, args.temperature_refine

    arms = {
        # Control: plain ProteinMPNN.
        "single_pass": lambda r, rng: refine.single_pass(r, temperature=t2, rng=rng),

        # The original method, which does not condition on its draft. Included
        # so its equivalence to the baseline is demonstrated, not asserted.
        "total_refresh": lambda r, rng: refine.total_refresh(r, t1, t2, rng=rng),

        # Corrected refinement, uncertainty-guided.
        "entropy_guided": lambda r, rng: refine.uncertainty_guided(
            r, t2, t2, policy="entropy", fraction=args.fraction, n_rounds=args.rounds, rng=rng),
        "margin_guided": lambda r, rng: refine.uncertainty_guided(
            r, t2, t2, policy="margin", fraction=args.fraction, n_rounds=args.rounds, rng=rng),

        # Targets the residues the project notes blame: those decoded first.
        "early_decoded": lambda r, rng: refine.uncertainty_guided(
            r, t2, t2, policy="early", fraction=args.fraction, n_rounds=args.rounds, rng=rng),

        # Matched-budget control. If a guided policy cannot beat this, the
        # guidance is contributing nothing and any gain is from extra sampling.
        "random_guided": lambda r, rng: refine.uncertainty_guided(
            r, t2, t2, policy="random", fraction=args.fraction, n_rounds=args.rounds, rng=rng),

        # Full iterated conditional modes to a fixed point.
        "icm_fixed_point": lambda r, rng: refine.iterate_to_fixed_point(
            r, t1, t2, max_sweeps=args.max_sweeps, mode="sequential", greedy=True, rng=rng),
    }
    if args.only:
        missing = set(args.only) - set(arms)
        if missing:
            raise SystemExit(f"unknown arm(s): {sorted(missing)}; available: {sorted(arms)}")
        arms = {k: v for k, v in arms.items() if k in args.only}
    return arms


def make_runner(args):
    """Build either a real ProteinMPNN runner or the toy stand-in."""
    if args.toy:
        from ipr.toy import ToyPottsModel

        model = ToyPottsModel(length=args.toy_length, seed=args.seed)
        return model, model.native, "toy_potts"

    if not args.pdb or not args.weights:
        raise SystemExit("--pdb and --weights are required unless --toy is given")

    from ipr.backends import Runner, featurize_pdb, load_model

    model, info = load_model(args.weights, device=args.device, backbone_noise=args.backbone_noise)
    print(f"loaded weights: k_neighbors={info['k_neighbors']}, training_noise={info['training_noise']}")
    structure = featurize_pdb(args.pdb, device=args.device)
    runner = Runner(model, structure, device=args.device)
    print(f"structure {structure.name}: length={structure.length}, designable={structure.n_designable}")
    return runner, runner.native, structure.name


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", type=Path, help="input PDB file")
    p.add_argument("--weights", type=Path, help="ProteinMPNN checkpoint (.pt)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--toy", action="store_true", help="run against the Potts stand-in instead")
    p.add_argument("--toy-length", type=int, default=76)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature-draft", type=float, default=0.5)
    p.add_argument("--temperature-refine", type=float, default=0.1)
    p.add_argument("--fraction", type=float, default=0.25, help="share of residues reconsidered per round")
    p.add_argument("--rounds", type=int, default=2, help="refinement rounds for guided arms")
    p.add_argument("--max-sweeps", type=int, default=8)
    p.add_argument("--backbone-noise", type=float, default=0.0)
    p.add_argument("--only", nargs="*", help="restrict to these arm names")
    p.add_argument("--out", type=Path, help="write per-trial results to this CSV")
    args = p.parse_args()

    runner, native, name = make_runner(args)
    arms = build_arms(args)

    print(f"\nrunning {len(arms)} arms x {args.trials} paired trials on {name}")
    print(f"temperatures: draft={args.temperature_draft}, refine={args.temperature_refine}; "
          f"fraction={args.fraction}, rounds={args.rounds}\n")

    records: list[dict] = []
    recoveries: dict[str, list[float]] = {arm: [] for arm in arms}

    for trial in range(args.trials):
        for arm, fn in arms.items():
            # Same seed per trial across arms: this is what makes the comparison paired.
            rng = np.random.default_rng(args.seed + trial)
            result = fn(runner, rng)
            rec = sequence_recovery(result.sequence, native, getattr(runner, "designable_mask", None))
            recoveries[arm].append(rec)
            records.append({
                "structure": name,
                "trial": trial,
                "arm": arm,
                "recovery": rec,
                "n_redesigned": result.n_redesigned,
                "n_changed": result.n_changed,
                "rounds": result.rounds,
                "converged": result.converged,
                "contraction_factor": refine.contraction_factor(result.trajectory),
            })
        print(f"  trial {trial + 1}/{args.trials} done", end="\r", flush=True)

    print("\n\nper-arm recovery")
    print("-" * 72)
    for arm, values in recoveries.items():
        s = summarize(values)
        print(f"  {arm:<20} mean={s['mean']:.2%}  sd={s['std']:.2%}  range=[{s['min']:.2%}, {s['max']:.2%}]")

    baseline = recoveries.get("single_pass")
    if baseline and len(arms) > 1:
        print("\npaired vs single_pass")
        print("-" * 72)
        for arm, values in recoveries.items():
            if arm == "single_pass":
                continue
            st = paired_difference(values, baseline)
            flag = "" if abs(st["t_statistic"]) >= 2 else "   (not distinguishable)"
            print(f"  {arm:<20} delta={st['mean_difference']:+.2%}  "
                  f"se={st['stderr']:.2%}  t={st['t_statistic']:+.2f}  "
                  f"win={st['win_rate']:.0%}{flag}")
        print(
            "\n  |t| < 2 means the arm is not distinguishable from baseline at this sample\n"
            "  size. One structure is one structure: a result here is a pilot, not a claim."
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {len(records)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
