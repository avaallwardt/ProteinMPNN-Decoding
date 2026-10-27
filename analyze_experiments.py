#!/usr/bin/env python3
"""Analyse the output of run_experiments.py.

The unit of analysis is the structure, not the trial. Trials within a structure
share a backbone and are not independent observations of "does this method
work", so the procedure is:

1. average each arm's recovery over trials within a structure,
2. take the per-structure paired difference against the baseline arm,
3. summarise those differences across structures.

Confidence intervals come from a bootstrap over structures, which makes no
normality assumption and is honest about small structure counts.

Usage::

    python experiments/analyze_experiments.py --csv results/experiments.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BASELINE = "baseline"
N_BOOT = 20000


def load(path: Path):
    per = defaultdict(list)          # (structure, arm) -> [recovery]
    nll = defaultdict(list)          # (structure, arm) -> [nll]
    lengths: dict[str, int] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["structure"], row["arm"])
            per[key].append(float(row["recovery"]))
            if row.get("nll"):
                nll[key].append(float(row["nll"]))
            lengths[row["structure"]] = int(row["length"])
    return per, nll, lengths


def bootstrap_ci(deltas: np.ndarray, rng, alpha=0.05):
    if deltas.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, deltas.size, size=(N_BOOT, deltas.size))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--baseline", default=BASELINE)
    args = p.parse_args()

    per, nll, lengths = load(args.csv)
    structures = sorted({k[0] for k in per})
    arms = sorted({k[1] for k in per})
    rng = np.random.default_rng(0)

    print(f"structures: {len(structures)}   arms: {len(arms)}")
    lens = [lengths[s] for s in structures]
    print(f"chain length: min={min(lens)} median={int(np.median(lens))} max={max(lens)}\n")

    # Per-structure mean recovery for each arm.
    means = {
        arm: np.array([np.mean(per[(s, arm)]) if (s, arm) in per else np.nan for s in structures])
        for arm in arms
    }

    base = means[args.baseline]
    print(f"{'arm':<18} {'mean rec':>9} {'vs base':>9} {'95% CI':>19} {'t':>7} {'wins':>6}")
    print("-" * 76)

    order = [args.baseline] + [a for a in arms if a != args.baseline]
    results = {}
    for arm in order:
        vals = means[arm]
        ok = ~np.isnan(vals) & ~np.isnan(base)
        mean_rec = np.nanmean(vals)
        if arm == args.baseline:
            print(f"{arm:<18} {mean_rec:>8.2%} {'--':>9} {'--':>19} {'--':>7} {'--':>6}")
            continue

        d = vals[ok] - base[ok]
        lo, hi = bootstrap_ci(d, rng)
        se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float("nan")
        t = d.mean() / se if se and se > 0 else float("nan")
        wins = (d > 0).mean()
        results[arm] = (d.mean(), lo, hi, t, wins, d.size)
        print(f"{arm:<18} {mean_rec:>8.2%} {d.mean():>+8.2%} "
              f"[{lo:>+7.2%},{hi:>+7.2%}] {t:>+7.2f} {wins:>5.0%}")

    print("\nCI is a bootstrap over structures. An interval excluding zero is the")
    print("minimum bar for claiming an effect.\n")

    # Highlight anything that clears the bar.
    clear = {a: r for a, r in results.items() if r[1] > 0}
    if clear:
        print("arms whose 95% CI excludes zero (improvement):")
        for arm, (m, lo, hi, t, w, n) in sorted(clear.items(), key=lambda kv: -kv[1][0]):
            print(f"  {arm:<18} {m:+.2%}  CI [{lo:+.2%}, {hi:+.2%}]  n={n} structures")
    else:
        print("no arm's 95% CI excludes zero.")

    worse = {a: r for a, r in results.items() if r[2] < 0}
    if worse:
        print("\narms significantly worse than baseline:")
        for arm, (m, lo, hi, t, w, n) in sorted(worse.items(), key=lambda kv: kv[1][0]):
            print(f"  {arm:<18} {m:+.2%}  CI [{lo:+.2%}, {hi:+.2%}]  n={n} structures")

    # NLL table, where recorded.
    nll_arms = [a for a in order if any((s, a) in nll for s in structures)]
    if nll_arms:
        print("\nmean NLL (lower is better), scored under fixed decoding orders")
        print("-" * 50)
        for arm in nll_arms:
            vals = [np.mean(nll[(s, arm)]) for s in structures if (s, arm) in nll]
            if vals:
                print(f"  {arm:<18} {np.mean(vals):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
