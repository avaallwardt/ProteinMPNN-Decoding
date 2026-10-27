#!/usr/bin/env python3
"""Re-analyse the results recorded in the original Colab notebook.

The numbers in ``notes/notebook_results.csv`` were transcribed verbatim from the
cell outputs of ``proteinmpnn_w_edits_3_24_26.ipynb``. Nothing here re-runs the
model; this script only computes the aggregate statistics that the notebook
reported trial-by-trial, so the claims in ``docs/findings.md`` can be checked
against the raw data.

Usage::

    python experiments/analyze_notebook_results.py
    python experiments/analyze_notebook_results.py --csv path/to/results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ipr.metrics import paired_difference, summarize  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "notes" / "notebook_results.csv"


def load(csv_path: Path) -> dict[tuple[str, str, str], list[float]]:
    """Group recorded values by (experiment, arm, metric), ordered by trial."""
    rows = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    rows.sort(key=lambda r: int(r["trial"]))

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["arm"], row["metric"])].append(float(row["value"]))
    return grouped


def fmt(value: float, metric: str) -> str:
    return f"{value:.2%}" if metric == "recovery" else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="results CSV to analyse")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"error: {args.csv} not found", file=sys.stderr)
        return 1

    grouped = load(args.csv)
    experiments = sorted({key[0] for key in grouped})

    print("Recorded results from proteinmpnn_w_edits_3_24_26.ipynb")
    print("Protein: 1UBQ (ubiquitin, 76 residues), single structure throughout.")
    print("=" * 78)

    for exp in experiments:
        arms = sorted({key[1] for key in grouped if key[0] == exp})
        metrics = sorted({key[2] for key in grouped if key[0] == exp})
        print(f"\n{exp}")
        print("-" * 78)

        for metric in metrics:
            better = "higher" if metric == "recovery" else "lower"
            print(f"  metric: {metric} ({better} is better)")
            for arm in arms:
                values = grouped.get((exp, arm, metric))
                if not values:
                    continue
                stats = summarize(values)
                print(
                    f"    {arm:<24} n={stats['n']:<3} "
                    f"mean={fmt(stats['mean'], metric)}  "
                    f"sd={fmt(stats['std'], metric)}  "
                    f"range=[{fmt(stats['min'], metric)}, {fmt(stats['max'], metric)}]"
                )

            # Paired comparison against the standard arm, where one exists.
            baseline_arm = "standard" if "standard" in arms else ("baseline" if "baseline" in arms else None)
            if baseline_arm is None:
                continue
            base = grouped.get((exp, baseline_arm, metric))
            for arm in arms:
                if arm == baseline_arm:
                    continue
                other = grouped.get((exp, arm, metric))
                if not other or len(other) != len(base):
                    continue
                # For recovery, positive difference favours the arm; for NLL it is the reverse.
                stats = (
                    paired_difference(other, base) if metric == "recovery"
                    else paired_difference(base, other)
                )
                direction = f"{arm} vs {baseline_arm}"
                print(
                    f"\n    paired: {direction}\n"
                    f"      mean difference favouring {arm}: {fmt(stats['mean_difference'], metric)}\n"
                    f"      sd of differences:               {fmt(stats['std_difference'], metric)}\n"
                    f"      standard error:                  {fmt(stats['stderr'], metric)}\n"
                    f"      t statistic:                     {stats['t_statistic']:.3f}\n"
                    f"      win rate for {arm}: {stats['win_rate']:.0%}"
                )

    print("\n" + "=" * 78)
    print(
        "Reading:\n"
        "  - On recovery, the per-trial spread is several times the mean difference in\n"
        "    both benchmarks (|t| = 0.30 and 0.10). Neither is distinguishable from the\n"
        "    single-pass baseline.\n"
        "  - On NLL, the two-pass arms are no better than baseline and trend worse. The\n"
        "    0.1 -> 0.1 control reaches |t| = 2.43, nominally worse than baseline, but\n"
        "    that is one marginal result among several comparisons on a single structure\n"
        "    and should not be read as an effect without replication.\n"
        "  - Every arm here redesigns all designable positions in its second pass, so\n"
        "    none of them conditions on its draft. See docs/findings.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
