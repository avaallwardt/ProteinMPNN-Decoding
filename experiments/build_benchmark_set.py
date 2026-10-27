#!/usr/bin/env python3
"""Assemble a benchmark set of clean single chains from locally available PDB files.

Sequence recovery is only interpretable on well-resolved, standard-residue
chains. This script scans candidate PDB files, keeps the chains that qualify,
drops duplicates by sequence, and writes a manifest.

Filters applied:

- chain length within ``--min-length`` .. ``--max-length``
- no unknown residues (ProteinMPNN's 'X' token) in the chain
- all backbone coordinates present and finite
- low-complexity chains rejected (a single residue type must not dominate)
- exact-duplicate sequences collapsed, keeping the first occurrence

Usage::

    python experiments/build_benchmark_set.py --out benchmark_set.json \\
        --search ProteinMPNN/inputs prody_src biopython_src/Tests pdbfixer_src
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def extract_chain(pdb_path: Path, chain: str, out_dir: Path) -> Path:
    """Write a single-chain PDB containing only the first model of ``chain``.

    ProteinMPNN's ``tied_featurize`` requires every chain in a parsed structure
    to be listed as either designable or visible. Designing one chain of a
    complex in isolation is simpler to set up, and more importantly it is the
    standard monomer design task, so each chain gets its own file.

    Only ATOM records are kept. HETATM records (ligands, waters, modified
    residues) are dropped, as are all models after the first.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_path.stem}_{chain}.pdb"

    lines = []
    in_later_model = False
    for line in pdb_path.read_text(errors="replace").splitlines():
        record = line[:6]
        if record == "MODEL ":
            # Everything after the first ENDMDL is a duplicate conformer.
            if in_later_model:
                break
            continue
        if record == "ENDMDL":
            in_later_model = True
            continue
        if record == "ATOM  " and len(line) > 21 and line[21] == chain:
            lines.append(line)

    lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def iter_candidates(roots: list[Path]) -> list[Path]:
    seen: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"  (skipping missing path {root})")
            continue
        seen.extend(sorted(root.rglob("*.pdb")))
    return seen


def chain_records(pdb_path: Path, parse_PDB):
    """Yield (chain_letter, sequence, n_residues) for each chain in a PDB file."""
    try:
        entries = parse_PDB(str(pdb_path))
    except Exception as exc:  # noqa: BLE001 - candidate files are untrusted input
        return [], f"parse failed: {type(exc).__name__}"
    if not entries:
        return [], "no entries"

    entry = entries[0]
    out = []
    for key, value in entry.items():
        if key.startswith("seq_chain_"):
            letter = key.removeprefix("seq_chain_")
            coords = entry.get(f"coords_chain_{letter}")
            out.append((letter, value, coords))
    return out, None


def coords_ok(coords: dict, letter: str) -> bool:
    """True if every backbone atom of the chain has finite coordinates."""
    if not coords:
        return False
    for atom in ("N", "CA", "C", "O"):
        key = f"{atom}_chain_{letter}"
        if key not in coords:
            return False
        arr = np.asarray(coords[key], dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            return False
    return True


def low_complexity(seq: str, max_single_fraction: float = 0.45) -> bool:
    """True if one residue type dominates the chain."""
    if not seq:
        return True
    counts = Counter(seq)
    return counts.most_common(1)[0][1] / len(seq) > max_single_fraction


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search", nargs="+", type=Path, required=True, help="directories to scan")
    p.add_argument("--mpnn", type=Path, default=Path("ProteinMPNN"), help="ProteinMPNN checkout")
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument("--max-length", type=int, default=250)
    p.add_argument("--max-chains", type=int, default=40)
    p.add_argument("--extract-dir", type=Path, default=Path("benchmark_chains"),
                   help="directory for the extracted single-chain PDB files")
    p.add_argument("--out", type=Path, default=Path("benchmark_set.json"))
    args = p.parse_args()

    sys.path.insert(0, str(args.mpnn.resolve()))
    from protein_mpnn_utils import parse_PDB  # noqa: E402

    candidates = iter_candidates(args.search)
    print(f"scanning {len(candidates)} candidate PDB files")

    kept: list[dict] = []
    seen_sequences: dict[str, str] = {}
    rejected = Counter()

    for path in candidates:
        chains, err = chain_records(path, parse_PDB)
        if err:
            rejected[err] += 1
            continue
        for letter, seq, coords in chains:
            if not isinstance(seq, str):
                rejected["sequence not a string"] += 1
                continue
            if not (args.min_length <= len(seq) <= args.max_length):
                rejected["length out of range"] += 1
                continue
            if "X" in seq.upper():
                rejected["unknown residues"] += 1
                continue
            if not coords_ok(coords, letter):
                rejected["missing or non-finite coordinates"] += 1
                continue
            if low_complexity(seq):
                rejected["low complexity"] += 1
                continue
            if seq in seen_sequences:
                rejected["duplicate sequence"] += 1
                continue

            seen_sequences[seq] = f"{path.name}:{letter}"
            kept.append({
                "id": f"{path.stem}_{letter}",
                "path": str(path),
                "chain": letter,
                "length": len(seq),
                "sequence": seq,
            })

    # Extract each surviving chain to its own file and re-verify it round-trips.
    verified = []
    for rec in kept:
        try:
            single = extract_chain(Path(rec["path"]), rec["chain"], args.extract_dir)
            entries = parse_PDB(str(single))
            seq = entries[0].get(f"seq_chain_{rec['chain']}") if entries else None
            if seq is None or len(seq) != rec["length"] or "X" in seq.upper():
                rejected["chain extraction changed the sequence"] += 1
                continue
            rec["source_path"] = rec["path"]
            rec["path"] = str(single)
            verified.append(rec)
        except Exception:  # noqa: BLE001
            rejected["chain extraction failed"] += 1
    kept = verified

    kept.sort(key=lambda r: r["length"])
    if len(kept) > args.max_chains:
        # Spread the selection across the length range rather than taking the shortest.
        idx = np.linspace(0, len(kept) - 1, args.max_chains).round().astype(int)
        kept = [kept[i] for i in sorted(set(idx))]

    print(f"\nkept {len(kept)} chains")
    print("rejections:")
    for reason, n in rejected.most_common():
        print(f"  {n:4d}  {reason}")

    if kept:
        lengths = [r["length"] for r in kept]
        print(f"\nlength: min={min(lengths)} median={int(np.median(lengths))} max={max(lengths)}")
        print("chains: " + ", ".join(r["id"] for r in kept))

    args.out.write_text(json.dumps(kept, indent=2))
    print(f"\nwrote {args.out}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
