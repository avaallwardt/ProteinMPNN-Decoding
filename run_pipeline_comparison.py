#!/usr/bin/env python3
"""End-to-end comparison: the original pipeline versus the corrected one.

The individual effects measured elsewhere in this repository are each small.
What a user of the pipeline actually experiences is their combination, and
combined effects are not guaranteed to be additive -- they have to be measured
together.

Arms
----
``original``
    The configuration from the original notebook: model built with
    ``k_neighbors=64`` against weights trained at 48, one sampling pass at
    T=0.1.
``corrected``
    Neighbour count read from the checkpoint, single pass at T=0.03.
``corrected_consensus``
    As ``corrected``, but decoding by marginal consensus: average the
    per-position distributions over N samples at T=0.5 and take the argmax.

Reporting both ``corrected`` and ``corrected_consensus`` separates the part of
the gain that comes from fixing a bug from the part that comes from the
decoding method. Quoting only the combined number would blur them.

Usage::

    python experiments/run_pipeline_comparison.py --set benchmark_set.json \\
        --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \\
        --n-samples 32 --reps 3 --out results/pipeline_comparison.csv
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
from ipr.metrics import sequence_recovery  # noqa: E402

sys.path.insert(0, str(Path("ProteinMPNN").resolve()))

from run_consensus import consensus_sequence  # noqa: E402


def build_mismatched_model(weights, device):
    """The notebook's model: correct weights, wrong neighbour count."""
    import torch
    from protein_mpnn_utils import ProteinMPNN

    ckpt = torch.load(str(weights), map_location=device, weights_only=False)
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                        augment_eps=0.0, k_neighbors=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--mpnn", type=Path, default=Path("ProteinMPNN"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--consensus-temperature", type=float, default=0.5)
    p.add_argument("--corrected-temperature", type=float, default=0.03)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.mpnn.resolve()))
    from ipr.backends import Runner, featurize_pdb, load_model

    records = json.loads(args.set.read_text())
    if args.limit:
        records = records[:args.limit]

    model48, info = load_model(args.weights, device=args.device)
    model64 = build_mismatched_model(args.weights, args.device)
    print(f"corrected model: k_neighbors={info['k_neighbors']}; original model: k_neighbors=64")
    print(f"consensus over {args.n_samples} samples at T={args.consensus_temperature}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["structure", "length", "arm", "trial", "recovery"]
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

            r48 = Runner(model48, struct, device=args.device)
            r64 = Runner(model64, struct, device=args.device)
            native = r48.native
            mask = r48.designable_mask
            got = {}

            def emit(arm, trial, seq):
                rv = sequence_recovery(seq, native, mask)
                w.writerow({"structure": rec["id"], "length": struct.length,
                            "arm": arm, "trial": trial, "recovery": rv})
                got.setdefault(arm, []).append(rv)

            for t in range(args.reps):
                s = args.seed + t * 100

                emit("original", t, refine.single_pass(
                    r64, temperature=0.1, rng=np.random.default_rng(s)).sequence)

                emit("corrected", t, refine.single_pass(
                    r48, temperature=args.corrected_temperature,
                    rng=np.random.default_rng(s)).sequence)

                seq, _, _ = consensus_sequence(r48, args.n_samples, args.consensus_temperature, s)
                emit("corrected_consensus", t, seq)

            fh.flush()
            o, c, cc = (np.mean(got[k]) for k in ("original", "corrected", "corrected_consensus"))
            print(f"[{i+1}/{len(records)}] {rec['id']:>22} L={struct.length:<4} "
                  f"orig={o:.1%} corr={c:.1%} consensus={cc:.1%} "
                  f"(delta {cc-o:+.1%})  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
