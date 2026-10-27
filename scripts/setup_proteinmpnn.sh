#!/usr/bin/env bash
# Fetch ProteinMPNN and verify the pieces this repository needs are present.
#
# ProteinMPNN is MIT licensed but is not vendored here, so that this repository
# tracks upstream rather than pinning a stale copy.
set -euo pipefail

REPO_URL="https://github.com/dauparas/ProteinMPNN.git"
TARGET="${1:-ProteinMPNN}"
WEIGHTS="vanilla_model_weights/v_48_020.pt"

if [[ -d "$TARGET" ]]; then
    echo "==> $TARGET already present, skipping clone"
else
    echo "==> cloning ProteinMPNN into $TARGET"
    git clone --depth 1 "$REPO_URL" "$TARGET"
fi

echo "==> checking for model weights"
if [[ -f "$TARGET/$WEIGHTS" ]]; then
    echo "    found $TARGET/$WEIGHTS"
else
    echo "    WARNING: $TARGET/$WEIGHTS not found."
    echo "    The vanilla weights ship with the repository; if they are missing,"
    echo "    check that git-lfs is installed and run 'git lfs pull' in $TARGET."
fi

echo
echo "==> to use the model, put ProteinMPNN on your import path:"
echo "    export PYTHONPATH=\"\$PWD/$TARGET:\$PYTHONPATH\""
echo
echo "==> the '48' in v_48_020 is the neighbour count the weights were trained"
echo "    with, and the '020' is the training backbone noise in angstroms."
echo "    ipr.backends.load_model reads the neighbour count from the checkpoint"
echo "    rather than hardcoding it -- see docs/findings.md section 4."
echo
echo "==> a test structure, if you want one:"
echo "    mkdir -p inputs && curl -o inputs/1UBQ.pdb https://files.rcsb.org/download/1UBQ.pdb"
