"""Amino-acid alphabet, matched to ProteinMPNN's ordering.

ProteinMPNN encodes sequences as integers over a 21-letter alphabet where
index 20 ('X') is the gap/unknown token. Every array in this package that has
a trailing dimension of 21 is laid out in this order.
"""

from __future__ import annotations

import numpy as np

ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"
N_TOKENS: int = len(ALPHABET)  # 21
GAP_INDEX: int = ALPHABET.index("X")  # 20

AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(ALPHABET)}
IDX_TO_AA: dict[int, str] = {i: aa for i, aa in enumerate(ALPHABET)}

# Hydrophobic set, used by the burial-stratified recovery metrics.
HYDROPHOBIC: frozenset[str] = frozenset("AVILMFWYC")


def encode(seq: str) -> np.ndarray:
    """Encode a one-letter sequence string to an int array of shape (L,).

    Unknown characters map to the gap index rather than raising, which keeps
    PDB files with noncanonical residues usable.
    """
    return np.array([AA_TO_IDX.get(c.upper(), GAP_INDEX) for c in seq], dtype=np.int64)


def decode(idx: np.ndarray) -> str:
    """Decode an int array of shape (L,) back to a one-letter string."""
    return "".join(IDX_TO_AA[int(i)] for i in np.asarray(idx).ravel())


def one_hot(idx: np.ndarray, n_tokens: int = N_TOKENS) -> np.ndarray:
    """One-hot encode an int array of shape (L,) to shape (L, n_tokens)."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.zeros((idx.shape[0], n_tokens), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out
