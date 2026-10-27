"""Decoding-order strategies, and how to force one on ProteinMPNN.

ProteinMPNN does not take a decoding order directly in ``sample``. It derives
one from the noise vector you hand it::

    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))

Two things follow. First, only ``|randn|`` matters, so the order is fully
controllable: give position ``p`` the value ``rank(p) + 1`` and it decodes at
that rank. Second, fixed positions (``chain_mask == 0``) collapse to ~0.0001 and
sort to the very front regardless of ``randn``, which is how they become
context for everything else.

:func:`randn_for_order` does the first of those, which turns "try a different
decoding order" from a random-seed lottery into a controlled experiment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Order = np.ndarray  # shape (L,), a permutation of range(L)


def random_order(length: int, rng: np.random.Generator) -> Order:
    """A uniformly random permutation -- ProteinMPNN's default behaviour."""
    return rng.permutation(length)


def n_to_c(length: int, rng: np.random.Generator | None = None) -> Order:
    """Decode strictly from the N-terminus to the C-terminus."""
    return np.arange(length)


def c_to_n(length: int, rng: np.random.Generator | None = None) -> Order:
    """Decode strictly from the C-terminus to the N-terminus."""
    return np.arange(length)[::-1].copy()


def from_scores(scores: np.ndarray, descending: bool = True) -> Order:
    """Order positions by a per-position score.

    Parameters
    ----------
    scores:
        Array of shape ``(L,)``.
    descending:
        If True, the highest-scoring position decodes first.

    Ties are broken by index, so the result is deterministic.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    return np.argsort(-s if descending else s, kind="stable")


def confident_first(confidence: np.ndarray) -> Order:
    """Decode the positions the model is most sure about first.

    The motivating idea from the project notes: the residues whose identity is
    most strongly determined by the backbone alone should be laid down first so
    they can serve as stable context, instead of being decided late when they
    are already over-constrained by arbitrary early picks.
    """
    return from_scores(confidence, descending=True)


def uncertain_first(confidence: np.ndarray) -> Order:
    """Decode the least certain positions first -- the control for :func:`confident_first`.

    Worth running: if confident-first helps, uncertain-first should hurt by a
    comparable margin. If both help, the effect is not about ordering.
    """
    return from_scores(confidence, descending=False)


def core_first(neighbor_counts: np.ndarray) -> Order:
    """Decode buried positions before exposed ones.

    From the notes: "If we held those things stable (probably in the core) that
    are fixed". Buried positions have the most constrained chemistry, so they
    are the natural anchors.
    """
    return from_scores(neighbor_counts, descending=True)


def surface_first(neighbor_counts: np.ndarray) -> Order:
    """Decode exposed positions before buried ones -- the control for :func:`core_first`."""
    return from_scores(neighbor_counts, descending=False)


def blocked_random(length: int, rng: np.random.Generator, block_size: int = 8) -> Order:
    """Shuffle contiguous blocks rather than individual residues.

    Keeps local sequence context intact while still varying global order, which
    separates "the order of nearby residues matters" from "the order of distant
    residues matters".
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    blocks = [np.arange(i, min(i + block_size, length)) for i in range(0, length, block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks) if blocks else np.arange(0)


def neighbor_counts(coords, cutoff: float = 10.0) -> np.ndarray:
    """Count CA atoms within ``cutoff`` angstroms of each residue's CA.

    Parameters
    ----------
    coords:
        Backbone coordinates shaped ``(L, 4, 3)`` (N, CA, C, O) or ``(B, L, 4, 3)``,
        or CA-only coordinates shaped ``(L, 3)``.
    cutoff:
        Distance cutoff in angstroms.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(L,)``. Self is excluded from the count.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[:, 1, :]  # CA is index 1 in the N/CA/C/O layout
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(f"could not interpret coords of shape {np.shape(coords)}")

    finite = np.isfinite(arr).all(axis=-1)
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    within = (dist < cutoff) & finite[:, None] & finite[None, :]
    np.fill_diagonal(within, False)
    return within.sum(axis=1).astype(np.int64)


def randn_for_order(order: Order, length: int | None = None):
    """Build the ``randn`` vector that makes ProteinMPNN decode in ``order``.

    Parameters
    ----------
    order:
        A permutation of ``range(L)``; ``order[k]`` is the position decoded at
        rank ``k``.
    length:
        Sequence length. Defaults to ``len(order)``.

    Returns
    -------
    torch.Tensor
        Shape ``(1, L)``, float32, suitable to pass as ``randn`` to
        ``model.sample``.

    Notes
    -----
    Values start at 1.0 rather than 0.0 so that no designable position can tie
    with the ~0.0001 that fixed positions collapse to.

    This controls the order *among designable positions only*. Fixed positions
    are forced to the front by ``chain_mask`` and cannot be reordered.
    """
    import torch

    order = np.asarray(order, dtype=np.int64).ravel()
    length = len(order) if length is None else length
    if not _is_permutation(order, length):
        raise ValueError("order must be a permutation of range(length)")

    randn = np.empty(length, dtype=np.float32)
    randn[order] = np.arange(1, length + 1, dtype=np.float32)
    return torch.from_numpy(randn).unsqueeze(0)


def order_from_randn(randn, chain_mask=None) -> Order:
    """Recover the decoding order ProteinMPNN would use from a ``randn`` vector.

    Mirrors the library's own ``argsort((chain_mask + 0.0001) * |randn|)``, which
    makes it possible to verify that a forced order was applied as intended.
    """
    r = np.abs(np.asarray(_detach(randn), dtype=np.float64)).ravel()
    cm = np.ones_like(r) if chain_mask is None else np.asarray(_detach(chain_mask), dtype=np.float64).ravel()
    return np.argsort((cm + 0.0001) * r, kind="stable")


STRATEGIES: dict[str, Callable] = {
    "random": random_order,
    "n_to_c": n_to_c,
    "c_to_n": c_to_n,
    "blocked_random": blocked_random,
}
"""Strategies that need only a length and an RNG.

Score-dependent strategies (:func:`core_first`, :func:`confident_first` and
their controls) are not listed here because they need per-position inputs; call
them directly.
"""


def _is_permutation(order: np.ndarray, length: int) -> bool:
    return order.shape == (length,) and np.array_equal(np.sort(order), np.arange(length))


def _detach(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return x.numpy() if hasattr(x, "numpy") else x
