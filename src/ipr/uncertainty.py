"""Per-position uncertainty measures and residue-selection policies.

The refinement loop needs an answer to "which residues should I reconsider?".
This module supplies the scores and the selection rules, kept separate from the
refinement loop so they can be swapped and ablated independently.

A note on which score to use. The original notebook thresholded on the
probability of the *picked* token. That conflates two different situations: a
position where the model has a clear second choice (genuinely contested) and a
position where probability mass is spread thinly over many residues (genuinely
unconstrained). Entropy and margin separate them, and they disagree often
enough to be worth ablating.
"""

from __future__ import annotations

import numpy as np

from .metrics import to_numpy


def _as_probs(probs) -> np.ndarray:
    """Normalize input to a ``(L, 21)`` probability array."""
    p = to_numpy(probs).astype(np.float64)
    if p.ndim == 3:
        p = p[0]
    if p.ndim != 2:
        raise ValueError(f"expected probabilities of shape (L, K), got {p.shape}")
    return p


def picked_probability(probs, sequence) -> np.ndarray:
    """Probability assigned to the token actually chosen at each position.

    This is the measure the original notebook used for its confidence threshold.
    """
    p = _as_probs(probs)
    seq = to_numpy(sequence).ravel()
    if seq.shape[0] != p.shape[0]:
        raise ValueError(f"length mismatch: probs {p.shape[0]} vs sequence {seq.shape[0]}")
    return p[np.arange(seq.shape[0]), seq]


def entropy(probs, normalize: bool = False) -> np.ndarray:
    """Shannon entropy of each position's distribution, in nats.

    Parameters
    ----------
    normalize:
        If True, divide by ``log(K)`` so the result lies in ``[0, 1]`` and is
        comparable across alphabets.

    High entropy means the model is spreading mass widely -- the position is
    weakly determined by its context.
    """
    p = _as_probs(probs)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=-1, keepdims=True)
    h = -(p * np.log(p)).sum(axis=-1)
    return h / np.log(p.shape[-1]) if normalize else h


def margin(probs) -> np.ndarray:
    """Gap between the top two probabilities at each position.

    A small margin means the model has a close runner-up: exactly the situation
    where extra sequence context is most likely to flip the decision, and so the
    most promising place to spend a refinement step.
    """
    p = _as_probs(probs)
    if p.shape[-1] < 2:
        return np.zeros(p.shape[0])
    top2 = np.partition(p, -2, axis=-1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def max_probability(probs) -> np.ndarray:
    """Probability of each position's most likely token, regardless of what was picked."""
    return _as_probs(probs).max(axis=-1)


def select_by_threshold(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """Positions whose confidence falls strictly below ``threshold``.

    .. warning::
       Threshold selection has no control over how many positions it returns.
       In the original notebook a threshold of 0.80 unlocked all 76 residues of
       ubiquitin, which turned an intended "surgical polish" into a full
       redesign without that being visible in the output. :func:`select_top_k`
       avoids that failure mode. If you do use a threshold, always report the
       count it selected.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    return np.flatnonzero(conf < threshold)


def select_top_k(uncertainty: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` most uncertain positions, highest uncertainty first.

    Pass a score where larger means less certain (entropy, or negated
    confidence). Ties break by index for determinism.
    """
    unc = np.asarray(uncertainty, dtype=np.float64).ravel()
    k = int(max(0, min(k, unc.shape[0])))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argsort(-unc, kind="stable")[:k]
    return np.sort(idx)


def select_fraction(uncertainty: np.ndarray, fraction: float) -> np.ndarray:
    """The most uncertain ``fraction`` of positions."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    n = np.asarray(uncertainty).ravel().shape[0]
    return select_top_k(uncertainty, int(round(n * fraction)))


def select_random(length: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random subset of ``k`` positions.

    The control arm for every uncertainty-guided policy. If guided selection
    does not beat this at matched ``k``, the guidance is doing nothing and the
    gain is coming from the extra sampling alone.
    """
    k = int(max(0, min(k, length)))
    return np.sort(rng.choice(length, size=k, replace=False)) if k else np.empty(0, dtype=np.int64)


def early_decoded(decoding_order, fraction: float) -> np.ndarray:
    """Positions decoded in the first ``fraction`` of a decoding order.

    These are the residues the project notes identify as the problem: decided
    with little or no sequence context, then never revisited. Targeting them is
    the most direct test of the compounding-error hypothesis.
    """
    order = to_numpy(decoding_order).ravel().astype(np.int64)
    cutoff = int(round(order.shape[0] * fraction))
    return np.sort(order[:cutoff]) if cutoff else np.empty(0, dtype=np.int64)


POLICIES = ("entropy", "margin", "picked_prob", "random", "early")
"""Names accepted by :func:`score_positions`."""


def score_positions(policy: str, *, probs=None, sequence=None, decoding_order=None) -> np.ndarray:
    """Return an uncertainty score per position under the named ``policy``.

    Larger values always mean *less* certain, so the result can be passed
    straight to :func:`select_top_k` whichever policy produced it.
    """
    if policy == "entropy":
        return entropy(probs)
    if policy == "margin":
        return -margin(probs)
    if policy == "picked_prob":
        return -picked_probability(probs, sequence)
    if policy == "early":
        order = to_numpy(decoding_order).ravel().astype(np.int64)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        return -rank.astype(np.float64)  # earliest-decoded scores highest
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
