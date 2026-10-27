"""Evaluation metrics: sequence recovery, likelihood, and uncertainty summaries.

All functions accept either NumPy arrays or PyTorch tensors and return plain
Python floats or NumPy arrays, so they can be used from the toy backend without
PyTorch installed.
"""

from __future__ import annotations

import numpy as np

from .alphabet import ALPHABET, HYDROPHOBIC


def to_numpy(x) -> np.ndarray:
    """Convert a tensor or array-like to a detached NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def sequence_recovery(predicted, native, mask=None) -> float:
    """Fraction of masked positions where the prediction matches the native residue.

    Parameters
    ----------
    predicted, native:
        Integer token arrays of the same shape.
    mask:
        Optional array of the same shape; positions where it is nonzero are
        scored. Defaults to scoring every position.

    Returns
    -------
    float
        Recovery in ``[0, 1]``. Returns ``nan`` if the mask selects nothing,
        rather than dividing by zero.
    """
    pred = to_numpy(predicted).ravel()
    nat = to_numpy(native).ravel()
    if pred.shape != nat.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {nat.shape}")

    weights = np.ones_like(pred, dtype=np.float64) if mask is None else to_numpy(mask).ravel().astype(np.float64)
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float(((pred == nat).astype(np.float64) * weights).sum() / denom)


def per_position_match(predicted, native) -> np.ndarray:
    """Boolean array marking positions where prediction and native agree."""
    return to_numpy(predicted).ravel() == to_numpy(native).ravel()


def hamming_distance(a, b) -> int:
    """Number of positions at which two token sequences differ."""
    x, y = to_numpy(a).ravel(), to_numpy(b).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return int((x != y).sum())


def negative_log_likelihood(log_probs, sequence, mask=None) -> float:
    """Mean negative log-likelihood of ``sequence`` under ``log_probs``, in nats.

    Parameters
    ----------
    log_probs:
        Array of shape ``(L, 21)`` or ``(1, L, 21)`` of natural-log probabilities.
    sequence:
        Integer token array of shape ``(L,)`` or ``(1, L)``.
    mask:
        Optional per-position weights.
    """
    lp = to_numpy(log_probs)
    if lp.ndim == 3:
        lp = lp[0]
    seq = to_numpy(sequence).ravel()
    if lp.shape[0] != seq.shape[0]:
        raise ValueError(f"length mismatch: log_probs {lp.shape[0]} vs sequence {seq.shape[0]}")

    picked = lp[np.arange(seq.shape[0]), seq]
    weights = np.ones_like(picked, dtype=np.float64) if mask is None else to_numpy(mask).ravel().astype(np.float64)
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float(-(picked * weights).sum() / denom)


def perplexity(log_probs, sequence, mask=None) -> float:
    """Exponentiated mean NLL. Lower is better; 21.0 is a uniform-random model."""
    return float(np.exp(negative_log_likelihood(log_probs, sequence, mask)))


def recovery_by_burial(predicted, native, neighbor_counts, threshold: int = 16) -> dict[str, float]:
    """Split recovery into buried and exposed positions.

    Buried positions are where the model is normally most confident, so a
    refinement method that only helps on the surface will show up here. The
    default threshold of 16 neighbours within 10 A is a conventional cutoff for
    a buried residue.
    """
    counts = to_numpy(neighbor_counts).ravel()
    buried = counts >= threshold
    return {
        "buried": sequence_recovery(predicted, native, buried.astype(np.float64)),
        "exposed": sequence_recovery(predicted, native, (~buried).astype(np.float64)),
        "n_buried": int(buried.sum()),
        "n_exposed": int((~buried).sum()),
    }


def composition(sequence) -> dict[str, float]:
    """Amino-acid composition of a token sequence, plus hydrophobic fraction.

    A refinement pass that improves likelihood by collapsing onto a few
    favoured residues will show up as a skewed composition even when recovery
    looks fine, so this is worth reporting alongside recovery.
    """
    seq = to_numpy(sequence).ravel()
    total = max(1, seq.shape[0])
    counts = np.bincount(seq, minlength=len(ALPHABET))
    out = {ALPHABET[i]: float(counts[i] / total) for i in range(len(ALPHABET))}
    out["hydrophobic_fraction"] = float(
        sum(counts[ALPHABET.index(a)] for a in HYDROPHOBIC) / total
    )
    return out


def summarize(values) -> dict[str, float]:
    """Mean, standard deviation, min, max and n for a list of trial results.

    Uses the sample standard deviation (``ddof=1``) because these are trials
    drawn from a larger population of random decoding orders, not a census.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def paired_difference(a, b) -> dict[str, float]:
    """Paired comparison of two arms run on the same random seeds.

    Returns the mean paired difference ``a - b``, its standard error, a
    two-sided t-statistic, and how often ``a`` wins. Pairing matters here:
    trial-to-trial variance from the random decoding order is large relative to
    the effect being measured, and pairing removes it.

    The t-statistic is reported without a p-value on purpose -- with n around 10
    and a single protein, the honest reading is the effect size and its spread,
    not a significance verdict.
    """
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arms must have equal length: {x.shape} vs {y.shape}")
    d = x - y
    n = d.size
    mean = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    stderr = sd / np.sqrt(n) if n > 1 and sd > 0 else 0.0
    return {
        "n": int(n),
        "mean_difference": mean,
        "std_difference": sd,
        "stderr": float(stderr),
        "t_statistic": float(mean / stderr) if stderr > 0 else float("nan"),
        "win_rate": float((d > 0).mean()) if n else float("nan"),
    }
