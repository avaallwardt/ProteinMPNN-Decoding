"""Tests for metrics, uncertainty policies, and decoding-order construction."""

from __future__ import annotations

import numpy as np
import pytest

from ipr import decoding_orders as do
from ipr import metrics, uncertainty
from ipr.alphabet import ALPHABET, decode, encode, one_hot

torch = pytest.importorskip("torch", reason="order forcing needs PyTorch", exc_type=ImportError) if False else None


# ---------------------------------------------------------------- alphabet
def test_encode_decode_roundtrip():
    seq = "MQIFVKTLTGKTITLEVEPSDTIENVK"
    assert decode(encode(seq)) == seq


def test_unknown_residues_map_to_gap():
    assert decode(encode("MZQ")) == "MXQ"


def test_one_hot_rows_sum_to_one():
    assert np.allclose(one_hot(encode("ACDE")).sum(axis=1), 1.0)


# ----------------------------------------------------------------- metrics
def test_recovery_is_exact_on_a_known_case():
    native = encode("AAAA")
    pred = encode("AACA")
    assert metrics.sequence_recovery(pred, native) == pytest.approx(0.75)


def test_recovery_respects_the_mask():
    native = encode("AAAA")
    pred = encode("AACC")
    mask = np.array([1, 1, 0, 0])
    assert metrics.sequence_recovery(pred, native, mask) == pytest.approx(1.0)


def test_empty_mask_returns_nan_not_a_crash():
    native, pred = encode("AA"), encode("AA")
    assert np.isnan(metrics.sequence_recovery(pred, native, np.zeros(2)))


def test_nll_of_a_confident_correct_prediction_is_near_zero():
    log_probs = np.log(np.full((4, 21), 1e-9))
    seq = encode("ACDE")
    log_probs[np.arange(4), seq] = np.log(1.0 - 20e-9)
    assert metrics.negative_log_likelihood(log_probs, seq) == pytest.approx(0.0, abs=1e-6)


def test_uniform_model_has_perplexity_equal_to_alphabet_size():
    log_probs = np.log(np.full((5, 21), 1.0 / 21))
    assert metrics.perplexity(log_probs, np.zeros(5, dtype=int)) == pytest.approx(21.0)


def test_paired_difference_detects_a_consistent_win():
    a = [0.55, 0.56, 0.54, 0.57]
    b = [0.50, 0.51, 0.49, 0.52]
    stats = metrics.paired_difference(a, b)
    assert stats["mean_difference"] == pytest.approx(0.05)
    assert stats["win_rate"] == 1.0


def test_paired_difference_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.paired_difference([1, 2], [1])


def test_summarize_reports_sample_std():
    stats = metrics.summarize([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(1.0)
    assert stats["n"] == 3


# ------------------------------------------------------------- uncertainty
def _probs_fixture() -> np.ndarray:
    """Four positions chosen so entropy and margin rank them differently.

    - 0: confident, one residue at 0.80.
    - 1: a contested pair at 0.45/0.44 -- tiny margin, moderate entropy.
    - 2: mass spread over nine residues -- large entropy, but a clear leader.
    - 3: near-certain, one residue at 0.95.
    """
    p = np.zeros((4, 21))
    p[0, 0] = 0.80
    p[0, 1:] = 0.20 / 20

    p[1, 0] = 0.45
    p[1, 1] = 0.44
    p[1, 2:] = 0.11 / 19

    p[2, 0] = 0.20
    p[2, 1:9] = 0.10
    p[2, 9:] = 0.0

    p[3, 5] = 0.95
    p[3, np.arange(21) != 5] = 0.05 / 20
    return p / p.sum(axis=1, keepdims=True)


def test_entropy_is_highest_for_the_most_diffuse_position():
    h = uncertainty.entropy(_probs_fixture())
    assert np.argmax(h) == 2


def test_entropy_is_lowest_for_the_most_confident_position():
    h = uncertainty.entropy(_probs_fixture())
    assert np.argmin(h) == 3


def test_normalized_entropy_is_bounded():
    h = uncertainty.entropy(_probs_fixture(), normalize=True)
    assert h.min() >= 0.0 and h.max() <= 1.0


def test_margin_is_smallest_where_the_top_two_are_close():
    m = uncertainty.margin(_probs_fixture())
    assert np.argmin(m) == 1


def test_entropy_and_margin_disagree():
    """The two measures rank different positions as least certain.

    Position 1 is a near-tie between two residues: small margin, but entropy is
    unremarkable because only two residues carry mass. Position 2 spreads mass
    over nine residues: high entropy, but a comfortable margin. A policy built
    on one will reconsider different residues than a policy built on the other,
    which is why the choice is worth ablating rather than assuming.
    """
    probs = _probs_fixture()
    assert np.argmax(uncertainty.entropy(probs)) != np.argmin(uncertainty.margin(probs))


def test_select_top_k_returns_exactly_k():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert uncertainty.select_top_k(scores, 2).size == 2


def test_select_top_k_is_deterministic_under_ties():
    scores = np.ones(5)
    np.testing.assert_array_equal(
        uncertainty.select_top_k(scores, 3), uncertainty.select_top_k(scores, 3)
    )


def test_select_top_k_clamps_to_available_positions():
    assert uncertainty.select_top_k(np.ones(3), 10).size == 3


def test_threshold_selection_can_select_everything():
    """The failure mode from the original notebook, made explicit."""
    confidence = np.full(76, 0.5)
    selected = uncertainty.select_by_threshold(confidence, threshold=0.80)
    assert selected.size == 76


def test_early_decoded_picks_the_first_positions_in_the_order():
    order = np.array([3, 1, 0, 2])
    np.testing.assert_array_equal(uncertainty.early_decoded(order, 0.5), np.array([1, 3]))


def test_score_positions_rejects_unknown_policy():
    with pytest.raises(ValueError):
        uncertainty.score_positions("nonsense", probs=_probs_fixture())


# ---------------------------------------------------------- decoding orders
def test_n_to_c_and_c_to_n_are_reverses():
    np.testing.assert_array_equal(do.n_to_c(5), do.c_to_n(5)[::-1])


def test_core_first_puts_the_most_buried_position_first():
    counts = np.array([2, 18, 7, 11])
    assert do.core_first(counts)[0] == 1


def test_surface_first_is_the_reverse_of_core_first():
    counts = np.array([2, 18, 7, 11])
    np.testing.assert_array_equal(do.core_first(counts), do.surface_first(counts)[::-1])


def test_blocked_random_is_a_permutation():
    order = do.blocked_random(20, np.random.default_rng(0), block_size=4)
    np.testing.assert_array_equal(np.sort(order), np.arange(20))


def test_neighbor_counts_on_a_known_geometry():
    """Three points on a line, 5 A apart, with a 10 A cutoff.

    The ends each see one neighbour within range and one at exactly 10 A, which
    the strict inequality excludes; the middle sees both.
    """
    coords = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    counts = do.neighbor_counts(coords, cutoff=10.0)
    np.testing.assert_array_equal(counts, np.array([1, 2, 1]))


def test_neighbor_counts_accepts_full_backbone_shape():
    coords = np.zeros((6, 4, 3))
    coords[:, 1, 0] = np.arange(6) * 2.0  # CA atoms along x
    assert do.neighbor_counts(coords, cutoff=5.0).shape == (6,)


def test_order_from_randn_matches_the_library_formula():
    """Mirror of ProteinMPNN's argsort, verified against a hand-computed case."""
    randn = np.array([0.9, 0.1, 0.5])
    np.testing.assert_array_equal(do.order_from_randn(randn), np.array([1, 2, 0]))


def test_fixed_positions_sort_to_the_front():
    randn = np.array([0.9, 0.1, 0.5])
    chain_mask = np.array([1.0, 1.0, 0.0])  # position 2 is fixed
    assert do.order_from_randn(randn, chain_mask)[0] == 2


def test_randn_for_order_requires_a_permutation():
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        do.randn_for_order(np.array([0, 0, 1]))


def test_randn_for_order_roundtrips():
    """The forced order must be the order ProteinMPNN would actually use."""
    pytest.importorskip("torch")
    order = np.array([3, 0, 2, 1])
    randn = do.randn_for_order(order)
    np.testing.assert_array_equal(do.order_from_randn(randn[0]), order)
