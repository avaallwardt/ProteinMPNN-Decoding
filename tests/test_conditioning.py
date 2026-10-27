"""Regression tests for the conditioning bug that motivated this repository.

The central claim of ``docs/findings.md`` is that passing a draft sequence with
an all-ones chain mask does not condition on the draft. These tests pin that
behaviour down so it cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipr import refine
from ipr.toy import ToyPottsModel

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def model() -> ToyPottsModel:
    return ToyPottsModel(length=30, seed=7)


def test_full_redesign_ignores_the_context_sequence(model):
    """With every position redesigned, the draft has no effect on the output.

    This is the bug. Two calls with the same RNG seed but wildly different
    context sequences must produce byte-identical results, because the context
    is discarded at every redesigned position.
    """
    draft_a = np.zeros(model.length, dtype=np.int64)
    draft_b = np.full(model.length, 5, dtype=np.int64)

    out_a = model.sample(temperature=0.5, context_sequence=draft_a, rng=np.random.default_rng(1))
    out_b = model.sample(temperature=0.5, context_sequence=draft_b, rng=np.random.default_rng(1))

    np.testing.assert_array_equal(out_a["S"], out_b["S"])


def test_partial_redesign_does_condition_on_the_context(model):
    """Holding some positions fixed makes the draft matter again."""
    redesign = np.arange(0, model.length, 2)
    draft_a = np.zeros(model.length, dtype=np.int64)
    draft_b = np.full(model.length, 5, dtype=np.int64)

    out_a = model.sample(temperature=0.5, redesign=redesign, context_sequence=draft_a, rng=np.random.default_rng(1))
    out_b = model.sample(temperature=0.5, redesign=redesign, context_sequence=draft_b, rng=np.random.default_rng(1))

    assert not np.array_equal(out_a["S"], out_b["S"]), (
        "redesigned positions should respond to the held context"
    )


def test_fixed_positions_keep_their_draft_values(model):
    """Positions outside the redesign set are copied verbatim from the draft."""
    redesign = np.array([1, 4, 9])
    draft = np.full(model.length, 3, dtype=np.int64)

    out = model.sample(temperature=0.1, redesign=redesign, context_sequence=draft, rng=np.random.default_rng(0))
    result = out["S"].ravel()

    held = np.setdiff1d(np.arange(model.length), redesign)
    np.testing.assert_array_equal(result[held], draft[held])


def test_probabilities_are_zero_at_fixed_positions(model):
    """Reproduces ProteinMPNN's masked-probability convention.

    Any selection policy that thresholds on these probabilities will see fixed
    positions as zero-confidence. Code must account for it rather than assume
    the array is fully populated.
    """
    redesign = np.array([2, 3])
    out = model.sample(temperature=0.1, redesign=redesign, rng=np.random.default_rng(0))
    probs = out["probs"][0]

    held = np.setdiff1d(np.arange(model.length), redesign)
    assert np.allclose(probs[held], 0.0)
    assert np.allclose(probs[redesign].sum(axis=-1), 1.0)


def test_empty_redesign_reproduces_the_input_exactly(model):
    """Redesigning nothing is a pure scoring pass."""
    draft = np.full(model.length, 7, dtype=np.int64)
    out = model.sample(temperature=0.1, redesign=[], context_sequence=draft, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out["S"].ravel(), draft)


def test_total_refresh_matches_a_single_pass_in_distribution(model):
    """The original method is statistically indistinguishable from one pass.

    Both arms are compared on mean energy of the resulting sequence. If
    ``total_refresh`` genuinely refined the draft it would land at lower energy
    than an independent sample at the same temperature; it does not.
    """
    n_trials = 40
    refresh, baseline = [], []
    for seed in range(n_trials):
        r = refine.total_refresh(model, 0.5, 0.1, rng=np.random.default_rng(seed))
        b = refine.single_pass(model, temperature=0.1, rng=np.random.default_rng(1000 + seed))
        refresh.append(model.energy(r.sequence))
        baseline.append(model.energy(b.sequence))

    pooled_std = np.sqrt(np.var(refresh, ddof=1) / n_trials + np.var(baseline, ddof=1) / n_trials)
    gap = abs(np.mean(refresh) - np.mean(baseline))
    assert gap < 3 * pooled_std, (
        f"total_refresh should not differ from a single pass: gap={gap:.3f}, 3se={3 * pooled_std:.3f}"
    )


def test_toy_sampling_is_reproducible_from_its_rng(model):
    """Same generator state, same sequence.

    The real backend had a bug here: ProteinMPNN samples tokens with
    torch.multinomial, which reads torch's global RNG, so seeding only the
    NumPy generator fixed the decoding order and left the token draws free.
    Two runs with the same seed returned different sequences, which quietly
    weakened every paired comparison. ipr.backends.Runner.sample now derives a
    torch seed from the generator. This test pins the same property for the toy
    backend, which has always drawn through its NumPy generator.
    """
    a = model.sample(temperature=0.5, rng=np.random.default_rng(3))["S"]
    b = model.sample(temperature=0.5, rng=np.random.default_rng(3))["S"]
    c = model.sample(temperature=0.5, rng=np.random.default_rng(4))["S"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
