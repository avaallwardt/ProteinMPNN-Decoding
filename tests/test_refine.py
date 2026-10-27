"""Tests for the refinement algorithms, using the Potts model as a positive control.

A Potts model has a true joint distribution, so iterated conditional modes is
guaranteed to decrease energy monotonically and reach a fixed point. If these
tests fail, the refinement logic is wrong -- there is no modelling ambiguity to
hide behind.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipr import refine, uncertainty
from ipr.metrics import hamming_distance, paired_difference, sequence_recovery
from ipr.toy import ToyPottsModel


@pytest.fixture(scope="module")
def model() -> ToyPottsModel:
    return ToyPottsModel(length=30, seed=11)


def test_icm_reaches_a_fixed_point(model):
    """Greedy sweeps must stop changing the sequence."""
    result = refine.iterate_to_fixed_point(
        model, temperature_draft=0.5, temperature_refine=0.01,
        max_sweeps=25, mode="sequential", greedy=True,
        rng=np.random.default_rng(3),
    )
    assert result.converged, f"ICM did not converge; trajectory={result.trajectory}"


def test_icm_decreases_energy(model):
    """Each ICM sweep is a descent step, so the final energy beats the draft."""
    rng = np.random.default_rng(5)
    result = refine.iterate_to_fixed_point(
        model, temperature_draft=0.5, temperature_refine=0.01,
        max_sweeps=25, mode="sequential", greedy=True, rng=rng,
    )
    assert model.energy(result.sequence) < model.energy(result.draft)


def test_changes_per_sweep_decrease(model):
    """A contracting update map changes fewer positions each sweep."""
    result = refine.iterate_to_fixed_point(
        model, temperature_draft=0.5, temperature_refine=0.01,
        max_sweeps=25, mode="sequential", greedy=True,
        rng=np.random.default_rng(13),
    )
    counts = [t["n_changed"] for t in result.trajectory if "n_changed" in t]
    assert counts[0] >= counts[-1]
    assert counts[-1] == 0


def test_contraction_factor_is_below_one_when_converging(model):
    """The empirical contraction estimate should be under 1 on a converging run."""
    result = refine.iterate_to_fixed_point(
        model, temperature_draft=0.5, temperature_refine=0.01,
        max_sweeps=25, mode="sequential", greedy=True,
        rng=np.random.default_rng(21),
    )
    factor = refine.contraction_factor(result.trajectory)
    if not np.isnan(factor):
        assert factor < 1.0


def test_conditional_refine_beats_total_refresh_on_energy(model):
    """Genuine conditioning improves on the non-conditioning method.

    This is the whole point of the repository: the corrected pass uses the
    draft, so it should land at lower energy than the pass that discards it.
    """
    n_trials = 30
    corrected, original = [], []
    for seed in range(n_trials):
        refined = refine.iterate_to_fixed_point(
            model, temperature_draft=0.5, temperature_refine=0.01,
            max_sweeps=20, mode="sequential", greedy=True,
            rng=np.random.default_rng(seed),
        )
        corrected.append(model.energy(refined.sequence))
        original.append(
            model.energy(refine.total_refresh(model, 0.5, 0.01, rng=np.random.default_rng(seed)).sequence)
        )

    stats = paired_difference(original, corrected)  # original - corrected, want > 0
    assert stats["mean_difference"] > 0, f"corrected refinement should win: {stats}"
    assert stats["win_rate"] > 0.7, f"corrected refinement should win most trials: {stats}"


def test_batched_mode_refuses_to_select_every_position(model):
    """Selecting all positions in batched mode leaves no context, so it must fail loudly.

    This is the guard that encodes the original bug: a pass that redesigns
    everything cannot be conditioning on anything.
    """
    with pytest.raises(ValueError, match="conditions on nothing"):
        refine.gibbs_sweep(
            model, model.native, None, temperature=0.01,
            mode="batched", greedy=True, rng=np.random.default_rng(0),
        )

    with pytest.raises(ValueError, match="requires fraction < 1.0"):
        refine.iterate_to_fixed_point(model, mode="batched", fraction=1.0)


def test_both_sweep_modes_lower_energy_on_a_subset(model):
    """On a strict subset, the cheap batched approximation still descends."""
    rng = np.random.default_rng(17)
    draft = refine.single_pass(model, temperature=0.5, rng=rng).sequence
    positions = np.arange(0, model.length, 3)  # sparse, so neighbours rarely collide

    for mode in ("sequential", "batched"):
        updated, _ = refine.gibbs_sweep(
            model, draft, positions, temperature=0.01, mode=mode, greedy=True,
            rng=np.random.default_rng(17),
        )
        assert model.energy(updated) <= model.energy(draft), f"{mode} sweep increased energy"


def test_uncertainty_guided_respects_its_budget(model):
    """The method must redesign the fraction it claims to redesign."""
    result = refine.uncertainty_guided(
        model, policy="entropy", fraction=0.25, n_rounds=1,
        rng=np.random.default_rng(2),
    )
    expected = int(round(model.length * 0.25))
    assert result.trajectory[-1]["n_redesigned"] == expected
    assert result.n_changed <= expected, "more positions changed than were unlocked"


def test_random_policy_is_a_valid_control(model):
    """The random-selection control runs and selects the same number of positions."""
    guided = refine.uncertainty_guided(model, policy="entropy", fraction=0.2, rng=np.random.default_rng(1))
    control = refine.uncertainty_guided(model, policy="random", fraction=0.2, rng=np.random.default_rng(1))
    assert guided.trajectory[-1]["n_redesigned"] == control.trajectory[-1]["n_redesigned"]


def test_sampling_does_not_claim_convergence(model):
    """Non-greedy Gibbs explores; it should not be reported as converged."""
    result = refine.iterate_to_fixed_point(
        model, temperature_draft=0.5, temperature_refine=1.0,
        max_sweeps=5, mode="sequential", greedy=False,
        rng=np.random.default_rng(4),
    )
    assert not result.converged


def test_single_pass_is_reproducible(model):
    """Same seed, same sequence -- every experiment depends on this."""
    a = refine.single_pass(model, temperature=0.3, rng=np.random.default_rng(99))
    b = refine.single_pass(model, temperature=0.3, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a.sequence, b.sequence)


def test_recovery_improves_toward_the_native_configuration(model):
    """On a model whose native sequence is a low-energy state, ICM raises recovery.

    This is the toy analogue of the sequence-recovery claim, and it holds here
    precisely because the conditionals are consistent.
    """
    n_trials = 25
    before, after = [], []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        draft = refine.single_pass(model, temperature=0.5, rng=rng).sequence
        refined, _ = refine.gibbs_sweep(
            model, draft, None, temperature=0.01, mode="sequential", greedy=True, rng=rng
        )
        before.append(sequence_recovery(draft, model.native))
        after.append(sequence_recovery(refined, model.native))

    stats = paired_difference(after, before)
    assert stats["mean_difference"] > 0, f"ICM should raise recovery on the toy model: {stats}"
