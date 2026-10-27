"""Refinement strategies for autoregressive sequence design.

Each strategy here is a function from a model wrapper to a sequence, plus a
record of what happened on the way. They share one interface so the benchmark
harness can treat them as interchangeable arms.

The backend only has to provide:

``length``
    int, sequence length.
``sample(temperature, order, redesign, context_sequence, rng)``
    returns ``{"S", "probs", "decoding_order"}``.
``log_probs(sequence, order, rng)``
    returns ``(1, L, 21)`` teacher-forced log-probabilities.

:class:`ipr.backends.Runner` and :class:`ipr.toy.ToyPottsModel` both satisfy it,
which is what lets the test suite exercise these algorithms without PyTorch or
model weights.

A note on what "refinement" means here
--------------------------------------
A pass only counts as refinement if the draft is in its conditioning set. In
ProteinMPNN that requires holding the draft positions fixed via ``chain_mask``;
resampling a position discards its draft value. :func:`total_refresh` is kept
as a deliberate counterexample -- it is what the original notebook ran, and it
conditions on nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from . import decoding_orders, uncertainty
from .metrics import hamming_distance, to_numpy


@dataclass
class RefinementResult:
    """A sequence plus an audit trail of how it was produced."""

    sequence: np.ndarray
    draft: np.ndarray | None = None
    n_redesigned: int = 0
    n_changed: int = 0
    rounds: int = 1
    trajectory: list[dict] = field(default_factory=list)
    method: str = ""

    @property
    def converged(self) -> bool:
        """True if the final round changed nothing -- a fixed point of the update map."""
        return bool(self.trajectory) and self.trajectory[-1].get("n_changed", -1) == 0


def _seq(out) -> np.ndarray:
    return to_numpy(out["S"]).ravel().astype(np.int64)


def _probs(out) -> np.ndarray:
    p = to_numpy(out["probs"])
    return p[0] if p.ndim == 3 else p


def _order(out) -> np.ndarray:
    return to_numpy(out["decoding_order"]).ravel().astype(np.int64)


# ---------------------------------------------------------------------- #
# Baseline
# ---------------------------------------------------------------------- #
def single_pass(runner, temperature: float = 0.1, order=None, rng=None) -> RefinementResult:
    """Standard ProteinMPNN: one autoregressive pass, no refinement.

    The control arm every other method must beat.
    """
    rng = rng or np.random.default_rng()
    out = runner.sample(temperature=temperature, order=order, rng=rng)
    seq = _seq(out)
    return RefinementResult(
        sequence=seq,
        n_redesigned=runner.length,
        rounds=1,
        method="single_pass",
        trajectory=[{"round": 0, "temperature": temperature, "n_redesigned": runner.length}],
    )


# ---------------------------------------------------------------------- #
# The notebook's original arm, preserved as a counterexample
# ---------------------------------------------------------------------- #
def total_refresh(
    runner,
    temperature_draft: float = 0.5,
    temperature_refine: float = 0.1,
    rng=None,
    shared_order: bool = False,
) -> RefinementResult:
    """Reproduce the original "total refresh" two-pass method exactly.

    .. warning::
       **This does not condition on the draft.** Pass 2 sets every designable
       position to ``chain_mask == 1``, and ProteinMPNN discards ``S_true`` at
       every such position. Pass 2 is therefore an independent sample at
       ``temperature_refine``, statistically identical to calling
       :func:`single_pass` at that temperature.

       It is kept so the ablation suite can demonstrate that equivalence rather
       than assert it. See ``docs/findings.md``.

    Parameters
    ----------
    shared_order:
        If True, reuse pass 1's decoding order for pass 2, as one variant in the
        original notebook did. This removes the only remaining difference
        between the two passes.
    """
    rng = rng or np.random.default_rng()
    order1 = decoding_orders.random_order(runner.length, rng)
    out1 = runner.sample(temperature=temperature_draft, order=order1, rng=rng)
    draft = _seq(out1)

    order2 = order1 if shared_order else decoding_orders.random_order(runner.length, rng)
    out2 = runner.sample(
        temperature=temperature_refine,
        order=order2,
        redesign=None,  # None => every position redesigned => draft discarded
        context_sequence=draft,
        rng=rng,
    )
    seq = _seq(out2)
    return RefinementResult(
        sequence=seq,
        draft=draft,
        n_redesigned=runner.length,
        n_changed=hamming_distance(seq, draft),
        rounds=2,
        method="total_refresh",
        trajectory=[
            {"round": 0, "temperature": temperature_draft, "n_redesigned": runner.length},
            {"round": 1, "temperature": temperature_refine, "n_redesigned": runner.length,
             "n_changed": hamming_distance(seq, draft), "conditions_on_draft": False},
        ],
    )


# ---------------------------------------------------------------------- #
# Corrected refinement
# ---------------------------------------------------------------------- #
def conditional_refine(
    runner,
    draft: np.ndarray,
    redesign: Sequence[int],
    temperature: float = 0.1,
    order=None,
    rng=None,
) -> RefinementResult:
    """Redesign ``redesign`` while genuinely conditioning on the rest of ``draft``.

    This is the pass the project set out to build. Positions outside
    ``redesign`` are held fixed, decoded first, and embedded as context, so each
    redesigned residue sees the full surrounding sequence -- including residues
    that were decided after it in the original pass.
    """
    rng = rng or np.random.default_rng()
    redesign = np.asarray(list(redesign), dtype=np.int64)

    out = runner.sample(
        temperature=temperature,
        order=order,
        redesign=redesign,
        context_sequence=draft,
        rng=rng,
    )
    seq = _seq(out)
    return RefinementResult(
        sequence=seq,
        draft=np.asarray(draft),
        n_redesigned=int(redesign.size),
        n_changed=hamming_distance(seq, draft),
        rounds=1,
        method="conditional_refine",
        trajectory=[{"round": 0, "temperature": temperature, "n_redesigned": int(redesign.size),
                     "n_changed": hamming_distance(seq, draft), "conditions_on_draft": True}],
    )


def uncertainty_guided(
    runner,
    temperature_draft: float = 0.1,
    temperature_refine: float = 0.1,
    policy: str = "entropy",
    fraction: float = 0.25,
    n_rounds: int = 1,
    rng=None,
) -> RefinementResult:
    """Draft once, then repeatedly redesign the least certain residues in context.

    Parameters
    ----------
    policy:
        Which uncertainty measure picks the residues. See
        :data:`ipr.uncertainty.POLICIES`. Use ``"random"`` for the matched-budget
        control arm -- if a guided policy cannot beat random selection at the
        same ``fraction``, the guidance contributes nothing.
    fraction:
        Share of positions reconsidered per round.
    n_rounds:
        Refinement rounds after the draft. Each round reselects from the current
        sequence, so uncertainty is recomputed as the sequence changes.
    """
    rng = rng or np.random.default_rng()
    out = runner.sample(temperature=temperature_draft, order=None, rng=rng)
    draft = _seq(out)
    probs, order = _probs(out), _order(out)

    k = int(round(runner.length * fraction))
    current = draft.copy()
    trajectory = [{"round": 0, "temperature": temperature_draft, "n_redesigned": runner.length}]

    for r in range(n_rounds):
        if policy == "random":
            positions = uncertainty.select_random(runner.length, k, rng)
        elif policy == "early":
            positions = uncertainty.early_decoded(order, fraction)
        else:
            scores = uncertainty.score_positions(policy, probs=probs, sequence=current, decoding_order=order)
            positions = uncertainty.select_top_k(scores, k)

        step = conditional_refine(runner, current, positions, temperature=temperature_refine, rng=rng)
        changed = hamming_distance(step.sequence, current)
        current = step.sequence

        # Refresh probabilities from a scoring pass so the next round selects on
        # the updated sequence rather than on stale draft probabilities.
        if r + 1 < n_rounds:
            probs = np.exp(to_numpy(runner.log_probs(current, rng=rng))[0])

        trajectory.append({
            "round": r + 1,
            "temperature": temperature_refine,
            "policy": policy,
            "n_redesigned": int(np.size(positions)),
            "n_changed": changed,
            "conditions_on_draft": True,
        })

    return RefinementResult(
        sequence=current,
        draft=draft,
        n_redesigned=k,
        n_changed=hamming_distance(current, draft),
        rounds=n_rounds + 1,
        method=f"uncertainty_guided[{policy}]",
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------- #
# Gibbs / ICM
# ---------------------------------------------------------------------- #
def gibbs_sweep(
    runner,
    sequence: np.ndarray,
    positions: Sequence[int] | None = None,
    temperature: float = 0.1,
    mode: Literal["sequential", "batched"] = "sequential",
    greedy: bool = False,
    rng=None,
) -> tuple[np.ndarray, int]:
    """One sweep of single-site updates, each conditioned on all other residues.

    Parameters
    ----------
    mode:
        ``"sequential"`` updates positions one at a time, each conditioned on
        the *current* value of every other residue. This is a true Gibbs sweep
        and costs one model call per position.

        ``"batched"`` updates every selected position from a single pass in
        which they are decoded last. Each sees all unselected residues plus the
        *pre-sweep* values of the other selected ones -- a Jacobi-style update.
        It costs one call per sweep instead of ``k``, at the price of being an
        approximation when selected positions are spatial neighbours.
    greedy:
        Take the argmax instead of sampling. With ``greedy=True`` this is
        iterated conditional modes (ICM), which converges monotonically in
        pseudo-likelihood to a local optimum. With ``greedy=False`` it is Gibbs
        sampling, which explores instead of converging.

    Returns
    -------
    tuple
        ``(sequence, n_changed)``.
    """
    rng = rng or np.random.default_rng()
    current = np.asarray(sequence, dtype=np.int64).copy()
    positions = np.arange(runner.length) if positions is None else np.asarray(list(positions), dtype=np.int64)
    if positions.size == 0:
        return current, 0

    if mode == "batched":
        others = np.setdiff1d(np.arange(runner.length), positions)
        if others.size == 0:
            raise ValueError(
                "batched mode with every position selected conditions on nothing: "
                "redesigning all positions in one pass discards the draft entirely "
                "(see ipr.conditioning). Use mode='sequential', or select a strict "
                "subset of positions."
            )
        rng.shuffle(others)
        order = np.concatenate([others, rng.permutation(positions)])
        out = runner.sample(
            temperature=temperature, order=order, redesign=positions,
            context_sequence=current, rng=rng,
        )
        updated = _seq(out)
        if greedy:
            # Re-derive argmax from the returned probabilities where they exist.
            probs = _probs(out)
            for p in positions:
                if probs[p].sum() > 0:
                    updated[p] = int(np.argmax(probs[p]))
        n_changed = hamming_distance(updated, current)
        return updated, n_changed

    if mode != "sequential":
        raise ValueError(f"unknown mode {mode!r}; expected 'sequential' or 'batched'")

    n_changed = 0
    for p in rng.permutation(positions):
        out = runner.sample(
            temperature=temperature, order=None, redesign=[int(p)],
            context_sequence=current, rng=rng,
        )
        if greedy:
            probs = _probs(out)[int(p)]
            new_token = int(np.argmax(probs)) if probs.sum() > 0 else int(current[p])
        else:
            new_token = int(_seq(out)[int(p)])
        if new_token != int(current[p]):
            n_changed += 1
            current[int(p)] = new_token

    return current, n_changed


def iterate_to_fixed_point(
    runner,
    temperature_draft: float = 0.5,
    temperature_refine: float = 0.1,
    max_sweeps: int = 10,
    mode: Literal["sequential", "batched"] = "sequential",
    greedy: bool = True,
    fraction: float = 1.0,
    rng=None,
) -> RefinementResult:
    """Sweep until the sequence stops changing, recording the trajectory.

    This is the fixed-point iteration from the project notes made concrete:
    treat the first pass as an initial guess, apply the update map repeatedly,
    and watch whether successive iterates contract toward a stable sequence.

    The trajectory records ``n_changed`` per sweep. If those counts fall roughly
    geometrically, the update map is behaving as a contraction on sequence space
    and the ratio between successive counts estimates the contraction factor --
    the quantity ``docs/theory.md`` calls ``L``. If they plateau at a nonzero
    value, the map is not contracting and no amount of iteration will converge.

    .. note::
       With ``greedy=False`` this samples rather than descends, and it should
       *not* be expected to reach a fixed point; the trajectory then measures
       mixing, not convergence.
    """
    if mode == "batched" and fraction >= 1.0:
        raise ValueError(
            "batched mode requires fraction < 1.0: selecting every position leaves "
            "no context to condition on. Use mode='sequential' for full sweeps."
        )

    rng = rng or np.random.default_rng()
    out = runner.sample(temperature=temperature_draft, order=None, rng=rng)
    draft = _seq(out)
    current = draft.copy()

    k = int(round(runner.length * fraction))
    trajectory = [{"round": 0, "temperature": temperature_draft, "n_redesigned": runner.length}]

    for sweep in range(max_sweeps):
        if fraction >= 1.0:
            positions = np.arange(runner.length)
        else:
            probs = np.exp(to_numpy(runner.log_probs(current, rng=rng))[0])
            positions = uncertainty.select_top_k(uncertainty.entropy(probs), k)

        current, n_changed = gibbs_sweep(
            runner, current, positions, temperature=temperature_refine,
            mode=mode, greedy=greedy, rng=rng,
        )
        trajectory.append({
            "round": sweep + 1,
            "temperature": temperature_refine,
            "n_redesigned": int(np.size(positions)),
            "n_changed": int(n_changed),
            "conditions_on_draft": True,
        })
        if n_changed == 0:
            break

    return RefinementResult(
        sequence=current,
        draft=draft,
        n_redesigned=k,
        n_changed=hamming_distance(current, draft),
        rounds=len(trajectory),
        method=f"fixed_point[{mode},{'icm' if greedy else 'gibbs'}]",
        trajectory=trajectory,
    )


def contraction_factor(trajectory: list[dict]) -> float:
    """Estimate the contraction factor from a fixed-point trajectory.

    Fits a geometric decay to the per-sweep change counts and returns the ratio.
    A value below 1 is consistent with a contraction mapping; at or above 1 is
    evidence against one. Returns ``nan`` when there are too few sweeps with
    nonzero changes to fit.

    This is an empirical estimate on one trajectory, not a proof. A Lipschitz
    constant is a supremum over all pairs of inputs; this is one path. Treat it
    as evidence that motivates the claim, never as the claim itself.
    """
    counts = [t["n_changed"] for t in trajectory if "n_changed" in t and t["n_changed"] > 0]
    if len(counts) < 2:
        return float("nan")
    ratios = [counts[i + 1] / counts[i] for i in range(len(counts) - 1) if counts[i] > 0]
    return float(np.mean(ratios)) if ratios else float("nan")


STRATEGIES = {
    "single_pass": single_pass,
    "total_refresh": total_refresh,
    "uncertainty_guided": uncertainty_guided,
    "fixed_point": iterate_to_fixed_point,
}
