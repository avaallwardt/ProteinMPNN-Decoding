"""Chain-mask construction for conditional refinement.

This is the most important module in the package, because the central bug in
the original notebook lives here.

Background
----------
``ProteinMPNN.sample`` resolves each position with this line (see
``protein_mpnn_utils.py``, inside the decoding loop)::

    S_t = (S_t*chain_mask_gathered + S_true_gathered*(1.0-chain_mask_gathered)).long()

So the semantics of ``chain_mask`` are:

===================  ==========================================================
``chain_mask == 1``  Position is **redesigned**. A fresh token is sampled and
                     ``S_true`` is discarded at that position.
``chain_mask == 0``  Position is **held fixed**. The token is copied verbatim
                     from ``S_true`` and, crucially, it is decoded *first* and
                     becomes conditioning context for every redesigned
                     position.
===================  ==========================================================

The ordering half of that follows from::

    decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))

Multiplying by ``chain_mask`` sends every fixed position to ~0.0001, so fixed
positions sort to the front of the decoding order and are embedded into ``h_S``
before any redesigned position is sampled.

The consequence
---------------
Passing a draft sequence as ``S_true`` with ``chain_mask`` all ones does
**not** condition on the draft. It discards the draft at every position and
produces an independent sample. To make the model actually use a draft as
context, the mask must be *inverted* relative to the intuition: zeros at the
positions you want the model to read, ones only at the positions you want it
to reconsider.

That is what :func:`refinement_mask` builds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def refinement_mask(
    designable: "torch.Tensor",
    redesign_positions: Sequence[int] | np.ndarray | "torch.Tensor",
) -> "torch.Tensor":
    """Build a ``chain_mask`` that redesigns some positions and conditions on the rest.

    Parameters
    ----------
    designable:
        The original ``chain_M`` from ``tied_featurize``, shape ``(B, L)``. A 1
        marks a position the task allows to be designed at all; a 0 marks a
        position that is structurally fixed (another chain, a missing region).
        Positions that are 0 here stay 0 in the result -- this function can
        only ever narrow the designable set, never widen it.
    redesign_positions:
        Indices along ``L`` to reconsider in this pass. Every other designable
        position is set to 0, which holds its draft token and feeds it to the
        model as context.

    Returns
    -------
    torch.Tensor
        A mask of shape ``(B, L)``, same dtype and device as ``designable``.

    Notes
    -----
    Passing an empty ``redesign_positions`` yields an all-zero mask, which makes
    ``sample`` a pure scoring pass: it reproduces the input sequence exactly and
    returns the conditional probabilities the model assigns to it.
    """
    torch = _torch()

    if designable.dim() != 2:
        raise ValueError(f"expected designable of shape (B, L), got {tuple(designable.shape)}")

    idx = _as_index_tensor(redesign_positions, device=designable.device)
    length = designable.shape[1]
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= length):
        raise IndexError(f"redesign_positions out of range for length {length}")

    selector = torch.zeros_like(designable)
    if idx.numel():
        selector[:, idx] = 1

    return designable * selector


def scoring_mask(designable: "torch.Tensor") -> "torch.Tensor":
    """Build an all-zero mask: hold every position, redesign nothing.

    Used to score an existing sequence. ``sample`` called with this mask returns
    the input sequence unchanged along with per-position probabilities computed
    with full sequence context.
    """
    torch = _torch()
    return torch.zeros_like(designable)


def total_refresh_mask(designable: "torch.Tensor") -> "torch.Tensor":
    """Build an all-ones mask: redesign everything, condition on nothing.

    .. warning::
       This is the mask used by the original notebook's "total refresh" second
       pass. It does **not** condition on the draft sequence. Every position is
       resampled from scratch and the draft is discarded, which makes the
       resulting pass an independent sample rather than a refinement. It is kept
       here so the ablation suite can reproduce that arm exactly and show it is
       equivalent to a fresh single pass -- see ``docs/findings.md``.
    """
    return designable.clone()


def fraction_to_count(length: int, fraction: float) -> int:
    """Convert a fraction of a sequence to a residue count, clamped to ``[0, length]``."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return int(max(0, min(length, round(length * fraction))))


def _as_index_tensor(positions, device) -> "torch.Tensor":
    torch = _torch()
    if isinstance(positions, torch.Tensor):
        return positions.to(device=device, dtype=torch.long).ravel()
    arr = np.asarray(list(positions), dtype=np.int64).ravel()
    return torch.as_tensor(arr, device=device, dtype=torch.long)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required for mask construction. Install it with "
            "`pip install torch`, or use ipr.toy for a dependency-free model."
        ) from exc
    return torch
