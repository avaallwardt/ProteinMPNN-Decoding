"""Thin, correct wrapper around ProteinMPNN.

Everything in this module exists to make three things easy and hard to get
wrong:

1. forcing an exact decoding order (see :mod:`ipr.decoding_orders`),
2. conditioning on a draft sequence (see :mod:`ipr.conditioning`),
3. reading out per-position probabilities that are actually populated.

Point 3 has a trap worth stating up front. ``ProteinMPNN.sample`` fills its
``probs`` output with::

    all_probs.scatter_(1, ..., (chain_mask_gathered[:,:,None] * probs[:,None,:]).float())

The probabilities are multiplied by the chain mask, so **every fixed position
reports a probability of exactly zero** -- not a low probability, an absent
one. Any code that thresholds on ``out['probs']`` without accounting for this
will treat all fixed positions as maximally uncertain. To score positions that
are being held fixed, use :meth:`Runner.log_probs`, which runs the teacher-forced
forward pass instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import conditioning, decoding_orders
from .metrics import to_numpy

DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 3


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required to run ProteinMPNN. Install it with "
            "`pip install torch`, or use ipr.toy.ToyPottsModel to exercise the "
            "refinement algorithms without it."
        ) from exc
    return torch


@dataclass
class Structure:
    """Featurized backbone plus the native sequence, for a single structure."""

    name: str
    X: "object"  # (1, L, 4, 3) backbone coordinates
    S_native: "object"  # (1, L) native tokens
    mask: "object"  # (1, L) 1 where a residue is resolved
    chain_M: "object"  # (1, L) 1 where designable
    chain_M_pos: "object"
    residue_idx: "object"
    chain_encoding_all: "object"
    omit_AA_mask: "object"
    bias_by_res: "object"

    @property
    def length(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_designable(self) -> int:
        return int(to_numpy(self.chain_M).sum())

    def neighbor_counts(self, cutoff: float = 10.0) -> np.ndarray:
        """CA neighbour count per residue, for burial-stratified metrics and core-first ordering."""
        return decoding_orders.neighbor_counts(to_numpy(self.X), cutoff=cutoff)


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    backbone_noise: float = 0.0,
    ca_only: bool = False,
):
    """Load a pretrained ProteinMPNN checkpoint.

    ``k_neighbors`` is read from the checkpoint rather than hardcoded. This
    matters: the ``v_48_020`` weights were trained with 48 neighbours, and
    instantiating the model with a different value silently changes the graph
    the network sees at inference without raising any error, because
    ``k_neighbors`` does not affect parameter shapes. The original notebook
    passed ``k_neighbors=64`` against these weights.

    Parameters
    ----------
    backbone_noise:
        Gaussian noise (angstroms) added to coordinates. Keep at 0.0 for
        deterministic evaluation; raise it for the protein-breathing
        experiments described in the project notes.
    """
    torch = _torch()
    from protein_mpnn_utils import ProteinMPNN

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    k_neighbors = int(checkpoint.get("num_edges", 48))

    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=DEFAULT_HIDDEN_DIM,
        edge_features=DEFAULT_HIDDEN_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        num_encoder_layers=DEFAULT_NUM_LAYERS,
        num_decoder_layers=DEFAULT_NUM_LAYERS,
        augment_eps=backbone_noise,
        k_neighbors=k_neighbors,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, {"k_neighbors": k_neighbors, "training_noise": checkpoint.get("noise_level")}


def featurize_pdb(
    pdb_path: str | Path,
    device: str = "cpu",
    designed_chains: Sequence[str] | None = None,
) -> Structure:
    """Parse a PDB file and featurize it into tensors ProteinMPNN accepts."""
    torch = _torch()
    from protein_mpnn_utils import StructureDatasetPDB, parse_PDB, tied_featurize

    pdb_dict_list = parse_PDB(str(pdb_path))
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)
    batch = [dataset[0]]
    name = batch[0]["name"]

    if designed_chains is None:
        designed_chains = [
            k.removeprefix("seq_chain_") for k in batch[0] if k.startswith("seq_chain_")
        ]
    chain_dict = {name: (list(designed_chains), [])}

    with torch.no_grad():
        feats = tied_featurize(batch=batch, device=device, chain_dict=chain_dict)

    X, S_native, mask, _lengths, chain_M, chain_encoding_all = feats[:6]
    chain_M_pos, omit_AA_mask, residue_idx = feats[10], feats[11], feats[12]

    return Structure(
        name=name,
        X=X,
        S_native=S_native,
        mask=mask,
        chain_M=chain_M,
        chain_M_pos=chain_M_pos,
        residue_idx=residue_idx,
        chain_encoding_all=chain_encoding_all,
        omit_AA_mask=omit_AA_mask,
        bias_by_res=torch.zeros([X.shape[0], X.shape[1], 21], device=device),
    )


class Runner:
    """Stateless-feeling interface to one model and one structure.

    Every method takes an explicit decoding order and an explicit set of
    positions to redesign, so no call depends on hidden random state. Pass an
    ``np.random.Generator`` to reproduce a run exactly.
    """

    def __init__(self, model, structure: Structure, device: str = "cpu"):
        self.model = model
        self.structure = structure
        self.device = device
        self._omit = np.zeros(21, dtype=np.float32)
        self._bias = np.zeros(21, dtype=np.float32)

    @property
    def length(self) -> int:
        return self.structure.length

    @property
    def native(self) -> np.ndarray:
        return to_numpy(self.structure.S_native).ravel().astype(np.int64)

    @property
    def designable_mask(self) -> np.ndarray:
        return to_numpy(self.structure.chain_M).ravel()

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(
        self,
        temperature: float = 0.1,
        order: np.ndarray | None = None,
        redesign: Sequence[int] | None = None,
        context_sequence=None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Sample a sequence, optionally conditioning on a draft.

        Parameters
        ----------
        temperature:
            Sampling temperature. Lower is greedier.
        order:
            Decoding order over designable positions. Random if omitted.
        redesign:
            Positions to resample. Every other designable position is **held
            fixed at its value in** ``context_sequence`` **and used as context**.
            If omitted, every designable position is redesigned, which means
            ``context_sequence`` is ignored entirely -- that is ProteinMPNN's
            standard single-pass behaviour.
        context_sequence:
            The sequence supplying fixed-position values. Defaults to the native
            sequence. For a refinement pass this is the draft from the previous
            pass.

        Returns
        -------
        dict
            ``S`` (1, L) sampled tokens, ``probs`` (1, L, 21) with **zeros at
            fixed positions**, and ``decoding_order`` (1, L).
        """
        torch = _torch()
        s = self.structure
        rng = rng or np.random.default_rng()

        context = s.S_native if context_sequence is None else _as_tensor_like(context_sequence, s.S_native)

        if redesign is None:
            chain_mask = s.chain_M
        else:
            chain_mask = conditioning.refinement_mask(s.chain_M, redesign)

        order = decoding_orders.random_order(s.length, rng) if order is None else np.asarray(order)
        randn = decoding_orders.randn_for_order(order, s.length).to(self.device)

        # ProteinMPNN draws tokens with torch.multinomial, which reads torch's
        # GLOBAL RNG. Seeding only the NumPy generator fixes the decoding order
        # and leaves the token draws unseeded, so two runs with the same `rng`
        # return different sequences. Deriving a torch seed here makes a call
        # fully reproducible from `rng` alone, which is what paired comparisons
        # between arms require.
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

        with torch.no_grad():
            out = self.model.sample(
                s.X,
                randn,
                context,
                chain_mask,
                s.chain_encoding_all,
                s.residue_idx,
                mask=s.mask,
                temperature=temperature,
                omit_AAs_np=self._omit,
                bias_AAs_np=self._bias,
                chain_M_pos=s.chain_M_pos,
                omit_AA_mask=s.omit_AA_mask,
                bias_by_res=s.bias_by_res,
            )
        return out

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def log_probs(self, sequence, order: np.ndarray | None = None, rng=None):
        """Teacher-forced log-probabilities for ``sequence`` under a given decoding order.

        Returns a ``(1, L, 21)`` tensor where entry ``t`` is
        ``log p(s_t | s_{<t in order}, X)``.

        .. warning::
           Because this is teacher-forced on the sequence you pass, calling it
           with the *native* sequence and reading ``argmax`` does **not** measure
           design recovery -- every position is conditioned on native context.
           The original notebook's 57.89% "pass 1 recovery" came from this path.
           For design recovery, use :meth:`sample`.
        """
        torch = _torch()
        s = self.structure
        rng = rng or np.random.default_rng()
        seq = _as_tensor_like(sequence, s.S_native)

        order = decoding_orders.random_order(s.length, rng) if order is None else np.asarray(order)
        order_t = torch.as_tensor(np.asarray(order, dtype=np.int64), device=self.device).unsqueeze(0)
        randn = decoding_orders.randn_for_order(order, s.length).to(self.device)

        with torch.no_grad():
            out = self.model(
                s.X,
                seq,
                s.mask,
                s.chain_M,
                s.residue_idx,
                s.chain_encoding_all,
                randn,
                use_input_decoding_order=True,
                decoding_order=order_t,
            )
        return out["log_probs"] if isinstance(out, dict) else out

    def conditional_log_probs_at(self, sequence, position: int, rng=None):
        """``log p(s_position | all other residues, X)`` -- a full-context conditional.

        Implemented by forcing a decoding order that places ``position`` last,
        so every other residue has already been embedded as context. This is the
        exact conditional a Gibbs sweep needs.

        Cost is one forward pass per position, so a full sweep over ``k``
        positions costs ``k`` forward passes. That is the honest price of exact
        conditionals; :func:`ipr.refine.gibbs_sweep` exposes a cheaper batched
        approximation for when it is too slow.
        """
        s = self.structure
        rng = rng or np.random.default_rng()
        others = np.array([p for p in range(s.length) if p != position], dtype=np.int64)
        rng.shuffle(others)
        order = np.concatenate([others, [position]])
        lp = self.log_probs(sequence, order=order)
        return lp[0, position]

    def score(self, sequence, order: np.ndarray | None = None, rng=None) -> dict:
        """Mean NLL and perplexity of ``sequence`` over designable positions."""
        from .metrics import negative_log_likelihood, perplexity

        lp = self.log_probs(sequence, order=order, rng=rng)
        mask = to_numpy(self.structure.chain_M).ravel()
        nll = negative_log_likelihood(lp, _as_numpy_seq(sequence), mask)
        return {"nll": nll, "perplexity": float(np.exp(nll)), "pseudo_perplexity": perplexity(lp, _as_numpy_seq(sequence), mask)}


def _as_tensor_like(sequence, reference):
    """Coerce a sequence to a tensor matching ``reference`` in shape, dtype and device."""
    torch = _torch()
    if isinstance(sequence, torch.Tensor):
        t = sequence
    else:
        t = torch.as_tensor(np.asarray(sequence, dtype=np.int64))
    t = t.to(device=reference.device, dtype=reference.dtype)
    return t if t.dim() == reference.dim() else t.unsqueeze(0)


def _as_numpy_seq(sequence) -> np.ndarray:
    return to_numpy(sequence).ravel().astype(np.int64)
