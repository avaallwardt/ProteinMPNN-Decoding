"""A dependency-free Potts-model stand-in for ProteinMPNN.

Why this exists
---------------
Every refinement algorithm in :mod:`ipr.refine` is a statement about
autoregressive decoding in general, not about ProteinMPNN in particular. Those
statements are testable without a GPU, model weights, or PyTorch -- and they
should be, because a bug in the refinement logic is otherwise indistinguishable
from a bug in the model wrapper.

:class:`ToyPottsModel` is a small pairwise Markov random field over a random
geometric contact graph. It is useful here for one specific reason: a Potts
model has a genuine joint distribution, so its single-site conditionals are
mutually consistent and Gibbs sampling provably converges to it. That makes it
a positive control. If a refinement method fails to improve likelihood on this
model, the method is broken; if it improves here but not on ProteinMPNN, the
difference is a property of ProteinMPNN, which is itself a finding.

What it is not
--------------
It is not a protein model, and no number produced from it belongs in a results
table about sequence recovery. Its "native" sequence is a low-energy
configuration found by annealing, not a biological sequence.

The class deliberately reproduces one ProteinMPNN quirk: ``sample`` returns
probabilities that are **zero at fixed positions**, because that trap has
already caused one bug in this project and code should be tested against it.
"""

from __future__ import annotations

import numpy as np

from .alphabet import N_TOKENS


class ToyPottsModel:
    """A pairwise MRF with ProteinMPNN's decoding interface.

    Parameters
    ----------
    length:
        Number of positions.
    n_tokens:
        Alphabet size; defaults to 21 to match ProteinMPNN.
    contact_cutoff:
        Distance cutoff defining the contact graph, in arbitrary units.
    coupling_strength:
        Scale of the pairwise terms relative to the single-site fields. Larger
        values make context matter more, which widens the gap between a
        single-pass decode and a fully-conditioned one.
    seed:
        Seed for the model parameters. Independent of the RNG passed to
        sampling calls, so the same model can be sampled many ways.
    """

    def __init__(
        self,
        length: int = 40,
        n_tokens: int = N_TOKENS,
        contact_cutoff: float = 1.1,
        coupling_strength: float = 1.5,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.length = int(length)
        self.n_tokens = int(n_tokens)
        self.coords = rng.normal(size=(self.length, 3))

        dist = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)
        self.contacts = (dist < contact_cutoff) & ~np.eye(self.length, dtype=bool)
        # Chain connectivity, so the graph is never disconnected.
        for i in range(self.length - 1):
            self.contacts[i, i + 1] = self.contacts[i + 1, i] = True

        self.h = rng.normal(scale=1.0, size=(self.length, self.n_tokens))
        j = rng.normal(scale=coupling_strength, size=(self.length, self.length, self.n_tokens, self.n_tokens))
        # Symmetrize so the energy is well defined, and zero out non-contacts.
        self.J = 0.5 * (j + j.transpose(1, 0, 3, 2))
        self.J *= self.contacts[:, :, None, None]

        self.native = self._anneal(rng)

    # ------------------------------------------------------------------ #
    # Energy
    # ------------------------------------------------------------------ #
    def energy(self, sequence) -> float:
        """Total energy of a configuration. Lower is more favourable."""
        s = np.asarray(sequence, dtype=np.int64)
        single = self.h[np.arange(self.length), s].sum()
        pair = self.J[np.arange(self.length)[:, None], np.arange(self.length)[None, :], s[:, None], s[None, :]]
        return float(single + 0.5 * (pair * self.contacts).sum())

    def _site_logits(self, position: int, sequence, visible: np.ndarray) -> np.ndarray:
        """Negative energy of each token at ``position`` given the visible context."""
        s = np.asarray(sequence, dtype=np.int64)
        neighbors = np.flatnonzero(self.contacts[position] & visible)
        logits = -self.h[position].copy()
        if neighbors.size:
            logits -= self.J[position, neighbors, :, s[neighbors]].sum(axis=0)
        return logits

    def _anneal(self, rng, n_sweeps: int = 300) -> np.ndarray:
        """Find a low-energy configuration to serve as the ground-truth sequence."""
        s = rng.integers(0, self.n_tokens, size=self.length)
        visible = np.ones(self.length, dtype=bool)
        for t in range(n_sweeps):
            temp = max(0.05, 2.0 * (1.0 - t / n_sweeps))
            for p in rng.permutation(self.length):
                logits = self._site_logits(p, s, visible & (np.arange(self.length) != p))
                s[p] = _sample_from_logits(logits, temp, rng)
        return s.astype(np.int64)

    # ------------------------------------------------------------------ #
    # ProteinMPNN-compatible interface
    # ------------------------------------------------------------------ #
    def sample(
        self,
        temperature: float = 0.1,
        order=None,
        redesign=None,
        context_sequence=None,
        rng=None,
    ) -> dict:
        """Autoregressively sample, holding non-redesigned positions as context.

        Mirrors :meth:`ipr.backends.Runner.sample`, including the convention
        that ``redesign=None`` redesigns everything and therefore ignores
        ``context_sequence`` entirely.
        """
        rng = rng or np.random.default_rng()
        context = (
            np.asarray(self.native if context_sequence is None else context_sequence, dtype=np.int64).copy()
        )

        if redesign is None:
            redesign_idx = np.arange(self.length)
        else:
            redesign_idx = np.asarray(list(redesign), dtype=np.int64)

        is_redesigned = np.zeros(self.length, dtype=bool)
        is_redesigned[redesign_idx] = True

        order = rng.permutation(self.length) if order is None else np.asarray(order, dtype=np.int64)
        rank = np.empty(self.length, dtype=np.int64)
        rank[order] = np.arange(self.length)

        # Fixed positions decode first and become context; redesigned positions
        # follow in the requested relative order. This is exactly what
        # ProteinMPNN's argsort((chain_mask + 1e-4) * |randn|) produces.
        fixed = np.flatnonzero(~is_redesigned)
        to_design = np.flatnonzero(is_redesigned)
        decoding_order = np.concatenate([fixed[np.argsort(rank[fixed])], to_design[np.argsort(rank[to_design])]])

        s = context.copy()
        visible = ~is_redesigned  # fixed positions are context from the start
        probs = np.zeros((self.length, self.n_tokens))

        for p in decoding_order:
            if not is_redesigned[p]:
                continue
            logits = self._site_logits(int(p), s, visible)
            dist = _softmax(logits / max(temperature, 1e-8))
            probs[p] = dist  # zero elsewhere, matching ProteinMPNN's masked probs
            s[p] = rng.choice(self.n_tokens, p=dist)
            visible[p] = True

        return {
            "S": s[None, :],
            "probs": probs[None, :, :],
            "decoding_order": decoding_order[None, :],
        }

    def log_probs(self, sequence, order=None, rng=None) -> np.ndarray:
        """Teacher-forced log-probabilities under a given decoding order.

        Entry ``(0, t, :)`` is ``log p(s_t | s_{<t in order})``. As with the real
        model, taking an argmax of this against the native sequence measures
        teacher-forced accuracy, not design recovery.
        """
        rng = rng or np.random.default_rng()
        s = np.asarray(sequence, dtype=np.int64).ravel()
        order = rng.permutation(self.length) if order is None else np.asarray(order, dtype=np.int64)

        out = np.zeros((self.length, self.n_tokens))
        visible = np.zeros(self.length, dtype=bool)
        for p in order:
            logits = self._site_logits(int(p), s, visible)
            out[p] = np.log(np.clip(_softmax(logits), 1e-12, None))
            visible[p] = True
        return out[None, :, :]

    def conditional_log_probs_at(self, sequence, position: int, rng=None) -> np.ndarray:
        """``log p(s_position | every other position)`` -- exact for a Potts model."""
        s = np.asarray(sequence, dtype=np.int64).ravel()
        visible = np.ones(self.length, dtype=bool)
        visible[position] = False
        logits = self._site_logits(int(position), s, visible)
        return np.log(np.clip(_softmax(logits), 1e-12, None))

    @property
    def designable_mask(self) -> np.ndarray:
        return np.ones(self.length)

    def neighbor_counts(self, cutoff: float | None = None) -> np.ndarray:
        """Contact count per position, the toy analogue of burial."""
        return self.contacts.sum(axis=1).astype(np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _sample_from_logits(logits: np.ndarray, temperature: float, rng) -> int:
    return int(rng.choice(logits.shape[0], p=_softmax(logits / max(temperature, 1e-8))))
