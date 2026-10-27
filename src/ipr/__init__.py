"""Iterative refinement of autoregressive protein sequence design.

Submodules
----------
:mod:`ipr.conditioning`
    Chain-mask construction. Read this first -- it explains why a draft
    sequence is only used as context when its positions are held fixed.
:mod:`ipr.decoding_orders`
    Decoding-order strategies, and how to force an exact order on ProteinMPNN.
:mod:`ipr.uncertainty`
    Per-position uncertainty measures and residue-selection policies.
:mod:`ipr.refine`
    The refinement strategies themselves, including a faithful reproduction of
    the original (non-conditioning) method for comparison.
:mod:`ipr.metrics`
    Recovery, likelihood, and paired-comparison statistics.
:mod:`ipr.backends`
    ProteinMPNN wrapper. Requires PyTorch and model weights.
:mod:`ipr.toy`
    Potts-model stand-in that implements the same interface with no
    dependencies, used by the test suite.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import conditioning, decoding_orders, metrics, refine, toy, uncertainty

__all__ = [
    "conditioning",
    "decoding_orders",
    "metrics",
    "refine",
    "toy",
    "uncertainty",
    "__version__",
]


def __getattr__(name: str):
    """Expose :mod:`ipr.backends` lazily so importing ``ipr`` never requires PyTorch."""
    if name == "backends":
        from . import backends

        return backends
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
