"""TimeVAE-specific enums.

``ResidualProjectionType`` lives here rather than in ``models.enums`` because
it is only used by TimeVAE's residual decoder branch (see
``ResidualConnection`` in ``chronocratic.models.layers.general``).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ResidualProjectionType"]


class ResidualProjectionType(StrEnum):
    """How TimeVAE's residual decoder maps its deconvolution output to ``(B, T, C)``.

    Attributes:
        DENSE: Upstream TimeVAE: flatten and apply ``Linear(C * L, C * T)``. Matches the
            original implementation exactly. Default, for parity. Costs O((C * T)^2)
            parameters — e.g. ~676 M at T=5200, C=5 (~10.8 GB of training memory) — so
            large ``sequence_length`` / ``input_dim`` combinations can exhaust memory.
        CROP: Keep the first ``T`` steps of the deconvolution output (which always has at
            least ``T``). The last deconvolution is linear so residuals can be negative.
            No extra parameters. Use this instead of DENSE when the upstream-parity cost
            is too high.
    """

    DENSE = "dense"
    CROP = "crop"
