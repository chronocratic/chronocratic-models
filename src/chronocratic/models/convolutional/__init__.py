"""Barrel for the convolutional model family."""

from __future__ import annotations

from .dilated import (
    AutoTCL,
    AutoTCLModelParameters,
    CoST,
    CoSTModelParameters,
    TS2Vec,
    TS2VecModelParameters,
)
from .standard import (
    FCN,
    MCLModelParameters,
    Series2Vec,
    Series2VecModelParameters,
    TSTCC,
    TSTCCModelParameters,
)

__all__ = [
    "FCN",
    "TSTCC",
    "AutoTCL",
    "AutoTCLModelParameters",
    "CoST",
    "CoSTModelParameters",
    "MCLModelParameters",
    "Series2Vec",
    "Series2VecModelParameters",
    "TS2Vec",
    "TS2VecModelParameters",
    "TSTCCModelParameters",
]
