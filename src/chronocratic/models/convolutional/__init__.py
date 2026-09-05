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
    MCL,
    MCLModelParameters,
    MHCCL,
    MHCCLModelParameters,
    Series2Vec,
    Series2VecModelParameters,
    SimCLR,
    SimCLRModelParameters,
    TSTCC,
    TSTCCModelParameters,
)

__all__ = [
    "MCL",
    "MHCCL",
    "TSTCC",
    "AutoTCL",
    "AutoTCLModelParameters",
    "CoST",
    "CoSTModelParameters",
    "MCLModelParameters",
    "MHCCLModelParameters",
    "Series2Vec",
    "Series2VecModelParameters",
    "SimCLR",
    "SimCLRModelParameters",
    "TS2Vec",
    "TS2VecModelParameters",
    "TSTCCModelParameters",
]
