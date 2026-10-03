"""Barrel for standard-convolution models (MCL, MHCCL, Series2Vec, SimCLR, TS-TCC)."""

from __future__ import annotations

from .mcl import MCL, MCLModelParameters
from .mhccl import MHCCL, MHCCLModelParameters
from .series2vec import Series2Vec, Series2VecModelParameters
from .simclr import SimCLR, SimCLRModelParameters
from .tstcc import TSTCC, TSTCCModelParameters

__all__ = [
    "MCL",
    "MHCCL",
    "TSTCC",
    "MCLModelParameters",
    "MHCCLModelParameters",
    "Series2Vec",
    "Series2VecModelParameters",
    "SimCLR",
    "SimCLRModelParameters",
    "TSTCCModelParameters",
]
