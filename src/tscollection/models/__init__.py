"""Convenient imports for the main model classes and their configs."""

from __future__ import annotations

from .convolutional import (
    AutoTCL,
    AutoTCLModelParameters,
    CoST,
    CoSTModelParameters,
    FCN,
    MCLModelParameters,
    Series2Vec,
    Series2VecModelParameters,
    TS2Vec,
    TS2VecModelParameters,
    TSTCC,
    TSTCCModelParameters,
)
from .generative import TimeVAE, TimeVAEModelParameters
from .recurrent import TimeNet, TimeNetModelParameters
from .transformer import TST, TSTModelParameters

__all__ = [
    'AutoTCL',
    'AutoTCLModelParameters',
    'CoST',
    'CoSTModelParameters',
    'FCN',
    'MCLModelParameters',
    'Series2Vec',
    'Series2VecModelParameters',
    'TS2Vec',
    'TS2VecModelParameters',
    'TST',
    'TSTCC',
    'TSTCCModelParameters',
    'TSTModelParameters',
    'TimeNet',
    'TimeNetModelParameters',
    'TimeVAE',
    'TimeVAEModelParameters',
]
