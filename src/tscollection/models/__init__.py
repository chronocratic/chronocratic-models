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
from .recurrent import (
    LSTMAutoEncoder,
    LSTMAutoEncoderModelParameters,
    TimeNet,
    TimeNetModelParameters,
)
from .transformer import TST, TSTModelParameters

__all__ = [
    'FCN',
    'TST',
    'TSTCC',
    'AutoTCL',
    'AutoTCLModelParameters',
    'CoST',
    'CoSTModelParameters',
    'LSTMAutoEncoder',
    'LSTMAutoEncoderModelParameters',
    'MCLModelParameters',
    'Series2Vec',
    'Series2VecModelParameters',
    'TS2Vec',
    'TS2VecModelParameters',
    'TSTCCModelParameters',
    'TSTModelParameters',
    'TimeNet',
    'TimeNetModelParameters',
    'TimeVAE',
    'TimeVAEModelParameters',
]
