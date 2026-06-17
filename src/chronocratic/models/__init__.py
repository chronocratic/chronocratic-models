"""Convenient imports for the main model classes and their configs."""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"  # fallback for editable installs before _version.py is generated

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
from .contracts import HasDecoder, HasEncoder
from .generative import TimeVAE, TimeVAEModelParameters
from .recurrent import TimeNet, TimeNetModelParameters
from .transformer import TST, TSTModelParameters

__all__ = [
    "AutoTCL",
    "AutoTCLModelParameters",
    "CoST",
    "CoSTModelParameters",
    "FCN",
    "HasDecoder",
    "HasEncoder",
    "MCLModelParameters",
    "Series2Vec",
    "Series2VecModelParameters",
    "TS2Vec",
    "TS2VecModelParameters",
    "TST",
    "TSTCC",
    "TSTCCModelParameters",
    "TSTModelParameters",
    "TimeNet",
    "TimeNetModelParameters",
    "TimeVAE",
    "TimeVAEModelParameters",
    "__version__",
]
