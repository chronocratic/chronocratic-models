"""Convenient imports for the main model classes and their configs."""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"  # fallback for editable installs before _version.py is generated

from .enums import EncodingOutputShape
from .convolutional import (
    AutoTCL,
    AutoTCLModelParameters,
    CoST,
    CoSTModelParameters,
    MCL,
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
    RecurrentAutoEncoder,
    RecurrentAutoEncoderModelParameters,
    TimeNet,
    TimeNetModelParameters,
)
from .transformer import TST, TSTModelParameters

__all__ = [
    "EncodingOutputShape",
    "MCL",
    "MCLModelParameters",
    "TST",
    "TSTCC",
    "TSTCCModelParameters",
    "AutoTCL",
    "AutoTCLModelParameters",
    "CoST",
    "CoSTModelParameters",
    "RecurrentAutoEncoder",
    "RecurrentAutoEncoderModelParameters",
    "Series2Vec",
    "Series2VecModelParameters",
    "TS2Vec",
    "TS2VecModelParameters",
    "TSTModelParameters",
    "TimeNet",
    "TimeNetModelParameters",
    "TimeVAE",
    "TimeVAEModelParameters",
    "__version__",
]
