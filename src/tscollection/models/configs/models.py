"""Central re-export of all model configuration classes.

Provides backward-compatible import paths for model config classes
that now live in per-model files. Consumers can use::

    from tscollection.models.configs.models import TS2VecModelParameters

instead of the per-model paths.
"""

__all__ = [
    'AutoTCLModelParameters',
    'CoSTModelParameters',
    'DilatedCNNModelParameters',
    'ModelParameters',
    'TS2VecModelParameters',
]

from tscollection.models.config import ModelParameters
from tscollection.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from tscollection.models.convolutional.dilated.config import DilatedCNNModelParameters
from tscollection.models.convolutional.dilated.cost.config import CoSTModelParameters
from tscollection.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters
