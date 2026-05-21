"""Provide convenient imports for the main model classes."""

from tscollection.models.cnn.dilated.autotcl.model import AutoTCL
from tscollection.models.cnn.dilated.cost.model import CoST
from tscollection.models.cnn.dilated.ts2vec.model import TS2Vec

__all__ = ["AutoTCL", "CoST", "TS2Vec"]
