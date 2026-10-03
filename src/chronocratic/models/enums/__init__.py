"""Enum types for encoding contracts, clustering, and layer configuration."""

from __future__ import annotations

from .blocks import ResidualBlockType
from .clustering import OutlierMaskMode
from .encoding import EncodingOutputShape
from .layers import NormalizationLayerType

__all__ = ["EncodingOutputShape", "NormalizationLayerType", "OutlierMaskMode", "ResidualBlockType"]
