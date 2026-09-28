"""Normalization layer type enum.

Single source of truth for normalization layer selection across encoders,
models, and config dataclasses. Two values capture the core distinction:
per-sample statistics versus batch-dependent statistics.

Mapping to nn modules
    CHANNEL: nn.GroupNorm(num_groups=1, ...) for conv encoders (MCL,
              TSTCC, Series2Vec). nn.LayerNorm for transformer encoder
              layers (TST). Both are batch-size independent.
    BATCH:   nn.BatchNorm1d / nn.BatchNorm2d for conv encoders.
             TransformerBatchNormEncoderLayer for transformer (TST).
"""

from __future__ import annotations

from enum import StrEnum


class NormalizationLayerType(StrEnum):
    """Type of normalization layer to instantiate.

    Attributes:
        CHANNEL: Per-sample normalization. Uses GroupNorm(1, C) for conv
            feature maps and LayerNorm for transformer sequences. Batch-size
            independent — safe at batch_size=1.
        BATCH: Batch-dependent normalization. Uses BatchNorm1d/BatchNorm2d
            for conv and a custom BatchNorm encoder layer for transformers.
    """

    CHANNEL = "channel"
    BATCH = "batch"
