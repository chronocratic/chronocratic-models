"""Encoding output-shape enum for encode() contracts.

Provides a shared, import-cycle-safe enum consumed by both mixin families
and model implementations.

Shape semantics
    VECTOR:   (N, D)      one representation vector per sample.
    SEQUENCE: (N, T, D)   one representation per timestep.
"""

from __future__ import annotations

import enum


class EncodingOutputShape(enum.Enum):
    """Requested shape of an encoder's output representation.

    Attributes:
        VECTOR: Returns a 2-D tensor of shape ``(N, D)`` with one
            representation vector per sample. Models that natively produce
            ``(N, T, D)`` apply their default reduction (e.g., last-step,
            mean-pool, or global average pooling).
        SEQUENCE: Returns a 3-D tensor of shape ``(N, T, D)`` with one
            representation per timestep. Models that natively produce flat
            vectors return a length-1 sequence with a warning.
    """

    VECTOR = "vector"
    SEQUENCE = "sequence"
