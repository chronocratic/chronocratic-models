"""Configuration for the RecurrentAutoEncoder model."""

from __future__ import annotations

__all__ = ["RecurrentAutoEncoderModelParameters"]

from dataclasses import dataclass
from typing import Literal

from chronocratic.models.recurrent.enums import RecurrentCellType


@dataclass(kw_only=True)
class RecurrentAutoEncoderModelParameters:
    """Configuration for the recurrent autoencoder model.

    Args:
        n_features: Number of input features (channels) per timestep.
        layers: Hidden sizes for each encoder RNN layer, e.g. ``[64, 32]``.
            The decoder uses the reversed order.
        recurrent_cell_type: RNN variant — LSTM, GRU, or RNN.
        dropout: Dropout probability applied between successive layers. A single
            float applies uniformly; a list must match ``len(layers)``.
        loss: Reconstruction objective — ``'mse'`` or ``'mae'``.
        learning_rate: Base learning rate for the Adam optimizer.
        sync_dist: Whether to sync logged metrics across devices.
    """

    n_features: int
    layers: tuple[int]
    recurrent_cell_type: RecurrentCellType = RecurrentCellType.LSTM
    dropout: float | list[float] = 0.2
    loss: Literal["mse", "mae"] = "mse"
    learning_rate: float = 1e-3
    sync_dist: bool = False
