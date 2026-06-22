"""Configuration for the RecurrentAutoEncoder model."""

from __future__ import annotations

__all__ = ["RecurrentAutoEncoderModelParameters"]

from dataclasses import dataclass

from chronocratic.models.recurrent.enums import OptimizerName, ReconstructionLoss, RecurrentCellType


@dataclass(kw_only=True)
class RecurrentAutoEncoderModelParameters:
    """Configuration for the recurrent autoencoder model.

    Args:
        input_dims: Number of input features (channels) per timestep.
        layers: Hidden sizes for each encoder RNN layer, e.g. ``(64, 32)``.
            The decoder uses the reversed order.
        recurrent_cell_type: RNN variant — LSTM, GRU, or RNN.
        dropout: Dropout probability applied between successive layers. A single
            float applies uniformly; a tuple must match ``len(layers)``.
        loss: Reconstruction objective — ``'mse'`` or ``'mae'``.
        optimizer: Optimizer — ``'adam'``, ``'adamw'``, or ``'radam'``.
        learning_rate: Base learning rate for the optimizer.
        sync_dist: Whether to sync logged metrics across devices.
    """

    input_dims: int
    layers: tuple[int, ...]
    recurrent_cell_type: RecurrentCellType = RecurrentCellType.LSTM
    dropout: float | tuple[float, ...] = 0.2
    loss: ReconstructionLoss = ReconstructionLoss.MSE
    optimizer: OptimizerName = OptimizerName.ADAM
    learning_rate: float = 1e-3
    sync_dist: bool = False
