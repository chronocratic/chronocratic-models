"""Configuration for the RecurrentAutoEncoder model."""

from __future__ import annotations

__all__ = ["RecurrentAutoEncoderModelParameters"]

from dataclasses import dataclass

from chronocratic.models.recurrent.enums import OptimizerName, ReconstructionLoss, RecurrentCellType


@dataclass(kw_only=True)
class RecurrentAutoEncoderModelParameters:
    """Configuration for the recurrent autoencoder model.

    Args:
        input_dim: Number of input features (channels) per timestep.
        layers: Hidden sizes for each encoder RNN layer, e.g. ``(16, 8)``.
            The decoder uses the reversed order. Defaults to ``(16, 8)``
            matching the source Recurrent-Autoencoder (latent_dim=8).
        recurrent_cell_type: RNN variant — LSTM, GRU, or RNN.
        dropout: Dropout probability applied between successive layers. A single
            float applies uniformly; a tuple must match ``len(layers)``.
        loss: Reconstruction objective — ``'mse'`` or ``'mae'``.
        optimizer: Optimizer — ``'adam'``, ``'adamw'``, or ``'radam'``.
        learning_rate: Base learning rate for the optimizer.
        sync_dist: Whether to sync logged metrics across devices.
        num_covariate_channels: Number of leading channels that are known
            covariates rather than targets, as prepended by the forecasting
            datamodules (input is ``[covariates | raw channels]``). They are
            still fed to the encoder but excluded from the reconstruction
            loss. The decoder keeps its full width. Must satisfy
            ``0 <= num_covariate_channels < input_dim``.
    """

    input_dim: int
    layers: tuple[int, ...] = (16, 8)
    recurrent_cell_type: RecurrentCellType = RecurrentCellType.LSTM
    dropout: float | tuple[float, ...] = 0.2
    loss: ReconstructionLoss = ReconstructionLoss.MSE
    optimizer: OptimizerName = OptimizerName.ADAM
    learning_rate: float = 1e-3
    sync_dist: bool = False
    num_covariate_channels: int = 0
