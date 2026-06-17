"""Configuration for the RecurrentAutoEncoder model.

Provides RecurrentAutoEncoderModelParameters with settings for the recurrent
encoder / decoder pair and the pretraining objective.
"""

from __future__ import annotations

__all__ = ["RecurrentAutoEncoderModelParameters"]

from dataclasses import dataclass, field

from chronocratic.models.recurrent.enums import RecurrentCellType
from chronocratic.models.recurrent.recurrentae.enums import RecurrentEncoderLoss


@dataclass(kw_only=True)
class RecurrentAutoEncoderModelParameters:
    """Configuration for the recurrent autoencoder model.

    Args:
        n_features: Number of input features (channels) per timestep.
        latent_dim: Dimensionality of the RNN hidden state used as the
            bottleneck representation.
        recurrent_cell_type: Recurrent cell variant — ``'LSTM'``, ``'GRU'``, or
            ``'RNN'``.
        loss: Training objective — ``'MSE'`` (mean squared error) or
            ``'MAE'`` (mean absolute error).
        learning_rate: Base learning rate for the Adam optimizer.
    """

    n_features: int
    latent_dim: int
    recurrent_cell_type: RecurrentCellType = RecurrentCellType.LSTM
    loss: RecurrentEncoderLoss = field(default=RecurrentEncoderLoss.MSE)
    learning_rate: float = 1e-3
