"""Configuration for the LSTM Autoencoder model.

Provides LSTMAutoEncoderModelParameters with settings for the recurrent
encoder / decoder pair and the pretraining objective.
"""

from __future__ import annotations

__all__ = ['LSTMAutoEncoderModelParameters']

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal


@dataclass(kw_only=True)
class LSTMAutoEncoderModelParameters:
    """Configuration for the recurrent autoencoder model.

    Args:
        n_features: Number of input features (channels) per timestep.
        latent_dim: Dimensionality of the RNN hidden state used as the
            bottleneck representation.
        rnn_type: Recurrent cell variant — ``'LSTM'``, ``'GRU'``, or
            ``'RNN'``.
        loss_type: Training objective — ``'MSE'`` (mean squared error) or
            ``'MAE'`` (mean absolute error).
        learning_rate: Base learning rate for the Adam optimizer.
    """

    n_features: int
    latent_dim: int
    rnn_type: Literal['LSTM', 'GRU', 'RNN'] = field(default='LSTM')
    loss_type: Literal['MSE', 'MAE'] = field(default='MSE')
    learning_rate: float = 1e-3
