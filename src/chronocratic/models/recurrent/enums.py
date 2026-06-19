__all__ = ["OptimizerName", "ReconstructionLoss", "RecurrentCellType"]

from enum import StrEnum


class OptimizerName(StrEnum):
    """Optimizer variants for recurrent autoencoder training."""

    ADAM = "adam"
    ADAMW = "adamw"
    RADAM = "radam"


class RecurrentCellType(StrEnum):
    """Recurrent cell variants for the autoencoder backbone."""

    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"


class ReconstructionLoss(StrEnum):
    """Reconstruction loss functions for the autoencoder objective."""

    MSE = "mse"
    MAE = "mae"
