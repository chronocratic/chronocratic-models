__all__ = ["RecurrentEncoderLoss"]

from enum import StrEnum


class RecurrentEncoderLoss(StrEnum):
    MSE = "mse"
    MAE = "mae"
