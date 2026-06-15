"""Configuration for the MCL (MixUp Contrastive Learning) model.

Provides MCLModelParameters with MCL-specific settings for the FCN
encoder and MixUp contrastive criterion.
"""

__all__ = ["MCLModelParameters"]

from dataclasses import dataclass


@dataclass(kw_only=True)
class MCLModelParameters:
    """Configuration for the MCL model.

    Args:
        n_in: Number of input features (channels) in the time series.
        output_dims: Number of output features produced by the encoder.
        alpha: Beta-distribution parameter controlling the MixUp
            interpolation coefficient.
        learning_rate: Base learning rate for the Adam optimizer.
    """

    n_in: int
    output_dims: int = 320
    alpha: float = 1.0
    learning_rate: float = 1e-3
