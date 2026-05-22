__all__ = [
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentationParameters',
]

from dataclasses import dataclass, field

from tscollection.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass
class CropShiftAugmentationParameters:
    """Parameters for :class:`CropShiftAugmentation`.

    Controls the temporal granularity of the random crop-and-shift
    augmentation used by TS2Vec.

    Args:
        temporal_unit: Controls the minimum crop length as
            ``2 ** (temporal_unit + 1)``. Defaults to ``0``.
    """

    temporal_unit: int = 0


@dataclass
class CosTRandomFunctionAugmentationParameters:
    """Parameters for :class:`CosTRandomFunctionAugmentation`.

    Controls the stochastic jitter/scale/shift transforms used by CoST.

    Args:
        sigma: Noise scale for jitter, magnitude for scale, offset
            for shift.
        p: Probability of applying each individual transform
            (default ``0.5``).
    """

    sigma: float
    p: float = 0.5


@dataclass
class AutoTCLNeuralNetworkAugmentationParameters:
    """Parameters for :class:`AutoTCLNeuralNetworkAugmentation`.

    Mirrors the constructor signature of
    :class:`AutoTCLAugmentationTimeSeriesEncoder`, allowing ``vars()``
    unpacking to instantiate the encoder directly.

    Args:
        input_dims: Number of input features (channels).
        output_dims: Number of output features produced by the encoder.
        kernel_sizes: DWT decomposition levels as kernel sizes. Empty
            list means the encoder selects levels automatically.
        hidden_dims: Number of hidden units in each encoder layer.
        feature_extractor_depth: Number of encoder layers.
        dropout_rate: Dropout probability after each encoder layer.
        conv_kernel_size: Size of the convolutional kernel.
        mask_mode: Strategy for masking input tokens.
        num_augmentation_channels: Number of augmentation output channels.
        gumbel_bias: Bias term for the Gumbel-sigmoid mask sampling.
        zeta: Scaling factor for the Gumbel temperature.
        gamma_zeta: Weight for the zeta regularization term.
        hard_mask: Whether to use a hard (binary) mask at inference time.
    """

    input_dims: int
    output_dims: int
    kernel_sizes: list[int] = field(default_factory=list)
    hidden_dims: int = 64
    feature_extractor_depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    num_augmentation_channels: int = 1
    gumbel_bias: float = 0.001
    zeta: float = 1.0
    gamma_zeta: float = 0.05
    hard_mask: bool = True
