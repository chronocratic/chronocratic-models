"""Configuration for the TS-TCC model.

Provides TSTCCModelParameters with all settings for the TCC encoder,
temporal contrast head, and NT-Xent contextual loss across the three
supported training modes.
"""

__all__ = ['TSTCCModelParameters']

from dataclasses import dataclass

from tscollection.models.convolutional.standard.tstcc.enums import TSTCCTrainingMode


@dataclass(kw_only=True)
class TSTCCModelParameters:
    """Configuration for the TS-TCC model.

    Args:
        input_channels: Number of input features (channels) in the time
            series.
        kernel_size: Convolutional kernel size used in the TCC encoder.
        stride: Convolutional stride used in the TCC encoder.
        final_out_channels: Number of channels produced by the final
            encoder block (also used as the temporal-contrast input dim).
        features_len: Length of the encoder feature map fed into the
            logits head.
        num_classes: Number of output classes for the supervised /
            fine-tuning logits head.
        dropout: Dropout probability applied inside the TCC encoder.
        tc_hidden_dim: Hidden dimensionality of the temporal-contrast
            module.
        tc_timesteps: Number of future timesteps the temporal-contrast
            module predicts.
        temperature: Temperature scaling for the NT-Xent contextual loss.
        use_cosine_similarity: Whether the NT-Xent loss uses cosine
            similarity (otherwise dot-product).
        training_mode: A ``TSTCCTrainingMode`` value. Controls which loss is
            active and which parameters are trainable.
        learning_rate: Base learning rate for the two Adam optimizers
            (encoder and temporal-contrast).
        lambda1: Weight of the temporal-contrast loss term in the
            self-supervised objective.
        lambda2: Weight of the contextual NT-Xent loss term in the
            self-supervised objective.
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
    """

    input_channels: int
    kernel_size: int
    stride: int
    final_out_channels: int
    features_len: int
    num_classes: int
    dropout: float = 0.35
    tc_hidden_dim: int = 100
    tc_timesteps: int = 6
    temperature: float = 0.2
    use_cosine_similarity: bool = True
    training_mode: TSTCCTrainingMode = TSTCCTrainingMode.SELF_SUPERVISED
    learning_rate: float = 3e-4
    lambda1: float = 1.0
    lambda2: float = 0.7
    sync_dist: bool = False
