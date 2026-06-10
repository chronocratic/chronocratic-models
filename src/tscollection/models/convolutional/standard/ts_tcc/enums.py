__all__ = ['TSTCCTrainingMode']

from enum import Enum


class TSTCCTrainingMode(Enum):
    """Training modes supported by the TS-TCC model.

    Attributes:
        SELF_SUPERVISED: Temporal + contextual contrastive pre-training on
            augmented views; labels are ignored.
        SUPERVISED: Standard cross-entropy training on labeled data.
        FINE_TUNING: Cross-entropy training with only the logits head
            trainable; backbone weights are frozen.
    """

    SELF_SUPERVISED = 'self_supervised'
    SUPERVISED = 'supervised'
    FINE_TUNING = 'fine_tuning'
