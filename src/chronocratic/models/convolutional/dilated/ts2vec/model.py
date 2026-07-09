__all__ = ["TS2Vec"]


from typing import cast

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from chronocratic.models.augmentation.base import AlignedPair, AugmentationProducer
from chronocratic.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin
from chronocratic.models.convolutional.dilated.encoders.encoders import TS2VecTimeSeriesEncoder
from chronocratic.models.convolutional.dilated.encoders.masking import MaskMode
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.losses import hierarchical_contrastive_loss
from chronocratic.models.utils import extract_features_from_batch, process_sample_length


class TS2Vec(pl.LightningModule, PoolingEncodingMixin):
    """TS2Vec model.

    Learns ordered representation through hierarchical contrastive
    learning at multiple scales. Uses dilated convolutions with
    masking strategies for self-supervised pretraining.

    Args:
        input_dim: Number of input features (channels).
        augmentation: Custom augmentation producer. Defaults to
            CropShiftProducer.
        hidden_dim: Number of hidden units in each encoder layer.
        representation_dim: Number of output features produced by the encoder.
        depth: Number of encoder layers.
        dropout_rate: Dropout probability applied after each encoder layer.
        conv_kernel_size: Size of the convolutional kernel in each layer.
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        max_train_length: Maximum sequence length; longer samples are
            truncated. ``None`` means no limit.
        temporal_unit: Token-level temporal unit index.
        sync_dist: Whether to synchronize metrics across distributed
            processes.

    Code source: https://github.com/zhihanyue/ts2vec
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        *,
        input_dim: int,
        augmentation: AugmentationProducer[AlignedPair] | None = None,
        hidden_dim: int = 64,
        representation_dim: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        max_train_length: int | None = None,
        temporal_unit: int = 0,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["augmentation"])

        self._representation_dim = representation_dim
        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._temporal_unit = temporal_unit
        self._sync_dist = sync_dist

        if augmentation is None:
            from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (  # noqa: PLC0415
                CropShiftProducer,
            )

            self._augmentation: AugmentationProducer[AlignedPair] = CropShiftProducer()
        else:
            self._augmentation = augmentation

        self.automatic_optimization = False

        self._encoder = TS2VecTimeSeriesEncoder(
            input_dim=input_dim,
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self._averaged_encoder = AveragedModel(self._encoder)
        self._averaged_encoder.update_parameters(self._encoder)

    @property
    def representation_dim(self) -> int:
        """Return the feature dim of the ``encode()`` output.

        This is the width of the representation vector produced by the
        encoder, matching the ``representation_dim`` configuration
        parameter.
        """
        return self._representation_dim

    @property
    def encoder(self) -> TS2VecTimeSeriesEncoder:
        """Return the averaged encoder used for inference.

        Matches the module returned by ``_get_encoder()`` so that the
        ``HasEncoder`` protocol is consistent with the encode() path.
        """
        return cast("TS2VecTimeSeriesEncoder", self._averaged_encoder)

    def configure_optimizers(self) -> AdamW:
        """Return the AdamW optimizer for the TS2Vec encoder."""
        optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        return optimizer

    def _calculate_encoder_loss(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        return hierarchical_contrastive_loss(
            instance_1=embeddings_1, instance_2=embeddings_2, temporal_unit=self._temporal_unit
        )

    def _encode_augmented_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment ``x`` and encode both views, slicing by ``overlap_length``.

        Clamps the overlap to the actual temporal length of both embeddings to
        prevent misaligned or truncated outputs when the augmentation produces
        sequences shorter than ``overlap_length``.
        """
        pair = self._augmentation.produce(x)

        encoder = self._encoder if self.training else self._averaged_encoder
        emb_1 = encoder(pair.first)
        emb_2 = encoder(pair.second)

        overlap = min(pair.overlap_length, emb_1.size(1), emb_2.size(1))
        emb_1 = emb_1[:, -overlap:]
        emb_2 = emb_2[:, :overlap]

        return emb_1, emb_2

    def training_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run one TS2Vec training step with manual optimization."""
        x = extract_features_from_batch(batch)

        optimizer = cast("torch.optim.Optimizer", self.optimizers())

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        embeddings_1, embeddings_2 = self._encode_augmented_views(x)

        train_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        optimizer.zero_grad()
        self.manual_backward(train_loss)
        optimizer.step()
        if isinstance(self._averaged_encoder, AveragedModel):
            self._averaged_encoder.update_parameters(self._encoder)

        return train_loss

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute and log the TS2Vec validation loss."""
        x = extract_features_from_batch(batch)

        embeddings_1, embeddings_2 = self._encode_augmented_views(x)

        val_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return val_loss
