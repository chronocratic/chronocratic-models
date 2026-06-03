__all__ = ['TS2Vec']


from typing import cast

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from tscollection.models.augmentation.base import AugmentationMethod
from tscollection.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin
from tscollection.models.convolutional.dilated.encoders.encoders import TS2VecTimeSeriesEncoder
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode
from tscollection.models.convolutional.dilated.ts2vec.losses import hierarchical_contrastive_loss
from tscollection.models.utils import extract_features_from_batch, process_sample_length


class TS2Vec(pl.LightningModule, PoolingEncodingMixin):
    """TS2Vec Model.

    Code source: https://github.com/zhihanyue/ts2vec
    """

    def __init__(
        self,
        *,
        input_dims: int,
        augmentation: AugmentationMethod | None = None,
        hidden_dims: int = 64,
        output_dims: int = 320,
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

        self.save_hyperparameters(ignore=['augmentation'])

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._temporal_unit = temporal_unit
        self._sync_dist = sync_dist

        if augmentation is None:
            from tscollection.models.convolutional.dilated.ts2vec.augmentation import (  # noqa: PLC0415
                CropShiftAugmentation,
            )

            self._augmentation: AugmentationMethod = CropShiftAugmentation()
        else:
            self._augmentation = augmentation

        self.automatic_optimization = False

        self._encoder = TS2VecTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self._averaged_encoder = AveragedModel(self._encoder)
        self._averaged_encoder.update_parameters(self._encoder)

    @property
    def encoder(self) -> TS2VecTimeSeriesEncoder:
        """Return the primary (non-averaged) encoder for inspection and checkpointing."""
        return cast('TS2VecTimeSeriesEncoder', self._encoder)

    def configure_optimizers(self) -> AdamW:
        """Return the AdamW optimizer for the TS2Vec encoder."""
        optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        return optimizer

    def _calculate_encoder_loss(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        return hierarchical_contrastive_loss(
            embeddings_1, embeddings_2, temporal_unit=self._temporal_unit
        )

    def _encode_augmented_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment ``x`` and encode both views, slicing by ``crop_length``."""
        views = self._augmentation.augment(x, temporal_unit=self._temporal_unit)
        crop_length = views.metadata['crop_length']

        encoder = self._encoder if self.training else self._averaged_encoder
        emb_1 = encoder(views.views[0])[:, -crop_length:]
        emb_2 = encoder(views.views[1])[:, :crop_length]

        return emb_1, emb_2

    def training_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Run one TS2Vec training step with manual optimization."""
        x = extract_features_from_batch(batch)

        optimizer = cast('torch.optim.Optimizer', self.optimizers())

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        embeddings_1, embeddings_2 = self._encode_augmented_views(x)

        train_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

        self.log(
            'train_loss',
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
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the TS2Vec validation loss."""
        x = extract_features_from_batch(batch)

        embeddings_1, embeddings_2 = self._encode_augmented_views(x)

        val_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

        self.log(
            'val_loss',
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return val_loss
