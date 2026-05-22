__all__ = ['TS2Vec']


import lightning.pytorch as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from tscollection.models.cnn.dilated._mixin.encoding import PoolingEncodingMixin
from tscollection.models.augmentation.enums import TS2VecAugmentationMode
from tscollection.models.augmentation.factories import TS2VecAugmentationMethodFactory
from tscollection.models.cnn.dilated.encoders.encoders import TS2VecTimeSeriesEncoder
from tscollection.models.cnn.dilated.encoders.masking import MaskMode
from tscollection.models.config import TS2VecModelParameters
from tscollection.models.losses import hierarchical_contrastive_loss
from tscollection.models.utils import extract_features_from_batch, process_sample_length


class TS2Vec(pl.LightningModule, PoolingEncodingMixin):
    def __init__(
        self,
        input_dims: int,
        augmentation_mode: TS2VecAugmentationMode,
        augmentation_method_params: dict,
        augmentation_mode_params: dict | None = None,  ## noqa: ARG002
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        max_train_length: int | None = None,
        temporal_unit: int = 0,
        sync_dist: bool = False,  ## noqa: FBT001 FBT002
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._temporal_unit = temporal_unit
        self._sync_dist = sync_dist

        self._augmentation_mode = augmentation_mode

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

        self._init_augmentation_method(augmentation_method_params)

        self._init_augmentation_mode_params()

    def configure_optimizers(self) -> AdamW:
        """Return the AdamW optimizer for the TS2Vec encoder."""
        optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        return optimizer

    @classmethod
    def from_config(cls, config: TS2VecModelParameters, **additional_kwargs: object) -> 'TS2Vec':
        """Instantiate TS2Vec from a typed config dataclass.

        Args:
            config: TS2Vec model parameters dataclass.
            **additional_kwargs: Extra keyword arguments forwarded to __init__.
                Typically includes augmentation_mode and augmentation_method_params.

        Returns:
            A configured TS2Vec model instance.
        """
        config_kwargs = vars(config)
        overlapping = set(config_kwargs) & set(additional_kwargs)
        if overlapping:
            msg = (
                f'from_config received overlapping keys between config and additional_kwargs: '
                f'{overlapping}. Remove them from one side.'
            )
            raise ValueError(msg)
        return cls(**config_kwargs, **additional_kwargs)  # type: ignore[arg-type]

    def _init_augmentation_method(self, augmentation_method_params: dict) -> None:
        self._augmentation_method = TS2VecAugmentationMethodFactory.get_augmentation_method(
            mode=self._augmentation_mode, params=augmentation_method_params
        )

    def _init_augmentation_mode_params(self) -> None:
        self.automatic_optimization = False

    def _calculate_encoder_loss(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        return hierarchical_contrastive_loss(
            embeddings_1, embeddings_2, temporal_unit=self._temporal_unit
        )

    def _crop_shift_augmentation_strategy(
        self, x: torch.Tensor, encoder: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augmented_subsequences_1, augmented_subsequences_2, crop_length = (
            self._augmentation_method.augment(data=x, temporal_unit=self._temporal_unit)
        )

        augmented_subsequences_1_embeddings = encoder(augmented_subsequences_1)
        augmented_subsequences_2_embeddings = encoder(augmented_subsequences_2)

        augmented_subsequences_1_embeddings = augmented_subsequences_1_embeddings[:, -crop_length:]

        augmented_subsequences_2_embeddings = augmented_subsequences_2_embeddings[:, :crop_length]

        return augmented_subsequences_1_embeddings, augmented_subsequences_2_embeddings

    def _execute_augmentation_strategy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self._encoder if self.training else self._averaged_encoder
        return self._crop_shift_augmentation_strategy(x=x, encoder=encoder)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  ## noqa: ARG002
        """Run one TS2Vec training step with manual optimization."""
        x = extract_features_from_batch(batch)

        optimizer = self.optimizers()

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        embeddings_1, embeddings_2 = self._execute_augmentation_strategy(x)

        train_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

        self.log(
            'train_loss',
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            self.manual_backward(train_loss)
            optimizer.step()
        else:
            msg = 'Expected optimizer to be an instance of torch.optim.Optimizer'
            raise TypeError(msg)
        if isinstance(self._averaged_encoder, AveragedModel):
            self._averaged_encoder.update_parameters(self._encoder)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  ## noqa: ARG002
        """Compute and log the TS2Vec validation loss."""
        x = extract_features_from_batch(batch)

        embeddings_1, embeddings_2 = self._execute_augmentation_strategy(x)

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
