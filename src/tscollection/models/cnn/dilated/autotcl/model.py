__all__ = ['AutoTCL']


from typing import cast

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from tscollection.models.augmentation import AugmentationMethod, TrainableAugmentation
from tscollection.models.cnn.dilated._mixin.encoding import PoolingEncodingMixin
from tscollection.models.cnn.dilated.encoders.encoders import AutoTCLTimeSeriesEncoder
from tscollection.models.cnn.dilated.encoders.masking import MaskMode
from tscollection.models.config import AutoTCLModelParameters
from tscollection.models.losses import info_nce_loss, local_info_nce_loss
from tscollection.models.utils import extract_features_from_batch, process_sample_length


class AutoTCL(pl.LightningModule, PoolingEncodingMixin):
    """Contrastive time-series encoder with a pluggable augmentation strategy.

    Accepts any ``AugmentationMethod`` instance in the constructor.
    When the augmentation is a ``TrainableAugmentation``, the model runs
    a two-phase training step: (1) aug-network self-training via the
    composed strategy, (2) uniform encoder training.
    """

    def __init__(
        self,
        input_dims: int,
        kernel_sizes: list[int],
        augmentation: AugmentationMethod,
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        max_train_length: int | None = None,
        meta_learning_rate: float = 1e-2,
        local_loss_weight: float = 0.1,
        sync_dist: bool = False,  # noqa: FBT001 FBT002
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['augmentation'])

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._meta_learning_rate = meta_learning_rate
        self._local_loss_weight = local_loss_weight
        self._sync_dist = sync_dist
        self._augmentation = augmentation

        self.automatic_optimization = False

        self._encoder = AutoTCLTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            kernel_sizes=kernel_sizes,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self._averaged_encoder = AveragedModel(self._encoder)

    def configure_optimizers(self) -> AdamW | list[AdamW]:
        """Return encoder optimizer(s); two optimizers when using TrainableAugmentation."""
        if isinstance(self._augmentation, TrainableAugmentation):
            main_optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
            meta_optimizer = self._augmentation.configure_optimizer(lr=self._meta_learning_rate)
            return [main_optimizer, meta_optimizer]
        return AdamW(self._encoder.parameters(), lr=self._learning_rate)

    @classmethod
    def from_config(cls, config: AutoTCLModelParameters, **additional_kwargs: object) -> 'AutoTCL':
        """Instantiate AutoTCL from a typed config dataclass.

        Args:
            config: AutoTCL model parameters dataclass.
            **additional_kwargs: Extra keyword arguments forwarded to __init__.
                Typically includes
                ``augmentation=AutoTCLNeuralNetworkAugmentation(params=..., strategy=...)``.

        Returns:
            A configured AutoTCL model instance.
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

    def _calculate_encoder_loss(
        self, x_embeddings: torch.Tensor, augmented_x_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute encoder contrastive loss combining InfoNCE and local InfoNCE.

        Encoder loss remains model-internal (D-05: model owns the choice).
        """
        local_loss = local_info_nce_loss(x_embeddings, augmented_x_embeddings)
        loss = info_nce_loss(x_embeddings, augmented_x_embeddings, temperature=1.0)
        return loss + self._local_loss_weight * local_loss

    def training_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor | None:
        """Run one AutoTCL training step with manual optimization.

        Two-phase training:
        1. Aug network self-training (only for TrainableAugmentation).
        2. Uniform encoder training (all augmentation types).
        """
        x = extract_features_from_batch(batch)
        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        # Cache optimizers once
        opts = self.optimizers()

        # Phase 1: Aug network self-training (TrainableAugmentation only)
        if isinstance(self._augmentation, TrainableAugmentation):
            main_opt, meta_opt = cast('list[AdamW]', opts)
            if self._augmentation._training_strategy.should_train(  # noqa: SLF001
                self.current_epoch, batch_idx
            ):
                self._encoder.eval()
                self._augmentation.train()
                aug_loss = self._augmentation.train_step(x, self._encoder, batch_idx)
                if aug_loss is not None:
                    meta_opt.zero_grad()
                    self.manual_backward(aug_loss)
                    meta_opt.step()

        # Phase 2: Uniform encoder training
        self._encoder.train()
        if isinstance(self._augmentation, TrainableAugmentation):
            self._augmentation.eval()

        views = self._augmentation.augment(x)
        aug_x = views.views[0]

        x_embeddings = self._encoder(x)
        aug_x_embeddings = self._encoder(aug_x)

        encoder_loss = self._calculate_encoder_loss(x_embeddings, aug_x_embeddings)

        main_opt = opts[0] if isinstance(opts, list) else opts
        main_opt = cast('AdamW', main_opt)
        main_opt.zero_grad()
        self.manual_backward(encoder_loss)
        main_opt.step()

        self._averaged_encoder.update_parameters(self._encoder)

        self.log(
            'enc_train_loss',
            encoder_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return encoder_loss

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> torch.Tensor:
        """Compute validation contrastive loss using the averaged encoder."""
        x = extract_features_from_batch(batch)

        views = self._augmentation.augment(x)
        aug_x = views.views[0]

        x_embeddings = self._averaged_encoder(x)
        aug_x_embeddings = self._averaged_encoder(aug_x)

        encoder_loss = self._calculate_encoder_loss(x_embeddings, aug_x_embeddings)

        self.log(
            'val_loss',
            encoder_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return encoder_loss
