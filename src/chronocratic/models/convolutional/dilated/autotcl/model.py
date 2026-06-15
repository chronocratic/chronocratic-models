__all__ = ['AutoTCL']


from typing import cast

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from chronocratic.models.augmentation.base import (
    AugmentationProducer,
    SingleView,
    TrainableAugmentationProducer,
)
from chronocratic.models.augmentation.trainable_support import (
    maybe_configure_augmentation_optimizer,
    maybe_train_augmentation,
)
from chronocratic.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin
from chronocratic.models.convolutional.dilated.autotcl.losses import (
    info_nce_loss,
    local_info_nce_loss,
)
from chronocratic.models.convolutional.dilated.encoders.encoders import AutoTCLTimeSeriesEncoder
from chronocratic.models.convolutional.dilated.encoders.masking import MaskMode
from chronocratic.models.utils import extract_features_from_batch, process_sample_length


class AutoTCL(pl.LightningModule, PoolingEncodingMixin):
    """AutoTCL Model.

    Code source: https://github.com/AslanDing/AutoTCL
    """

    def __init__(
        self,
        *,
        input_dims: int,
        kernel_sizes: list[int] | None = None,
        augmentation: AugmentationProducer[SingleView] | None = None,
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
        sync_dist: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['augmentation'])

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._meta_learning_rate = meta_learning_rate
        self._local_loss_weight = local_loss_weight
        self._sync_dist = sync_dist

        if augmentation is None:
            from chronocratic.models.convolutional.dilated.autotcl.augmentation import (  # noqa: PLC0415
                AutoTCLNeuralNetworkAugmentation,
                AutoTCLNeuralNetworkAugmentationParameters,
                RIPTrainingStrategy,
            )

            self._augmentation: AugmentationProducer[SingleView] = (
                AutoTCLNeuralNetworkAugmentation(
                    params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=input_dims),
                    training_strategy=RIPTrainingStrategy(),
                )
            )
        else:
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
        self._averaged_encoder.update_parameters(self._encoder)

    def configure_optimizers(self) -> AdamW | list[AdamW]:
        """Return encoder optimizer(s); two optimizers when using trainable aug."""
        main_optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        aug_opt = maybe_configure_augmentation_optimizer(
            self._augmentation, lr=self._meta_learning_rate
        )
        if aug_opt is not None:
            return [main_optimizer, cast('AdamW', aug_opt)]
        return main_optimizer

    def _calculate_encoder_loss(
        self, x_embeddings: torch.Tensor, augmented_x_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute encoder contrastive loss combining InfoNCE and local InfoNCE.

        Encoder loss remains model-internal.
        """
        local_loss = local_info_nce_loss(x_embeddings, augmented_x_embeddings)
        loss = info_nce_loss(x_embeddings, augmented_x_embeddings, temperature=1.0)
        return loss + self._local_loss_weight * local_loss

    def training_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor | None:
        """Run one AutoTCL training step with manual optimization.

        Two-phase training:
        1. Aug network self-training (via centralized maybe_train_augmentation gate).
        2. Uniform encoder training (all augmentation types).
        """
        x = extract_features_from_batch(batch)
        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        opts = self.optimizers()

        # Phase 1: Aug network self-training (centralized gate, not isinstance in model)
        aug_loss = maybe_train_augmentation(
            self._augmentation,
            x=x, encoder=self._encoder,
            epoch=self.current_epoch, batch_idx=batch_idx,
        )
        if aug_loss is not None:
            main_opt, meta_opt = cast('list[AdamW]', opts)
            meta_opt.zero_grad()
            self.manual_backward(aug_loss)
            meta_opt.step()
            self.log(
                'aug_train_loss',
                aug_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self._sync_dist,
            )

        # Phase 2: Uniform encoder training
        self._encoder.train()

        view = self._augmentation.produce(x)
        aug_x = view.view

        # Defensive device transfer — original model called .to(x.device)
        if aug_x.device != x.device:
            aug_x = aug_x.to(x.device)

        x_embeddings = self._encoder(x)
        aug_x_embeddings = self._encoder(aug_x)

        encoder_loss = self._calculate_encoder_loss(x_embeddings, aug_x_embeddings)

        main_opt = opts[0] if isinstance(opts, list) else opts
        main_opt = cast('AdamW', main_opt)
        main_opt.zero_grad()
        self.manual_backward(encoder_loss)
        main_opt.step()

        self._averaged_encoder.update_parameters(self._encoder)  # type: ignore  # noqa: PGH003

        self.log(
            'train_loss',
            encoder_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return encoder_loss

    def _eval_mutual_information(self, batch: torch.Tensor) -> float:
        """Evaluate mutual information between original and augmented data.

        Diagnostic method for research use. Sets augmentation to eval
        mode during measurement and restores previous state afterward.

        Args:
            batch: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            MI estimate (L1-out loss) as a float.
        """
        from chronocratic.models.convolutional.dilated.autotcl.utils import (  # noqa: PLC0415
            calculate_mutual_information,
        )

        # Diagnostic method: branch on trainable producer to manage eval/train mode
        if isinstance(self._augmentation, TrainableAugmentationProducer):
            prev_mode = self._augmentation.training
            self._augmentation.eval()
        mi = calculate_mutual_information(
            batch=batch,
            augmentation_method=self._augmentation,
            max_train_length=self._max_train_length,
        )
        # Restore training mode for diagnostic method
        if isinstance(self._augmentation, TrainableAugmentationProducer):
            self._augmentation.train(prev_mode)
        return mi

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> torch.Tensor:
        """Compute validation contrastive loss using the averaged encoder."""
        x = extract_features_from_batch(batch)

        view = self._augmentation.produce(x)
        aug_x = view.view

        # Defensive device transfer — original model called .to(x.device)
        if aug_x.device != x.device:
            aug_x = aug_x.to(x.device)

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
