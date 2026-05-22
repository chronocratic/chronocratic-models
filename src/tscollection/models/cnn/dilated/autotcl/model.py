__all__ = ['AutoTCL']


from typing import cast

import lightning.pytorch as pl
import torch
from torch.nn import functional as F  ## noqa: N812
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from tscollection.models.cnn.dilated._mixin.encoding import PoolingEncodingMixin
from tscollection.models.augmentation.enums import (
    AutoTCLAugmentationMode,
    AutoTCLNeuralNetworkAugmentationTrainingMode,
)
from tscollection.models.augmentation.factories import AutoTCLAugmentationMethodFactory
from tscollection.models.augmentation.strategies import (
    AugmentationMethod,
    AutoTCLAugmentationTimeSeriesEncoder,
    AutoTCLNeuralNetworkAugmentation,
)
from tscollection.models.cnn.dilated.autotcl.utils import (
    calculate_mutual_information,
    calculate_regular_consistency,
)
from tscollection.models.cnn.dilated.encoders.encoders import AutoTCLTimeSeriesEncoder
from tscollection.models.cnn.dilated.encoders.masking import MaskMode
from tscollection.models.config import AutoTCLModelParameters
from tscollection.models.losses import (
    info_nce_loss,
    local_info_nce_loss,
    maximum_mean_discrepancy_with_gaussian_kernel_loss,
)
from tscollection.models.utils import extract_features_from_batch, process_sample_length


class AutoTCL(pl.LightningModule, PoolingEncodingMixin):
    def __init__(
        self,
        input_dims: int,
        kernel_sizes: list[int],
        augmentation_mode: AutoTCLAugmentationMode,
        augmentation_method_params: dict,
        augmentation_mode_params: dict | None = None,
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        max_train_length: int | None = None,
        sync_dist: bool = False,  # noqa: FBT001 FBT002
    ) -> None:
        """Contrastive time-series encoder with a pluggable augmentation strategy."""
        super().__init__()

        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._sync_dist = sync_dist
        self._augmentation_mode = augmentation_mode

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

        self._init_augmentation_method(augmentation_method_params)

        self._init_augmentation_mode_params(augmentation_mode_params)

    # -------------- Setup functions --------------

    def _init_augmentation_method(self, augmentation_method_params: dict) -> None:
        self._augmentation_method: AugmentationMethod = (
            AutoTCLAugmentationMethodFactory.get_augmentation_method(
                mode=self._augmentation_mode, params=augmentation_method_params
            )
        )

        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            if not isinstance(self._augmentation_method, AutoTCLNeuralNetworkAugmentation):
                msg = f'Augmentation method for mode {self._augmentation_mode} \
                    must be an instance of AutoTCLNeuralNetworkAugmentation.'
                raise ValueError(msg)
            self._augmentation_model: AutoTCLAugmentationTimeSeriesEncoder = (
                self._augmentation_method.get_model()
            )

    def _init_augmentation_mode_params(self, augmentation_mode_param: dict | None) -> None:
        if augmentation_mode_param is None:
            augmentation_mode_param = {}

        self.automatic_optimization = False

        self._local_loss_weight = augmentation_mode_param.get('local_loss_weight', 0.1)

        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            self._regular_consistency_weight = augmentation_mode_param.get(
                'regular_consistency_weight', 0.001
            )
            self._regularization_weight = augmentation_mode_param.get(
                'regularization_weight', 0.001
            )
            self._regularization_threshold = augmentation_mode_param.get(
                'regularization_threshold', 0.4
            )
            self._augmentation_network_training_ratio_step = augmentation_mode_param.get(
                'augmentation_network_training_ratio_step', 1
            )
            self._meta_learning_rate = augmentation_mode_param.get('meta_learning_rate', 1e-2)
            self._augmentation_network_training_mode = augmentation_mode_param.get(
                'augmentation_network_training_mode',
                AutoTCLNeuralNetworkAugmentationTrainingMode.RELEVANT_INFORMATION_PRINCIPLE,
            )

    def _configure_optimizers_default(self) -> AdamW:
        return AdamW(self._encoder.parameters(), lr=self._learning_rate)

    def _configure_optimizers_neural_network_augmentation(self) -> list[AdamW]:
        augmentation_network_params = self._augmentation_model.get_parameters()

        main_optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        meta_optimizer = AdamW(augmentation_network_params, lr=self._meta_learning_rate)

        return [main_optimizer, meta_optimizer]

    def configure_optimizers(self) -> AdamW | list[AdamW]:
        """Return encoder optimizer(s); two optimizers in neural-network augmentation mode."""
        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            return self._configure_optimizers_neural_network_augmentation()
        return self._configure_optimizers_default()

    @classmethod
    def from_config(cls, config: AutoTCLModelParameters, **additional_kwargs: object) -> 'AutoTCL':
        """Instantiate AutoTCL from a typed config dataclass.

        Args:
            config: AutoTCL model parameters dataclass.
            **additional_kwargs: Extra keyword arguments forwarded to __init__.
                Typically includes augmentation_mode and augmentation_method_params.

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

    # -------------- End of setup functions --------------

    # -------------- Loss calculation functions --------------

    def _eval_mutual_information(self, batch: torch.Tensor) -> float:

        augmentation_network_mode = self._augmentation_model.training
        self._augmentation_model.eval()

        mutual_information = calculate_mutual_information(
            batch=batch,
            augmentation_method=self._augmentation_method,
            max_train_length=self._max_train_length,
        )

        self._augmentation_model.train(augmentation_network_mode)

        return mutual_information

    def _calculate_encoder_loss(
        self, x_embeddings: torch.Tensor, augmented_x_embeddings: torch.Tensor
    ) -> torch.Tensor:
        local_loss = local_info_nce_loss(x_embeddings, augmented_x_embeddings)
        loss = info_nce_loss(x_embeddings, augmented_x_embeddings, temperature=1.0)
        total_loss = loss + self._local_loss_weight * local_loss

        return total_loss

    def _augmentation_loss_network_augmentation_relevant_information_principle(
        self,
        x_embeddings: torch.Tensor,
        augmented_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:

        vx_distance = maximum_mean_discrepancy_with_gaussian_kernel_loss(
            x_embeddings, augmented_x_embeddings
        )
        regular_consistency = calculate_regular_consistency(weights=augmentation_factor)
        regularization_loss = F.relu(
            torch.sum(augmentation_factor, dim=-1).mean() - self._regularization_threshold
        )

        aug_total_loss = (
            vx_distance
            + self._regularization_weight * regularization_loss
            + self._regular_consistency_weight * regular_consistency
        )

        return aug_total_loss

    def _augmentation_loss_neural_network_augmentation_adversarial(
        self, x_embeddings: torch.Tensor, augmented_x_embeddings: torch.Tensor
    ) -> torch.Tensor:
        aug_total_loss = -1 * info_nce_loss(x_embeddings, augmented_x_embeddings, temperature=1.0)
        return aug_total_loss

    def _calculate_augmentation_loss_neural_network_augmentation(
        self,
        x_embeddings: torch.Tensor,
        augmented_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._augmentation_network_training_mode
            == AutoTCLNeuralNetworkAugmentationTrainingMode.RELEVANT_INFORMATION_PRINCIPLE
        ):
            return self._augmentation_loss_network_augmentation_relevant_information_principle(
                x_embeddings, augmented_x_embeddings, augmentation_factor
            )
        if (
            self._augmentation_network_training_mode
            == AutoTCLNeuralNetworkAugmentationTrainingMode.ADVERSARIAL
        ):
            return self._augmentation_loss_neural_network_augmentation_adversarial(
                x_embeddings, augmented_x_embeddings
            )
        msg = f'Logic for training mode {self._augmentation_network_training_mode} not implemented.'
        raise ValueError(msg)

    # -------------- End of loss calculation functions --------------

    # -------------- Training step functions --------------

    def _training_step_function_default(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> torch.Tensor:
        x = extract_features_from_batch(batch)

        optimizer = self.optimizers()

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        augmented_x = self._augmentation_method.augment(x)

        augmented_x = (
            augmented_x.to(x.device) if isinstance(augmented_x, torch.Tensor) else augmented_x
        )  ## could be substituted by ignoring the typehint

        x_embeddings = self._encoder(x)
        augmented_x_embeddings = self._encoder(augmented_x)

        encoder_total_loss = self._calculate_encoder_loss(x_embeddings, augmented_x_embeddings)

        self.log(
            'train_loss',
            encoder_total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        optimizer = cast('torch.optim.Optimizer', optimizer)
        optimizer.zero_grad()
        self.manual_backward(encoder_total_loss)
        optimizer.step()
        self._averaged_encoder.update_parameters(self._encoder)  # ty: ignore[call-non-callable]

        return encoder_total_loss

    def _training_step_function_neural_network_augmentation(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> None:
        x = extract_features_from_batch(batch)

        main_optimizer, meta_optimizer = cast('list[AdamW]', self.optimizers())

        augmentation_total_loss = None

        if self.current_epoch % self._augmentation_network_training_ratio_step == 0:
            self._encoder.eval()
            self._augmentation_model.train()

            features = self._augmentation_model(x)
            augmentation_factor = features['augmentation_factor']
            augmented_x = features['augmented_data']
            x_embeddings = self._encoder(x)
            augmented_x_embeddings = self._encoder(augmented_x)

            augmentation_total_loss = self._calculate_augmentation_loss_neural_network_augmentation(
                x_embeddings, augmented_x_embeddings, augmentation_factor
            )

            meta_optimizer.zero_grad()
            self.manual_backward(augmentation_total_loss)
            meta_optimizer.step()

        self._encoder.train()
        self._augmentation_model.eval()

        augmented_x = self._augmentation_method.augment(x)

        x_embeddings = self._encoder(x)
        augmented_x_embeddings = self._encoder(augmented_x)

        encoder_total_loss = self._calculate_encoder_loss(x_embeddings, augmented_x_embeddings)

        main_optimizer.zero_grad()
        self.manual_backward(encoder_total_loss)
        main_optimizer.step()

        self._averaged_encoder.update_parameters(self._encoder)  # ty: ignore[call-non-callable]

        if augmentation_total_loss is not None:
            self.log_dict(
                {'enc_train_loss': encoder_total_loss, 'aug_train_loss': augmentation_total_loss},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self._sync_dist,
            )
        else:
            self.log_dict(
                {'enc_train_loss': encoder_total_loss},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self._sync_dist,
            )

    def _exec_training_step_function(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor | None:
        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            return self._training_step_function_neural_network_augmentation(batch, batch_idx)
        return self._training_step_function_default(batch, batch_idx)

    # -------------- End of training step functions ---------------

    def training_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor | None:
        """Run one training step and update both the encoder and the averaged encoder."""
        total_loss = self._exec_training_step_function(batch, batch_idx)
        return total_loss

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> torch.Tensor:
        """Compute validation contrastive loss using the averaged encoder."""
        x = extract_features_from_batch(batch)

        augmented_x = self._augmentation_method.augment(x)
        if isinstance(augmented_x, torch.Tensor):
            augmented_x = augmented_x.to(x.device)

        x_embeddings = self._averaged_encoder(x)
        augmented_x_embeddings = self._averaged_encoder(augmented_x)

        encoder_total_loss = self._calculate_encoder_loss(x_embeddings, augmented_x_embeddings)

        self.log_dict(
            {'val_loss': encoder_total_loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        return encoder_total_loss
