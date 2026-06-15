__all__ = ['CoST']

import itertools
from typing import cast

import lightning.pytorch as pl
import numpy as np
import torch
from torch import fft, nn
import torch.nn.functional as F  # noqa: N812
from torch.optim import SGD

from chronocratic.models.augmentation.base import (
    Augmentation,
    AugmentationProducer,
    ViewPair,
)
from chronocratic.models.convolutional.dilated._mixin.encoding import DecompositionEncodingMixin
from chronocratic.models.convolutional.dilated.cost.utils import compute_amplitude_and_phase
from chronocratic.models.convolutional.dilated.encoders.encoders import CoSTTimeSeriesEncoder
from chronocratic.models.convolutional.dilated.encoders.masking import MaskMode
from chronocratic.models.losses import instance_contrastive_loss
from chronocratic.models.utils import extract_features_from_batch, process_sample_length


class CoST(pl.LightningModule, DecompositionEncodingMixin):
    """CoST Model.

    Code source: https://github.com/salesforce/CoST
    """

    def __init__(
        self,
        *,
        input_dims: int,
        sequence_length: int,
        kernel_sizes: list[int] | None = None,
        augmentation: AugmentationProducer[ViewPair] | None = None,
        max_train_length: int = 201,
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        seasonal_loss_weight: float = 0.1,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['augmentation'])

        if kernel_sizes is None:
            kernel_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._sync_dist = sync_dist

        if augmentation is None:
            from chronocratic.models.augmentation.producers import (  # noqa: PLC0415
                IndependentPair,
            )
            from chronocratic.models.convolutional.dilated.cost.augmentation import (  # noqa: PLC0415
                CosTRandomFunctionAugmentation,
            )

            self._augmentation: AugmentationProducer[ViewPair] = IndependentPair(
                aug=CosTRandomFunctionAugmentation()
            )
        elif isinstance(augmentation, Augmentation):
            # Backward compat: wrap plain Augmentation in IndependentPair
            from chronocratic.models.augmentation.producers import (  # noqa: PLC0415
                IndependentPair,
            )

            self._augmentation: AugmentationProducer[ViewPair] = IndependentPair(
                aug=augmentation
            )
        else:
            self._augmentation = augmentation

        self.automatic_optimization = False

        # Seeded RNG for reproducible temporal index selection
        self._rng = np.random.default_rng(seed=int(torch.random.initial_seed()))

        length = min(max_train_length, sequence_length)

        self.query_encoder = CoSTTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            kernel_sizes=kernel_sizes,
            length=length,
            mask_mode=mask_mode,
        )

        component_dims = self.query_encoder.component_dims

        self.query_projection_head = nn.Sequential(
            nn.Linear(component_dims, component_dims),
            nn.ReLU(),
            nn.Linear(component_dims, component_dims),
        )

        self.key_encoder = CoSTTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            kernel_sizes=kernel_sizes,
            length=length,
            mask_mode=mask_mode,
        )

        self.key_projection_head = nn.Sequential(
            nn.Linear(component_dims, component_dims),
            nn.ReLU(),
            nn.Linear(component_dims, component_dims),
        )

        for param_query_encoder, param_key_encoder in zip(
            self.query_encoder.parameters(), self.key_encoder.parameters(), strict=True
        ):
            param_key_encoder.data.copy_(param_query_encoder.data)  # initialize
            param_key_encoder.requires_grad = False  # not update by gradient
        for param_query_projection_head, param_key_projection_head in zip(
            self.query_projection_head.parameters(),
            self.key_projection_head.parameters(),
            strict=True,
        ):
            param_key_projection_head.data.copy_(param_query_projection_head.data)  # initialize
            param_key_projection_head.requires_grad = False  # not update by gradient

        self.queue: torch.Tensor
        self.queue_insert_index: torch.Tensor

        self.register_buffer('queue', F.normalize(torch.randn(component_dims, queue_size), dim=0))
        self.register_buffer('queue_insert_index', torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.seasonal_loss_weight = seasonal_loss_weight

    def configure_optimizers(self) -> SGD:
        """Return SGD optimizer over trainable query encoder and projection head parameters."""
        model_params = itertools.chain(
            self.query_encoder.parameters(), self.query_projection_head.parameters()
        )

        model_params = filter(lambda p: p.requires_grad, model_params)

        optimizer = SGD(model_params, lr=self._learning_rate, momentum=0.9, weight_decay=1e-4)
        return optimizer

    def _compute_contrastive_loss(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        negative_key_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # positive logits: Nx1
        positive_logits = torch.einsum('nc,nc->n', [query_embeddings, key_embeddings]).unsqueeze(-1)
        # negative logits: NxK
        negative_logits = torch.einsum('nc,ck->nk', [query_embeddings, negative_key_embeddings])

        # logits: Nx(1+K) ## noqa: ERA001
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Momentum update for key encoder."""
        for param_query_encoder, param_key_encoder in zip(
            self.query_encoder.parameters(), self.key_encoder.parameters(), strict=True
        ):
            param_key_encoder.data = (
                param_key_encoder.data * self.momentum
                + param_query_encoder.data * (1 - self.momentum)
            )
        for param_query_projection_head, param_key_projection_head in zip(
            self.query_projection_head.parameters(),
            self.key_projection_head.parameters(),
            strict=True,
        ):
            param_key_projection_head.data = (
                param_key_projection_head.data * self.momentum
                + param_query_projection_head.data * (1 - self.momentum)
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        batch_size = keys.shape[0]

        if self.queue_size % batch_size != 0:
            msg = f'queue_size ({self.queue_size}) must be divisible by batch_size ({batch_size})'
            raise ValueError(msg)

        ptr = int(self.queue_insert_index.item())

        # replace keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            first_chunk = self.queue_size - ptr
            self.queue[:, ptr:] = keys.T[:, :first_chunk]
            self.queue[:, : batch_size - first_chunk] = keys.T[:, first_chunk:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_insert_index[0] = ptr

    def _compute_total_loss(
        self, query: torch.Tensor, key: torch.Tensor, *, update_key_encoder: bool = True
    ) -> torch.Tensor:
        # compute query features
        random_index = self._rng.integers(0, query.shape[1])

        query_trend, query_seasonality = self.query_encoder(query)
        if query_trend is not None:
            query_trend = F.normalize(
                self.query_projection_head(query_trend[:, random_index]), dim=-1
            )

        # compute key features
        with torch.no_grad():  # no gradient for keys
            if update_key_encoder:
                self._momentum_update_key_encoder()  # update key encoder
            key_trend, key_seasonality = self.key_encoder(key)
            if key_trend is not None:
                key_trend = F.normalize(
                    self.key_projection_head(key_trend[:, random_index]), dim=-1
                )

        loss = self._compute_contrastive_loss(query_trend, key_trend, self.queue.detach().clone())
        if update_key_encoder:
            self._dequeue_and_enqueue(key_trend)

        query_seasonality = F.normalize(query_seasonality, dim=-1)
        _, key_seasonality = self.query_encoder(key)
        key_seasonality = F.normalize(key_seasonality, dim=-1)

        query_seasonality_freq = fft.rfft(query_seasonality, dim=1)
        key_seasonality_freq = fft.rfft(key_seasonality, dim=1)
        query_seasonality_amplitude, query_seasonality_phase = compute_amplitude_and_phase(
            query_seasonality_freq
        )
        key_seasonality_amplitude, key_seasonality_phase = compute_amplitude_and_phase(
            key_seasonality_freq
        )

        seasonal_loss = instance_contrastive_loss(
            query_seasonality_amplitude, key_seasonality_amplitude
        ) + instance_contrastive_loss(query_seasonality_phase, key_seasonality_phase)

        loss += self.seasonal_loss_weight * (seasonal_loss / 2)

        return loss

    def training_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> torch.Tensor:
        """Augment the batch twice, compute the contrastive loss, perform a manual update step."""
        x = extract_features_from_batch(batch)

        optimizer = cast('torch.optim.Optimizer', self.optimizers())

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        pair = self._augmentation.produce(x)
        query, key = pair.first, pair.second

        train_loss = self._compute_total_loss(query, key, update_key_encoder=True)

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

        return train_loss

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
        batch_idx: int,  ## noqa: ARG002
    ) -> torch.Tensor:
        """Compute and log the contrastive validation loss without updating model parameters."""
        x = extract_features_from_batch(batch)

        pair = self._augmentation.produce(x)
        query, key = pair.first, pair.second

        with torch.no_grad():
            val_loss = self._compute_total_loss(query, key, update_key_encoder=False)

        self.log(
            'val_loss',
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return val_loss
