from __future__ import annotations

__all__ = ['Series2Vec']

from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn

from tscollection.models._mixin import BasicEncodingMixin

if TYPE_CHECKING:
    from collections.abc import Callable

from tscollection.models.convolutional.standard.series2vec.filters import filter_frequencies
from tscollection.models.convolutional.standard.series2vec.losses import (
    pairwise_euclidean_distances,
    pairwise_soft_dtw_distances,
    pretraining_loss,
)
from tscollection.models.convolutional.standard.series2vec.network import Series2VecNetwork
from tscollection.models.distances.soft_dtw import SoftDTW
from tscollection.models.utils import extract_features_from_batch


def _get_optimizer(name: str) -> type[torch.optim.Optimizer]:
    if name == 'Adam':
        return torch.optim.Adam
    if name == 'RAdam':
        return torch.optim.RAdam
    if name == 'AdamW':
        return torch.optim.AdamW
    msg = f'Unknown optimizer: {name}'
    raise ValueError(msg)


class Series2Vec(pl.LightningModule, BasicEncodingMixin):
    """Lightning wrapper for Series2Vec pretraining.

    The public input shape is ``(batch, time, channels)``.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/Navidfoumani/Series2Vec.
    """

    def __init__(
        self,
        input_dims: int,
        embedding_dims: int,
        num_heads: int,
        feedforward_dims: int,
        representation_dims: int,
        dropout_rate: float,
        encoder_kernel_size: int = 8,
        learning_rate: float = 1e-3,
        soft_dtw_gamma: float = 0.1,
        *,
        sync_dist: bool = False,
        optimizer_name: str = 'RAdam',
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._soft_dtw_gamma = soft_dtw_gamma
        self._sync_dist = sync_dist
        self._optimizer_name = optimizer_name
        self._weight_decay = weight_decay

        self.network = Series2VecNetwork(
            input_dims=input_dims,
            embedding_dims=embedding_dims,
            num_heads=num_heads,
            feedforward_dims=feedforward_dims,
            representation_dims=representation_dims,
            dropout_rate=dropout_rate,
            encoder_kernel_size=encoder_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Series2Vec representations for ``x``."""
        return self.network(x)

    def _get_encoder(self) -> Callable[..., torch.Tensor]:
        """Expose ``Series2VecNetwork.encode`` to ``BasicEncodingMixin.encode``.

        It concatenates the temporal and frequency-domain representations.
        """
        return self.network.encode

    def _get_encoder_module(self) -> nn.Module:
        """Underlying module for state management — ``network.encode`` is a bound method."""
        return self.network

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """Add a trailing singleton dim so the shape is ``(batch, 1, 2 * representation_dims)``."""
        return output.unsqueeze(1)

    def _build_soft_dtw(self, x: torch.Tensor) -> SoftDTW:
        return SoftDTW(use_cuda=x.is_cuda and torch.cuda.is_available(), gamma=self._soft_dtw_gamma)

    def _calculate_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_distances, frequency_distances, _, _ = self.network.pretrain_forward(x)
        target_temporal_distances = pairwise_soft_dtw_distances(self._build_soft_dtw(x), x)
        filtered_frequency_data = filter_frequencies(x.detach().cpu(), training=self.training).to(
            x.device
        )
        target_frequency_distances = pairwise_euclidean_distances(filtered_frequency_data)
        return pretraining_loss(
            temporal_distances=temporal_distances,
            frequency_distances=frequency_distances,
            target_temporal_distances=target_temporal_distances,
            target_frequency_distances=target_frequency_distances,
        )

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the Series2Vec pretraining loss for one batch."""
        x = extract_features_from_batch(batch)
        train_loss, temporal_loss, frequency_loss = self._calculate_loss(x)
        self.log(
            'train_loss',
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log('train_temporal_loss', temporal_loss, on_epoch=True, sync_dist=self._sync_dist)
        self.log('train_frequency_loss', frequency_loss, on_epoch=True, sync_dist=self._sync_dist)
        return train_loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the Series2Vec validation loss for one batch."""
        x = extract_features_from_batch(batch)
        val_loss, temporal_loss, frequency_loss = self._calculate_loss(x)
        self.log(
            'val_loss',
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log('val_temporal_loss', temporal_loss, on_epoch=True, sync_dist=self._sync_dist)
        self.log('val_frequency_loss', frequency_loss, on_epoch=True, sync_dist=self._sync_dist)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the configured optimizer for Series2Vec pretraining."""
        optimizer_cls = _get_optimizer(self._optimizer_name)
        kwargs: dict = {'lr': self._learning_rate, 'weight_decay': self._weight_decay}
        return optimizer_cls(self.parameters(), **kwargs)

    @property
    def representation_dim(self) -> int:
        """Flattened representation size (temporal + frequency concatenated).

        Returns:
            ``2 * representation_dims`` — the output dimension of
            :meth:`Series2VecNetwork.encode`.
        """
        return 2 * self.network.branch_representation_dim
