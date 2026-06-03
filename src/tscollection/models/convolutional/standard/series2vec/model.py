from __future__ import annotations

__all__ = ['Series2Vec']

import lightning.pytorch as pl
import torch

from tscollection.models.convolutional.two_dimensional.series2vec.filters import filter_frequencies
from tscollection.models.convolutional.two_dimensional.series2vec.losses import (
    pairwise_euclidean_distances,
    pairwise_soft_dtw_distances,
    pretraining_loss,
)
from tscollection.models.convolutional.two_dimensional.series2vec.network import Series2VecNetwork
from tscollection.models.distances.soft_dtw import SoftDTW


def _extract_features_from_batch(batch: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, tuple | list):
        return batch[0]
    msg = f'Unsupported batch format: {type(batch)}'
    raise ValueError(msg)


def _get_optimizer(name):
    if name == 'Adam':
        return torch.optim.Adam
    if name == 'RAdam':
        return torch.optim.RAdam
    if name == 'AdamW':
        return torch.optim.AdamW
    msg = f'Unknown optimizer: {name}'
    raise ValueError(msg)


class Series2Vec(pl.LightningModule):
    """Lightning wrapper for Series2Vec pretraining.

    The public input shape is ``(batch, time, channels)``.
    """

    def __init__(
        self,
        input_dims: int,
        num_classes: int,
        embedding_dims: int,
        num_heads: int,
        feedforward_dims: int,
        representation_dims: int,
        dropout_rate: float,
        encoder_kernel_size: int = 8,
        learning_rate: float = 1e-3,
        soft_dtw_gamma: float = 0.1,
        sync_dist: bool = False,
        optimizer_name: str = 'Adam',
        weight_decay: float = 0.0,
        warmup: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.soft_dtw_gamma = soft_dtw_gamma
        self.sync_dist = sync_dist
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.output_representation = False

        self.network = Series2VecNetwork(
            input_dims=input_dims,
            num_classes=num_classes,
            embedding_dims=embedding_dims,
            num_heads=num_heads,
            feedforward_dims=feedforward_dims,
            representation_dims=representation_dims,
            dropout_rate=dropout_rate,
            encoder_kernel_size=encoder_kernel_size,
        )

    def switch_to_representation_mode(self) -> None:
        self.output_representation = True

    def switch_to_training_mode(self) -> None:
        self.output_representation = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_representation:
            return self.network.encode(x).unsqueeze(1)
        return self.network(x)

    def _build_soft_dtw(self, x: torch.Tensor) -> SoftDTW:
        return SoftDTW(use_cuda=x.is_cuda and torch.cuda.is_available(), gamma=self.soft_dtw_gamma)

    def _calculate_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_distances, frequency_distances, _, _ = self.network.pretrain_forward(x)
        target_temporal_distances = pairwise_soft_dtw_distances(self._build_soft_dtw(x), x)
        filtered_frequency_data = filter_frequencies(x.detach().cpu()).to(x.device)
        target_frequency_distances = pairwise_euclidean_distances(filtered_frequency_data)
        return pretraining_loss(
            temporal_distances=temporal_distances,
            frequency_distances=frequency_distances,
            target_temporal_distances=target_temporal_distances,
            target_frequency_distances=target_frequency_distances,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = _extract_features_from_batch(batch)
        train_loss, temporal_loss, frequency_loss = self._calculate_loss(x)
        self.log(
            'train_loss',
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.log('train_temporal_loss', temporal_loss, on_epoch=True, sync_dist=self.sync_dist)
        self.log('train_frequency_loss', frequency_loss, on_epoch=True, sync_dist=self.sync_dist)
        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = _extract_features_from_batch(batch)
        val_loss, temporal_loss, frequency_loss = self._calculate_loss(x)
        self.log(
            'val_loss',
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.log('val_temporal_loss', temporal_loss, on_epoch=True, sync_dist=self.sync_dist)
        self.log('val_frequency_loss', frequency_loss, on_epoch=True, sync_dist=self.sync_dist)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_cls = _get_optimizer(self.optimizer_name)
        kwargs: dict = {'lr': self.learning_rate, 'weight_decay': self.weight_decay}
        if self.optimizer_name == 'AdamW':
            kwargs['warmup'] = self.warmup
        return optimizer_cls(self.parameters(), **kwargs)
