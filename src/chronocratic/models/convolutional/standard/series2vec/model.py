from __future__ import annotations

__all__ = ["Series2Vec"]

import lightning.pytorch as pl
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.series2vec.filters import filter_frequencies
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.utils.helpers import _warn_sequence_fallback
from chronocratic.models.convolutional.standard.series2vec.losses import (
    pairwise_euclidean_distances,
    pairwise_soft_dtw_distances,
    pretraining_loss,
)
from chronocratic.models.convolutional.standard.series2vec.network import Series2VecNetwork
from chronocratic.models.distances.soft_dtw import SoftDTW
from chronocratic.models.utils import extract_features_from_batch


def _get_optimizer(name: str) -> type[torch.optim.Optimizer]:
    if name == "Adam":
        return torch.optim.Adam
    if name == "RAdam":
        return torch.optim.RAdam
    if name == "AdamW":
        return torch.optim.AdamW
    msg = f"Unknown optimizer: {name}"
    raise ValueError(msg)


class Series2Vec(pl.LightningModule, BasicEncodingMixin):
    """Lightning wrapper for Series2Vec pretraining.

    The public input shape is ``(batch, time, channels)``.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/Navidfoumani/Series2Vec.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR}
    )

    def __init__(
        self,
        input_dims: int,
        embedding_dims: int = 16,
        num_heads: int = 8,
        feedforward_dims: int = 256,
        representation_dims: int = 320,
        dropout_rate: float = 0.01,
        encoder_kernel_size: int = 8,
        learning_rate: float = 1e-3,
        soft_dtw_gamma: float = 0.1,
        *,
        sync_dist: bool = False,
        optimizer_name: str = "RAdam",
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

    @property
    def encoder(self) -> nn.Module:
        """Return the Series2Vec network for inspection and checkpointing."""
        return self.network

    def _get_encoder(self) -> nn.Module:
        """Return the Series2Vec network for ``BasicEncodingMixin.encode``."""
        return self.network

    def _encode_batch(
        self,
        encoder: nn.Module,
        batch_x: torch.Tensor,
        *,
        output: EncodingOutputShape = EncodingOutputShape.VECTOR,
    ) -> torch.Tensor:
        """Return flat representation for VECTOR, unsqueeze for SEQUENCE.

        Args:
            encoder: The Series2VecNetwork module.
            batch_x: Batch tensor of shape ``(B, seq_len, input_dims)``.
            output: Requested output shape. Defaults to VECTOR (2-D).

        Returns:
            Representations of shape ``(B, 2 * representation_dims)`` for
            VECTOR or ``(B, 1, 2 * representation_dims)`` for SEQUENCE.
        """
        flat = encoder.encode(batch_x)  # (B, D) - D=2*representation_dims
        if output == EncodingOutputShape.VECTOR:
            return flat  # (B, D) — VECTOR
        elif output == EncodingOutputShape.SEQUENCE:
            _warn_sequence_fallback(type(self))
            return flat.unsqueeze(1)  # (B, 1, D) — SEQUENCE (fake temporal axis)
        else:
            raise ValueError(
                f"Series2Vec does not support output={output}; "
                f"supported: {type(self).supported_outputs}"
            )

    def _build_soft_dtw(self, x: torch.Tensor) -> SoftDTW:
        # SoftDTW's CUDA kernel has no MPS equivalent; for MPS (x.is_cuda is False)
        # this correctly falls back to the CPU path. Do not add an MPS branch.
        return SoftDTW(use_cuda=x.is_cuda and torch.cuda.is_available(), gamma=self._soft_dtw_gamma)

    def _calculate_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_distances, frequency_distances, _, _ = self.network.pretrain_forward(x)
        target_temporal_distances = pairwise_soft_dtw_distances(self._build_soft_dtw(x), x)
        filtered_frequency_data = filter_frequencies(x.detach(), training=self.training)
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
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log("train_temporal_loss", temporal_loss, on_epoch=True, sync_dist=self._sync_dist)
        self.log("train_frequency_loss", frequency_loss, on_epoch=True, sync_dist=self._sync_dist)
        return train_loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the Series2Vec validation loss for one batch."""
        x = extract_features_from_batch(batch)
        val_loss, temporal_loss, frequency_loss = self._calculate_loss(x)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log("val_temporal_loss", temporal_loss, on_epoch=True, sync_dist=self._sync_dist)
        self.log("val_frequency_loss", frequency_loss, on_epoch=True, sync_dist=self._sync_dist)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the configured optimizer for Series2Vec pretraining."""
        optimizer_cls = _get_optimizer(self._optimizer_name)
        kwargs: dict = {"lr": self._learning_rate, "weight_decay": self._weight_decay}
        return optimizer_cls(self.parameters(), **kwargs)

    @property
    def representation_dim(self) -> int:
        """Flattened representation size (temporal + frequency concatenated).

        Returns:
            ``2 * representation_dims`` — the output dimension of
            :meth:`Series2VecNetwork.encode`.
        """
        return 2 * self.network.branch_representation_dim
