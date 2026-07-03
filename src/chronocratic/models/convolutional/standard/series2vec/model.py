from __future__ import annotations

__all__ = ["Series2Vec"]

import lightning.pytorch as pl
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.series2vec.filters import filter_frequencies
from chronocratic.models.convolutional.standard.series2vec.losses import (
    pairwise_euclidean_distances,
    pairwise_soft_dtw_distances,
    pretraining_loss,
)
from chronocratic.models.convolutional.standard.series2vec.network import Series2VecNetwork
from chronocratic.models.distances.soft_dtw import SoftDTW
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.utils import extract_features_from_batch
from chronocratic.models.utils.helpers import _warn_sequence_fallback

# Minimum windows to split a singleton batch into: K<3 gives <2 unique pairwise
# distances, which _distance_normalizer detaches (no min-max range), so no gradient.
_MIN_SPLIT_COUNT = 3


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

    The encoder defaults to GroupNorm (``norm="layer"``), ensuring correct
    gradient flow at batch_size=1 (unlike BatchNorm, which degenerates with
    zero variance statistics for single-sample batches). Pass ``norm="batch"``
    to reproduce the upstream BatchNorm architecture exactly.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/Navidfoumani/Series2Vec.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
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
        singleton_split_count: int = 3,
        norm: str = "layer",
        sync_dist: bool = False,
        optimizer_name: str = "RAdam",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        if singleton_split_count < _MIN_SPLIT_COUNT:
            msg = (
                f"singleton_split_count must be >= {_MIN_SPLIT_COUNT}, "
                f"got {singleton_split_count}"
            )
            raise ValueError(msg)
        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._soft_dtw_gamma = soft_dtw_gamma
        self._sync_dist = sync_dist
        self._optimizer_name = optimizer_name
        self._weight_decay = weight_decay
        self._singleton_split_count = singleton_split_count
        self._encoder_kernel_size = encoder_kernel_size

        self.network = Series2VecNetwork(
            input_dims=input_dims,
            embedding_dims=embedding_dims,
            num_heads=num_heads,
            feedforward_dims=feedforward_dims,
            representation_dims=representation_dims,
            dropout_rate=dropout_rate,
            encoder_kernel_size=encoder_kernel_size,
            norm=norm,
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
            Representations of shape ``(B, representation_dims)`` for
            VECTOR or ``(B, 1, representation_dims)`` for SEQUENCE.
        """
        if output not in type(self).supported_outputs:
            msg = (
                f"Series2Vec does not support output={output}; "
                f"supported: {type(self).supported_outputs}"
            )
            raise ValueError(msg)
        flat = encoder.encode(batch_x)  # (B, D) - D=2*representation_dims
        if output == EncodingOutputShape.VECTOR:
            return flat  # (B, D) — VECTOR
        _warn_sequence_fallback(type(self))
        return flat.unsqueeze(1)  # (B, 1, D) — SEQUENCE (fake temporal axis)

    def _build_soft_dtw(self, x: torch.Tensor) -> SoftDTW:
        # SoftDTW's CUDA kernel has no MPS equivalent; for MPS (x.is_cuda is False)
        # this correctly falls back to the CPU path. Do not add an MPS branch.
        return SoftDTW(use_cuda=x.is_cuda and torch.cuda.is_available(), gamma=self._soft_dtw_gamma)

    def _ensure_pairable_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Split a singleton ``(1, L, C)`` batch into K windows for pairwise loss.

        The Series2Vec loss matches the batch's pairwise distance matrix, so a
        single long series (``B == 1``) yields no pairs and no learning signal.
        Splitting it into ``singleton_split_count`` contiguous, non-overlapping
        sub-series manufactures a real batch. No-op when ``B > 1`` or when the
        series is too short to form windows of at least the encoder kernel size
        (the caller's zero-loss fallback then handles the degenerate case).

        Args:
            x: Input batch of shape ``(B, L, C)``.

        Returns:
            ``(K, L // K, C)`` when ``x`` is a splittable singleton, else ``x``.
        """
        if x.size(0) != 1:
            return x
        k = self._singleton_split_count
        window_len = x.size(1) // k
        if window_len < self._encoder_kernel_size:
            return x
        return x[0, : k * window_len].reshape(k, window_len, x.size(2))

    def _calculate_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._ensure_pairable_batch(x)
        temporal_distances, frequency_distances, _, _ = self.network.pretrain_forward(x)
        target_temporal_distances = pairwise_soft_dtw_distances(self._build_soft_dtw(x), x)
        filtered_frequency_data = filter_frequencies(x.detach(), training=self.training)
        target_frequency_distances = pairwise_euclidean_distances(filtered_frequency_data)
        result = pretraining_loss(
            temporal_distances=temporal_distances,
            frequency_distances=frequency_distances,
            target_temporal_distances=target_temporal_distances,
            target_frequency_distances=target_frequency_distances,
        )
        # Fallback for series too short to split (window_len < kernel): pretraining_loss
        # returns new_tensor(0.0), disconnected from the graph, which crashes backward().
        # temporal_distances.sum() * 0.0 produces zero with a grad_fn reaching the encoder.
        if not result[0].requires_grad:
            dummy = temporal_distances.sum() * 0.0
            result = (dummy, dummy, dummy)
        return result

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
            ``representation_dims`` — the output dimension of
            :meth:`Series2VecNetwork.encode`, matching the constructor parameter.
        """
        return self.network.representation_dim
