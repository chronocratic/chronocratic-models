"""Lightweight shared encoding mixin for fixed-length sequence models.

Provides a uniform ``encode(data, batch_size, num_workers)`` API for models that
process whole sequences in a single forward pass (TST, TimeVAE, TimeNet,
MCL, TS-TCC, Series2Vec). Models that need sliding-window inference, multi-scale
pooling, or mask-mode handling (the dilated trio: TS2Vec, AutoTCL, CoST) should
use the heavier mixins under ``convolutional/dilated/_mixin/`` instead.

Subclasses implement :meth:`SimpleEncodingMixin._encode_batch` — one method
returning the representation for a single batch tensor. The mixin handles
DataLoader iteration, eval-mode toggling, ``inference_mode``, and result
concatenation.
"""

from __future__ import annotations

__all__ = ['SimpleEncodingMixin']

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, TensorDataset


class SimpleEncodingMixin(ABC):
    """Uniform ``encode()`` API for fixed-length sequence models.

    Designed to be mixed into a ``lightning.pytorch.LightningModule``. The
    LightningModule is expected to provide:

    - ``self.device`` — used to move batches onto the model's device.
    - ``self.training`` flag and ``self.eval()`` / ``self.train(mode)`` —
      standard ``nn.Module`` toggles used to put the model into inference
      mode and restore the prior training state on exit.

    Concrete subclasses implement :meth:`_encode_batch`. They are free to
    project the input through any subset of their submodules — typically the
    feature-producing trunk before any task-specific head.
    """

    # Attribute provided by the host LightningModule. Declared here so type
    # checkers see the contract; not set at the mixin level.
    device: torch.device

    @abstractmethod
    def _encode_batch(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Return the representation for one batch.

        Called from within ``torch.inference_mode()`` and with the model in
        eval mode. The implementation is responsible for moving ``batch_x``
        onto the correct device and producing whatever tensor shape the
        downstream consumer expects.

        Args:
            batch_x: Input batch tensor as yielded by a ``TensorDataset`` over
                the ``data`` argument of :meth:`encode`.

        Returns:
            The batch's representation tensor (any shape). The mixin will move
            it to CPU and concatenate it with the other batches' outputs along
            dim 0.
        """

    @torch.inference_mode()
    def encode(
        self,
        data: torch.Tensor,
        batch_size: int,
        num_workers: int = 0,
    ) -> torch.Tensor:
        """Extract representations for ``data`` in mini-batches.

        Iterates the input through a ``DataLoader``, applies
        :meth:`_encode_batch` per batch under ``inference_mode``, and
        concatenates the per-batch outputs on dim 0. The model's prior
        training-mode state is preserved across the call.

        Args:
            data: Input tensor of shape ``(N, ...)`` — leading dim is the
                sample dimension, the rest is whatever the model expects.
            batch_size: Mini-batch size for inference.
            num_workers: Number of DataLoader workers.

        Returns:
            CPU tensor of shape ``(N, ...)`` — concatenation of per-batch
            representations along dim 0.
        """
        was_training = self.training
        self.eval()
        try:
            loader = DataLoader(
                TensorDataset(data),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
            outputs: list[torch.Tensor] = []
            for (batch_x,) in loader:
                outputs.append(self._encode_batch(batch_x).cpu())
            return torch.cat(outputs, dim=0)
        finally:
            self.train(was_training)
