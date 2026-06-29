"""Lightweight shared encoding mixin for fixed-length sequence models.

Provides a uniform ``encode(data, batch_size, num_workers)`` API for models that
process whole sequences in a single forward pass (TST, TimeVAE, TimeNet,
MCL, TS-TCC, Series2Vec). Models that need sliding-window inference, multi-scale
pooling, or mask-mode handling (the dilated trio: TS2Vec, AutoTCL, CoST) should
use the heavier mixins under ``convolutional/dilated/_mixin/`` instead.

Subclasses expose their encoder via :meth:`BasicEncodingMixin._get_encoder`
and customize batch-to-representation logic via :meth:`_encode_batch`. The
mixin itself owns the DataLoader iteration, eval/inference mode handling,
device placement, encoder invocation, and result concatenation.

The two-hook contract separates concerns cleanly:
- ``_get_encoder() -> nn.Module`` identifies the module whose train/eval
  state should be toggled during encoding.
- ``_encode_batch(encoder, batch_x) -> Tensor`` defines how one on-device
  batch becomes a representation tensor.
"""

from __future__ import annotations

__all__ = ["BasicEncodingMixin"]

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BasicEncodingMixin(ABC):
    """Uniform ``encode()`` API for fixed-length sequence models.

    Designed to be mixed into a ``lightning.pytorch.LightningModule``. The
    LightningModule is expected to provide ``self.device`` so that batches
    can be moved onto the model's device.

    Subclasses implement exactly two hooks:
    :meth:`_get_encoder` (required, returns the ``nn.Module`` whose
    train/eval state to toggle) and :meth:`_encode_batch` (optional override,
    maps one on-device batch to a representation tensor).

    The mixin manages the ``training`` / ``eval`` state of the *encoder
    module* (not the whole LightningModule) so that ``encode()`` does not
    perturb submodules that are unrelated to inference (e.g. contrastive
    heads, auxiliary loss modules).
    """

    # Attribute provided by the host LightningModule. Declared here so type
    # checkers see the contract; not set at the mixin level.
    device: torch.device

    @abstractmethod
    def _get_encoder(self) -> nn.Module:
        """Return the module whose train/eval state ``encode()`` toggles.

        Must be an ``nn.Module`` (not a bound method). Called once per
        ``encode()`` invocation.
        """

    def _encode_batch(self, encoder: nn.Module, batch_x: torch.Tensor) -> torch.Tensor:
        """Map one on-device batch to its representation tensor.

        Default: ``encoder(batch_x)`` (the module's ``forward`` IS the
        representation path). Override when the rep path is not ``forward``
        or needs post-processing (e.g. tuple unpacking, slicing, pooling).

        Args:
            encoder: The module returned by :meth:`_get_encoder`.
            batch_x: Batch tensor already moved to ``self.device``.

        Returns:
            Representation tensor for this batch.
        """
        return encoder(batch_x)

    def encode_batch(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Encode one on-device batch in a single forward pass.

        Differentiable: gradients flow back to ``batch_x`` when it requires
        grad. Returns an on-device tensor (no CPU transfer). Does NOT toggle
        the encoder's train/eval state — the caller owns that. Put the
        encoder in ``eval()`` before a gradient attack loop.

        Args:
            batch_x: Batch tensor; moved to ``self.device`` internally.

        Returns:
            Representation tensor, on ``self.device``.
        """
        return self._encode_batch(self._get_encoder(), batch_x.to(self.device))

    def encode(
        self,
        data: torch.Tensor,
        batch_size: int,
        num_workers: int = 0,
        *,
        gradient_enabled: bool = False,
    ) -> torch.Tensor:
        """Extract representations for ``data`` in mini-batches.

        Iterates ``data`` through a ``DataLoader``, moves each batch to
        ``self.device``, invokes the encoder via the ``_encode_batch`` hook,
        and concatenates the per-batch outputs on dim 0. The model's prior
        training-mode state is preserved across the call.

        By default the encoding loop runs under ``torch.inference_mode()``,
        which severs the autograd graph. Set ``gradient_enabled=True`` to
        preserve gradients (e.g. for adversarial attacks or contrastive
        view comparison); the encoder stays in ``eval()`` regardless.

        Args:
            data: Input tensor of shape ``(N, ...)`` — leading dim is the
                sample dimension, the rest is whatever the model expects.
            batch_size: Mini-batch size for inference.
            num_workers: Number of DataLoader workers.
            gradient_enabled: When True, keep the autograd graph alive by
                using ``nullcontext()`` instead of ``inference_mode()``.
                The encoder remains in ``eval()`` to ensure deterministic
                behavior (no dropout, frozen BN stats). Default False.

        Returns:
            Tensor of shape ``(N, ...)`` on the same device as ``data``,
            concatenation of per-batch representations along dim 0.
        """
        encoder = self._get_encoder()
        was_training = encoder.training
        encoder.eval()
        grad_ctx = nullcontext() if gradient_enabled else torch.inference_mode()
        try:
            with grad_ctx:
                loader = DataLoader(
                    TensorDataset(data),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                outputs = [self.encode_batch(batch_x).to(data.device) for (batch_x,) in loader]
                return torch.cat(outputs, dim=0)
        finally:
            encoder.train(was_training)
