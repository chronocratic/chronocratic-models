"""Lightweight shared encoding mixin for fixed-length sequence models.

Provides a uniform ``encode(data, batch_size, num_workers)`` API for models that
process whole sequences in a single forward pass (TST, TimeVAE, TimeNet,
MCL, TS-TCC, Series2Vec). Models that need sliding-window inference, multi-scale
pooling, or mask-mode handling (the dilated trio: TS2Vec, AutoTCL, CoST) should
use the heavier mixins under ``convolutional/dilated/_mixin/`` instead.

Subclasses expose their encoder via :meth:`BasicEncodingMixin._get_encoder`
and, when needed, customize input preparation and output post-processing via
:meth:`_prepare_inputs` and :meth:`_postprocess`. The mixin itself owns the
DataLoader iteration, eval/inference mode handling, device placement, encoder
invocation, and result concatenation.
"""

from __future__ import annotations

__all__ = ['BasicEncodingMixin']

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from collections.abc import Callable


class BasicEncodingMixin(ABC):
    """Uniform ``encode()`` API for fixed-length sequence models.

    Designed to be mixed into a ``lightning.pytorch.LightningModule``. The
    LightningModule is expected to provide ``self.device`` so that batches
    can be moved onto the model's device.

    Subclasses implement :meth:`_get_encoder` (required) and may override
    :meth:`_get_encoder_module`, :meth:`_prepare_inputs`, and
    :meth:`_postprocess` to adapt the contract to encoders that take extra
    arguments or return non-tensor structures.

    The mixin manages the ``training`` / ``eval`` state of the *encoder
    module* (not the whole LightningModule) so that ``encode()`` does not
    perturb submodules that are unrelated to inference (e.g. contrastive
    heads, auxiliary loss modules).
    """

    # Attribute provided by the host LightningModule. Declared here so type
    # checkers see the contract; not set at the mixin level.
    device: torch.device

    @abstractmethod
    def _get_encoder(self) -> Callable[..., Any]:
        """Return the callable used to produce per-batch representations.

        Typically a submodule (e.g. ``self._encoder``, ``self.encoder``) or a
        bound method (e.g. ``self.get_representations``). Called once per
        ``encode()`` invocation.
        """

    def _get_encoder_module(self) -> nn.Module:
        """Return the ``nn.Module`` whose train/eval state should be toggled.

        Default: returns :meth:`_get_encoder` when it already is an
        ``nn.Module``. Override when :meth:`_get_encoder` returns a bound
        method (e.g. TST's ``self.get_representations``) — the underlying
        module then has to be identified explicitly so ``encode()`` can
        toggle its state surgically instead of the whole LightningModule's.
        """
        encoder = self._get_encoder()
        if isinstance(encoder, nn.Module):
            return encoder
        msg = (
            '_get_encoder() returned a non-Module callable. Override '
            '_get_encoder_module() to return the underlying nn.Module.'
        )
        raise NotImplementedError(msg)

    def _prepare_inputs(self, batch_x: torch.Tensor) -> tuple[Any, ...]:
        """Return the positional args to pass to the encoder.

        Default: ``(batch_x,)``. Override when the encoder needs additional
        inputs (e.g. padding masks) or a dtype cast.

        Args:
            batch_x: Batch tensor already moved to ``self.device``.

        Returns:
            Tuple of positional arguments fed to the encoder as
            ``encoder(*args)``.
        """
        return (batch_x,)

    def _postprocess[T](self, output: T) -> T:
        """Return the final representation tensor from the encoder output.

        Default: identity (the encoder is assumed to return a tensor).
        Override when the encoder returns a tuple, dict, or otherwise needs
        a final reshape (e.g. ``.unsqueeze(1)``).

        Args:
            output: Whatever the encoder returned for one batch.

        Returns:
            The representation tensor for this batch. The mixin will move it
            to CPU and concatenate with the other batches' outputs along dim 0.
        """
        return output

    @torch.inference_mode()
    def encode(self, data: torch.Tensor, batch_size: int, num_workers: int = 0) -> torch.Tensor:
        """Extract representations for ``data`` in mini-batches.

        Iterates ``data`` through a ``DataLoader``, moves each batch to
        ``self.device``, invokes the encoder via the
        ``_get_encoder`` / ``_prepare_inputs`` / ``_postprocess`` hooks under
        ``inference_mode``, and concatenates the per-batch outputs on dim 0.
        The model's prior training-mode state is preserved across the call.

        Args:
            data: Input tensor of shape ``(N, ...)`` — leading dim is the
                sample dimension, the rest is whatever the model expects.
            batch_size: Mini-batch size for inference.
            num_workers: Number of DataLoader workers.

        Returns:
            CPU tensor of shape ``(N, ...)`` — concatenation of per-batch
            representations along dim 0.
        """
        encoder_module = self._get_encoder_module()
        was_training = encoder_module.training
        encoder_module.eval()
        data_device = data.device
        try:
            loader = DataLoader(
                TensorDataset(data), batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )
            outputs: list[torch.Tensor] = []
            for (batch_x,) in loader:
                batch_on_device = batch_x.to(self.device)
                args = self._prepare_inputs(batch_on_device)
                output = encoder_module(*args)
                outputs.append(self._postprocess(output).to(data_device))
            return torch.cat(outputs, dim=0)
        finally:
            encoder_module.train(was_training)
