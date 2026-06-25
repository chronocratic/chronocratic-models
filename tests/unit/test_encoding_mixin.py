"""Tests for BasicEncodingMixin 2-hook contract and gradient_enabled.

Verifies the mixin interface after refactor from 4-hook (_get_encoder +
_get_encoder_module + _prepare_inputs + _postprocess) to 2-hook
(_get_encoder + _encode_batch), and the gradient_enabled toggle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from chronocratic.models._mixin.encoding import BasicEncodingMixin

if TYPE_CHECKING:
    pass


class _TestModel(BasicEncodingMixin, nn.Module):
    """Minimal concrete subclass for testing mixin behavior."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.device: torch.device = torch.device("cpu")
        self._encoder = nn.Linear(10, 8)

    def _get_encoder(self) -> nn.Module:
        return self._encoder


class TestMixinTwoHookContract:
    """BasicEncodingMixin exposes exactly two hooks after refactor."""

    def test_has_get_encoder_hook(self) -> None:
        """_get_encoder is an abstract method on the mixin."""
        assert hasattr(BasicEncodingMixin, "_get_encoder")

    def test_has_encode_batch_hook(self) -> None:
        """_encode_batch exists as a concrete method on the mixin."""
        assert hasattr(BasicEncodingMixin, "_encode_batch")

    def test_encode_batch_default_calls_encoder(self) -> None:
        """Default _encode_batch passes batch_x through encoder."""
        model = _TestModel()
        batch_x = torch.randn(2, 10)
        result = model._encode_batch(model._encoder, batch_x)
        assert torch.allclose(result, model._encoder(batch_x))

    def test_get_encoder_module_deleted(self) -> None:
        """_get_encoder_module no longer exists on the mixin."""
        assert not hasattr(BasicEncodingMixin, "_get_encoder_module")

    def test_prepare_inputs_deleted(self) -> None:
        """_prepare_inputs no longer exists on the mixin."""
        assert not hasattr(BasicEncodingMixin, "_prepare_inputs")

    def test_postprocess_deleted(self) -> None:
        """_postprocess no longer exists on the mixin."""
        assert not hasattr(BasicEncodingMixin, "_postprocess")


class TestEncodeGradientEnabled:
    """gradient_enabled toggle controls autograd graph preservation."""

    def test_default_gradient_disabled(self) -> None:
        """encode() defaults to gradient_enabled=False; output has no grad."""
        model = _TestModel()
        data = torch.randn(4, 10)
        result = model.encode(data, batch_size=2)
        assert not result.requires_grad

    def test_gradient_enabled_false_severs_graph(self) -> None:
        """explicit gradient_enabled=False severs autograd graph."""
        model = _TestModel()
        data = torch.randn(4, 10)
        result = model.encode(data, batch_size=2, gradient_enabled=False)
        assert not result.requires_grad

    def test_gradient_enabled_true_preserves_graph(self) -> None:
        """gradient_enabled=True keeps autograd graph alive."""
        model = _TestModel()
        data = torch.randn(4, 10).requires_grad_(True)
        result = model.encode(data, batch_size=2, gradient_enabled=True)
        assert result.requires_grad

    def test_gradient_enabled_true_backprop_to_input(self) -> None:
        """Loss can backprop through encode() to input when gradient_enabled=True."""
        model = _TestModel()
        model._encoder.weight.data.fill_(0.1)  # deterministic weights
        data = torch.randn(4, 10).requires_grad_(True)
        result = model.encode(data, batch_size=2, gradient_enabled=True)
        loss = result.pow(2).sum()
        loss.backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


class TestEncodeStateRestoration:
    """Encoder train/eval state is saved and restored by encode()."""

    def test_encoder_returns_to_train_after_encode(self) -> None:
        """Encoder that was in train mode is restored after encode()."""
        model = _TestModel()
        model._encoder.train()
        data = torch.randn(4, 10)
        model.encode(data, batch_size=2)
        assert model._encoder.training is True

    def test_encoder_returns_to_eval_after_encode(self) -> None:
        """Encoder that was in eval mode is restored after encode()."""
        model = _TestModel()
        model._encoder.eval()
        data = torch.randn(4, 10)
        model.encode(data, batch_size=2)
        assert model._encoder.training is False

    def test_encoder_runs_in_eval_during_encode(self) -> None:
        """Encoder is in eval mode during the encode() loop."""
        model = _TestModel()
        model._encoder.train()
        training_during_encode: list[bool] = []

        class _StateChecker(nn.Module):
            def forward(inner_self, x: torch.Tensor) -> torch.Tensor:
                training_during_encode.append(inner_self.training)
                return x

        model._encoder = _StateChecker()
        data = torch.randn(4, 10)
        model.encode(data, batch_size=2)
        # All batches should see encoder.training == False
        assert len(training_during_encode) > 0
        assert all(not t for t in training_during_encode)


class TestEncodeOutputShape:
    """encode() output shape matches input batch dimension."""

    def test_output_batch_dim_matches_input(self) -> None:
        """Output batch dimension equals input batch dimension."""
        model = _TestModel()
        data = torch.randn(7, 10)
        result = model.encode(data, batch_size=2)
        assert result.shape[0] == 7

    def test_output_shape_with_different_batch_sizes(self) -> None:
        """encode() produces same output shape regardless of batch_size."""
        model = _TestModel()
        data = torch.randn(10, 10)
        result_single = model.encode(data, batch_size=10)
        result_multi = model.encode(data, batch_size=3)
        assert result_single.shape == result_multi.shape

    def test_output_device_matches_input_device(self) -> None:
        """Output tensor is on the same device as input data."""
        model = _TestModel()
        data = torch.randn(4, 10)
        result = model.encode(data, batch_size=2)
        assert result.device == data.device
