"""Tests for encode_batch() — DataLoader-free single-batch encoding primitive.

Verifies the public encode_batch() API on both encoding mixin families:
BasicEncodingMixin (Family A) and BaseEncodingMixin (Family B, dilated).

Covers all 6 spec categories (§8 of chronocratic-encode-batch-primitive.md):
1. Numerical equivalence
2. Differentiability
3. On-device return
4. No state mutation
5. Decomposition guard
6. encode() regression
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chronocratic.models._mixin.encoding import BasicEncodingMixin
from chronocratic.models.convolutional.dilated._mixin.encoding import (
    BaseEncodingMixin,
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)


# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------


class _BasicTestModel(BasicEncodingMixin, nn.Module):
    """Concrete BasicEncodingMixin for Family A tests."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.device: torch.device = torch.device("cpu")
        self._encoder = nn.Linear(10, 8)

    def _get_encoder(self) -> nn.Module:
        return self._encoder


class _PoolingTestModel(nn.Module, PoolingEncodingMixin):
    """Concrete PoolingEncodingMixin for Family B pooling tests."""

    def __init__(self, seq_len: int = 16, features: int = 3) -> None:
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self._pool_size = seq_len
        self._pool_type = "max"
        self._averaged_encoder = _DummyEncoder(out_dim=features)

    def _get_encoder(self) -> nn.Module:
        return self._averaged_encoder


class _DummyEncoder(nn.Module):
    """Encoder that returns (batch, features, seq_len) for pooling tests."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self._out_dim = out_dim
        self._linear = nn.Linear(3, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask_mode: str | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        # Output shape: (batch, seq_len, features) → transpose to (batch, features, seq_len)
        out = self._linear(x)
        return out.transpose(1, 2)


class _DecompositionTestModel(nn.Module, DecompositionEncodingMixin):
    """Concrete DecompositionEncodingMixin for Family B decomposition tests."""

    def __init__(self, seq_len: int = 16, trend_dim: int = 4, seas_dim: int = 4) -> None:
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self._trend_dim = trend_dim
        self._seas_dim = seas_dim
        self.query_encoder = _DummyDecompositionEncoder(
            trend_dim=trend_dim, seas_dim=seas_dim
        )


class _DummyDecompositionEncoder(nn.Module):
    """Encoder that returns (trend, seasonality) tuple for decomposition tests."""

    def __init__(self, trend_dim: int, seas_dim: int) -> None:
        super().__init__()
        self._trend_dim = trend_dim
        self._seas_dim = seas_dim
        self._trend_linear = nn.Linear(3, trend_dim)
        self._seas_linear = nn.Linear(3, seas_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask_mode: str | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self._trend_linear(x)
        seas = self._seas_linear(x)
        return trend, seas


# ---------------------------------------------------------------------------
# 1. Numerical equivalence
# ---------------------------------------------------------------------------


class TestNumericalEquivalence:
    """encode_batch(x) matches encode(x, batch_size>=len(x)) output."""

    def test_basic_equivalence(self) -> None:
        """BasicEncodingMixin: encode_batch matches encode for single batch."""
        model = _BasicTestModel()
        data = torch.randn(4, 10)
        batch_result = model.encode_batch(data)
        encode_result = model.encode(data, batch_size=4, num_workers=0)
        assert torch.allclose(batch_result.cpu(), encode_result, atol=1e-6)

    def test_basic_equivalence_shape(self) -> None:
        """BasicEncodingMixin: output shapes match."""
        model = _BasicTestModel()
        data = torch.randn(8, 10)
        batch_result = model.encode_batch(data)
        encode_result = model.encode(data, batch_size=8, num_workers=0)
        assert batch_result.shape == encode_result.shape


# ---------------------------------------------------------------------------
# 2. Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    """encode_batch preserves autograd graph back to input."""

    def test_basic_gradient_flows(self) -> None:
        """BasicEncodingMixin: grad_fn exists and backward populates x.grad."""
        model = _BasicTestModel()
        model._encoder.weight.data.fill_(0.1)
        x = torch.randn(2, 10).requires_grad_(True)
        out = model.encode_batch(x)
        assert out.grad_fn is not None
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_pooling_gradient_flows(self) -> None:
        """PoolingEncodingMixin: backward through encode_batch works."""
        model = _PoolingTestModel()
        encoder = model._get_encoder()
        encoder.eval()
        x = torch.randn(2, 16, 3).requires_grad_(True)
        out = model.encode_batch(x)
        assert out.grad_fn is not None
        out.sum().backward()
        assert x.grad is not None

    def test_decomposition_gradient_flows(self) -> None:
        """DecompositionEncodingMixin: backward through encode_batch works."""
        model = _DecompositionTestModel()
        encoder = model._get_encoder()
        encoder.eval()
        x = torch.randn(2, 16, 3).requires_grad_(True)
        out = model.encode_batch(x)
        assert out.grad_fn is not None
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# 3. On-device return
# ---------------------------------------------------------------------------


class TestOnDeviceReturn:
    """encode_batch returns on self.device, not CPU."""

    def test_basic_returns_on_device(self) -> None:
        """BasicEncodingMixin: output device matches model device."""
        model = _BasicTestModel()
        data = torch.randn(2, 10)
        result = model.encode_batch(data)
        assert result.device == model.device

    def test_pooling_returns_on_device(self) -> None:
        """PoolingEncodingMixin: output device matches model device."""
        model = _PoolingTestModel()
        data = torch.randn(2, 16, 3)
        result = model.encode_batch(data)
        assert result.device == model.device

    def test_decomposition_returns_on_device(self) -> None:
        """DecompositionEncodingMixin: output device matches model device."""
        model = _DecompositionTestModel()
        data = torch.randn(2, 16, 3)
        result = model.encode_batch(data)
        assert result.device == model.device


# ---------------------------------------------------------------------------
# 4. No state mutation
# ---------------------------------------------------------------------------


class TestNoStateMutation:
    """encode_batch does not toggle train/eval state."""

    def test_basic_preserves_training_state(self) -> None:
        """BasicEncodingMixin: encoder training flag unchanged."""
        model = _BasicTestModel()
        model._encoder.train()
        data = torch.randn(2, 10)
        model.encode_batch(data)
        assert model._encoder.training is True

    def test_basic_preserves_eval_state(self) -> None:
        """BasicEncodingMixin: encoder eval flag unchanged."""
        model = _BasicTestModel()
        model._encoder.eval()
        data = torch.randn(2, 10)
        model.encode_batch(data)
        assert model._encoder.training is False

    def test_pooling_preserves_training_state(self) -> None:
        """PoolingEncodingMixin: encoder training flag unchanged."""
        model = _PoolingTestModel()
        model._averaged_encoder._linear.train()
        data = torch.randn(2, 16, 3)
        model.encode_batch(data)
        assert model._averaged_encoder._linear.training is True

    def test_decomposition_preserves_training_state(self) -> None:
        """DecompositionEncodingMixin: encoder training flag unchanged."""
        model = _DecompositionTestModel()
        model.query_encoder._trend_linear.train()
        data = torch.randn(2, 16, 3)
        model.encode_batch(data)
        assert model.query_encoder._trend_linear.training is True


# ---------------------------------------------------------------------------
# 5. Decomposition guard
# ---------------------------------------------------------------------------


class TestDecompositionGuard:
    """Decomposition models reject unsupported encoding_window values."""

    def test_rejects_multiscale(self) -> None:
        """encode_batch with multiscale on decomposition model raises ValueError."""
        model = _DecompositionTestModel()
        data = torch.randn(2, 16, 3)
        with pytest.raises(ValueError, match="multiscale"):
            model.encode_batch(data, encoding_window="multiscale")

    def test_accepts_full_series(self) -> None:
        """encode_batch with full_series on decomposition model works."""
        model = _DecompositionTestModel()
        data = torch.randn(2, 16, 3)
        result = model.encode_batch(data, encoding_window="full_series")
        assert isinstance(result, torch.Tensor)

    def test_accepts_none(self) -> None:
        """encode_batch with None encoding_window on decomposition model works."""
        model = _DecompositionTestModel()
        data = torch.randn(2, 16, 3)
        result = model.encode_batch(data, encoding_window=None)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# 6. encode() regression
# ---------------------------------------------------------------------------


class TestEncodeRegression:
    """encode() behavior unchanged after encode_batch refactor."""

    def test_encode_output_device_unchanged(self) -> None:
        """encode() still returns on input data device (CPU)."""
        model = _BasicTestModel()
        data = torch.randn(4, 10)
        result = model.encode(data, batch_size=2, num_workers=0)
        assert result.device == data.device

    def test_encode_state_restoration(self) -> None:
        """encode() still restores encoder training state."""
        model = _BasicTestModel()
        model._encoder.train()
        data = torch.randn(4, 10)
        model.encode(data, batch_size=2, num_workers=0)
        assert model._encoder.training is True

    def test_encode_gradient_enabled(self) -> None:
        """encode() gradient_enabled=True still preserves autograd."""
        model = _BasicTestModel()
        data = torch.randn(4, 10).requires_grad_(True)
        result = model.encode(data, batch_size=2, num_workers=0, gradient_enabled=True)
        assert result.requires_grad
