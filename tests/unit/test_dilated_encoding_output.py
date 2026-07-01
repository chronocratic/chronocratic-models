"""Tests for dilated BaseEncodingMixin output parameter and encoding_window derivation.

Verifies that the dilated mixin encode_batch() and encode() accept the keyword-only
`output` parameter (EncodingOutputShape), derive encoding_window from it, and respect
explicit encoding_window precedence.

TDD for plan 10-04, Task 1.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.convolutional.dilated._mixin.encoding import BaseEncodingMixin


class _TestableEncodingModel(BaseEncodingMixin, torch.nn.Module):
    """Concrete model for testing BaseEncodingMixin output param behavior."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self._averaged_encoder = torch.nn.Linear(10, 10)

    def _get_eval_method(self):
        return self._evaluate_for_test

    def _evaluate_for_test(
        self, input_tensor: torch.Tensor, mask=None, slicing=None, encoding_window=None
    ) -> torch.Tensor:
        """Echo encoding_window into output shape so derivation is testable."""
        batch, seq, features = input_tensor.shape
        if encoding_window == "full_series":
            # Return (B, 1, F) so encode_batch.squeeze(1) produces (B, F)
            return torch.ones(batch, 1, features, device=self.device)
        else:
            # encoding_window is None (SEQUENCE)
            return torch.ones(batch, seq, features, device=self.device)


@pytest.fixture
def model() -> _TestableEncodingModel:
    return _TestableEncodingModel()


class TestDilatedEncodeBatchSignature:
    """BaseEncodingMixin.encode_batch has output keyword-only param."""

    def test_encode_batch_has_output_param(self) -> None:
        sig = inspect.signature(BaseEncodingMixin.encode_batch)
        assert "output" in sig.parameters

    def test_output_defaults_to_vector(self) -> None:
        sig = inspect.signature(BaseEncodingMixin.encode_batch)
        param = sig.parameters["output"]
        assert param.default == EncodingOutputShape.VECTOR

    def test_output_is_keyword_only(self) -> None:
        sig = inspect.signature(BaseEncodingMixin.encode_batch)
        param = sig.parameters["output"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_encode_has_output_param(self) -> None:
        sig = inspect.signature(BaseEncodingMixin.encode)
        assert "output" in sig.parameters

    def test_encode_output_defaults_to_vector(self) -> None:
        sig = inspect.signature(BaseEncodingMixin.encode)
        param = sig.parameters["output"]
        assert param.default == EncodingOutputShape.VECTOR

    def test_encode_output_is_keyword_only(self) -> None:
        sig = inspect.signature(BaseEncodingMixin.encode)
        param = sig.parameters["output"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


class TestEncodeBatchOutputDerivation:
    """output param drives encoding_window derivation in encode_batch."""

    def test_default_output_derives_full_series(self, model: _TestableEncodingModel) -> None:
        """encode_batch(batch_x) with default output=VECTOR derives encoding_window='full_series'."""
        batch_x = torch.randn(2, 10, 5)
        result = model.encode_batch(batch_x)
        # VECTOR: (B, 1, F) squeezed to (B, F) -> 2D
        assert result.ndim == 2

    def test_sequence_output_derives_none(self, model: _TestableEncodingModel) -> None:
        """encode_batch(batch_x, output=SEQUENCE) derives encoding_window=None."""
        batch_x = torch.randn(2, 10, 5)
        result = model.encode_batch(batch_x, output=EncodingOutputShape.SEQUENCE)
        # SEQUENCE: returns (B, seq, F) -> 3D
        assert result.ndim == 3

    def test_explicit_encoding_window_overrides_output(self, model: _TestableEncodingModel) -> None:
        """encoding_window='full_series' takes precedence over output=SEQUENCE."""
        batch_x = torch.randn(2, 10, 5)
        result = model.encode_batch(
            batch_x, output=EncodingOutputShape.SEQUENCE, encoding_window="full_series"
        )
        # explicit encoding_window wins, so we get squeezed VECTOR output
        assert result.ndim == 2


class TestEncodeOutputDerivation:
    """output param drives encoding_window derivation in encode()."""

    def test_encode_default_output_is_vector(self, model: _TestableEncodingModel) -> None:
        """encode(data, batch_size=N) with default output=VECTOR returns 2D."""
        data = torch.randn(4, 10, 5)
        result = model.encode(data, batch_size=2, num_workers=0)
        assert result.ndim == 2

    def test_encode_sequence_returns_3d(self, model: _TestableEncodingModel) -> None:
        """encode(data, batch_size=N, output=SEQUENCE) returns 3D."""
        data = torch.randn(4, 10, 5)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3

    def test_encode_explicit_encoding_window_overrides(self, model: _TestableEncodingModel) -> None:
        """encoding_window takes precedence over output in encode()."""
        data = torch.randn(4, 10, 5)
        result = model.encode(
            data,
            batch_size=2,
            num_workers=0,
            output=EncodingOutputShape.SEQUENCE,
            encoding_window="full_series",
        )
        assert result.ndim == 2


class TestSupportedOutputs:
    """BaseEncodingMixin.supported_outputs declares native capability."""

    def test_supported_outputs_exists(self) -> None:
        assert hasattr(BaseEncodingMixin, "supported_outputs")

    def test_supported_outputs_contains_vector(self) -> None:
        assert EncodingOutputShape.VECTOR in BaseEncodingMixin.supported_outputs

    def test_supported_outputs_contains_sequence(self) -> None:
        assert EncodingOutputShape.SEQUENCE in BaseEncodingMixin.supported_outputs
