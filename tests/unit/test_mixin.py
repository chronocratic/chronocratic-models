"""Tests for the encoding mixin hierarchy (Base/Pooling/Decomposition).

Covers importability, class structure, polymorphic dispatch,
and bug-fix verification (persistent_workers, sliding window shape).
"""

import abc
import inspect
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from chronocratic.models.convolutional.dilated._mixin.encoding import (
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _DummyEncoder(nn.Module):
    """Minimal encoder that returns a shaped tensor for testing."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        mask_mode: "Callable[..., torch.Tensor] | None" = None,  # noqa: ARG002
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        return torch.randn(batch, seq, self.output_dim, device=x.device)


class _DecompositionEncoder(nn.Module):
    """Minimal decomposition encoder returning (trend, seasonality) tuple."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.component_dim = output_dim  # Match CoST convention

    def forward(
        self,
        x: torch.Tensor,
        mask_mode: "Callable[..., torch.Tensor] | None" = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        dim = self.output_dim
        trend = torch.randn(batch, seq, dim, device=x.device)
        seasonality = torch.randn(batch, seq, dim, device=x.device)
        return trend, seasonality


# ---------------------------------------------------------------------------
# Import tests (MIX-01, MIX-02, MIX-03)
# ---------------------------------------------------------------------------


class TestMixinImports:
    """Verify all three mixin classes are importable from _mixin.encoding."""

    def test_pooling_mixin_import(self) -> None:
        from chronocratic.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin

        assert PoolingEncodingMixin is not None

    def test_decomposition_mixin_import(self) -> None:
        from chronocratic.models.convolutional.dilated._mixin.encoding import (
            DecompositionEncodingMixin,
        )

        assert DecompositionEncodingMixin is not None

    def test_base_mixin_import(self) -> None:
        from chronocratic.models.convolutional.dilated._mixin.encoding import BaseEncodingMixin

        assert BaseEncodingMixin is not None


# ---------------------------------------------------------------------------
# Class structure tests
# ---------------------------------------------------------------------------


class TestMixinHierarchy:
    """Verify the three-class hierarchy and ABC enforcement."""

    @pytest.fixture(autouse=True)
    def _load_classes(self) -> None:
        from chronocratic.models.convolutional.dilated._mixin.encoding import (
            BaseEncodingMixin,
            DecompositionEncodingMixin,
            PoolingEncodingMixin,
        )

        self.BaseEncodingMixin = BaseEncodingMixin
        self.PoolingEncodingMixin = PoolingEncodingMixin
        self.DecompositionEncodingMixin = DecompositionEncodingMixin

    def test_pooling_is_subclass_of_base(self) -> None:
        assert issubclass(self.PoolingEncodingMixin, self.BaseEncodingMixin)

    def test_decomposition_is_subclass_of_base(self) -> None:
        assert issubclass(self.DecompositionEncodingMixin, self.BaseEncodingMixin)

    def test_base_is_abc(self) -> None:
        assert issubclass(self.BaseEncodingMixin, abc.ABC)


# ---------------------------------------------------------------------------
# Concrete test models (minimal subclasses for behavior testing)
# ---------------------------------------------------------------------------


class _PoolingTestModel(PoolingEncodingMixin, nn.Module):  # type: ignore[misc]
    """Minimal pooling-based model for testing."""

    device: torch.device

    def __init__(self) -> None:
        super().__init__()
        self._averaged_encoder = _DummyEncoder(output_dim=64)
        self.device = torch.device("cpu")


class _DecompositionTestModel(DecompositionEncodingMixin, nn.Module):  # type: ignore[misc]
    """Minimal decomposition-based model for testing."""

    device: torch.device

    def __init__(self) -> None:
        super().__init__()
        self.query_encoder = _DecompositionEncoder(output_dim=64)
        self.device = torch.device("cpu")


@pytest.fixture
def pooling_model() -> _PoolingTestModel:
    """Create a minimal pooling-based model for testing."""
    return _PoolingTestModel()


@pytest.fixture
def decomposition_model() -> _DecompositionTestModel:
    """Create a minimal decomposition-based model for testing."""
    return _DecompositionTestModel()


# ---------------------------------------------------------------------------
# Polymorphic dispatch tests
# ---------------------------------------------------------------------------


class TestPolymorphicDispatch:
    """Verify _get_eval_method and _get_encoder return correct implementations."""

    def test_pooling_get_eval_method_returns_pooling(
        self, pooling_model: _PoolingTestModel
    ) -> None:
        method = pooling_model._get_eval_method()
        assert method == pooling_model._evaluate_with_pooling

    def test_decomposition_get_eval_method_returns_concat(
        self, decomposition_model: _DecompositionTestModel
    ) -> None:
        method = decomposition_model._get_eval_method()
        assert method == decomposition_model._evaluate_with_feature_concatenation

    def test_pooling_get_slice_returns_real_slice(self, pooling_model: _PoolingTestModel) -> None:
        s = pooling_model._get_slice(sliding_padding=10, sliding_length=20)
        assert s == slice(10, 30)

    def test_decomposition_get_slice_returns_real_slice(
        self, decomposition_model: _DecompositionTestModel
    ) -> None:
        s = decomposition_model._get_slice(sliding_padding=10, sliding_length=20)
        assert s == slice(10, 30)

    def test_pooling_get_encoder_returns_averaged_encoder(
        self, pooling_model: _PoolingTestModel
    ) -> None:
        encoder = pooling_model._get_encoder()
        assert encoder is pooling_model._averaged_encoder


# ---------------------------------------------------------------------------
# Sliding-window SEQUENCE shape tests (over-emit regression)
# ---------------------------------------------------------------------------


class TestSlidingSequenceShape:
    """Sliding SEQUENCE encode must preserve the time axis for both mixins.

    Regression for the CoST over-emit bug: DecompositionEncodingMixin ignored the
    per-window slice, so concatenation produced (sliding_padding + sliding_length)*T
    timesteps instead of T.
    """

    @pytest.mark.parametrize("model_fixture", ["pooling_model", "decomposition_model"])
    def test_sliding_sequence_preserves_time_length(
        self, model_fixture: str, request: "pytest.FixtureRequest"
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_samples, time_len, channels = 2, 8, 3
        data = torch.randn(num_samples, time_len, channels)

        out = model._compute_sliding_representations(
            input_tensor=data,
            sliding_length=1,
            sliding_padding=3,
            causal=True,
            mask=None,
            encoding_window=None,  # SEQUENCE
            num_samples=num_samples,
            batch_size=num_samples,  # >= num_samples avoids the buffering path
        )

        assert out.shape[0] == num_samples
        assert out.shape[1] == time_len


# ---------------------------------------------------------------------------
# Encode behavior tests
# ---------------------------------------------------------------------------


class TestEncodeBehavior:
    """Verify encode() does not mutate instance state and uses polymorphic dispatch."""

    def test_encode_no_state_mutation(self, pooling_model: _PoolingTestModel) -> None:
        """encode() should not set self._encoder or self._eval_method."""
        data = torch.randn(2, 10, 3)
        # Ensure these attrs don't exist before encode
        if hasattr(pooling_model, "_encoder"):
            del pooling_model._encoder
        if hasattr(pooling_model, "_eval_method"):
            del pooling_model._eval_method

        pooling_model.encode(data=data, batch_size=2, num_workers=0)

        # After encode(), no instance-level _encoder or _eval_method should be set
        assert not hasattr(pooling_model, "_encoder") or (
            "_encoder" not in pooling_model.__dict__
        ), "encode() should not set self._encoder instance attribute"
        assert not hasattr(pooling_model, "_eval_method") or (
            "_eval_method" not in pooling_model.__dict__
        ), "encode() should not set self._eval_method instance attribute"

    def test_encode_uses_polymorphic_dispatch(self, pooling_model: _PoolingTestModel) -> None:
        """encode() delegates to encode_batch() which calls _get_eval_method()."""
        encode_source = inspect.getsource(pooling_model.encode)
        assert "_get_encoder()" in encode_source or "self._get_encoder" in encode_source
        assert "encode_batch(" in encode_source, "encode() should delegate to encode_batch()"


# ---------------------------------------------------------------------------
# Bug-fix verification tests (source inspection)
# ---------------------------------------------------------------------------


class TestBugFixes:
    """Verify the two runtime bug fixes are present in the source."""

    @pytest.fixture(autouse=True)
    def _mixin_source(self) -> None:
        import pathlib

        mixin_file = (
            pathlib.Path(__file__).parents[2]
            / "src"
            / "chronocratic"
            / "models"
            / "convolutional"
            / "dilated"
            / "_mixin"
            / "encoding.py"
        )
        self.source = mixin_file.read_text()

    def test_persistent_workers_condition(self) -> None:
        """DataLoader must use persistent_workers=num_workers > 0."""
        assert "persistent_workers=num_workers > 0" in self.source

    def test_sliding_window_transpose(self) -> None:
        """_compute_sliding_representations full_series path uses .transpose(1, 2)."""
        assert "transpose(1, 2)" in self.source


# ---------------------------------------------------------------------------
# Decomposition validation tests
# ---------------------------------------------------------------------------


class TestDecompositionValidation:
    """Verify _evaluate_with_feature_concatenation raises for invalid encoding_window."""

    def test_invalid_encoding_window_raises(
        self, decomposition_model: _DecompositionTestModel
    ) -> None:
        data = torch.randn(2, 10, 3)
        with pytest.raises(ValueError, match="encoding_window"):
            decomposition_model._evaluate_with_feature_concatenation(
                input_tensor=data, mask=None, slicing=None, encoding_window="multiscale"
            )

    def test_none_encoding_window_ok(self, decomposition_model: _DecompositionTestModel) -> None:
        data = torch.randn(2, 10, 3)
        result = decomposition_model._evaluate_with_feature_concatenation(
            input_tensor=data, mask=None, slicing=None, encoding_window=None
        )
        assert isinstance(result, torch.Tensor)

    def test_full_series_encoding_window_ok(
        self, decomposition_model: _DecompositionTestModel
    ) -> None:
        data = torch.randn(2, 10, 3)
        result = decomposition_model._evaluate_with_feature_concatenation(
            input_tensor=data, mask=None, slicing=None, encoding_window="full_series"
        )
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Source-level compliance tests
# ---------------------------------------------------------------------------


class TestSourceCompliance:
    """Verify mixin adaptations in source code."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        import pathlib

        mixin_file = (
            pathlib.Path(__file__).parents[2]
            / "src"
            / "chronocratic"
            / "models"
            / "convolutional"
            / "dilated"
            / "_mixin"
            / "encoding.py"
        )
        self.source = mixin_file.read_text()

    def test_no_hasattr_branching(self) -> None:
        assert "hasattr" not in self.source

    def test_no_encoder_none_guard(self) -> None:
        assert "encoder is None" not in self.source

    def test_uses_logger_private(self) -> None:
        assert "_logger = logging" in self.source

    def test_has_expected_input_rank_constant(self) -> None:
        assert "_EXPECTED_INPUT_DIM" in self.source

    def test_has_override_decorator(self) -> None:
        assert "@override" in self.source

    def test_type_checking_mask_mode(self) -> None:
        assert "TYPE_CHECKING" in self.source
