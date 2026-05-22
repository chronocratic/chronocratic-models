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
        self.component_dims = output_dim  # Match CoST convention

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
        from tscollection.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin

        assert PoolingEncodingMixin is not None

    def test_decomposition_mixin_import(self) -> None:
        from tscollection.models.convolutional.dilated._mixin.encoding import DecompositionEncodingMixin

        assert DecompositionEncodingMixin is not None

    def test_base_mixin_import(self) -> None:
        from tscollection.models.convolutional.dilated._mixin.encoding import BaseEncodingMixin

        assert BaseEncodingMixin is not None


# ---------------------------------------------------------------------------
# Class structure tests
# ---------------------------------------------------------------------------

class TestMixinHierarchy:
    """Verify the three-class hierarchy and ABC enforcement."""

    @pytest.fixture(autouse=True)
    def _load_classes(self) -> None:
        from tscollection.models.convolutional.dilated._mixin.encoding import (
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

@pytest.fixture
def pooling_model() -> nn.Module:
    """Create a minimal pooling-based model for testing."""
    from tscollection.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin

    class _PoolingTestModel(PoolingEncodingMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._averaged_encoder = _DummyEncoder(output_dim=64)
            self.device = torch.device('cpu')

    return _PoolingTestModel()


@pytest.fixture
def decomposition_model() -> nn.Module:
    """Create a minimal decomposition-based model for testing."""
    from tscollection.models.convolutional.dilated._mixin.encoding import DecompositionEncodingMixin

    class _DecompositionTestModel(DecompositionEncodingMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_encoder = _DecompositionEncoder(output_dim=64)
            self.device = torch.device('cpu')

    return _DecompositionTestModel()


# ---------------------------------------------------------------------------
# Polymorphic dispatch tests
# ---------------------------------------------------------------------------

class TestPolymorphicDispatch:
    """Verify _get_eval_method and _get_encoder return correct implementations."""

    def test_pooling_get_eval_method_returns_pooling(
        self, pooling_model: nn.Module
    ) -> None:
        method = pooling_model._get_eval_method()
        assert method == pooling_model._evaluate_with_pooling

    def test_decomposition_get_eval_method_returns_concat(
        self, decomposition_model: nn.Module
    ) -> None:
        method = decomposition_model._get_eval_method()
        assert method == decomposition_model._evaluate_with_feature_concatenation

    def test_pooling_get_slice_returns_real_slice(
        self, pooling_model: nn.Module
    ) -> None:
        s = pooling_model._get_slice(sliding_padding=10, sliding_length=20)
        assert s == slice(10, 30)

    def test_decomposition_get_slice_returns_none(
        self, decomposition_model: nn.Module
    ) -> None:
        s = decomposition_model._get_slice(sliding_padding=10, sliding_length=20)
        assert s is None

    def test_pooling_get_encoder_returns_averaged_encoder(
        self, pooling_model: nn.Module
    ) -> None:
        encoder = pooling_model._get_encoder()
        assert encoder is pooling_model._averaged_encoder


# ---------------------------------------------------------------------------
# Encode behavior tests
# ---------------------------------------------------------------------------

class TestEncodeBehavior:
    """Verify encode() does not mutate instance state and uses polymorphic dispatch."""

    def test_encode_no_state_mutation(self, pooling_model: nn.Module) -> None:
        """encode() should not set self._encoder or self._eval_method."""
        data = torch.randn(2, 10, 3)
        # Ensure these attrs don't exist before encode
        if hasattr(pooling_model, '_encoder'):
            del pooling_model._encoder
        if hasattr(pooling_model, '_eval_method'):
            del pooling_model._eval_method

        pooling_model.encode(data=data, batch_size=2, num_workers=0)

        # After encode(), no instance-level _encoder or _eval_method should be set
        assert not hasattr(pooling_model, '_encoder') or (
            '_encoder' not in pooling_model.__dict__
        ), "encode() should not set self._encoder instance attribute"
        assert not hasattr(pooling_model, '_eval_method') or (
            '_eval_method' not in pooling_model.__dict__
        ), "encode() should not set self._eval_method instance attribute"

    def test_encode_uses_polymorphic_dispatch(self, pooling_model: nn.Module) -> None:
        """encode() calls _get_encoder() and _get_eval_method() each invocation."""
        source = inspect.getsource(pooling_model.encode)
        assert '_get_encoder()' in source or 'self._get_encoder' in source
        assert '_get_eval_method()' in source or 'self._get_eval_method' in source


# ---------------------------------------------------------------------------
# Bug-fix verification tests (source inspection)
# ---------------------------------------------------------------------------

class TestBugFixes:
    """Verify the two runtime bug fixes are present in the source."""

    def test_persistent_workers_condition(self) -> None:
        """DataLoader must use persistent_workers=num_workers > 0."""
        import pathlib

        mixin_file = pathlib.Path(__file__).parents[1] / 'src' / 'tscollection' / 'models' / 'convolutional' / 'dilated' / '_mixin' / 'encoding.py'
        source = mixin_file.read_text()
        assert 'persistent_workers=num_workers > 0' in source

    def test_sliding_window_transpose(self) -> None:
        """_compute_sliding_representations full_series path uses .transpose(1, 2)."""
        import pathlib

        mixin_file = pathlib.Path(__file__).parents[1] / 'src' / 'tscollection' / 'models' / 'convolutional' / 'dilated' / '_mixin' / 'encoding.py'
        source = mixin_file.read_text()
        assert 'transpose(1, 2)' in source


# ---------------------------------------------------------------------------
# Decomposition validation tests
# ---------------------------------------------------------------------------

class TestDecompositionValidation:
    """Verify _evaluate_with_feature_concatenation raises for invalid encoding_window."""

    def test_invalid_encoding_window_raises(self, decomposition_model: nn.Module) -> None:
        data = torch.randn(2, 10, 3)
        with pytest.raises(ValueError, match='encoding_window'):
            decomposition_model._evaluate_with_feature_concatenation(
                input_tensor=data,
                mask=None,
                slicing=None,
                encoding_window='multiscale',
            )

    def test_none_encoding_window_ok(self, decomposition_model: nn.Module) -> None:
        data = torch.randn(2, 10, 3)
        result = decomposition_model._evaluate_with_feature_concatenation(
            input_tensor=data,
            mask=None,
            slicing=None,
            encoding_window=None,
        )
        assert isinstance(result, torch.Tensor)

    def test_full_series_encoding_window_ok(self, decomposition_model: nn.Module) -> None:
        data = torch.randn(2, 10, 3)
        result = decomposition_model._evaluate_with_feature_concatenation(
            input_tensor=data,
            mask=None,
            slicing=None,
            encoding_window='full_series',
        )
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Source-level compliance tests
# ---------------------------------------------------------------------------

class TestSourceCompliance:
    """Verify D-05 adaptations at the source level."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        import pathlib

        mixin_file = pathlib.Path(__file__).parents[1] / 'src' / 'tscollection' / 'models' / 'convolutional' / 'dilated' / '_mixin' / 'encoding.py'
        self.source = mixin_file.read_text()

    def test_no_hasattr_branching(self) -> None:
        assert 'hasattr' not in self.source

    def test_no_encoder_none_guard(self) -> None:
        assert 'encoder is None' not in self.source

    def test_uses_logger_private(self) -> None:
        assert '_logger = logging' in self.source

    def test_has_expected_input_dims_constant(self) -> None:
        assert '_EXPECTED_INPUT_DIMS' in self.source

    def test_has_override_decorator(self) -> None:
        assert '@override' in self.source

    def test_type_checking_mask_mode(self) -> None:
        assert 'TYPE_CHECKING' in self.source
