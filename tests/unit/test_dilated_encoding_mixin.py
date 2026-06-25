"""Tests for dilated BaseEncodingMixin gradient_enabled support.

Verifies that the dilated mixin encode() accepts gradient_enabled kwarg
and uses nullcontext/inference_mode branching, consistent with BasicEncodingMixin.
"""

from __future__ import annotations

import inspect

import pytest

from chronocratic.models.convolutional.dilated._mixin.encoding import BaseEncodingMixin


class TestDilatedMixinGradientEnabledSignature:
    """BaseEncodingMixin.encode() has gradient_enabled keyword argument."""

    def test_encode_has_gradient_enabled_param(self) -> None:
        """encode() signature includes gradient_enabled."""
        sig = inspect.signature(BaseEncodingMixin.encode)
        assert "gradient_enabled" in sig.parameters

    def test_gradient_enabled_defaults_to_false(self) -> None:
        """gradient_enabled defaults to False (backward compatible)."""
        sig = inspect.signature(BaseEncodingMixin.encode)
        param = sig.parameters["gradient_enabled"]
        assert param.default is False

    def test_gradient_enabled_is_keyword_only(self) -> None:
        """gradient_enabled must be passed as keyword argument."""
        sig = inspect.signature(BaseEncodingMixin.encode)
        param = sig.parameters["gradient_enabled"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


class TestDilatedMixinUsesNullcontext:
    """Dilated encode() body uses nullcontext for gradient_enabled=True."""

    def test_nullcontext_import_present(self) -> None:
        """nullcontext is imported in the dilated encoding module."""
        import chronocratic.models.convolutional.dilated._mixin.encoding as mod

        assert hasattr(mod, "nullcontext")

    def test_encode_source_contains_nullcontext(self) -> None:
        """encode() source code references nullcontext."""
        source = inspect.getsource(BaseEncodingMixin.encode)
        assert "nullcontext" in source

    def test_encode_source_contains_gradient_enabled(self) -> None:
        """encode() source code references gradient_enabled."""
        source = inspect.getsource(BaseEncodingMixin.encode)
        assert "gradient_enabled" in source
