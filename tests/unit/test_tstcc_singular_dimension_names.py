"""Tests for TSTCC dimension parameter naming (singular convention).

Verifies the rename from plural dimension names to singular:
- input_dims -> input_dim
- output_dims -> representation_dim

These tests enforce decisions D-01 (plural to singular), D-02 (unify to
representation_dim), and D-03 (property returns encode output).
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

from chronocratic.models.convolutional.standard.tstcc.config import TSTCCModelParameters
from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC


class TestConfigSingularDimensionNames:
    """TSTCCModelParameters uses singular dimension field names."""

    def test_input_dim_field_exists(self) -> None:
        """Config has input_dim (not input_dims)."""
        fields = TSTCCModelParameters.__dataclass_fields__
        assert "input_dim" in fields, "Expected 'input_dim' field in TSTCCModelParameters"
        assert "input_dims" not in fields, "'input_dims' should be renamed to 'input_dim'"

    def test_representation_dim_field_exists(self) -> None:
        """Config has representation_dim (not output_dims)."""
        fields = TSTCCModelParameters.__dataclass_fields__
        assert (
            "representation_dim" in fields
        ), "Expected 'representation_dim' field in TSTCCModelParameters"
        assert (
            "output_dims" not in fields
        ), "'output_dims' should be renamed to 'representation_dim'"

    def test_config_constructs_with_singular_names(self) -> None:
        """TSTCCModelParameters accepts singular param names."""
        config = TSTCCModelParameters(input_dim=3, representation_dim=64)
        assert config.input_dim == 3
        assert config.representation_dim == 64

    def test_config_validation_input_dim(self) -> None:
        """Validator rejects non-positive input_dim."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            TSTCCModelParameters(input_dim=0, representation_dim=64)

    def test_config_validation_representation_dim(self) -> None:
        """Validator rejects non-positive representation_dim."""
        with pytest.raises(ValueError, match="representation_dim must be positive"):
            TSTCCModelParameters(input_dim=3, representation_dim=-1)


class TestModelSingularParams:
    """TSTCC model __init__ uses singular dimension parameter names."""

    def test_model_accepts_input_dim(self) -> None:
        """TSTCC.__init__ accepts input_dim."""
        sig = inspect.signature(TSTCC.__init__)
        assert "input_dim" in sig.parameters
        assert "input_dims" not in sig.parameters

    def test_model_accepts_representation_dim(self) -> None:
        """TSTCC.__init__ accepts representation_dim (not output_dims)."""
        sig = inspect.signature(TSTCC.__init__)
        assert "representation_dim" in sig.parameters
        assert "output_dims" not in sig.parameters

    def test_model_instantiates_with_singular_names(self) -> None:
        """TSTCC can be created using singular param names."""
        model = TSTCC(input_dim=3, conv_kernel_size=8, stride=4, representation_dim=16)
        assert model is not None

    def test_model_from_config_vars(self) -> None:
        """TSTCC(**vars(TSTCCModelParameters(...))) succeeds."""
        config = TSTCCModelParameters(
            input_dim=3,
            conv_kernel_size=8,
            stride=4,
            representation_dim=16,
        )
        # Extract only __init__-compatible fields (skip training-only params)
        init_sig = inspect.signature(TSTCC.__init__)
        init_params = set(init_sig.parameters.keys()) - {"self", "augmentation"}
        kwargs = {k: v for k, v in vars(config).items() if k in init_params}
        model = TSTCC(**kwargs)
        assert model is not None


class TestEncoderSingularParams:
    """TCCEncoder uses singular dimension parameter names."""

    def test_encoder_accepts_input_dim(self) -> None:
        """TCCEncoder.__init__ accepts input_dim."""
        sig = inspect.signature(TCCEncoder.__init__)
        assert "input_dim" in sig.parameters
        assert "input_dims" not in sig.parameters

    def test_encoder_accepts_representation_dim(self) -> None:
        """TCCEncoder.__init__ accepts representation_dim (not output_dims)."""
        sig = inspect.signature(TCCEncoder.__init__)
        assert "representation_dim" in sig.parameters
        assert "output_dims" not in sig.parameters

    def test_encoder_has_representation_dim_attribute(self) -> None:
        """TCCEncoder stores representation_dim attribute."""
        encoder = TCCEncoder(input_dim=3, conv_kernel_size=8, stride=1, representation_dim=64)
        assert hasattr(encoder, "representation_dim")
        assert encoder.representation_dim == 64

    def test_encoder_forward_output_shape(self) -> None:
        """Encoder forward returns (B, representation_dim, L')."""
        encoder = TCCEncoder(input_dim=3, conv_kernel_size=8, stride=4, representation_dim=16)
        x = torch.randn(4, 256, 3)
        out = encoder(x)
        assert out.shape[1] == 16  # representation_dim


class TestRepresentationDimProperty:
    """TSTCC.representation_dim returns encode output feature width."""

    def test_property_returns_encoder_representation_dim(self) -> None:
        """representation_dim property delegates to encoder.representation_dim."""
        model = TSTCC(input_dim=3, conv_kernel_size=8, stride=4, representation_dim=32)
        assert model.representation_dim == 32
        assert model._encoder.representation_dim == 32

    def test_property_matches_encode_vector_output(self) -> None:
        """representation_dim matches the feature dim of encode() VECTOR output."""
        rep_dim = 48
        model = TSTCC(input_dim=3, conv_kernel_size=8, stride=4, representation_dim=rep_dim)
        assert model.representation_dim == rep_dim
        data = torch.randn(2, 256, 3)
        encoded = model.encode(data, batch_size=2, num_workers=0)
        assert encoded.shape[1] == rep_dim

    def test_property_matches_encode_sequence_feature_dim(self) -> None:
        """representation_dim matches the feature dim of SEQUENCE output."""
        rep_dim = 48
        model = TSTCC(input_dim=3, conv_kernel_size=8, stride=4, representation_dim=rep_dim)
        data = torch.randn(2, 256, 3)
        from chronocratic.models.enums.encoding import EncodingOutputShape

        encoded = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert encoded.shape[2] == rep_dim
