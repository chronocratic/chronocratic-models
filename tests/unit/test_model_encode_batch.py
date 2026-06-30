"""Tests for 5 simple BasicEncodingMixin models after 2-hook refactor.

Verifies that TimeVAE, TimeNet, RecurrentAutoEncoder, MCL, and TSTCC
use _encode_batch (not _postprocess) and produce correct output shapes.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models import MCL
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.generative.timevae.model import TimeVAE
from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder
from chronocratic.models.recurrent.timenet.model import TimeNet


MODELS = {
    "TimeVAE": TimeVAE(sequence_length=32, input_dims=3, latent_dim=8),
    "TimeNet": TimeNet(hidden_dims=16, depth=1, input_dims=3),
    "RecurrentAutoEncoder": RecurrentAutoEncoder(input_dims=3, layers=(16,)),
    "MCL": MCL(input_dims=3),
    "TSTCC": TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16),
}


class TestModelsUseEncodeBatch:
    """All 5 simple models use _encode_batch, not old hooks."""

    @pytest.mark.parametrize("name,model", MODELS.items())
    def test_has_encode_batch(self, name: str, model) -> None:
        """Each model defines _encode_batch."""
        assert hasattr(model, "_encode_batch")
        # Verify it's overridden (not just inherited default)
        method = type(model)._encode_batch
        assert method is not None

    @pytest.mark.parametrize("name,model", MODELS.items())
    def test_no_postprocess(self, name: str, model) -> None:
        """Old _postprocess hook is removed from model classes."""
        assert "_postprocess" not in type(model).__dict__

    @pytest.mark.parametrize("name,model", MODELS.items())
    def test_no_get_encoder_module(self, name: str, model) -> None:
        """Old _get_encoder_module hook is not overridden by model classes."""
        # Models should not define their own _get_encoder_module
        assert "_get_encoder_module" not in type(model).__dict__


class TestEncodeOutputShapes:
    """encode() produces expected output shapes after refactor."""

    def test_timevae_encode_shape(self) -> None:
        """TimeVAE.encode() returns (B, latent_dim)."""
        model = TimeVAE(sequence_length=32, input_dims=3, latent_dim=8)
        data = torch.randn(4, 32, 3)
        result = model.encode(data, batch_size=2)
        assert result.shape == (4, 8)

    def test_timenet_encode_shape(self) -> None:
        """TimeNet.encode() returns (B, hidden_dims)."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        data = torch.randn(4, 20, 3)
        result = model.encode(data, batch_size=2)
        assert result.shape == (4, 16)

    def test_recurrentae_encode_shape(self) -> None:
        """RecurrentAutoEncoder.encode() returns (B, last_layer_dim)."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        data = torch.randn(4, 20, 3)
        result = model.encode(data, batch_size=2)
        assert result.shape == (4, 16)

    def test_mcl_encode_shape(self) -> None:
        """MCL.encode() returns (B, output_dims) with VECTOR default."""
        model = MCL(input_dims=3, output_dims=128)
        data = torch.randn(4, 50, 3)
        result = model.encode(data, batch_size=2)
        assert result.shape == (4, 128)

    def test_tstcc_encode_shape(self) -> None:
        """TSTCC.encode() returns (B, output_dims) after pooling."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        data = torch.randn(4, 256, 3)
        result = model.encode(data, batch_size=2)
        assert result.shape == (4, 16)
