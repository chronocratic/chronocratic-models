"""Tests for TST and Series2Vec after 2-hook contract refactor (D-02).

Verifies that the two complex models (which previously returned bound methods
from _get_encoder()) now follow the clean 2-hook contract:
- _get_encoder() returns an nn.Module instance
- _encode_batch(encoder, batch_x) handles model-specific encoding
- Old hooks (_get_encoder_module, _prepare_inputs, _postprocess) are removed
"""

from __future__ import annotations

import torch
from torch import nn

from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.transformer.tst.model import TST


class TestTSTTwoHookContract:
    """TST follows the 2-hook encoding contract."""

    def test_get_encoder_returns_nn_module(self) -> None:
        """_get_encoder returns an nn.Module (TSTransformerEncoder), not a bound method."""
        model = TST(input_dims=3, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        encoder = model._get_encoder()
        assert isinstance(encoder, nn.Module), (
            f"_get_encoder must return nn.Module, got {type(encoder).__name__}"
        )

    def test_get_encoder_returns_encoder_not_self(self) -> None:
        """_get_encoder returns self._encoder, not self."""
        model = TST(input_dims=3, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        encoder = model._get_encoder()
        assert encoder is model._encoder

    def test_encode_batch_calls_encode_representations(self) -> None:
        """_encode_batch builds padding mask and calls encoder.encode_representations."""
        model = TST(input_dims=3, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        encoder = model._get_encoder()
        batch_x = torch.randn(2, 10, 3)
        result = model._encode_batch(encoder, batch_x)
        # VECTOR default: mean-pool over seq_len -> (B, hidden_dims)
        expected_shape = (2, 8)
        assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"

    def test_encode_batch_uses_batch_x_device_for_mask(self) -> None:
        """_encode_batch builds padding mask on batch_x.device, not self.device."""
        model = TST(input_dims=3, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        encoder = model._get_encoder()
        batch_x = torch.randn(2, 10, 3)
        result = model._encode_batch(encoder, batch_x)
        assert result.device == batch_x.device

    def test_no_get_encoder_module_override(self) -> None:
        """Old _get_encoder_module hook is removed from TST."""
        assert "_get_encoder_module" not in TST.__dict__, (
            "TST should not override _get_encoder_module after 2-hook refactor"
        )

    def test_no_prepare_inputs_override(self) -> None:
        """Old _prepare_inputs hook is removed from TST."""
        assert "_prepare_inputs" not in TST.__dict__, (
            "TST should not override _prepare_inputs after 2-hook refactor"
        )

    def test_encode_output_shape(self) -> None:
        """encode() produces (B, hidden_dims) output with VECTOR default."""
        model = TST(input_dims=3, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        data = torch.randn(4, 10, 3)
        result = model.encode(data, batch_size=2)
        expected_shape = (4, 8)  # (B, hidden_dims) — VECTOR default
        assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"


class TestSeries2VecTwoHookContract:
    """Series2Vec follows the 2-hook encoding contract."""

    def test_get_encoder_returns_nn_module(self) -> None:
        """_get_encoder returns an nn.Module (Series2VecNetwork), not a bound method."""
        model = Series2Vec(
            input_dims=3,
            embedding_dims=8,
            num_heads=2,
            representation_dims=4,
            encoder_kernel_size=8,
        )
        encoder = model._get_encoder()
        assert isinstance(encoder, nn.Module), (
            f"_get_encoder must return nn.Module, got {type(encoder).__name__}"
        )

    def test_get_encoder_returns_network(self) -> None:
        """_get_encoder returns self.network, not self.network.encode."""
        model = Series2Vec(
            input_dims=3,
            embedding_dims=8,
            num_heads=2,
            representation_dims=4,
            encoder_kernel_size=8,
        )
        encoder = model._get_encoder()
        assert encoder is model.network

    def test_encode_batch_calls_encoder_encode(self) -> None:
        """_encode_batch calls encoder.encode(batch_x) without unsqueeze for VECTOR."""
        model = Series2Vec(
            input_dims=3,
            embedding_dims=8,
            num_heads=2,
            representation_dims=4,
            encoder_kernel_size=8,
        )
        encoder = model._get_encoder()
        batch_x = torch.randn(2, 20, 3)
        result = model._encode_batch(encoder, batch_x)
        # VECTOR default: no unsqueeze -> (B, 2*rep_dims)
        expected_shape = (2, 8)  # 2 * 4 = 8
        assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"

    def test_no_get_encoder_module_override(self) -> None:
        """Old _get_encoder_module hook is removed from Series2Vec."""
        assert "_get_encoder_module" not in Series2Vec.__dict__, (
            "Series2Vec should not override _get_encoder_module after 2-hook refactor"
        )

    def test_no_postprocess_override(self) -> None:
        """Old _postprocess hook is removed from Series2Vec."""
        assert "_postprocess" not in Series2Vec.__dict__, (
            "Series2Vec should not override _postprocess after 2-hook refactor"
        )

    def test_encode_output_shape(self) -> None:
        """encode() produces (B, 2*representation_dims) output with VECTOR default."""
        model = Series2Vec(
            input_dims=3,
            embedding_dims=8,
            num_heads=2,
            representation_dims=4,
            encoder_kernel_size=8,
        )
        data = torch.randn(4, 20, 3)
        result = model.encode(data, batch_size=2)
        expected_shape = (4, 8)  # VECTOR: (B, 2*rep_dims)
        assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
