"""Tests for Series2Vec GroupNorm switch (BatchNorm -> GroupNorm).

Verifies that DisjoinEncoder uses GroupNorm instead of BatchNorm,
ensuring correct gradient flow at batch_size=1.
"""

import inspect

import pytest
import torch

from chronocratic.models.convolutional.standard.series2vec.encoder import DisjoinEncoder
from chronocratic.models.convolutional.standard.series2vec.network import Series2VecNetwork
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


class TestDisjoinEncoderGroupNorm:
    """DisjoinEncoder uses GroupNorm(1, C) for all normalization layers."""

    def test_temporal_cnn_uses_group_norm(self) -> None:
        encoder = DisjoinEncoder(
            input_dims=3, embedding_dims=16, representation_dims=32, kernel_size=8
        )
        norm = encoder.temporal_CNN[1]
        assert isinstance(norm, torch.nn.GroupNorm)
        assert norm.num_groups == 1
        assert norm.num_channels == 16

    def test_spatial_cnn_uses_group_norm(self) -> None:
        encoder = DisjoinEncoder(
            input_dims=3, embedding_dims=16, representation_dims=32, kernel_size=8
        )
        norm = encoder.spatial_CNN[1]
        assert isinstance(norm, torch.nn.GroupNorm)
        assert norm.num_groups == 1
        assert norm.num_channels == 16

    def test_rep_cnn_uses_group_norm(self) -> None:
        encoder = DisjoinEncoder(
            input_dims=3, embedding_dims=16, representation_dims=32, kernel_size=8
        )
        norm = encoder.rep_CNN[1]
        assert isinstance(norm, torch.nn.GroupNorm)
        assert norm.num_groups == 1
        assert norm.num_channels == 32

    def test_no_batch_norm_in_encoder(self) -> None:
        encoder = DisjoinEncoder(
            input_dims=3, embedding_dims=16, representation_dims=32, kernel_size=8
        )
        for module in encoder.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))

    def test_forward_batch_size_one(self) -> None:
        """GroupNorm must work at batch_size=1 (unlike BatchNorm)."""
        encoder = DisjoinEncoder(
            input_dims=3, embedding_dims=16, representation_dims=32, kernel_size=8
        )
        x = torch.randn(1, 3, 20)  # (batch, channels, time) — time must be >= kernel_size
        out = encoder(x)
        # temporal: Conv2d(1,16,(1,8),valid) on (1,1,3,20) -> (1,16,3,13)
        # spatial: Conv2d(16,16,(3,1),valid) on (1,16,3,13) -> (1,16,1,13)
        # rep: Conv1d(16,32,3) on (1,16,13) -> (1,32,11)
        assert out.shape == (1, 32, 11)

    def test_gradient_flow_batch_size_one(self) -> None:
        """Verify gradients flow through GroupNorm at batch_size=1."""
        encoder = DisjoinEncoder(
            input_dims=3, embedding_dims=16, representation_dims=32, kernel_size=8
        )
        x = torch.randn(1, 3, 20, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_no_norm_parameter_in_init(self) -> None:
        """DisjoinEncoder should not accept a norm parameter."""
        sig = inspect.signature(DisjoinEncoder.__init__)
        param_names = list(sig.parameters.keys())
        assert "norm" not in param_names


class TestSeries2VecNetworkGroupNorm:
    """Series2VecNetwork creates DisjoinEncoder instances with GroupNorm."""

    def test_both_encoders_use_group_norm(self) -> None:
        network = Series2VecNetwork(
            input_dims=3, embedding_dims=16, representation_dims=32, encoder_kernel_size=8
        )
        assert isinstance(network.embed_layer.temporal_CNN[1], torch.nn.GroupNorm)
        assert isinstance(network.embed_layer_f.spatial_CNN[1], torch.nn.GroupNorm)

    def test_no_batch_norm_in_network(self) -> None:
        network = Series2VecNetwork(
            input_dims=3, embedding_dims=16, representation_dims=32, encoder_kernel_size=8
        )
        for module in network.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))

    def test_encode_batch_size_one(self) -> None:
        network = Series2VecNetwork(
            input_dims=3, embedding_dims=16, representation_dims=32, encoder_kernel_size=8
        )
        x = torch.randn(1, 20, 3)  # (batch, time, channels) — time must be >= kernel_size
        out = network.encode(x)
        assert out.shape == (1, 64)  # (batch, 2 * representation_dims)


class TestSeries2VecGroupNorm:
    """Series2Vec model has no norm parameter and uses GroupNorm internally."""

    def test_default_model_uses_group_norm(self) -> None:
        model = Series2Vec(input_dims=3)
        assert isinstance(model.network.embed_layer.temporal_CNN[1], torch.nn.GroupNorm)

    def test_no_batch_norm_in_model(self) -> None:
        model = Series2Vec(input_dims=3)
        for module in model.network.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))

    def test_no_norm_parameter(self) -> None:
        sig = inspect.signature(Series2Vec.__init__)
        param_names = list(sig.parameters.keys())
        assert "norm" not in param_names

    def test_encode_batch_size_one(self) -> None:
        model = Series2Vec(input_dims=3)
        x = torch.randn(1, 50, 3)  # (batch, time, channels) — time >= kernel_size
        out = model.network.encode(x)
        assert out.shape == (1, 640)  # (batch, 2 * 320)
