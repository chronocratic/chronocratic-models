"""Tests for backbone representation_dim properties.

Verifies that each backbone exposes a correct ``representation_dim``
property matching the actual flattened output of its forward pass.
"""

from __future__ import annotations

import torch

from tscollection.models.supervised import (
    make_series2vec_supervised,
    make_tst_supervised,
    make_tstcc_supervised,
    RepresentationBackbone,
)
from tscollection.models.convolutional.standard.series2vec.model import Series2Vec
from tscollection.models.convolutional.standard.tstcc.model import TSTCC
from tscollection.models.transformer.tst.model import TST


class TestTSTRepresentationDim:
    """Verify TST.representation_dim matches the flattened output."""

    def test_representation_dim_matches_forward(self) -> None:
        """Flattened output size equals model.representation_dim."""
        model = TST(
            feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1, dim_feedforward=16
        )
        x = torch.randn(2, 10, 2)
        padding_masks = torch.ones(2, 10, dtype=torch.bool)
        reps = model.get_representations(x, padding_masks)
        # Zero padding (all valid here), then flatten
        reps_masked = reps * padding_masks.unsqueeze(-1)
        flat = reps_masked.reshape(reps_masked.shape[0], -1)
        assert flat.shape[1] == model.representation_dim

    def test_representation_dim_value(self) -> None:
        """representation_dim = d_model * max_len."""
        model = TST(feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1)
        assert model.representation_dim == 8 * 10

    def test_satisfies_protocol(self) -> None:
        """TST is an instance of RepresentationBackbone."""
        model = TST(feat_dim=2, max_seq_len=5, d_model=4, n_heads=1, num_layers=1)
        assert isinstance(model, RepresentationBackbone)


class TestSeries2VecRepresentationDim:
    """Verify Series2Vec.representation_dim matches the network output."""

    def test_representation_dim_matches_forward(self) -> None:
        """Network.encode output dim equals model.representation_dim."""
        model = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        x = torch.randn(2, 20, 2)
        reps = model.network.encode(x)
        assert reps.shape[1] == model.representation_dim

    def test_representation_dim_value(self) -> None:
        """representation_dim = 2 * representation_dims."""
        model = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        assert model.representation_dim == 2 * 4


class TestTSTCCRepresentationDim:
    """Verify TSTCC.representation_dim matches the encoder's logits input."""

    def test_representation_dim_equals_logits_in_features(self) -> None:
        """representation_dim returns the encoder's logits layer in_features."""
        model = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=4,
            num_classes=3,
        )
        # Per design spec: use logits.in_features as the source of truth
        assert model.representation_dim == model._encoder.logits.in_features  # noqa: SLF001

    def test_representation_dim_value(self) -> None:
        """representation_dim = final_out_channels * features_len."""
        model = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=4,
            num_classes=3,
        )
        assert model.representation_dim == 16 * 4


class TestFactoriesWithRealBackbones:
    """Smoke tests: factories produce working SupervisedModule with real backbones."""

    def test_tst_factory_works(self) -> None:
        """make_tst_supervised with a real TST backbone."""
        backbone = TST(feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1)
        module = make_tst_supervised(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
        )
        x = torch.randn(2, 10, 2)
        padding_masks = torch.ones(2, 10, dtype=torch.bool)
        out = module(x, padding_masks)
        assert out.shape == (2, 3)

    def test_series2vec_factory_works(self) -> None:
        """make_series2vec_supervised with a real Series2Vec backbone."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_supervised(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
        )
        x = torch.randn(2, 20, 2)
        out = module(x)
        assert out.shape == (2, 3)

    def test_tstcc_factory_works(self) -> None:
        """make_tstcc_supervised with a real TSTCC backbone."""
        backbone = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=4,
            num_classes=3,
        )
        module = make_tstcc_supervised(
            backbone, num_outputs=5, task='classification', freeze_backbone=False
        )
        # Verify module construction works (head uses backbone.representation_dim)
        assert module._head._fc.in_features == backbone.representation_dim  # noqa: SLF001
        assert module._head._fc.out_features == 5  # noqa: SLF001
