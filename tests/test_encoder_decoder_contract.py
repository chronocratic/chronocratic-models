"""Conformance tests for the encoder/decoder extraction contract.

Verifies that all model classes satisfy the ``HasEncoder`` and (where
applicable) ``HasDecoder`` protocols via runtime-checkable ``isinstance``
checks, and that the ``.encoder`` / ``.decoder`` properties return actual
``nn.Module`` instances.

Construction parameters are minimal — only enough to instantiate each
model without triggering side effects (e.g. augmentation training).
"""

from __future__ import annotations

import pytest
from torch import nn

from chronocratic.models import (
    AutoTCL,
    CoST,
    FCN,
    HasDecoder,
    HasEncoder,
    Series2Vec,
    TimeNet,
    TimeVAE,
    TS2Vec,
    TST,
    TSTCC,
)


class TestHasEncoderConformance:
    """All 9 models must satisfy HasEncoder via isinstance."""

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (FCN, {"n_in": 1}),
            (TST, {"feat_dim": 1, "max_seq_len": 100}),
            (
                TSTCC,
                {
                    "input_channels": 1,
                    "kernel_size": 5,
                    "stride": 1,
                    "final_out_channels": 16,
                    "features_len": 12,
                    "num_classes": 10,
                },
            ),
            (TS2Vec, {"input_dims": 1}),
            (AutoTCL, {"input_dims": 1, "kernel_sizes": [3]}),
            (CoST, {"input_dims": 1, "sequence_length": 100, "kernel_sizes": [3]}),
            (
                Series2Vec,
                {
                    "input_dims": 1,
                    "embedding_dims": 64,
                    "num_heads": 2,
                    "feedforward_dims": 128,
                    "representation_dims": 64,
                    "dropout_rate": 0.1,
                },
            ),
            (TimeVAE, {"seq_len": 100, "feat_dim": 1, "latent_dim": 10}),
            (TimeNet, {"hidden_dims": 64, "num_layers": 1, "feat_dim": 1}),
        ],
        ids=[
            "FCN",
            "TST",
            "TSTCC",
            "TS2Vec",
            "AutoTCL",
            "CoST",
            "Series2Vec",
            "TimeVAE",
            "TimeNet",
        ],
    )
    def test_model_satisfies_has_encoder(self, model_cls: type, kwargs: dict) -> None:
        """Each model instance must satisfy the HasEncoder protocol."""
        model = model_cls(**kwargs)
        assert isinstance(model, HasEncoder), (
            f"{model_cls.__name__} does not satisfy HasEncoder protocol"
        )

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (FCN, {"n_in": 1}),
            (TST, {"feat_dim": 1, "max_seq_len": 100}),
            (
                TSTCC,
                {
                    "input_channels": 1,
                    "kernel_size": 5,
                    "stride": 1,
                    "final_out_channels": 16,
                    "features_len": 12,
                    "num_classes": 10,
                },
            ),
            (TS2Vec, {"input_dims": 1}),
            (AutoTCL, {"input_dims": 1, "kernel_sizes": [3]}),
            (CoST, {"input_dims": 1, "sequence_length": 100, "kernel_sizes": [3]}),
            (
                Series2Vec,
                {
                    "input_dims": 1,
                    "embedding_dims": 64,
                    "num_heads": 2,
                    "feedforward_dims": 128,
                    "representation_dims": 64,
                    "dropout_rate": 0.1,
                },
            ),
            (TimeVAE, {"seq_len": 100, "feat_dim": 1, "latent_dim": 10}),
            (TimeNet, {"hidden_dims": 64, "num_layers": 1, "feat_dim": 1}),
        ],
        ids=[
            "FCN",
            "TST",
            "TSTCC",
            "TS2Vec",
            "AutoTCL",
            "CoST",
            "Series2Vec",
            "TimeVAE",
            "TimeNet",
        ],
    )
    def test_encoder_returns_nn_module(self, model_cls: type, kwargs: dict) -> None:
        """model.encoder must return an nn.Module instance."""
        model = model_cls(**kwargs)
        encoder = model.encoder
        assert isinstance(encoder, nn.Module), (
            f"{model_cls.__name__}.encoder returned {type(encoder).__name__}, expected nn.Module"
        )


class TestHasDecoderConformance:
    """Only TimeVAE and TimeNet satisfy HasDecoder."""

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (TimeVAE, {"seq_len": 100, "feat_dim": 1, "latent_dim": 10}),
            (TimeNet, {"hidden_dims": 64, "num_layers": 1, "feat_dim": 1}),
        ],
        ids=["TimeVAE", "TimeNet"],
    )
    def test_model_satisfies_has_decoder(self, model_cls: type, kwargs: dict) -> None:
        """Decoder models must satisfy the HasDecoder protocol."""
        model = model_cls(**kwargs)
        assert isinstance(model, HasDecoder), (
            f"{model_cls.__name__} does not satisfy HasDecoder protocol"
        )

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (TimeVAE, {"seq_len": 100, "feat_dim": 1, "latent_dim": 10}),
            (TimeNet, {"hidden_dims": 64, "num_layers": 1, "feat_dim": 1}),
        ],
        ids=["TimeVAE", "TimeNet"],
    )
    def test_decoder_returns_nn_module(self, model_cls: type, kwargs: dict) -> None:
        """model.decoder must return an nn.Module instance."""
        model = model_cls(**kwargs)
        decoder = model.decoder
        assert isinstance(decoder, nn.Module), (
            f"{model_cls.__name__}.decoder returned {type(decoder).__name__}, expected nn.Module"
        )


class TestEncoderOnlyModelsNoDecoder:
    """Encoder-only models must NOT satisfy HasDecoder."""

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (FCN, {"n_in": 1}),
            (TST, {"feat_dim": 1, "max_seq_len": 100}),
            (
                TSTCC,
                {
                    "input_channels": 1,
                    "kernel_size": 5,
                    "stride": 1,
                    "final_out_channels": 16,
                    "features_len": 12,
                    "num_classes": 10,
                },
            ),
            (TS2Vec, {"input_dims": 1}),
            (AutoTCL, {"input_dims": 1, "kernel_sizes": [3]}),
            (CoST, {"input_dims": 1, "sequence_length": 100, "kernel_sizes": [3]}),
            (
                Series2Vec,
                {
                    "input_dims": 1,
                    "embedding_dims": 64,
                    "num_heads": 2,
                    "feedforward_dims": 128,
                    "representation_dims": 64,
                    "dropout_rate": 0.1,
                },
            ),
        ],
        ids=["FCN", "TST", "TSTCC", "TS2Vec", "AutoTCL", "CoST", "Series2Vec"],
    )
    def test_encoder_only_model_not_has_decoder(self, model_cls: type, kwargs: dict) -> None:
        """Encoder-only models must NOT satisfy the HasDecoder protocol."""
        model = model_cls(**kwargs)
        assert not isinstance(model, HasDecoder), (
            f"{model_cls.__name__} unexpectedly satisfies HasDecoder (encoder-only model)"
        )
