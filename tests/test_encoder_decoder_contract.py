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
    RecurrentAutoEncoder,
    Series2Vec,
    TimeNet,
    TimeVAE,
    TS2Vec,
    TST,
    TSTCC,
)
from chronocratic.models.protocols import HasDecoder, HasEncoder

# ---------------------------------------------------------------------------
# Model construction specs (shared across parametrize calls)
# ---------------------------------------------------------------------------

ENCODER_MODEL_SPECS: list[tuple[type, dict, str]] = [
    (FCN, {"input_dims": 1}, "FCN"),
    (TST, {"input_dims": 1, "sequence_length": 100}, "TST"),
    (TSTCC, {"input_dims": 1, "conv_kernel_size": 5, "stride": 1, "output_dims": 16}, "TSTCC"),
    (TS2Vec, {"input_dims": 1}, "TS2Vec"),
    (AutoTCL, {"input_dims": 1, "kernel_sizes": (3,)}, "AutoTCL"),
    (CoST, {"input_dims": 1, "sequence_length": 100, "kernel_sizes": (3,)}, "CoST"),
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
        "Series2Vec",
    ),
    (TimeVAE, {"sequence_length": 100, "input_dims": 1, "latent_dim": 10}, "TimeVAE"),
    (TimeNet, {"hidden_dims": 64, "depth": 1, "input_dims": 1}, "TimeNet"),
    (RecurrentAutoEncoder, {"input_dims": 1, "layers": (64,)}, "RecurrentAutoEncoder"),
]

ENCODER_ONLY_MODEL_SPECS: list[tuple[type, dict, str]] = [
    (FCN, {"input_dims": 1}, "FCN"),
    (TST, {"input_dims": 1, "sequence_length": 100}, "TST"),
    (TSTCC, {"input_dims": 1, "conv_kernel_size": 5, "stride": 1, "output_dims": 16}, "TSTCC"),
    (TS2Vec, {"input_dims": 1}, "TS2Vec"),
    (AutoTCL, {"input_dims": 1, "kernel_sizes": (3,)}, "AutoTCL"),
    (CoST, {"input_dims": 1, "sequence_length": 100, "kernel_sizes": (3,)}, "CoST"),
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
        "Series2Vec",
    ),
]

DECODER_MODEL_SPECS: list[tuple[type, dict, str]] = [
    (TimeVAE, {"sequence_length": 100, "input_dims": 1, "latent_dim": 10}, "TimeVAE"),
    (TimeNet, {"hidden_dims": 64, "depth": 1, "input_dims": 1}, "TimeNet"),
    (RecurrentAutoEncoder, {"input_dims": 1, "layers": (64,)}, "RecurrentAutoEncoder"),
]


class TestHasEncoderConformance:
    """All 9 models must satisfy HasEncoder via isinstance."""

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "model_id"),
        ENCODER_MODEL_SPECS,
        ids=[spec[2] for spec in ENCODER_MODEL_SPECS],
    )
    def test_model_satisfies_has_encoder(
        self, model_cls: type, kwargs: dict, model_id: str
    ) -> None:
        """Each model instance must satisfy the HasEncoder protocol."""
        model = model_cls(**kwargs)
        assert isinstance(model, HasEncoder), (
            f"{model_cls.__name__} does not satisfy HasEncoder protocol"
        )

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "model_id"),
        ENCODER_MODEL_SPECS,
        ids=[spec[2] for spec in ENCODER_MODEL_SPECS],
    )
    def test_encoder_returns_nn_module(self, model_cls: type, kwargs: dict, model_id: str) -> None:
        """model.encoder must return an nn.Module instance."""
        model = model_cls(**kwargs)
        encoder = model.encoder
        assert isinstance(encoder, nn.Module), (
            f"{model_cls.__name__}.encoder returned {type(encoder).__name__}, expected nn.Module"
        )


class TestHasDecoderConformance:
    """Only TimeVAE and TimeNet satisfy HasDecoder."""

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "model_id"),
        DECODER_MODEL_SPECS,
        ids=[spec[2] for spec in DECODER_MODEL_SPECS],
    )
    def test_model_satisfies_has_decoder(
        self, model_cls: type, kwargs: dict, model_id: str
    ) -> None:
        """Decoder models must satisfy the HasDecoder protocol."""
        model = model_cls(**kwargs)
        assert isinstance(model, HasDecoder), (
            f"{model_cls.__name__} does not satisfy HasDecoder protocol"
        )

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "model_id"),
        DECODER_MODEL_SPECS,
        ids=[spec[2] for spec in DECODER_MODEL_SPECS],
    )
    def test_decoder_returns_nn_module(self, model_cls: type, kwargs: dict, model_id: str) -> None:
        """model.decoder must return an nn.Module instance."""
        model = model_cls(**kwargs)
        decoder = model.decoder
        assert isinstance(decoder, nn.Module), (
            f"{model_cls.__name__}.decoder returned {type(decoder).__name__}, expected nn.Module"
        )


class TestEncoderOnlyModelsNoDecoder:
    """Encoder-only models must NOT satisfy HasDecoder."""

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "model_id"),
        ENCODER_ONLY_MODEL_SPECS,
        ids=[spec[2] for spec in ENCODER_ONLY_MODEL_SPECS],
    )
    def test_encoder_only_model_not_has_decoder(
        self, model_cls: type, kwargs: dict, model_id: str
    ) -> None:
        """Encoder-only models must NOT satisfy the HasDecoder protocol."""
        model = model_cls(**kwargs)
        assert not isinstance(model, HasDecoder), (
            f"{model_cls.__name__} unexpectedly satisfies HasDecoder (encoder-only model)"
        )
