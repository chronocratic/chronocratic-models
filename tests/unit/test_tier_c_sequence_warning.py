"""Tests for Tier C models (Series2Vec, MCL, TimeVAE) output-aware encoding.

Verifies:
- VECTOR returns native flat (N, D) — no phantom temporal axis.
- SEQUENCE returns fake (N, 1, D) with once-per-class warning.
- supported_outputs declares {VECTOR} for Tier C models.
- Warning dedup fires once per class.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from chronocratic.models import EncodingOutputShape
from chronocratic.models.convolutional.standard.mcl.model import MCL
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.generative.timevae.model import TimeVAE
from chronocratic.models.utils.helpers import _warned_sequence_fallback


@pytest.fixture
def series2vec_model() -> Series2Vec:
    return Series2Vec(representation_dims=32, input_dims=3)


@pytest.fixture
def mcl_model() -> MCL:
    return MCL(input_dims=3, output_dims=64)


@pytest.fixture
def timevae_model() -> TimeVAE:
    return TimeVAE(sequence_length=32, input_dims=3, latent_dim=8)


@pytest.fixture
def batch_input() -> torch.Tensor:
    """Small batch for encoding tests."""
    return torch.randn(2, 32, 3)


class TestSeries2VecVectorOutput:
    """Series2Vec VECTOR returns (B, 2*representation_dims)."""

    def test_vector_shape(self, series2vec_model: Series2Vec, batch_input: torch.Tensor) -> None:
        encoder = series2vec_model._get_encoder()
        result = series2vec_model._encode_batch(
            encoder, batch_input, output=EncodingOutputShape.VECTOR
        )
        expected_dim = series2vec_model.representation_dim  # 2 * representation_dims
        assert result.shape == (2, expected_dim)

    def test_vector_is_2d(self, series2vec_model: Series2Vec, batch_input: torch.Tensor) -> None:
        encoder = series2vec_model._get_encoder()
        result = series2vec_model._encode_batch(
            encoder, batch_input, output=EncodingOutputShape.VECTOR
        )
        assert result.ndim == 2


class TestSeries2VecSequenceOutput:
    """Series2Vec SEQUENCE returns (B, 1, 2*representation_dims) with warning."""

    def test_sequence_shape(self, series2vec_model: Series2Vec, batch_input: torch.Tensor) -> None:
        encoder = series2vec_model._get_encoder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = series2vec_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            expected_dim = series2vec_model.representation_dim  # 2 * representation_dims
            assert result.shape == (2, 1, expected_dim)
            assert len(w) == 1
            assert "SEQUENCE" in str(w[0].message)

    def test_sequence_is_3d(self, series2vec_model: Series2Vec, batch_input: torch.Tensor) -> None:
        encoder = series2vec_model._get_encoder()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = series2vec_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert result.ndim == 3


class TestSeries2VecSupportedOutputs:
    """Series2Vec.supported_outputs declares {VECTOR}."""

    def test_supported_outputs_contains_vector(self, series2vec_model: Series2Vec) -> None:
        assert EncodingOutputShape.VECTOR in series2vec_model.supported_outputs

    def test_supported_outputs_is_frozenset(self, series2vec_model: Series2Vec) -> None:
        assert isinstance(series2vec_model.supported_outputs, frozenset)


class TestMCLVectorOutput:
    """MCL VECTOR returns (B, output_dims)."""

    def test_vector_shape(self, mcl_model: MCL, batch_input: torch.Tensor) -> None:
        encoder = mcl_model._get_encoder()
        result = mcl_model._encode_batch(
            encoder, batch_input, output=EncodingOutputShape.VECTOR
        )
        assert result.shape == (2, 64)

    def test_vector_is_2d(self, mcl_model: MCL, batch_input: torch.Tensor) -> None:
        encoder = mcl_model._get_encoder()
        result = mcl_model._encode_batch(
            encoder, batch_input, output=EncodingOutputShape.VECTOR
        )
        assert result.ndim == 2


class TestMCLSequenceOutput:
    """MCL SEQUENCE returns (B, 1, output_dims) with warning."""

    def test_sequence_shape(self, mcl_model: MCL, batch_input: torch.Tensor) -> None:
        encoder = mcl_model._get_encoder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mcl_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert result.shape == (2, 1, 64)
            assert len(w) == 1
            assert "SEQUENCE" in str(w[0].message)

    def test_sequence_is_3d(self, mcl_model: MCL, batch_input: torch.Tensor) -> None:
        encoder = mcl_model._get_encoder()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mcl_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert result.ndim == 3


class TestMCLSupportedOutputs:
    """MCL.supported_outputs declares {VECTOR}."""

    def test_supported_outputs_contains_vector(self, mcl_model: MCL) -> None:
        assert EncodingOutputShape.VECTOR in mcl_model.supported_outputs

    def test_supported_outputs_is_frozenset(self, mcl_model: MCL) -> None:
        assert isinstance(mcl_model.supported_outputs, frozenset)


class TestTimeVAEVectorOutput:
    """TimeVAE VECTOR returns (B, latent_dim)."""

    def test_vector_shape(self, timevae_model: TimeVAE, batch_input: torch.Tensor) -> None:
        encoder = timevae_model._get_encoder()
        result = timevae_model._encode_batch(
            encoder, batch_input, output=EncodingOutputShape.VECTOR
        )
        assert result.shape == (2, 8)

    def test_vector_is_2d(self, timevae_model: TimeVAE, batch_input: torch.Tensor) -> None:
        encoder = timevae_model._get_encoder()
        result = timevae_model._encode_batch(
            encoder, batch_input, output=EncodingOutputShape.VECTOR
        )
        assert result.ndim == 2


class TestTimeVAESequenceOutput:
    """TimeVAE SEQUENCE returns (B, 1, latent_dim) with warning."""

    def test_sequence_shape(self, timevae_model: TimeVAE, batch_input: torch.Tensor) -> None:
        encoder = timevae_model._get_encoder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = timevae_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert result.shape == (2, 1, 8)
            assert len(w) == 1
            assert "SEQUENCE" in str(w[0].message)

    def test_sequence_is_3d(self, timevae_model: TimeVAE, batch_input: torch.Tensor) -> None:
        encoder = timevae_model._get_encoder()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = timevae_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert result.ndim == 3


class TestTimeVAESupportedOutputs:
    """TimeVAE.supported_outputs declares {VECTOR}."""

    def test_supported_outputs_contains_vector(self, timevae_model: TimeVAE) -> None:
        assert EncodingOutputShape.VECTOR in timevae_model.supported_outputs

    def test_supported_outputs_is_frozenset(self, timevae_model: TimeVAE) -> None:
        assert isinstance(timevae_model.supported_outputs, frozenset)


class TestWarningDedup:
    """Warning fires once per class."""

    def setup_method(self) -> None:
        """Clear the warning tracker so dedup tests are isolated."""
        _warned_sequence_fallback.clear()

    def test_series2vec_warns_once(self, series2vec_model: Series2Vec, batch_input: torch.Tensor) -> None:
        encoder = series2vec_model._get_encoder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # First call: warning expected
            series2vec_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            # Second call: no additional warning (dedup)
            series2vec_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert len(w) == 1

    def test_mcl_warns_once(self, mcl_model: MCL, batch_input: torch.Tensor) -> None:
        encoder = mcl_model._get_encoder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mcl_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            mcl_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert len(w) == 1

    def test_timevae_warns_once(self, timevae_model: TimeVAE, batch_input: torch.Tensor) -> None:
        encoder = timevae_model._get_encoder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            timevae_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            timevae_model._encode_batch(
                encoder, batch_input, output=EncodingOutputShape.SEQUENCE
            )
            assert len(w) == 1
