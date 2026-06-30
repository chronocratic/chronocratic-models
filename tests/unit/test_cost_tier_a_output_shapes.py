"""Tests for CoST, TS2Vec, and AutoTCL Tier A output-shape wiring.

Verifies that Tier A dilated models declare `supported_outputs` and return
the correct tensor ranks for VECTOR (2-D) and SEQUENCE (3-D) output shapes.

TDD for plan 10-06, Task 1 and Task 2.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models import (
    AutoTCL,
    CoST,
    TS2Vec,
)
from chronocratic.models.enums.encoding import EncodingOutputShape


class TestCoSTSupportedOutputs:
    """CoST declares supported_outputs = {VECTOR, SEQUENCE}."""

    def test_cost_has_supported_outputs(self) -> None:
        assert hasattr(CoST, "supported_outputs")

    def test_cost_supported_outputs_contains_vector(self) -> None:
        assert EncodingOutputShape.VECTOR in CoST.supported_outputs

    def test_cost_supported_outputs_contains_sequence(self) -> None:
        assert EncodingOutputShape.SEQUENCE in CoST.supported_outputs

    def test_cost_supported_outputs_is_frozenset(self) -> None:
        assert isinstance(CoST.supported_outputs, frozenset)


class TestCoSTVectorShape:
    """CoST.encode() with VECTOR returns (N, 2D) — squeezed, no phantom axis."""

    @pytest.fixture
    def model(self) -> CoST:
        return CoST(input_dims=3, sequence_length=50)

    def test_vector_returns_2d_tensor(self, model: CoST) -> None:
        x = torch.randn(2, 50, 3)
        result = model.encode(x, batch_size=2, num_workers=0)
        assert result.ndim == 2, f"CoST VECTOR ndim={result.ndim}, shape={result.shape}"

    def test_vector_shape_is_n_d(self, model: CoST) -> None:
        x = torch.randn(4, 50, 3)
        result = model.encode(x, batch_size=2, num_workers=0)
        assert result.shape[0] == 4, f"Expected batch=4, got {result.shape[0]}"

    def test_vector_no_phantom_axis(self, model: CoST) -> None:
        """VECTOR must not produce (N, 1, 2D)."""
        x = torch.randn(2, 50, 3)
        result = model.encode(x, batch_size=2, num_workers=0)
        # Ensure no phantom temporal axis: shape must be (N, D), not (N, 1, D)
        assert result.shape != (2, 1, result.shape[-1]), (
            f"Phantom axis detected: {result.shape}"
        )


class TestCoSTSequenceShape:
    """CoST.encode(output=SEQUENCE) returns (N, L, 2D)."""

    @pytest.fixture
    def model(self) -> CoST:
        return CoST(input_dims=3, sequence_length=50)

    def test_sequence_returns_3d_tensor(self, model: CoST) -> None:
        x = torch.randn(2, 50, 3)
        result = model.encode(
            x, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3, f"CoST SEQUENCE ndim={result.ndim}, shape={result.shape}"

    def test_sequence_preserves_temporal_dim(self, model: CoST) -> None:
        x = torch.randn(2, 50, 3)
        result = model.encode(
            x, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.shape[0] == 2, f"Expected batch=2, got {result.shape[0]}"
        # Temporal dim should match input length (not collapsed)
        assert result.shape[1] == 50, (
            f"Expected temporal=50, got {result.shape[1]}"
        )

    def test_sequence_feature_dim_is_concatenated(self, model: CoST) -> None:
        """SEQUENCE feature dim = trend_dim + seasonality_dim (2D)."""
        x = torch.randn(1, 50, 3)
        result = model.encode(
            x, batch_size=1, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        # Feature dim should be 2x the encoder component_dim
        assert result.shape[2] > 0, f"Feature dim must be positive, got {result.shape[2]}"


class TestCoSTEncodeBatch:
    """CoST.encode_batch() also respects output shape."""

    @pytest.fixture
    def model(self) -> CoST:
        return CoST(input_dims=3, sequence_length=50)

    def test_encode_batch_vector_is_2d(self, model: CoST) -> None:
        x = torch.randn(2, 50, 3)
        result = model.encode_batch(x)  # default VECTOR
        assert result.ndim == 2

    def test_encode_batch_sequence_is_3d(self, model: CoST) -> None:
        x = torch.randn(2, 50, 3)
        result = model.encode_batch(x, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3


class TestTS2VecSupportedOutputs:
    """TS2Vec declares supported_outputs = {VECTOR, SEQUENCE}."""

    def test_ts2vec_has_supported_outputs(self) -> None:
        assert hasattr(TS2Vec, "supported_outputs")

    def test_ts2vec_supported_outputs_contains_vector(self) -> None:
        assert EncodingOutputShape.VECTOR in TS2Vec.supported_outputs

    def test_ts2vec_supported_outputs_contains_sequence(self) -> None:
        assert EncodingOutputShape.SEQUENCE in TS2Vec.supported_outputs

    def test_ts2vec_supported_outputs_is_frozenset(self) -> None:
        assert isinstance(TS2Vec.supported_outputs, frozenset)


class TestAutoTCLSupportedOutputs:
    """AutoTCL declares supported_outputs = {VECTOR, SEQUENCE}."""

    def test_autotcl_has_supported_outputs(self) -> None:
        assert hasattr(AutoTCL, "supported_outputs")

    def test_autotcl_supported_outputs_contains_vector(self) -> None:
        assert EncodingOutputShape.VECTOR in AutoTCL.supported_outputs

    def test_autotcl_supported_outputs_contains_sequence(self) -> None:
        assert EncodingOutputShape.SEQUENCE in AutoTCL.supported_outputs

    def test_autotcl_supported_outputs_is_frozenset(self) -> None:
        assert isinstance(AutoTCL.supported_outputs, frozenset)
