"""Tests for DecompositionEncodingMixin SEQUENCE output support.

Verifies that CoST's _evaluate_with_feature_concatenation handles both
VECTOR (last-step concat) and SEQUENCE (full-sequence concat) output shapes.

TDD for plan 10-04, Task 2.
"""

from __future__ import annotations

import torch

from chronocratic.models.convolutional.dilated._mixin.encoding import DecompositionEncodingMixin


class _TestableDecompositionModel(DecompositionEncodingMixin, torch.nn.Module):
    """Concrete decomposition model for testing SEQUENCE concat behavior."""

    def __init__(self, feature_dim: int = 4) -> None:
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.feature_dim = feature_dim
        # DecompositionEncodingMixin._get_encoder() returns self.query_encoder
        self.query_encoder = _DecompositionEncoder(feature_dim)

    def _get_encoder(self):
        """Override to return decomposition encoder callable."""
        return self.query_encoder


class _DecompositionEncoder(torch.nn.Module):
    """Fake decomposition encoder that produces (trend, seasonality) pairs."""

    def __init__(self, feature_dim: int = 4) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, mask_mode=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce (trend, seasonality) pairs of shape (B, L, D)."""
        batch, seq = x.shape[:2]
        trend = torch.ones(batch, seq, self.feature_dim, device=x.device)
        seasonality = torch.ones(batch, seq, self.feature_dim, device=x.device) * 2
        return trend, seasonality


class TestCostSequenceConcat:
    """DecompositionEncodingMixin handles SEQUENCE (full-sequence concat)."""

    def test_full_series_returns_squeezed_vector(self) -> None:
        """encoding_window='full_series' (VECTOR) returns (B, 2D) after squeeze."""
        model = _TestableDecompositionModel(feature_dim=4)
        batch_x = torch.randn(2, 10, 4)
        result = model.encode_batch(batch_x)  # default output=VECTOR -> full_series
        # After squeeze: (B, 2D)
        assert result.shape == (2, 8), f"Expected (2, 8), got {result.shape}"

    def test_sequence_returns_full_seq_concat(self) -> None:
        """encoding_window=None (SEQUENCE) returns (B, L, 2D) full-sequence concat."""
        from chronocratic.models.enums.encoding import EncodingOutputShape

        model = _TestableDecompositionModel(feature_dim=4)
        batch_x = torch.randn(2, 10, 4)
        result = model.encode_batch(batch_x, output=EncodingOutputShape.SEQUENCE)
        # Full sequence concat: (B, L, 2D)
        assert result.shape == (2, 10, 8), f"Expected (2, 10, 8), got {result.shape}"

    def test_full_series_is_last_step(self) -> None:
        """VECTOR output uses last-step values (not mean-pooled)."""
        model = _TestableDecompositionModel(feature_dim=2)
        batch_x = torch.randn(1, 10, 2)
        result = model.encode_batch(batch_x)
        # Result should be (1, 4) — squeezed last-step concat (2 trend + 2 seasonality)
        assert result.shape == (1, 4), f"Expected (1, 4), got {result.shape}"

    def test_sequence_concatenates_trend_and_seasonality(self) -> None:
        """SEQUENCE concat produces correct feature dimensions."""
        from chronocratic.models.enums.encoding import EncodingOutputShape

        model = _TestableDecompositionModel(feature_dim=5)
        batch_x = torch.randn(3, 8, 5)
        result = model.encode_batch(batch_x, output=EncodingOutputShape.SEQUENCE)
        # 3 samples, 8 timesteps, 10 features (5 trend + 5 seasonality)
        assert result.shape == (3, 8, 10), f"Expected (3, 8, 10), got {result.shape}"

    def test_encode_sequence_via_encode_method(self) -> None:
        """encode() with output=SEQUENCE returns (N, L, 2D)."""
        from chronocratic.models.enums.encoding import EncodingOutputShape

        model = _TestableDecompositionModel(feature_dim=4)
        data = torch.randn(4, 10, 4)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3
        assert result.shape == (4, 10, 8), f"Expected (4, 10, 8), got {result.shape}"
