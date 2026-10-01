"""Pure-function tests for ensure_pairable_batch."""

import pytest
import torch

from chronocratic.models.utils import ensure_pairable_batch


class TestEnsurePairableBatch:
    @pytest.fixture
    def x_singleton(self) -> torch.Tensor:
        """Singleton batch (1, 300, 3)."""
        return torch.randn(1, 300, 3)

    @pytest.fixture
    def x_multi(self) -> torch.Tensor:
        """Multi-sample batch (4, 100, 3)."""
        return torch.randn(4, 100, 3)

    def test_batch_gt_1_returns_same_object(self, x_multi: torch.Tensor) -> None:
        """B > 1 is returned unchanged — identity, not just equality."""
        out = ensure_pairable_batch(x_multi)
        assert out is x_multi

    def test_splits_singleton_into_correct_shape(self, x_singleton: torch.Tensor) -> None:
        """(1, 300, 3) with split_count=3 returns (3, 100, 3)."""
        out = ensure_pairable_batch(x_singleton, split_count=3)
        assert out.shape == (3, 100, 3)

    def test_content_preserved_in_time_order(self, x_singleton: torch.Tensor) -> None:
        """Windows preserve original time-ordered content."""
        out = ensure_pairable_batch(x_singleton, split_count=3)
        assert torch.equal(out[0], x_singleton[0, :100])
        assert torch.equal(out[1], x_singleton[0, 100:200])
        assert torch.equal(out[2], x_singleton[0, 200:300])

    def test_non_divisible_length_truncates(self) -> None:
        """(1, 100, 3) with split_count=3 returns (3, 33, 3); last timestep dropped."""
        x = torch.arange(300.0).reshape(1, 100, 3)
        out = ensure_pairable_batch(x, split_count=3)
        assert out.shape == (3, 33, 3)
        assert torch.equal(out[2, -1], x[0, 98])

    def test_min_window_len_blocks_split(self) -> None:
        """(1, 10, 3) with split_count=3, min_window_len=5 returns x unchanged (3 < 5)."""
        x = torch.randn(1, 10, 3)
        out = ensure_pairable_batch(x, split_count=3, min_window_len=5)
        assert out is x

    def test_split_count_2_works(self) -> None:
        """split_count=2 (boundary of useful range) produces (2, 150, 3)."""
        x = torch.randn(1, 300, 3)
        out = ensure_pairable_batch(x, split_count=2)
        assert out.shape == (2, 150, 3)
        assert torch.equal(out[0], x[0, :150])
        assert torch.equal(out[1], x[0, 150:300])


class TestMinBatchSize:
    def test_batch_1_default_args_unchanged_behavior(self) -> None:
        """B=1, default min_batch_size=2: identical to the pre-existing behavior."""
        x = torch.randn(1, 300, 3)
        out = ensure_pairable_batch(x, split_count=3)
        assert out.shape == (3, 100, 3)

    def test_batch_2_default_args_unchanged(self) -> None:
        """B=2, default min_batch_size=2: returned unchanged (SimCLR/TS-TCC regression guard)."""
        x = torch.randn(2, 300, 3)
        out = ensure_pairable_batch(x, split_count=3)
        assert out is x

    def test_batch_2_min_batch_size_3_splits(self) -> None:
        """B=2, min_batch_size=3, split_count=3, L=30 -> shape (6, 10, C)."""
        x = torch.randn(2, 30, 3)
        out = ensure_pairable_batch(x, split_count=3, min_batch_size=3)
        assert out.shape == (6, 10, 3)
        assert torch.equal(out[0], x[0, 0:10])
        assert torch.equal(out[3], x[1, 0:10])

    def test_batch_3_min_batch_size_3_unchanged(self) -> None:
        """B=3, min_batch_size=3: already meets the requirement, returned unchanged."""
        x = torch.randn(3, 30, 3)
        out = ensure_pairable_batch(x, split_count=3, min_batch_size=3)
        assert out is x

    def test_window_too_short_with_min_batch_size_unchanged(self) -> None:
        """B=2, min_batch_size=3, window below min_window_len: returned unchanged."""
        x = torch.randn(2, 10, 3)
        out = ensure_pairable_batch(x, split_count=3, min_batch_size=3, min_window_len=5)
        assert out is x
