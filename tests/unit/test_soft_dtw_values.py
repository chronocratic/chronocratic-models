"""Parity tests for the O(T)-memory pairwise soft-DTW value routine.

``SoftDTW(use_cuda=False, ...)`` (the vendored, full-table implementation) is
the test oracle throughout: ``pairwise_soft_dtw_values`` must match it to
floating-point tolerance while using O(T) memory per pair instead of O(T^2).
"""

import pytest
import torch

from chronocratic.models.utils.distances.soft_dtw import pairwise_soft_dtw_values
from chronocratic.models.utils.distances.soft_dtw.soft_dtw_cuda import SoftDTW


def _oracle(x: torch.Tensor, *, gamma: float, bandwidth: float | None) -> torch.Tensor:
    rows, cols = torch.tril_indices(x.size(0), x.size(0), offset=-1)
    dtw = SoftDTW(use_cuda=False, gamma=gamma, bandwidth=bandwidth)
    return dtw(x[rows], x[cols])


class TestParity:
    @pytest.mark.parametrize("gamma", [0.1, 1.0])
    def test_parity_no_pruning(self, gamma: float) -> None:
        torch.manual_seed(0)
        x = torch.randn(5, 37, 3)
        actual = pairwise_soft_dtw_values(x, gamma=gamma)
        expected = _oracle(x, gamma=gamma, bandwidth=None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_parity_with_bandwidth(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(5, 37, 3)
        actual = pairwise_soft_dtw_values(x, gamma=0.1, bandwidth=5)
        expected = _oracle(x, gamma=0.1, bandwidth=5)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_parity_univariate(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(4, 25, 1)
        actual = pairwise_soft_dtw_values(x, gamma=0.1)
        expected = _oracle(x, gamma=0.1, bandwidth=None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


class TestPairOrder:
    def test_matches_tril_indices_order(self) -> None:
        x = torch.arange(4.0).view(4, 1, 1).expand(4, 10, 2).contiguous()
        actual = pairwise_soft_dtw_values(x, gamma=0.1)
        expected = _oracle(x, gamma=0.1, bandwidth=None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


class TestDegenerateBatch:
    def test_empty_for_batch_lt_2(self) -> None:
        x = torch.randn(1, 10, 3)
        out = pairwise_soft_dtw_values(x, gamma=0.1)
        assert out.shape == (0,)


class TestDtypeDeviceGrad:
    def test_dtype_preserved(self) -> None:
        x = torch.randn(3, 10, 2, dtype=torch.float32)
        out = pairwise_soft_dtw_values(x, gamma=0.1)
        assert out.dtype == torch.float32

    def test_no_gradient(self) -> None:
        x = torch.randn(3, 10, 2, requires_grad=True)
        out = pairwise_soft_dtw_values(x, gamma=0.1)
        assert out.requires_grad is False


class TestValidation:
    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_invalid_gamma_raises(self, gamma: float) -> None:
        x = torch.randn(3, 10, 2)
        with pytest.raises(ValueError, match="gamma"):
            pairwise_soft_dtw_values(x, gamma=gamma)

    @pytest.mark.parametrize("bandwidth", [0.0, -3.0])
    def test_invalid_bandwidth_raises(self, bandwidth: float) -> None:
        x = torch.randn(3, 10, 2)
        with pytest.raises(ValueError, match="bandwidth"):
            pairwise_soft_dtw_values(x, gamma=0.1, bandwidth=bandwidth)
