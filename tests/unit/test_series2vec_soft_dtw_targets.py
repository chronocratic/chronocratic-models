"""Series2Vec-level tests for the O(T)-memory soft-DTW target wiring.

Covers: target parity against the old-style oracle call, a real training
step with gradient flow, bandwidth constructor validation/wiring, and the
B=2 silent-zero-loss regression (§3.2/§6.6 of the memory-fix spec).
"""

import os
import subprocess
import sys

import pytest
import torch

from chronocratic.models.convolutional.standard.series2vec.losses import pairwise_soft_dtw_distances
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.utils.distances.soft_dtw.soft_dtw_cuda import SoftDTW

_MEMORY_TEST_SCRIPT = """
import resource, sys
import torch
from chronocratic.models.utils.distances.soft_dtw import pairwise_soft_dtw_values

x = torch.randn(8, 7500, 1)
pairwise_soft_dtw_values(x, gamma=0.1)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_gb = rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9
print(peak_gb)
"""


def _make_series2vec(**overrides: object) -> Series2Vec:
    kwargs: dict = {
        "input_dim": 3,
        "embedding_dim": 8,
        "representation_dim": 16,
        "temporal_kernel_size": 4,
        "num_heads": 2,
        "feedforward_dim": 32,
    }
    kwargs.update(overrides)
    return Series2Vec(**kwargs)


class TestTargetsMatchOracle:
    def test_targets_equal_old_oracle_targets(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 40, 2)
        actual = pairwise_soft_dtw_distances(x, gamma=0.1)

        rows, cols = torch.tril_indices(6, 6, offset=-1)
        oracle = SoftDTW(use_cuda=False, gamma=0.1)
        expected = oracle(x[rows], x[cols])
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


class TestTrainingStepGradient:
    def test_training_step_has_finite_loss_and_gradient(self) -> None:
        model = _make_series2vec()
        model.train()
        x = torch.randn(4, 64, 3)
        loss = model.training_step(x, 0)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any(torch.any(g != 0) for g in grads)


class TestBandwidthWiring:
    def test_bandwidth_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="soft_dtw_bandwidth"):
            _make_series2vec(soft_dtw_bandwidth=0)

    def test_bandwidth_set_trains_one_step(self) -> None:
        model = _make_series2vec(soft_dtw_bandwidth=4)
        model.train()
        x = torch.randn(4, 64, 3)
        loss = model.training_step(x, 0)
        assert torch.isfinite(loss)


class TestBatchTwoLearns:
    def test_batch_2_gives_nonzero_loss_and_gradient(self) -> None:
        """B=2 must not silently collapse to loss==0.0 (§3.2: min-max normalizer
        collapses a single pair to a constant, killing the gradient)."""
        model = _make_series2vec()
        model.train()
        x = torch.randn(2, 64, 3)
        loss = model.training_step(x, 0)
        assert loss.item() > 0.0
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any(torch.any(g != 0) for g in grads)


class TestMemoryRegression:
    @pytest.mark.skipif(
        os.environ.get("CHRONOCRATIC_MEMORY_TESTS") != "1", reason="opt-in memory test"
    )
    def test_soft_dtw_values_peak_rss_under_2gb(self) -> None:
        """(8, 7500, 1) targets stay under 2GB peak RSS; the old full-table code needed ~30GB.

        Runs in a subprocess: RSS is a per-process high-water mark that other tests would
        pollute if measured in-process.
        """
        result = subprocess.run(  # noqa: S603 — fixed literal command, no untrusted input
            [sys.executable, "-c", _MEMORY_TEST_SCRIPT], capture_output=True, text=True, check=True
        )
        peak_gb = float(result.stdout.strip().splitlines()[-1])
        assert peak_gb < 2.0, f"peak RSS {peak_gb:.2f} GB exceeds 2 GB budget"
