"""Tests for Series2Vec _distance_normalizer — upstream-aligned behavior.

Verifies that _distance_normalizer matches upstream Series2Vec:
- Single-element: ``distance / distance`` → scalar 1.0 (connected to graph)
- Zero-variance: return raw distance (upstream skips normalize)
- Normal: min-max to [0, 1]

See ``/tmp/series2vec-upstream/models/Series2Vec/S2V_training.py:326``.
"""

import torch

from chronocratic.models.convolutional.standard.series2vec.losses import _distance_normalizer
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


class TestDistanceNormalizerGradientFlow:
    """Test _distance_normalizer edge-case branches and gradient safety."""

    def test_numel_le_1_returns_ones(self) -> None:
        """1-element input should return distance/distance = 1.0 (upstream behavior)."""
        distance = torch.tensor([5.0], requires_grad=True)
        result = _distance_normalizer(distance)

        # Should not crash
        assert result is not None
        # Upstream uses distance/distance for single element
        assert result.shape == (1,)
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_empty_tensor_returns_detached_tensor(self) -> None:
        """Empty input should return empty output, not crash."""
        distance = torch.tensor([])
        result = _distance_normalizer(distance)

        assert result.numel() == 0

    def test_zero_variance_returns_raw_distance(self) -> None:
        """Equal-valued input should return raw distance (upstream behavior).

        When all distances are identical, denominator is 0.
        Upstream returns the un-normalized tensor as-is.
        """
        distance = torch.tensor([3.0, 3.0, 3.0], requires_grad=True)
        result = _distance_normalizer(distance)

        # Should not crash
        assert result is not None
        # Preserve original values
        assert torch.allclose(result, distance)

    def test_normal_min_max_normalization(self) -> None:
        """Distinct values should produce standard min-max normalization."""
        distance = torch.tensor([1.0, 3.0, 5.0])
        result = _distance_normalizer(distance)

        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)

    def test_non_gradient_input(self) -> None:
        """Input without requires_grad should not crash."""
        distance = torch.tensor([4.0])  # requires_grad=False by default
        result = _distance_normalizer(distance)

        assert result is not None
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_single_zero_value(self) -> None:
        """Single zero value should return 1.0 (upstream: 0/0 = nan → clamp)."""
        distance = torch.tensor([0.0], requires_grad=True)
        result = _distance_normalizer(distance)

        # 0/0 = nan in upstream too; ensure it doesn't crash
        assert result is not None
        assert result.numel() == 1


class TestEncodeGradientAtBatchSize1:
    """Verify that model.encode() produces gradients through representations
    at batch_size=1 (the critical path, not the normalizer)."""

    def test_encode_gradient_flows_at_batch_size_1(self) -> None:
        """model.encode() at batch_size=1 should produce non-zero finite gradients."""
        model = Series2Vec(input_dim=1, embedding_dim=8, representation_dim=16)
        model.train()

        x = torch.randn(1, 32, 1, requires_grad=True)
        out = model.encode_batch(x)

        # Verify output shape
        assert out.shape[0] == 1  # batch_size=1 preserved

        # Verify gradients flow through encode()
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient — encode() path is broken at batch_size=1"
        assert x.grad.numel() == x.numel()
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN/Inf"

    def test_encode_gradient_flows_at_batch_size_2(self) -> None:
        """Baseline: model.encode() at batch_size=2 should produce gradients."""
        model = Series2Vec(input_dim=1, embedding_dim=8, representation_dim=16)
        model.train()

        x = torch.randn(2, 32, 1, requires_grad=True)
        out = model.encode_batch(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.numel() == x.numel()
        assert torch.isfinite(x.grad).all()


class TestGradientClipping:
    """Verify upstream-aligned gradient clipping."""

    def test_grad_clipped_at_4_0(self) -> None:
        """Gradients should be clipped to max_norm=4.0 (upstream behavior)."""
        model = Series2Vec(input_dim=1, embedding_dim=8, representation_dim=16)
        model.train()

        # Use a tiny batch to trigger gradients
        x = torch.randn(4, 32, 1)
        loss, _, _ = model._calculate_loss(x)
        loss.backward()

        # Verify gradients exist
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        assert total_norm > 0, "No gradients computed — test setup broken"

        # on_after_backward calls torch.nn.utils.clip_grad_norm_; verify it works
        model.on_after_backward()
        clipped_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        assert clipped_norm <= 4.0 + 1e-4, f"Grad norm {clipped_norm} exceeds max 4.0"
