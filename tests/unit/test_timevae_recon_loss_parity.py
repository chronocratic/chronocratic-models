# ruff: noqa: D, PLR2004, S101
"""Regression: TimeVAE _get_reconstruction_loss uses SUM semantics.

Ensures that the per-element and per-axis terms operate on the same
scale (both sums), matching the original TensorFlow implementation.
See Finding 1 in the TimeVAE audit (masked_reconstruction_loss mean
vs sum divergence).
"""

import torch

from chronocratic.models.generative.timevae.vae_base import BaseVariationalAutoencoder


class _DummyVAE(BaseVariationalAutoencoder):
    """Minimal subclass so that _get_reconstruction_loss can be called."""

    def __init__(self, *, sequence_length: int, input_dim: int) -> None:
        super().__init__(
            sequence_length=sequence_length,
            input_dim=input_dim,
            latent_dim=4,
        )
        self._encoder = torch.nn.Identity()
        self._decoder = torch.nn.Identity()

    def _build_encoder(self) -> torch.nn.Module:
        return self._encoder

    def _build_decoder(self) -> torch.nn.Module:
        return self._decoder


class TestTimeVAEReconLossSumSemantics:
    """Verify _get_reconstruction_loss matches TF reduce_sum semantics."""

    def test_all_valid_equals_sum(self) -> None:
        """With no padding, loss equals element-wise sum + axis sum."""
        vae = _DummyVAE(sequence_length=10, input_dim=3)
        x = torch.randn(2, 10, 3)
        x_recons = x * 0.9  # small errors
        err = (x - x_recons) ** 2

        # Manually compute expected SUM
        expected_elem = err.sum()
        x_r = x.mean(dim=2)
        x_c_r = x_recons.mean(dim=2)
        expected_axis = (x_r - x_c_r) ** 2
        expected_axis_sum = expected_axis.sum()
        expected_total = expected_elem + expected_axis_sum

        # Call with all-True mask
        keep_mask = torch.ones(2, 10, dtype=torch.bool)
        actual = vae._get_reconstruction_loss(x, x_recons, keep_mask=keep_mask)
        torch.testing.assert_close(actual, expected_total)

    def test_masked_excludes_padding(self) -> None:
        """Masked-out timesteps contribute 0 to per-element sum."""
        vae = _DummyVAE(sequence_length=10, input_dim=3)
        x = torch.ones(1, 10, 3)
        x_recons = torch.zeros(1, 10, 3)  # err = 1.0 everywhere
        keep_mask = torch.ones(1, 10, dtype=torch.bool)
        keep_mask[:, 5:] = False  # mask last half

        loss = vae._get_reconstruction_loss(x, x_recons, keep_mask=keep_mask)
        # Per-element: 5 valid × 3 channels × 1.0 = 15.0
        # Per-axis: mean over C=3 → still 1.0 vs 0.0 for all 10 steps
        # axis_err = (1.0 - 0.0)^2 = 1.0 per step, summed over 1×10 = 10.0
        # Total = 15.0 + 10.0 = 25.0
        torch.testing.assert_close(loss, torch.tensor(25.0))

    def test_multivariate_same_scale(self) -> None:
        """Per-element and per-axis terms scale consistently across C values."""
        vae = _DummyVAE(sequence_length=10, input_dim=5)
        x = torch.ones(2, 10, 5)
        x_recons = x * 0.8  # err = 0.04 everywhere
        keep_mask = torch.ones(2, 10, dtype=torch.bool)

        loss = vae._get_reconstruction_loss(x, x_recons, keep_mask=keep_mask)
        # Per-element sum: 2×10×5 × 0.04 = 4.0
        # Per-axis: mean over C=5 → 1.0 vs 0.8 → err = 0.04, sum over 2×10 = 0.8
        # Total = 4.0 + 0.8 = 4.8
        torch.testing.assert_close(loss, torch.tensor(4.8))

    def test_no_mask_same_as_masked(self) -> None:
        """keep_mask=None path (sum) matches keep_mask=all-True path."""
        vae = _DummyVAE(sequence_length=10, input_dim=3)
        x = torch.randn(2, 10, 3)
        x_recons = torch.randn(2, 10, 3)

        loss_none = vae._get_reconstruction_loss(x, x_recons, keep_mask=None)
        loss_mask = vae._get_reconstruction_loss(
            x, x_recons, keep_mask=torch.ones(2, 10, dtype=torch.bool)
        )
        torch.testing.assert_close(loss_none, loss_mask)
