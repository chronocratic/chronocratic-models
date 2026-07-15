"""TimeVAE short-sequence encoder tests.

Verifies:
1. _timevae_encoder_output_length correctly computes conv^N(T).
2. TimeVAE auto-clamps conv_stride when encoder output < 2.
3. Forward pass succeeds on short sequences (T=8).
4. Gradients flow on short sequences.
5. Clear error for degenerate sequences that cannot be saved.
6. NaN defense in _step with masked reconstruction loss.
"""

import pytest
import torch
import warnings

from chronocratic.models import TimeVAE
from chronocratic.models.generative.timevae.model import _timevae_encoder_output_length


# ---------------------------------------------------------------------------
# Task 1 / Task 2: Encoder output length + auto-clamp
# ---------------------------------------------------------------------------


class TestEncoderOutputLengthFormula:
    """_timevae_encoder_output_length matches conv^N(T) with conv(L)=(L-1)//stride+1."""

    @pytest.mark.parametrize(
        ("seq_len", "num_layers", "stride", "expected"),
        [
            (32, 3, 2, 4),  # 32->16->8->4
            (22, 3, 2, 3),  # 22->11->6->3
            (8, 3, 2, 1),  # 8->4->2->1
            (100, 3, 2, 13),  # 100->50->25->13
            (32, 3, 1, 32),  # stride=1 preserves length
            (8, 3, 1, 8),  # stride=1 preserves length
        ],
    )
    def test_formula_values(
        self, seq_len: int, num_layers: int, stride: int, expected: int
    ) -> None:
        result = _timevae_encoder_output_length(seq_len, num_layers, stride)
        assert result == expected, (
            f"encoder_output_length({seq_len}, {num_layers}, {stride}) = {result}, expected {expected}"
        )

    def test_exported_from_model_module(self) -> None:
        """The helper must also be available from the model module."""
        from chronocratic.models.generative.timevae.model import (
            _timevae_encoder_output_length as exported,
        )

        assert exported(32, 3, 2) == 4
        assert exported(8, 3, 2) == 1


class TestTimeVAEAutoClampStride:
    """TimeVAE auto-clamps conv_stride when encoder output < 2."""

    def test_stride_reduced_with_warning(self) -> None:
        """sequence_length=8 with default 3 layers → stride reduced to 1, UserWarning emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = TimeVAE(sequence_length=8, input_dim=3, latent_dim=8)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, (
            f"Expected UserWarning when stride is reduced, got {len(user_warnings)} warnings"
        )

        # Verify stride was actually reduced
        assert model.conv_stride == 1, (
            f"Expected conv_stride=1 after auto-clamp, got {model.conv_stride}"
        )

    def test_stride_not_reduced_when_safe(self) -> None:
        """sequence_length=32 → stride stays at default (2), no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = TimeVAE(sequence_length=32, input_dim=3, latent_dim=8)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        stride_warnings = [x for x in user_warnings if "stride" in str(x.message).lower()]
        assert len(stride_warnings) == 0, (
            f"No stride warning expected for T=32, got {stride_warnings}"
        )
        assert model.conv_stride == 2

    def test_degenerate_sequence_error(self) -> None:
        """Extremely short sequence that cannot be saved even with stride=1 → clear error."""
        # sequence_length=1: even stride=1 gives T'=1 with 3 layers, which is < 2
        with pytest.raises(ValueError, match="sequence_length"):
            TimeVAE(sequence_length=1, input_dim=3, latent_dim=8)


# ---------------------------------------------------------------------------
# Task 1 / Task 2: Forward pass + gradients on short sequences
# ---------------------------------------------------------------------------


class TestTimeVAEForwardOnShortSequence:
    """TimeVAE forward pass succeeds on short sequences after auto-clamp."""

    @pytest.mark.parametrize("seq_len", [8, 12, 22])
    def test_forward_succeeds(self, seq_len: int) -> None:
        model = TimeVAE(sequence_length=seq_len, input_dim=3, latent_dim=8)
        x = torch.randn(2, seq_len, 3)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, seq_len, 3), (
            f"Expected output shape (2, {seq_len}, 3), got {output.shape}"
        )
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    @pytest.mark.parametrize("seq_len", [8, 12])
    def test_gradients_flow(self, seq_len: int) -> None:
        """Backward produces non-zero finite gradients on short sequences."""
        model = TimeVAE(sequence_length=seq_len, input_dim=3, latent_dim=8)
        model.train()
        x = torch.randn(2, seq_len, 3, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradient did not flow back to input"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
        assert not torch.all(x.grad == 0), "Gradient is all zeros (degenerate encoder)"


# ---------------------------------------------------------------------------
# Task 3: NaN defense + masked reconstruction loss
# ---------------------------------------------------------------------------


class TestTimeVAENaNDefense:
    """NaN-padded batches produce finite loss with masked reconstruction."""

    def test_nan_padded_batch_finite_loss(self) -> None:
        """training_step on NaN-padded batch produces finite loss."""
        model = TimeVAE(sequence_length=16, input_dim=3, latent_dim=8)
        model.train()

        # Create a batch with NaN padding on the last 4 timesteps
        batch = torch.randn(2, 16, 3)
        batch[:, -4:, :] = float("nan")

        loss, recon_loss, kl_loss = model._step(batch)

        assert torch.isfinite(loss), f"Total loss is not finite: {loss}"
        assert torch.isfinite(recon_loss), f"Reconstruction loss is not finite: {recon_loss}"
        assert torch.isfinite(kl_loss), f"KL loss is not finite: {kl_loss}"

    def test_all_nan_timesteps_masked(self) -> None:
        """Batch where some timesteps are fully NaN should still produce finite loss."""
        model = TimeVAE(sequence_length=16, input_dim=3, latent_dim=8)
        model.train()

        batch = torch.randn(2, 16, 3)
        # Sample 0: last 12 timesteps are NaN
        batch[0, 4:, :] = float("nan")
        # Sample 1: last 8 timesteps are NaN
        batch[1, 8:, :] = float("nan")

        loss, recon_loss, kl_loss = model._step(batch)

        assert torch.isfinite(loss), f"Total loss is not finite: {loss}"

    def test_no_nan_unchanged_behavior(self) -> None:
        """Batch with no NaN should produce the same finite loss as before."""
        model = TimeVAE(sequence_length=16, input_dim=3, latent_dim=8)
        model.train()

        batch = torch.randn(2, 16, 3)
        loss, recon_loss, kl_loss = model._step(batch)

        assert torch.isfinite(loss), f"Total loss is not finite: {loss}"
