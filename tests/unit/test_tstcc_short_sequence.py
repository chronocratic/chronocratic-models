"""TSTCC short-sequence encoder tests.

Verifies that TSTCC correctly computes its encoder output sequence length,
auto-clamps temporal contrast timesteps when the encoder shrinks the sequence
below the default, and handles NaN-padded inputs without producing infinite
losses.

Covers:
- Encoder output length: probed against the live encoder across stride/kernel configs
- Auto-clamp of temporal_contrast_timesteps at init and at runtime
- Forward pass on InsectWingbeat shape (seq_len=22, channels=200)
- TemporalContrast.forward with clamped timesteps
- Gradient flow on short sequences
- NaN defense in _compute_loss
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from chronocratic.models import TSTCC
from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.model import _tstcc_encoder_output_length

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tstcc_insect(**overrides: object) -> TSTCC:
    """Create a TSTCC model configured for InsectWingbeat parameters.

    Default params: input_dim=200, conv_kernel_size=8,
    temporal_contrast_timesteps=6 (will be clamped at init).
    """
    kwargs: dict[str, object] = {
        "input_dim": 200,
        "conv_kernel_size": 8,
        "temporal_contrast_timesteps": 6,
        **overrides,
    }
    return TSTCC(**kwargs)


def _make_nan_padded_batch(
    seq_len: int, channels: int, batch_size: int, pad_timesteps: int
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Create a batch with trailing-NaN timesteps simulating variable-length padding."""
    data = torch.randn(batch_size, seq_len, channels)
    # Fill last `pad_timesteps` timesteps with NaN (all channels)
    if pad_timesteps > 0:
        data[:, -pad_timesteps:, :] = float("nan")
    labels = torch.zeros(batch_size, dtype=torch.long)
    return (data, labels)


# ---------------------------------------------------------------------------
# Test 1: Encoder output length formula
# ---------------------------------------------------------------------------


class TestEncoderOutputLengthFormula:
    """The length _clamp_timesteps trusts must equal what the encoder produces.

    Asserted against the live encoder, never against a restatement of its
    geometry. The previous version of this test checked a pool-only reference
    formula -- a copy of the same misconception as the code under test -- so it
    passed while the real function was wrong in 40 of 48 configs, ignoring
    stride entirely and overestimating L' by 3x at stride=4.
    """

    @pytest.mark.parametrize("seq_len", [8, 22, 32, 50, 128])
    @pytest.mark.parametrize("stride", [1, 2, 4])
    @pytest.mark.parametrize("inner_kernels", [(8, 8), (3, 3)])
    def test_matches_real_encoder_output(
        self, seq_len: int, stride: int, inner_kernels: tuple[int, int]
    ) -> None:
        """_tstcc_encoder_output_length equals the encoder's actual output length."""
        encoder = TCCEncoder(
            input_dim=3,
            conv_kernel_size=8,
            stride=stride,
            representation_dim=8,
            dropout_rate=0.0,
            encoder_channels=(8, 16),
            encoder_inner_kernels=inner_kernels,
        ).eval()
        with torch.no_grad():
            real = encoder(torch.zeros(1, seq_len, 3)).shape[-1]
        assert _tstcc_encoder_output_length(encoder, seq_len, 3) == real

    def test_probe_restores_encoder_training_mode(self) -> None:
        """Probing must not leave the encoder in eval() -- it runs during __init__."""
        encoder = TCCEncoder(
            input_dim=3,
            conv_kernel_size=8,
            stride=1,
            representation_dim=8,
            dropout_rate=0.0,
            encoder_channels=(8, 16),
            encoder_inner_kernels=(8, 8),
        )
        assert encoder.training
        _tstcc_encoder_output_length(encoder, 32, 3)
        assert encoder.training, "probe leaked eval() state onto the encoder"


class TestClampTimestepsGuard:
    """The clamp must actually prevent TemporalContrast's seq_len > timestep error."""

    def test_stride_config_is_clamped_not_crashed(self) -> None:
        """stride=4 shrinks L' 3x; the guard must catch it rather than raise.

        Regression: the old pool-only length formula returned 17 here (ignoring
        stride), so the clamp declined to fire and _compute_loss raised
        "seq_len (6) must be > timestep (12)" -- the exact error it guards.
        """
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = TSTCC(
                input_dim=3,
                sequence_length=128,
                representation_dim=8,
                encoder_channels=(8, 16),
                encoder_inner_kernels=(8, 8),
                temporal_contrast_hidden_dim=8,
                temporal_contrast_timesteps=12,
                conv_kernel_size=8,
                stride=4,
            )
        with torch.no_grad():
            real = model._encoder(torch.zeros(1, 128, 3)).shape[-1]
        assert real > model.temporal_contrast_timesteps
        batch = (torch.randn(4, 128, 3), torch.zeros(4, dtype=torch.long))
        assert torch.isfinite(model._compute_loss(batch))


# ---------------------------------------------------------------------------
# Test 2: TSTCC auto-clamp timesteps at init
# ---------------------------------------------------------------------------


class TestTSTCCAutoClampTimesteps:
    """Creating TSTCC with InsectWingbeat params should clamp timesteps and warn."""

    def test_tstcc_clamps_timesteps_at_init(self) -> None:
        """temporal_contrast_timesteps clamped from 6 to 4 for seq_len=22.

        The encoder's real output length for seq_len=22 is 5, so 4 is the
        largest valid value (TemporalContrast needs L' > timesteps). This
        previously asserted 3, because the pool-only length formula reported
        L'=4 and the clamp discarded a timestep of capacity it did not need to.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = _make_tstcc_insect(sequence_length=22)

        assert hasattr(model, "temporal_contrast_timesteps")
        assert model.temporal_contrast_timesteps == 4, (
            f"Expected timesteps clamped to 4 for seq_len=22, "
            f"got {model.temporal_contrast_timesteps}"
        )
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "timesteps" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# Test 3: TSTCC forward on InsectWingbeat shape
# ---------------------------------------------------------------------------


class TestTSTCCForwardOnInsectWingbeat:
    """Forward pass on InsectWingbeat shape (batch=2, seq=22, channels=200)."""

    def test_forward_no_crash(self) -> None:
        """forward((2, 22, 200)) succeeds without ValueError from TemporalContrast."""
        model = _make_tstcc_insect(sequence_length=22)
        model.eval()
        x = torch.randn(2, 22, 200)
        with torch.no_grad():
            output = model.encode_batch(x)
        assert output.shape[0] == 2
        assert output.shape[-1] == model.representation_dim


# ---------------------------------------------------------------------------
# Test 4: TemporalContrast no crash with clamped timesteps
# ---------------------------------------------------------------------------


class TestTSTCCTemporalContrastNoCrash:
    """TemporalContrast.forward with clamped timesteps should not trigger assertion."""

    def test_temporal_contrast_forward_succeeds(self) -> None:
        """Clamped timesteps (3) < encoder output (4), so no ValueError."""
        model = _make_tstcc_insect(sequence_length=22)
        model.eval()
        x = torch.randn(2, 22, 200)
        # Encoder output should be 4, timesteps clamped to 3 -> 4 > 3, safe
        with torch.no_grad():
            features = model._encoder(x)  # TCCEncoder expects (B, T, C), outputs (B, rep, L')
        actual_seq_len = features.shape[-1]
        assert actual_seq_len > model.temporal_contrast_timesteps, (
            f"Encoder output seq_len ({actual_seq_len}) must be > "
            f"clamped timesteps ({model.temporal_contrast_timesteps})"
        )


# ---------------------------------------------------------------------------
# Test 5: Gradient flow on short sequences
# ---------------------------------------------------------------------------


class TestTSTCCGradientsOnShortSeq:
    """Backward through _compute_loss on short sequences produces valid gradients."""

    def test_gradients_nonzero_finite(self) -> None:
        """backward produces non-zero finite gradients for seq_len=22."""
        model = _make_tstcc_insect(sequence_length=22)
        model.train()
        x = torch.randn(2, 22, 200, requires_grad=True)
        labels = torch.zeros(2, dtype=torch.long)
        batch = (x, labels)

        loss = model._compute_loss(batch)
        loss.backward()

        assert x.grad is not None, "Gradient did not flow back to input"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"


# ---------------------------------------------------------------------------
# Test 6: NaN defense in _compute_loss
# ---------------------------------------------------------------------------


class TestTSTCCNaNDefense:
    """NaN-padded batches should produce finite loss after zero-fill."""

    def test_nan_padded_batch_finite_loss(self) -> None:
        """training_step on NaN-padded batch produces finite loss."""
        model = _make_tstcc_insect(sequence_length=22)
        model.train()
        batch = _make_nan_padded_batch(seq_len=22, channels=200, batch_size=2, pad_timesteps=5)

        loss = model._compute_loss(batch)
        assert torch.isfinite(loss), f"Loss is not finite for NaN-padded batch: {loss.item()}"
        assert math.isfinite(loss.item()), f"Loss.item() is not finite: {loss.item()}"
