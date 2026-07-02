"""Batch-size=1 gradient flow tests for all 10 models.

Verifies that every model supports single-sample encoding with gradient flow
in training mode.  This catches:

1. BatchNorm degeneracy at B=1 (zero-variance statistics).
2. Gradient-breaking patterns such as detached tensors in the forward path.
3. Shape mismatches that only appear with tiny batch sizes.

All models use their default ``norm="layer"`` setting (GroupNorm / LayerNorm),
so BatchNorm degeneracy should not occur.  Tests run in **.train()** mode to
exercise normalization-layer behavior under the worst-case statistics regime.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models import (
    AutoTCL,
    CoST,
    MCL,
    RecurrentAutoEncoder,
    Series2Vec,
    TS2Vec,
    TST,
    TSTCC,
    TimeNet,
    TimeVAE,
)


# ---------------------------------------------------------------------------
# Model factories — minimal configs that produce a working instance
# ---------------------------------------------------------------------------

def _make_mcl() -> MCL:
    return MCL(
        input_dims=3,
        output_dims=16,
        encoder_channels=(16, 32, 16),
        encoder_kernels=(5, 3, 3),
        encoder_dilations=(1, 2, 4),
        projection_dims=16,
    )


def _make_tstcc() -> TSTCC:
    return TSTCC(
        input_dims=3,
        conv_kernel_size=5,
        stride=2,
        output_dims=16,
        encoder_channels=(8, 16),
        encoder_inner_kernels=(5, 5),
        temporal_contrast_hidden_dim=32,
        temporal_contrast_timesteps=2,
    )


def _make_series2vec() -> Series2Vec:
    return Series2Vec(
        input_dims=3,
        embedding_dims=8,
        representation_dims=16,
        encoder_kernel_size=4,
        num_heads=2,
        feedforward_dims=32,
    )


def _make_tst() -> TST:
    return TST(
        input_dims=3,
        sequence_length=32,
        hidden_dims=16,
        num_heads=2,
        depth=1,
        feedforward_dims=32,
    )


def _make_ts2vec() -> TS2Vec:
    return TS2Vec(
        input_dims=3,
        hidden_dims=16,
        output_dims=32,
        depth=2,
    )


def _make_cost() -> CoST:
    return CoST(
        input_dims=3,
        sequence_length=32,
        hidden_dims=16,
        output_dims=32,
        depth=2,
        max_train_length=32,
        queue_size=8,
    )


def _make_autotcl() -> AutoTCL:
    return AutoTCL(
        input_dims=3,
        hidden_dims=16,
        output_dims=32,
        depth=2,
        max_train_length=32,
    )


def _make_timevae() -> TimeVAE:
    return TimeVAE(
        sequence_length=32,
        input_dims=3,
        latent_dim=8,
        hidden_layer_sizes=(16, 32),
    )


def _make_timenet() -> TimeNet:
    return TimeNet(
        input_dims=3,
        hidden_dims=16,
        depth=1,
    )


def _make_recurrentae() -> RecurrentAutoEncoder:
    return RecurrentAutoEncoder(input_dims=3, layers=(16,))


# Mapping: model name -> (factory, input_shape)
# All shapes use batch_size=1, sequence_length=32, input_dims=3.
MODEL_PARAMS: dict[str, tuple] = {
    "MCL": (_make_mcl, (1, 32, 3)),
    "TSTCC": (_make_tstcc, (1, 32, 3)),
    "Series2Vec": (_make_series2vec, (1, 32, 3)),
    "TST": (_make_tst, (1, 32, 3)),
    "TS2Vec": (_make_ts2vec, (1, 32, 3)),
    "CoST": (_make_cost, (1, 32, 3)),
    "AutoTCL": (_make_autotcl, (1, 32, 3)),
    "TimeVAE": (_make_timevae, (1, 32, 3)),
    "TimeNet": (_make_timenet, (1, 32, 3)),
    "RecurrentAE": (_make_recurrentae, (1, 32, 3)),
}

MODEL_NAMES = list(MODEL_PARAMS.keys())


@pytest.fixture(params=MODEL_NAMES, ids=lambda name: name)
def model_name(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture yielding each model name."""
    return request.param


@pytest.fixture
def model(model_name: str):
    """Fresh model instance in training mode."""
    factory, _ = MODEL_PARAMS[model_name]
    m = factory()
    m.train()
    return m


@pytest.fixture
def input_tensor(model_name: str) -> torch.Tensor:
    """Single-sample input tensor with requires_grad=True."""
    _, shape = MODEL_PARAMS[model_name]
    return torch.randn(shape, requires_grad=True)


# ---------------------------------------------------------------------------
# Gradient flow assertions
# ---------------------------------------------------------------------------

class TestBatchOneGradientFlow:
    """encode_batch(x) at batch_size=1 preserves gradient flow for all 10 models."""

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_gradient_not_none(
        self, model_name: str, model, input_tensor: torch.Tensor
    ) -> None:
        """x.grad is populated after backward through encode_batch."""
        output = model.encode_batch(input_tensor)
        output.sum().backward()
        assert input_tensor.grad is not None, (
            f"{model_name}: gradient did not flow back to input at batch_size=1"
        )

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_gradient_finite(
        self, model_name: str, model, input_tensor: torch.Tensor
    ) -> None:
        """All gradient values are finite (no NaN / Inf) at batch_size=1."""
        output = model.encode_batch(input_tensor)
        output.sum().backward()
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all(), (
            f"{model_name}: gradient contains NaN or Inf at batch_size=1"
        )

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_gradient_not_all_zero(
        self, model_name: str, model, input_tensor: torch.Tensor
    ) -> None:
        """Gradient is not identically zero (normalization is not degenerate)."""
        output = model.encode_batch(input_tensor)
        output.sum().backward()
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0), (
            f"{model_name}: gradient is all zeros at batch_size=1 "
            "(likely BatchNorm degeneracy or detached forward path)"
        )


# ---------------------------------------------------------------------------
# Training mode preservation
# ---------------------------------------------------------------------------

class TestTrainModePreserved:
    """encode_batch must not toggle the encoder's train/eval state."""

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_encoder_stays_in_train_mode(
        self, model_name: str, model, input_tensor: torch.Tensor
    ) -> None:
        """Encoder remains in training mode after encode_batch."""
        # Verify model is in train mode
        assert model.training, f"{model_name}: model should be in train mode"
        _ = model.encode_batch(input_tensor)
        assert model.training, (
            f"{model_name}: encode_batch toggled model to eval mode"
        )
