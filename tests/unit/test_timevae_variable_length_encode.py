"""TimeVAE variable-length encoding tests.

Verifies that the encoder accepts inputs of any length via adaptive temporal
pooling, and that the pool is an exact identity at the nominal sequence_length.
"""

import warnings

import torch

from chronocratic.models import TimeVAE
from chronocratic.models.enums.encoding import EncodingOutputShape


class TestPoolTargetAndDenseDim:
    """Pool target matches the conv output at nominal length."""

    def test_pool_target_and_dense_dim(self) -> None:
        model = TimeVAE(sequence_length=128, input_dim=3, latent_dim=4)

        assert model.encoder.pooled_length == 16
        assert model.encoder.encoder_last_dense_dim == 3200


class TestPoolIdentityAtNominalLength:
    """Pooling 16 -> 16 bins is exact identity at nominal length."""

    def test_pool_identity_at_nominal_length(self) -> None:
        model = TimeVAE(sequence_length=128, input_dim=3, latent_dim=4)

        # The encoder sequential ends with [..., ReLU, AdaptiveAvgPool1d, Flatten]
        # Pool is at [-2], Flatten is at [-1].
        pool = model.encoder.encoder[-2]
        assert isinstance(pool, torch.nn.AdaptiveAvgPool1d), (
            f"Expected AdaptiveAvgPool1d at encoder[-2], got {type(pool).__name__}"
        )

        # Feed a (batch, channels, 16) tensor through the pool.
        # 16 -> 16 is one element per bin, so it must be exact identity.
        x = torch.randn(4, 200, 16)
        with torch.no_grad():
            y = pool(x)

        assert y.shape == (4, 200, 16)
        assert torch.equal(y, x), "AdaptiveAvgPool1d(16) is not identity on input of width 16"


class TestEncodeMultipleLengthsSingleModel:
    """One model encodes 96, 168, 336 timesteps."""

    def test_encode_multiple_lengths_single_model(self) -> None:
        model = TimeVAE(
            sequence_length=128, input_dim=3, latent_dim=4, hidden_layer_sizes=(4, 8, 16)
        )

        for window_length in [96, 168, 336]:
            data = torch.randn(8, window_length, 3)
            reps = model.encode(data, batch_size=4)
            assert reps.shape == (8, model.latent_dim), (
                f"Expected shape (8, {model.latent_dim}), got {reps.shape} "
                f"for window_length={window_length}"
            )
            assert torch.isfinite(reps).all(), (
                f"Non-finite values in representations for window_length={window_length}"
            )


class TestEncodeShorterThanNominalUpsamples:
    """T=96 path upsamples conv output (12 -> 16) and produces finite results."""

    def test_encode_shorter_than_nominal_upsamples(self) -> None:
        model = TimeVAE(
            sequence_length=128, input_dim=3, latent_dim=4, hidden_layer_sizes=(4, 8, 16)
        )

        # T=96 -> conv output 12 -> pooled to 16 (upsampling)
        data = torch.randn(8, 96, 3)
        reps = model.encode(data, batch_size=4)

        assert reps.shape == (8, model.latent_dim)
        assert torch.isfinite(reps).all()


class TestEncodeBatchOutputShapes:
    """_encode_batch produces correct shapes for VECTOR and SEQUENCE at T=168."""

    def test_encode_batch_output_shapes(self) -> None:
        model = TimeVAE(
            sequence_length=128, input_dim=3, latent_dim=4, hidden_layer_sizes=(4, 8, 16)
        )
        data = torch.randn(8, 168, 3)

        # VECTOR shape
        vec = model._encode_batch(model._encoder, data, output=EncodingOutputShape.VECTOR)
        assert vec.shape == (8, 4), f"VECTOR shape mismatch: {vec.shape}"

        # SEQUENCE shape
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            seq = model._encode_batch(model._encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert seq.shape == (8, 1, 4), f"SEQUENCE shape mismatch: {seq.shape}"


class TestTrainingStepUnchanged:
    """Training step still produces finite loss after the pool insertion."""

    def test_training_step_unchanged(self) -> None:
        torch.manual_seed(0)
        model = TimeVAE(
            sequence_length=128,
            input_dim=3,
            latent_dim=4,
            hidden_layer_sizes=(4, 8, 16),
            max_train_length=128,
        )
        model.train()

        batch = torch.randn(2, 512, 3)
        loss = model.training_step(batch, 0)

        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.requires_grad, "Loss does not require gradients"


class TestGradientsFlowThroughPool:
    """Backward through encode_batch at non-nominal length produces finite gradients."""

    def test_gradients_flow_through_pool(self) -> None:
        model = TimeVAE(
            sequence_length=128, input_dim=3, latent_dim=4, hidden_layer_sizes=(4, 8, 16)
        )

        x = torch.randn(2, 168, 3, requires_grad=True)
        reps = model.encode_batch(x)
        reps.sum().backward()

        assert x.grad is not None, "Gradient did not flow back to input"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
