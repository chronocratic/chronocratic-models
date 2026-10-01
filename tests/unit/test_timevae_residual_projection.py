"""Tests for TimeVAE's residual_projection option (crop vs. dense).

Default ``residual_projection=ResidualProjectionType.DENSE`` matches upstream
TimeVAE's ``Linear(C·L, C·T)`` exactly, at the cost of O((C*T)^2) parameters.
``ResidualProjectionType.CROP`` is the memory-saving alternative added here:
it crops ResidualConnection's deconvolution output instead, adding no
``final_dense`` layer. See the memory-fix spec §8.
"""

import warnings

import pytest
import torch

from chronocratic.models.generative.timevae.enums import ResidualProjectionType
from chronocratic.models.generative.timevae.model import TimeVAE


class TestDefaultIsDenseMatchesUpstream:
    def test_final_dense_attribute_present_by_default(self) -> None:
        model = TimeVAE(sequence_length=64, input_dim=3, hidden_layer_sizes=(8, 16, 32))
        assert model.residual_projection is ResidualProjectionType.DENSE
        assert hasattr(model.decoder.residual_conn, "final_dense")


class TestCropHasNoFinalDense:
    def test_no_final_dense_attribute(self) -> None:
        model = TimeVAE(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.CROP,
        )
        assert not hasattr(model.decoder.residual_conn, "final_dense")


class TestOutputShapeAcrossLengths:
    @pytest.mark.parametrize("sequence_length", [24, 64, 1000, 1001, 7500])
    def test_decoder_output_shape_and_finite(self, sequence_length: int) -> None:
        model = TimeVAE(
            sequence_length=sequence_length,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.CROP,
        )
        z = torch.randn(2, model.latent_dim)
        out = model.decoder(z)
        assert out.shape == (2, sequence_length, 3)
        assert torch.isfinite(out).all()

    def test_stride_auto_clamp_length_still_works(self) -> None:
        with pytest.warns(UserWarning, match="stride"):
            model = TimeVAE(
                sequence_length=8,
                input_dim=3,
                hidden_layer_sizes=(8, 16, 32),
                residual_projection=ResidualProjectionType.CROP,
            )
        z = torch.randn(2, model.latent_dim)
        out = model.decoder(z)
        assert out.shape == (2, 8, 3)
        assert torch.isfinite(out).all()


class TestResidualCanBeNegative:
    def test_deterministic_negative_residual(self) -> None:
        model = TimeVAE(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.CROP,
        )
        last_deconv = model.decoder.residual_conn.deconv_layers[-1]
        with torch.no_grad():
            last_deconv.weight.zero_()
            last_deconv.bias.fill_(-1.0)
        z = torch.randn(2, model.latent_dim)
        residuals = model.decoder.residual_conn(z)
        assert torch.all(residuals == -1.0)


class TestCropLayoutChannelsCorrect:
    def test_channel_c_outputs_constant_c(self) -> None:
        model = TimeVAE(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.CROP,
        )
        last_deconv = model.decoder.residual_conn.deconv_layers[-1]
        with torch.no_grad():
            last_deconv.weight.zero_()
            last_deconv.bias.copy_(torch.arange(3, dtype=last_deconv.bias.dtype))
        z = torch.randn(2, model.latent_dim)
        residuals = model.decoder.residual_conn(z)
        for c in range(3):
            assert torch.all(residuals[:, :, c] == c)


class TestDenseModeUnchanged:
    def test_final_dense_exists_with_expected_shape(self) -> None:
        model = TimeVAE(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.DENSE,
        )
        final_dense = model.decoder.residual_conn.final_dense
        assert final_dense.out_features == 64 * 3
        z = torch.randn(2, model.latent_dim)
        out = model.decoder(z)
        assert out.shape == (2, 64, 3)


class TestEnumAccepted:
    def test_enum_crop(self) -> None:
        model = TimeVAE(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.CROP,
        )
        assert model.residual_projection is ResidualProjectionType.CROP

    def test_enum_dense(self) -> None:
        model = TimeVAE(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            residual_projection=ResidualProjectionType.DENSE,
        )
        assert model.residual_projection is ResidualProjectionType.DENSE


class TestResidualConnectionRequiresCoercedEnum:
    """ResidualConnection is internal: the model coerces string/enum input (spec §8.3,
    'the layer is internal, the model chooses; explicit is better'). Callers must pass
    the already-coerced enum member directly."""

    def test_enum_dense_projection_works_standalone(self) -> None:
        from chronocratic.models.generative.timevae.layers import ResidualConnection

        rc = ResidualConnection(
            sequence_length=64,
            input_dim=3,
            hidden_layer_sizes=(8, 16, 32),
            latent_dim=8,
            encoder_last_dense_dim=32,
            projection=ResidualProjectionType.DENSE,
        )
        z = torch.randn(2, 8)
        out = rc(z)
        assert out.shape == (2, 64, 3)


class TestParameterBudget:
    def test_crop_mode_param_count_small(self) -> None:
        """Dense mode (the default) allocates ~678M params at this shape; crop avoids it."""
        model = TimeVAE(
            sequence_length=5200, input_dim=5, residual_projection=ResidualProjectionType.CROP
        )
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params < 5_000_000


class TestTrainingStepGradient:
    def test_gradient_reaches_last_deconv(self) -> None:
        model = TimeVAE(sequence_length=64, input_dim=3, hidden_layer_sizes=(8, 16, 32))
        model.train()
        x = torch.randn(4, 64, 3)
        loss = model.training_step(x, 0)
        loss.backward()
        last_deconv = model.decoder.residual_conn.deconv_layers[-1]
        assert last_deconv.weight.grad is not None
        assert torch.any(last_deconv.weight.grad != 0)


class TestDenseMemoryWarning:
    def test_warns_when_final_dense_exceeds_threshold(self) -> None:
        with pytest.warns(UserWarning, match="residual_projection"):
            TimeVAE(sequence_length=5200, input_dim=5)  # default DENSE, ~678M params

    def test_no_warning_for_small_dense_shape(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TimeVAE(
                sequence_length=64,
                input_dim=3,
                hidden_layer_sizes=(8, 16, 32),
                residual_projection=ResidualProjectionType.DENSE,
            )

    def test_no_warning_for_crop(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TimeVAE(
                sequence_length=5200, input_dim=5, residual_projection=ResidualProjectionType.CROP
            )
