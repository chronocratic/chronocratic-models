"""Tests for TimeVAE's residual_projection option (crop vs. dense).

Default ``residual_projection="crop"`` removes ResidualConnection's
O((C*T)^2) ``final_dense`` layer; ``"dense"`` reproduces upstream TimeVAE
exactly. See the memory-fix spec §8.
"""

import pytest
import torch

from chronocratic.models.enums.layers import ResidualProjectionType
from chronocratic.models.generative.timevae.model import TimeVAE


class TestDefaultIsCropNoFinalDense:
    def test_no_final_dense_attribute(self) -> None:
        model = TimeVAE(sequence_length=64, input_dim=3, hidden_layer_sizes=(8, 16, 32))
        assert not hasattr(model.decoder.residual_conn, "final_dense")


class TestOutputShapeAcrossLengths:
    @pytest.mark.parametrize("sequence_length", [24, 64, 1000, 1001, 7500])
    def test_decoder_output_shape_and_finite(self, sequence_length: int) -> None:
        model = TimeVAE(
            sequence_length=sequence_length, input_dim=3, hidden_layer_sizes=(8, 16, 32)
        )
        z = torch.randn(2, model.latent_dim)
        out = model.decoder(z)
        assert out.shape == (2, sequence_length, 3)
        assert torch.isfinite(out).all()

    def test_stride_auto_clamp_length_still_works(self) -> None:
        with pytest.warns(UserWarning, match="stride"):
            model = TimeVAE(sequence_length=8, input_dim=3, hidden_layer_sizes=(8, 16, 32))
        z = torch.randn(2, model.latent_dim)
        out = model.decoder(z)
        assert out.shape == (2, 8, 3)
        assert torch.isfinite(out).all()


class TestResidualCanBeNegative:
    def test_deterministic_negative_residual(self) -> None:
        model = TimeVAE(sequence_length=64, input_dim=3, hidden_layer_sizes=(8, 16, 32))
        last_deconv = model.decoder.residual_conn.deconv_layers[-1]
        with torch.no_grad():
            last_deconv.weight.zero_()
            last_deconv.bias.fill_(-1.0)
        z = torch.randn(2, model.latent_dim)
        residuals = model.decoder.residual_conn(z)
        assert torch.all(residuals == -1.0)


class TestCropLayoutChannelsCorrect:
    def test_channel_c_outputs_constant_c(self) -> None:
        model = TimeVAE(sequence_length=64, input_dim=3, hidden_layer_sizes=(8, 16, 32))
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
        from chronocratic.models.layers.general import ResidualConnection

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
        """The actual bug: dense mode allocates ~678M params at this shape."""
        model = TimeVAE(sequence_length=5200, input_dim=5)
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
