# ruff: noqa: D, PLR2004, S101
"""Verify Conv layer init matches Keras GlorotUniform + zero bias.

Original TF uses GlorotUniform (Xavier) for all layers, not just Dense.
PyTorch Conv1d defaults to KaimingUniform with non-zero bias, producing
1.7-2.9× wider initial weight ranges and shifting ReLU thresholds.
"""

import math

import torch

from chronocratic.models.generative.timevae import TimeVAE


class TestConvInit:
    """Conv layers must use Xavier (Glorot) init with zero bias."""

    def _find_first_conv_encoder(self, model: TimeVAE) -> torch.nn.Conv1d:
        """Get the first Conv1d in the encoder Sequential."""
        for mod in model._encoder.encoder:
            if isinstance(mod, torch.nn.Conv1d):
                return mod
        msg = "No Conv1d found in encoder"
        raise RuntimeError(msg)

    def test_encoder_conv_weights_are_xavier(self) -> None:
        """Encoder Conv1d weights use XavierUniform, not Kaiming."""
        model = TimeVAE(sequence_length=16, input_dim=1, hidden_layer_sizes=(50,))
        conv = self._find_first_conv_encoder(model)

        # XavierUniform bound = sqrt(6 / (fan_in + fan_out))
        fan_in = conv.in_channels * conv.kernel_size[0]
        fan_out = conv.out_channels * conv.kernel_size[0]
        expected_bound = math.sqrt(6.0 / (fan_in + fan_out))

        actual_max = conv.weight.abs().max().item()
        assert actual_max <= expected_bound * 1.01

    def test_encoder_conv_bias_is_zero(self) -> None:
        """Encoder Conv1d bias is initialized to zero."""
        model = TimeVAE(sequence_length=16, input_dim=1, hidden_layer_sizes=(50,))
        conv = self._find_first_conv_encoder(model)
        assert torch.all(conv.bias == 0.0)

    def test_decoder_residual_convtranspose_bias_is_zero(self) -> None:
        """Decoder ConvTranspose1d bias is initialized to zero."""
        model = TimeVAE(
            sequence_length=16,
            input_dim=1,
            hidden_layer_sizes=(50, 100),
            use_residual_conn=True,
        )
        residual = model._decoder.residual_conn
        for deconv in residual.deconv_layers:
            assert torch.all(deconv.bias == 0.0)

    def test_decoder_residual_convtranspose_weights_xavier(self) -> None:
        """Decoder ConvTranspose1d weights use XavierUniform."""
        model = TimeVAE(
            sequence_length=16,
            input_dim=1,
            hidden_layer_sizes=(50, 100),
            use_residual_conn=True,
        )
        deconv = model._decoder.residual_conn.deconv_layers[0]
        fan_in = deconv.in_channels * deconv.kernel_size[0]
        fan_out = deconv.out_channels * deconv.kernel_size[0]
        expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
        actual_max = deconv.weight.abs().max().item()
        assert actual_max <= expected_bound * 1.01
