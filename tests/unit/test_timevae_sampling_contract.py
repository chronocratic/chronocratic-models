# ruff: noqa: D, PLR2004, S101
"""Verify TimeVAE _step uses sampled z in both train and eval.

Original TF always feeds reparameterized z to decoder. Using z_mean
only in validation masks encoder divergence (NaN not caught by
monitoring) and distorts val_loss relative to train_loss.
"""

import torch

from chronocratic.models.generative.timevae import TimeVAE


class TestSamplingContract:
    """_step must route sampled z to decoder in all modes."""

    def _capture_decoder_input(self, model: TimeVAE, batch: torch.Tensor) -> torch.Tensor:
        """Run _step and return the tensor the decoder received."""
        received = {}
        handle = model._decoder.register_forward_hook(
            lambda _, input_h, __: received.update({"arg": input_h[0].clone()})
        )
        model._step(batch)
        handle.remove()
        return received["arg"]

    def test_training_decoder_input_is_not_z_mean(self) -> None:
        """During training, decoder input must differ from z_mean (sampled)."""
        model = TimeVAE(sequence_length=16, input_dim=1, latent_dim=4)
        model.train()
        batch = torch.randn(2, 16, 1)

        decoder_input = self._capture_decoder_input(model, batch)
        # Run encoder separately to get z_mean for comparison
        z_mean, _, _ = model._encoder(batch)
        # They should NOT be equal (sampling adds noise)
        assert not torch.allclose(decoder_input, z_mean)

    def test_validation_decoder_input_is_not_z_mean(self) -> None:
        """During validation, decoder input must differ from z_mean (TF parity)."""
        model = TimeVAE(sequence_length=16, input_dim=1, latent_dim=4)
        model.eval()
        batch = torch.randn(2, 16, 1)

        decoder_input = self._capture_decoder_input(model, batch)
        z_mean, _, _ = model._encoder(batch)
        assert not torch.allclose(decoder_input, z_mean)

    def test_forward_uses_z_mean(self) -> None:
        """forward() is deterministic: it uses z_mean, not sampled z."""
        model = TimeVAE(sequence_length=16, input_dim=1, latent_dim=4)
        model.eval()
        batch = torch.randn(2, 16, 1)

        received = {}
        handle = model._decoder.register_forward_hook(
            lambda _, input_h, __: received.update({"arg": input_h[0].clone()})
        )
        with torch.no_grad():
            model.forward(batch)
        handle.remove()

        with torch.no_grad():
            z_mean, _, _ = model._encoder(batch)
        torch.testing.assert_close(received["arg"], z_mean)

    def test_validation_loss_varies_with_sampling(self) -> None:
        """Validation loss should vary across calls due to sampling noise.

        If z_mean were used (deterministic), repeated _step calls on the same
        batch would produce identical losses. With sampled z, they differ.
        """
        model = TimeVAE(sequence_length=16, input_dim=1, latent_dim=4)
        model.eval()
        batch = torch.randn(2, 16, 1)

        losses = []
        for _ in range(3):
            with torch.no_grad():
                loss, _, _ = model._step(batch)
            losses.append(loss.item())

        assert not all(abs(losses[0] - l) < 1e-6 for l in losses[1:])
