"""Tests for BaseVariationalAutoencoder canonical parameter names.

Validates that vae_base.py uses sequence_length, input_dims, reconstruction_weight
instead of seq_len, feat_dim, reconstruction_wt.
"""

from __future__ import annotations

from chronocratic.models.generative.timevae.model import TimeVAE
from chronocratic.models.generative.timevae.vae_base import BaseVariationalAutoencoder


class TestBaseVAECanonicalNames:
    """BaseVariationalAutoencoder uses sequence_length, input_dims, reconstruction_weight."""

    def test_timevae_has_sequence_length(self) -> None:
        """Instance has self.sequence_length after rename."""
        model = TimeVAE(sequence_length=100, input_dims=1, latent_dim=10, reconstruction_weight=3.0)
        assert model.sequence_length == 100

    def test_timevae_has_input_dims(self) -> None:
        """Instance has self.input_dims after rename."""
        model = TimeVAE(sequence_length=100, input_dims=3, latent_dim=10)
        assert model.input_dims == 3

    def test_timevae_has_reconstruction_weight(self) -> None:
        """Instance has self.reconstruction_weight after rename."""
        model = TimeVAE(sequence_length=100, input_dims=1, latent_dim=10, reconstruction_weight=2.5)
        assert model.reconstruction_weight == 2.5

    def test_no_seq_len_attribute(self) -> None:
        """self.seq_len should no longer exist on the base class."""
        model = TimeVAE(sequence_length=100, input_dims=1, latent_dim=10)
        assert not hasattr(model, "seq_len")

    def test_no_feat_dim_attribute(self) -> None:
        """self.feat_dim should no longer exist on the base class."""
        model = TimeVAE(sequence_length=100, input_dims=1, latent_dim=10)
        assert not hasattr(model, "feat_dim")


class TestFullModelInstantiation:
    """TimeVAE instantiates and forward-passes with canonical names."""

    def test_full_instantiation(self) -> None:
        """TimeVAE(sequence_length=100, input_dims=1, latent_dim=10) works."""
        model = TimeVAE(sequence_length=100, input_dims=1, latent_dim=10)
        assert model is not None

    def test_forward_pass(self) -> None:
        """Model produces output with correct shape after rename."""
        import torch

        model = TimeVAE(sequence_length=96, input_dims=2, latent_dim=8)
        model.eval()
        x = torch.randn(4, 96, 2)  # (batch, sequence_length, input_dims)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 96, 2)
