"""Tests for TimeVAE config field renames and tuple type conversions.

Validates D-01 (plural→singular: input_dim), D-03 (representation_dim property),
D-04 (latent_dim unchanged) for TimeVAEModelParameters and TimeVAE.__init__.
"""

from __future__ import annotations

from chronocratic.models import TimeVAE, TimeVAEModelParameters


class TestConfigSingularInputDim:
    """TimeVAEModelParameters uses input_dim (singular, D-01)."""

    def test_config_with_singular_input_dim(self) -> None:
        """Config creates with input_dim (singular), latent_dim unchanged."""
        config = TimeVAEModelParameters(sequence_length=100, input_dim=1, latent_dim=10)
        assert config.sequence_length == 100
        assert config.input_dim == 1
        assert config.latent_dim == 10

    def test_config_reconstruction_weight_default(self) -> None:
        """reconstruction_weight defaults to 3.0."""
        config = TimeVAEModelParameters(sequence_length=100, input_dim=1, latent_dim=10)
        assert config.reconstruction_weight == 3.0

    def test_hidden_layer_sizes_is_tuple(self) -> None:
        """hidden_layer_sizes defaults to tuple (50, 100, 200)."""
        config = TimeVAEModelParameters(sequence_length=100, input_dim=1, latent_dim=10)
        assert config.hidden_layer_sizes == (50, 100, 200)
        assert isinstance(config.hidden_layer_sizes, tuple)

    def test_custom_seasonality_uses_tuple_type(self) -> None:
        """custom_seasonality accepts tuple of tuples."""
        config = TimeVAEModelParameters(
            sequence_length=100, input_dim=1, latent_dim=10, custom_seasonality=((4, 7), (2, 14))
        )
        assert config.custom_seasonality == ((4, 7), (2, 14))


class TestConfigSplatToModel:
    """vars(TimeVAEModelParameters(...)) passes to TimeVAE() without errors."""

    def test_splat_config_to_model(self) -> None:
        """TimeVAE(**vars(config)) instantiates without TypeError."""
        config = TimeVAEModelParameters(sequence_length=100, input_dim=1, latent_dim=10)
        model = TimeVAE(**vars(config))
        assert model is not None

    def test_splat_config_with_all_options(self) -> None:
        """Full config splat works with all parameters."""
        config = TimeVAEModelParameters(
            sequence_length=96,
            input_dim=3,
            latent_dim=8,
            reconstruction_weight=2.5,
            learning_rate=5e-4,
            hidden_layer_sizes=(32, 64, 128),
            trend_poly=2,
            custom_seasonality=((4, 7),),
            use_residual_conn=True,
        )
        model = TimeVAE(**vars(config))
        assert model is not None


class TestModelCanonicalParams:
    """TimeVAE.__init__ accepts singular input_dim (D-01)."""

    def test_model_with_singular_input_dim(self) -> None:
        """TimeVAE accepts input_dim (singular) and reconstruction_weight."""
        model = TimeVAE(sequence_length=100, input_dim=1, latent_dim=10, reconstruction_weight=2.0)
        assert model is not None

    def test_model_tuple_hidden_sizes(self) -> None:
        """TimeVAE accepts tuple for hidden_layer_sizes."""
        model = TimeVAE(
            sequence_length=100, input_dim=1, latent_dim=10, hidden_layer_sizes=(32, 64, 128)
        )
        assert model.hidden_layer_sizes == (32, 64, 128)


class TestRepresentationDimProperty:
    """TimeVAE exposes representation_dim returning latent_dim (D-03, D-04)."""

    def test_representation_dim_returns_latent_dim(self) -> None:
        """representation_dim property equals latent_dim."""
        model = TimeVAE(sequence_length=100, input_dim=1, latent_dim=16)
        assert model.representation_dim == 16

    def test_representation_dim_with_default_latent(self) -> None:
        """representation_dim returns default latent_dim (8)."""
        model = TimeVAE(sequence_length=100, input_dim=1)
        assert model.representation_dim == 8

    def test_representation_dim_independent_of_input_dim(self) -> None:
        """representation_dim is latent_dim, not input_dim."""
        model = TimeVAE(sequence_length=100, input_dim=5, latent_dim=12)
        assert model.representation_dim == 12
        assert model.representation_dim != 5
