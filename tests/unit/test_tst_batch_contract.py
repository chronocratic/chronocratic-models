"""Tests for TST masking_ratio parameter validation and config sync.

Verifies that TST (Time Series Transformer) accepts the `masking_ratio`
parameter, validates its bounds per D-03, stores it correctly, and keeps
the config dataclass in sync with the model init so that
``TST(**vars(config))`` continues to work.
"""

import pytest

from chronocratic.models.transformer.tst.config import TSTModelParameters
from chronocratic.models.transformer.tst.model import TST


@pytest.fixture
def tst_model() -> TST:
    """Create a small TST model for testing."""
    return TST(
        input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, feedforward_dim=32
    )


class TestMaskingRatio:
    """TST masking_ratio parameter validation and storage (D-01, D-03)."""

    def test_default_masking_ratio_is_0_15(self, tst_model: TST) -> None:
        """Default masking_ratio is 0.15."""
        assert tst_model._masking_ratio == 0.15

    def test_custom_masking_ratio_is_stored(self) -> None:
        """Custom masking_ratio value is stored correctly."""
        model = TST(
            input_dim=3,
            sequence_length=16,
            hidden_dim=8,
            num_heads=2,
            depth=1,
            feedforward_dim=32,
            masking_ratio=0.3,
        )
        assert model._masking_ratio == 0.3

    def test_masking_ratio_zero_raises_value_error(self) -> None:
        """masking_ratio=0.0 raises ValueError (D-03)."""
        with pytest.raises(ValueError, match="masking_ratio"):
            TST(
                input_dim=3,
                sequence_length=16,
                hidden_dim=8,
                num_heads=2,
                depth=1,
                feedforward_dim=32,
                masking_ratio=0.0,
            )

    def test_masking_ratio_one_raises_value_error(self) -> None:
        """masking_ratio=1.0 raises ValueError (D-03)."""
        with pytest.raises(ValueError, match="masking_ratio"):
            TST(
                input_dim=3,
                sequence_length=16,
                hidden_dim=8,
                num_heads=2,
                depth=1,
                feedforward_dim=32,
                masking_ratio=1.0,
            )

    def test_masking_ratio_negative_raises_value_error(self) -> None:
        """masking_ratio=-0.1 raises ValueError (D-03)."""
        with pytest.raises(ValueError, match="masking_ratio"):
            TST(
                input_dim=3,
                sequence_length=16,
                hidden_dim=8,
                num_heads=2,
                depth=1,
                feedforward_dim=32,
                masking_ratio=-0.1,
            )

    def test_masking_ratio_above_one_raises_value_error(self) -> None:
        """masking_ratio=1.5 raises ValueError (D-03)."""
        with pytest.raises(ValueError, match="masking_ratio"):
            TST(
                input_dim=3,
                sequence_length=16,
                hidden_dim=8,
                num_heads=2,
                depth=1,
                feedforward_dim=32,
                masking_ratio=1.5,
            )

    def test_config_has_masking_ratio_field(self) -> None:
        """TSTModelParameters has masking_ratio field with default 0.15."""
        config = TSTModelParameters(input_dim=3, sequence_length=16)
        assert config.masking_ratio == 0.15

    def test_tst_from_config_vars_works(self) -> None:
        """TST(**vars(TSTModelParameters(...))) works without error (config/init sync)."""
        config = TSTModelParameters(
            input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, feedforward_dim=32
        )
        model = TST(**vars(config))
        assert isinstance(model, TST)
        assert model._masking_ratio == 0.15
