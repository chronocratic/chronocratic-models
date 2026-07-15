"""Tests for TST masking_ratio parameter validation and config sync.

Verifies that TST (Time Series Transformer) accepts the `masking_ratio`
parameter, validates its bounds per D-03, stores it correctly, and keeps
the config dataclass in sync with the model init so that
``TST(**vars(config))`` continues to work.
"""

import pytest
import torch

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


class TestMakeMaskedInputs:
    """TST._make_masked_inputs Bernoulli masking (D-01, D-02)."""

    @pytest.fixture
    def model(self) -> TST:
        """Create a small TST model for testing."""
        return TST(
            input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, feedforward_dim=32
        )

    def test_input_is_masked(self, model: TST) -> None:
        """masked_x has zeros at masked positions, original values at kept positions."""
        x = torch.full((4, 64, 3), 5.0)
        masked_x, _targets, target_masks, _padding = model._make_masked_inputs(x)
        assert (masked_x[target_masks] == 0.0).all()
        assert (masked_x[~target_masks] == 5.0).all()

    def test_targets_are_unmasked(self, model: TST) -> None:
        """targets equals original x exactly."""
        x = torch.randn(4, 16, 3)
        _masked_x, targets, _target_masks, _padding = model._make_masked_inputs(x)
        assert torch.equal(targets, x)

    def test_masking_ratio_approx(self) -> None:
        """Masked fraction is ~15% over a large sample (within 0.13..0.17)."""
        torch.manual_seed(42)
        model = TST(
            input_dim=8,
            sequence_length=64,
            hidden_dim=16,
            num_heads=2,
            depth=1,
            feedforward_dim=32,
            masking_ratio=0.15,
        )
        x = torch.randn(256, 64, 8)
        _masked_x, _targets, target_masks, _padding = model._make_masked_inputs(x)
        masked_fraction = target_masks.float().mean().item()
        assert 0.13 <= masked_fraction <= 0.17

    def test_mask_is_stochastic(self, model: TST) -> None:
        """Two calls on the same input produce different target_masks."""
        x = torch.randn(4, 16, 3)
        _, _targets1, masks1, _padding1 = model._make_masked_inputs(x)
        _, _targets2, masks2, _padding2 = model._make_masked_inputs(x)
        assert not torch.equal(masks1, masks2)

    def test_padding_masks_shape_and_dtype(self, model: TST) -> None:
        """padding_masks: (B, T), bool, all True, on x.device."""
        x = torch.randn(4, 16, 3)
        _masked_x, _targets, _target_masks, padding_masks = model._make_masked_inputs(x)
        assert padding_masks.shape == (4, 16)
        assert padding_masks.dtype == torch.bool
        assert padding_masks.all()
        assert padding_masks.device == x.device

    def test_return_tuple_length(self, model: TST) -> None:
        """_make_masked_inputs returns exactly 4 elements."""
        x = torch.randn(4, 16, 3)
        result = model._make_masked_inputs(x)
        assert len(result) == 4
