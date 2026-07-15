"""Tests for TST masking_ratio parameter validation and config sync.

Verifies that TST (Time Series Transformer) accepts the `masking_ratio`
parameter, validates its bounds per D-03, stores it correctly, and keeps
the config dataclass in sync with the model init so that
``TST(**vars(config))`` continues to work.
"""

from lightning.pytorch import Trainer
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models.transformer.tst.config import TSTModelParameters
from chronocratic.models.transformer.tst.model import TST


@pytest.fixture
def tst_model() -> TST:
    """Create a small TST model for testing."""
    return TST(
        input_dim=3, sequence_length=32, hidden_dim=8, num_heads=2, depth=1, feedforward_dim=32
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


class TestBatchFormats:
    """_compute_loss accepts any batch format via extract_features_from_batch."""

    def test_bare_tensor(self, tst_model: TST) -> None:
        """_compute_loss(torch.randn(4, 32, 3)) returns a finite scalar."""
        x = torch.randn(4, 32, 3)
        loss = tst_model._compute_loss(x)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_two_tuple(self, tst_model: TST) -> None:
        """_compute_loss((X, y)) returns a finite scalar; y is ignored."""
        x = torch.randn(4, 32, 3)
        y = torch.randint(0, 5, (4,))
        loss = tst_model._compute_loss((x, y))
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_two_tuple_matches_bare(self, tst_model: TST) -> None:
        """Seeded RNG: same loss for bare X and (X, y) tuple."""
        torch.manual_seed(42)
        x = torch.randn(4, 32, 3)
        y = torch.randint(0, 5, (4,))

        torch.manual_seed(42)
        loss_bare = tst_model._compute_loss(x)

        torch.manual_seed(42)
        loss_tuple = tst_model._compute_loss((x, y))

        assert torch.allclose(loss_bare, loss_tuple)

    def test_batch_size_one(self, tst_model: TST) -> None:
        """(1, 32, 3) input produces a finite scalar."""
        x = torch.randn(1, 32, 3)
        loss = tst_model._compute_loss(x)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestEmptyMaskGuard:
    """Empty mask returns zero scalar with gradient path (NaN defense)."""

    def test_empty_mask_returns_zero_not_nan(self, tst_model: TST) -> None:
        """Force empty mask by setting _masking_ratio=0 so nothing is masked.
        Assert loss == 0, isfinite, and requires_grad."""
        original_ratio = tst_model._masking_ratio
        try:
            tst_model._masking_ratio = 0.0
            x = torch.randn(2, 4, 3, requires_grad=True)
            loss = tst_model._compute_loss(x)
            assert loss == 0.0, f"Expected zero loss for empty mask, got {loss.item()}"
            assert torch.isfinite(loss)
            assert loss.requires_grad
        finally:
            tst_model._masking_ratio = original_ratio


class TestGradientFlow:
    """Loss backward populates non-zero encoder gradients."""

    def test_loss_backward_populates_grads(self, tst_model: TST) -> None:
        """_compute_loss -> backward -> encoder params have non-None, non-zero grads."""
        tst_model.train()
        x = torch.randn(4, 16, 3, requires_grad=True)
        loss = tst_model._compute_loss(x)
        loss.backward()

        has_grad = False
        for _name, param in tst_model._encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No encoder parameter received non-zero gradients"


class TestTrainingIntegration:
    """End-to-end Lightning Trainer integration with (X, y) batch format."""

    def test_lightning_trainer_two_batch_fit(self) -> None:
        """Trainer(fast_dev_run=True) over a DataLoader of (X, y) 2-tuples completes.

        This reproduces the exact path that raised the original
        ``ValueError: not enough values to unpack (expected 5, got 2)``.
        """
        model = TST(
            input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, feedforward_dim=32
        )
        x = torch.randn(8, 16, 3)
        y = torch.randint(0, 5, (8,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = Trainer(fast_dev_run=True, enable_progress_bar=False, logger=False)
        # Must not raise ValueError from 5-tuple unpack
        trainer.fit(model, dataloader)

    def test_trainer_logs_train_loss(self) -> None:
        """After fit, train_loss was logged via trainer.callback_metrics."""
        model = TST(
            input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, feedforward_dim=32
        )
        x = torch.randn(8, 16, 3)
        y = torch.randint(0, 5, (8,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = Trainer(fast_dev_run=True, enable_progress_bar=False, logger=False)
        trainer.fit(model, dataloader)

        assert "train_loss" in trainer.callback_metrics, "train_loss was not logged during training"
        assert torch.isfinite(trainer.callback_metrics["train_loss"])


class TestClassDocstring:
    """Verify TST class docstring reflects current batch contract."""

    def test_docstring_mentions_extract_features_from_batch(self) -> None:
        """Class docstring references extract_features_from_batch."""
        doc = TST.__doc__
        assert "extract_features_from_batch" in doc, (
            "Class docstring should mention extract_features_from_batch"
        )

    def test_docstring_does_not_mention_5_tuple_batch(self) -> None:
        """Class docstring no longer claims 5-tuple batch format."""
        doc = TST.__doc__
        assert "Batch format expected from the DataLoader" not in doc, (
            "Old 5-tuple batch format text should be removed"
        )
        assert "padding_masks, IDs" not in doc, "Old 5-tuple unpack reference should be removed"

    def test_docstring_mentions_internal_masking(self) -> None:
        """Class docstring states masking is generated internally."""
        doc = TST.__doc__
        assert "internally" in doc, "Class docstring should state masking is generated internally"

    def test_docstring_mentions_masking_ratio_arg(self) -> None:
        """masking_ratio appears in the Args section of the docstring."""
        doc = TST.__doc__
        assert "masking_ratio" in doc, "Class docstring Args should mention masking_ratio"
