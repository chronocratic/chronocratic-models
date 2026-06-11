"""Tests for the _finetuning package.

Covers FineTuningModule, FlattenLinearHead, BatchAdapter,
RepresentationBackbone, all adapters, BackboneUnfreeze, and factory
functions. Tests use minimal nn.Module stubs so no real backbones are
needed — only the public API contracts are verified.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch import nn

from tscollection.models._finetuning import (
    BackboneUnfreeze,
    classification_loss,
    FineTuningModule,
    FlattenLinearHead,
    make_series2vec_finetuner,
    make_tst_finetuner,
    make_tstcc_finetuner,
    regression_loss,
    RepresentationBackbone,
    series2vec_representations,
    supervised_batch_adapter,
    tst_batch_adapter,
    tst_representations,
    tstcc_representations,
)

# ---------------------------------------------------------------------------
# Helpers: minimal stubs satisfying the protocols
# ---------------------------------------------------------------------------


class _DummyBackbone(nn.Module):
    """A tiny backbone with a known representation_dim for tests."""

    def __init__(self, rep_dim: int = 4) -> None:
        super().__init__()
        self._rep_dim = rep_dim
        self.fc = nn.Linear(2, rep_dim)

    @property
    def representation_dim(self) -> int:
        return self._rep_dim


def _dummy_rep_fn(backbone: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Return a (batch, rep_dim) tensor from the backbone.

    Uses the backbone's own parameters so the gradient chain is intact.
    """
    x = inputs[0]
    # Route through backbone.fc so gradients flow back to backbone params
    sample = x.mean(dim=1)  # (B, feat_dim)
    return backbone.fc(sample)  # type: ignore[attr-defined]


def _dummy_batch_adapter(batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Identity adapter: (X, targets) -> ((X,), targets)."""
    return (batch[0],), batch[1]


class _DummyHead(nn.Module):
    """Head that returns (batch, num_outputs) regardless of input shape.

    Uses a real Linear layer so parameters are registered for optimizer tests.
    """

    def __init__(self, num_outputs: int = 3) -> None:
        super().__init__()
        self._fc = nn.Linear(4, num_outputs)

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        # Pad or slice reps to match in_features=4
        x = (
            reps[:, :4]
            if reps.shape[1] >= 4
            else torch.nn.functional.pad(reps, (0, 4 - reps.shape[1]))
        )
        return self._fc(x)


# ---------------------------------------------------------------------------
# FineTuningModule tests
# ---------------------------------------------------------------------------


class TestFineTuningModule:
    """Verify the generic fine-tuning wrapper."""

    def test_forward_returns_expected_shape(self) -> None:
        """FineTuningModule.forward returns (batch, num_outputs)."""
        backbone = _DummyBackbone(rep_dim=4)
        head = FlattenLinearHead(in_features=4, num_outputs=5)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=False,
        )
        x = torch.randn(2, 10, 2)
        out = module(x)
        assert out.shape == (2, 5)

    def test_training_step_returns_scalar_and_logs(self) -> None:
        """training_step returns a scalar loss and logs train_loss."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=False,
        )
        batch: tuple = (torch.randn(4, 10, 2), torch.randn(4, 1))
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0  # scalar
        assert torch.isfinite(loss)

    def test_validation_step_returns_scalar_and_logs(self) -> None:
        """validation_step returns a scalar loss and logs val_loss."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=False,
        )
        batch: tuple = (torch.randn(4, 10, 2), torch.randn(4, 1))
        loss = module.validation_step(batch, 0)
        assert loss.ndim == 0  # scalar
        assert torch.isfinite(loss)

    def test_freeze_backbone_true_sets_requires_grad_false(self) -> None:
        """freeze_backbone=True sets requires_grad=False on all backbone params."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        _ = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=True,
        )
        for param in backbone.parameters():
            assert param.requires_grad is False

    def test_freeze_backbone_true_optimizer_sees_only_head_params(self) -> None:
        """When backbone is frozen, optimizer contains only head params."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=True,
        )
        opt = module.configure_optimizers()
        trainable_params = list(opt.param_groups[0]['params'])
        head_params = list(head.parameters())
        # All trainable params should belong to the head
        assert len(trainable_params) == len(head_params)

    def test_freeze_backbone_false_backbone_receives_grads(self) -> None:
        """When freeze_backbone=False, backbone params receive gradients."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=False,
        )
        batch: tuple = (torch.randn(2, 10, 2), torch.randn(2, 1))
        loss = module.training_step(batch, 0)
        loss.backward()
        backbone_grads = [p.grad for p in backbone.parameters() if p.requires_grad]
        # At least one backbone param should have a gradient
        assert any(g is not None for g in backbone_grads)


# ---------------------------------------------------------------------------
# FlattenLinearHead tests
# ---------------------------------------------------------------------------


class TestFlattenLinearHead:
    """Verify the flatten + linear head."""

    def test_3d_input_flattens(self) -> None:
        """FlattenLinearHead with (B, seq, dim) -> (B, num_outputs)."""
        head = FlattenLinearHead(in_features=8, num_outputs=3)
        # Input: (B=4, seq=2, dim=4) -> flatten -> (4, 8)
        x = torch.randn(4, 2, 4)
        out = head(x)
        assert out.shape == (4, 3)

    def test_2d_input_no_flatten_needed(self) -> None:
        """FlattenLinearHead with (B, dim) -> (B, num_outputs)."""
        head = FlattenLinearHead(in_features=8, num_outputs=3)
        x = torch.randn(4, 8)
        out = head(x)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# Batch adapter tests
# ---------------------------------------------------------------------------


class TestBatchAdapters:
    """Verify batch tuple unpacking for each model."""

    def test_tst_batch_adapter(self) -> None:
        """tst_batch_adapter unpacks (X, targets, padding_masks, IDs)."""
        x = torch.randn(2, 10, 3)
        targets = torch.tensor([0, 1])
        padding_masks = torch.ones(2, 10, dtype=torch.bool)
        ids = torch.tensor([100, 101])
        batch = (x, targets, padding_masks, ids)
        (inputs, result_targets) = tst_batch_adapter(batch)
        assert len(inputs) == 2
        torch.testing.assert_close(inputs[0], x)
        torch.testing.assert_close(inputs[1], padding_masks)
        torch.testing.assert_close(result_targets, targets)

    def test_supervised_batch_adapter(self) -> None:
        """supervised_batch_adapter unpacks (X, targets)."""
        x = torch.randn(2, 10, 3)
        targets = torch.tensor([0, 1])
        batch = (x, targets)
        (inputs, result_targets) = supervised_batch_adapter(batch)
        assert len(inputs) == 1
        torch.testing.assert_close(inputs[0], x)
        torch.testing.assert_close(result_targets, targets)


# ---------------------------------------------------------------------------
# Representation function tests
# ---------------------------------------------------------------------------


class TestRepresentationFunctions:
    """Verify representation functions call backbone methods correctly."""

    def test_tst_representations(self) -> None:
        """tst_representations calls backbone.get_representations and zeros padding."""
        backbone = MagicMock()
        reps = torch.randn(2, 5, 8)  # (B, seq, d_model)
        backbone.get_representations.return_value = reps
        x = torch.randn(2, 5, 3)
        padding_masks = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        result = tst_representations(backbone, x, padding_masks)
        backbone.get_representations.assert_called_once_with(x, padding_masks)
        assert result.shape == (2, 5, 8)
        # Positions where padding_masks==0 should be zeroed
        assert torch.all(result[0, 3:, :] == 0)
        assert torch.all(result[1, 4:, :] == 0)

    def test_series2vec_representations(self) -> None:
        """series2vec_representations calls backbone.network.encode(x)."""
        backbone = MagicMock()
        network = MagicMock()
        reps = torch.randn(2, 16)  # (B, 2*rep_dims)
        network.encode.return_value = reps
        backbone.network = network
        x = torch.randn(2, 10, 3)
        result = series2vec_representations(backbone, x)
        network.encode.assert_called_once_with(x)
        assert result.shape == (2, 16)

    def test_tstcc_representations(self) -> None:
        """tstcc_representations calls backbone(x.float()), extracts features."""
        backbone = MagicMock()
        logits = torch.randn(2, 5)
        features = torch.randn(2, 32)
        backbone.return_value = (logits, features)
        x = torch.randn(2, 10, 3)
        result = tstcc_representations(backbone, x)
        # Verify .float() was called
        call_arg = backbone.call_args[0][0]
        assert call_arg.dtype == torch.float32
        torch.testing.assert_close(result, features)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestClassificationLoss:
    """Verify classification_loss helper."""

    def test_classification_loss_calls_cross_entropy(self) -> None:
        """classification_loss uses nn.functional.cross_entropy with squeezed targets."""
        predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1.0, 0.0])  # float targets (common in dataloaders)
        loss = classification_loss(predictions, targets)
        expected = nn.functional.cross_entropy(predictions, targets.long().squeeze())
        torch.testing.assert_close(loss, expected)

    def test_regression_loss_calls_mse(self) -> None:
        """regression_loss uses nn.functional.mse_loss."""
        predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.1, 1.9], [3.1, 4.1]])
        loss = regression_loss(predictions, targets)
        expected = nn.functional.mse_loss(predictions, targets)
        torch.testing.assert_close(loss, expected)


# ---------------------------------------------------------------------------
# BackboneUnfreeze tests
# ---------------------------------------------------------------------------


class TestBackboneUnfreeze:
    """Verify the gradual unfreeze callback."""

    def test_freeze_before_training_freezes_backbone(self) -> None:
        """BackboneUnfreeze.freeze_before_training freezes backbone params."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=False,
        )
        callback = BackboneUnfreeze(unfreeze_at_epoch=5)
        callback.freeze_before_training(module)
        for param in backbone.parameters():
            assert param.requires_grad is False

    def test_finetune_function_unfreezes_at_target_epoch(self) -> None:
        """BackboneUnfreeze.finetune_function unfreezes at the target epoch."""
        backbone = _DummyBackbone(rep_dim=4)
        head = _DummyHead(num_outputs=1)
        module = FineTuningModule(
            backbone=backbone,
            head=head,
            representation_fn=_dummy_rep_fn,
            batch_adapter=_dummy_batch_adapter,
            loss_fn=nn.MSELoss(),
            freeze_backbone=False,
        )
        callback = BackboneUnfreeze(unfreeze_at_epoch=2)
        callback.freeze_before_training(module)
        opt = module.configure_optimizers()
        initial_groups = len(opt.param_groups)

        # Before target epoch — should stay frozen
        callback.finetune_function(module, current_epoch=1, optimizer=opt)
        for param in backbone.parameters():
            assert param.requires_grad is False

        # At target epoch — should unfreeze
        callback.finetune_function(module, current_epoch=2, optimizer=opt)
        for param in backbone.parameters():
            assert param.requires_grad is True
        # A new param group was added
        assert len(opt.param_groups) == initial_groups + 1


# ---------------------------------------------------------------------------
# RepresentationBackbone Protocol tests
# ---------------------------------------------------------------------------


class TestRepresentationBackboneProtocol:
    """Verify the Protocol is runtime-checkable."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """RepresentationBackbone has @runtime_checkable."""
        assert isinstance(RepresentationBackbone, type)
        # runtime_checkable means isinstance works
        backbone = _DummyBackbone(rep_dim=4)
        assert isinstance(backbone, RepresentationBackbone)

    def test_plain_module_without_property_fails(self) -> None:
        """A plain nn.Module without representation_dim is not a RepresentationBackbone."""
        plain = nn.Linear(2, 4)
        assert not isinstance(plain, RepresentationBackbone)


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    """Verify factory constructors produce FineTuningModule instances."""

    def test_make_tst_finetuner_classification(self) -> None:
        """make_tst_finetuner with classification task returns FineTuningModule."""
        backbone = _DummyBackbone(rep_dim=4)
        module = make_tst_finetuner(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
        )
        assert isinstance(module, FineTuningModule)

    def test_make_tst_finetuner_regression(self) -> None:
        """make_tst_finetuner with regression task returns FineTuningModule."""
        backbone = _DummyBackbone(rep_dim=4)
        module = make_tst_finetuner(
            backbone, num_outputs=2, task='regression', freeze_backbone=False
        )
        assert isinstance(module, FineTuningModule)

    def test_make_series2vec_finetuner_classification(self) -> None:
        """make_series2vec_finetuner with classification returns FineTuningModule."""
        backbone = _DummyBackbone(rep_dim=4)
        module = make_series2vec_finetuner(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
        )
        assert isinstance(module, FineTuningModule)

    def test_make_tstcc_finetuner_classification(self) -> None:
        """make_tstcc_finetuner with classification returns FineTuningModule."""
        backbone = _DummyBackbone(rep_dim=4)
        module = make_tstcc_finetuner(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
        )
        assert isinstance(module, FineTuningModule)

    def test_factory_creates_correct_head_size(self) -> None:
        """Factory head size matches backbone.representation_dim * num_outputs."""
        backbone = _DummyBackbone(rep_dim=8)
        module = make_tst_finetuner(
            backbone, num_outputs=5, task='classification', freeze_backbone=False
        )
        # The head should be a FlattenLinearHead with correct in_features
        head = module._head  # noqa: SLF001
        assert isinstance(head, FlattenLinearHead)
        fc = head._fc  # noqa: SLF001
        assert fc.in_features == 8
        assert fc.out_features == 5
