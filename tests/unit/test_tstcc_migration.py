"""Tests for TS-TCC enum removal and downstream mode migration.

Per D-03: remove TSTCCTrainingMode entirely; model becomes single-purpose
(pretrain only). Downstream tasks move to FineTuningModule.
"""

from __future__ import annotations

import pytest
import torch

from tscollection.models._finetuning import make_tstcc_finetuner
from tscollection.models.convolutional.standard.tstcc.model import TSTCC


class TestTSTCCEnumRemoved:
    """Verify TSTCCTrainingMode enum is no longer importable."""

    def test_enum_import_removed(self) -> None:
        """TSTCCTrainingMode is no longer importable from tstcc.enums."""
        with pytest.raises(ImportError):
            from tscollection.models.convolutional.standard.tstcc.enums import (  # noqa: F401, PLC0415
                TSTCCTrainingMode,
            )

    def test_enum_not_in_barrel(self) -> None:
        """TSTCCTrainingMode is no longer in the tstcc package barrel."""
        with pytest.raises(ImportError):
            from tscollection.models.convolutional.standard.tstcc import (  # noqa: F401, PLC0415
                TSTCCTrainingMode,
            )


class TestTSTCCModelCleaned:
    """Verify TSTCC model no longer accepts training_mode."""

    def test_model_no_training_mode_param(self) -> None:
        """TSTCC.__init__ no longer accepts training_mode."""
        import inspect

        sig = inspect.signature(TSTCC.__init__)
        assert 'training_mode' not in sig.parameters

    def test_only_self_supervised_contrastive(self) -> None:
        """TSTCC._compute_loss produces contrastive loss (no supervised branch)."""
        model = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=4,
            num_classes=3,
        )
        # Build a batch with labels — labels should be ignored
        x = torch.randn(4, 2, 128)
        labels = torch.randint(0, 3, (4,))
        batch = (x, labels)
        loss = model._compute_loss(batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestTSTCCConfigCleaned:
    """Verify TSTCCModelParameters no longer has training_mode."""

    def test_config_no_training_mode_field(self) -> None:
        """TSTCCModelParameters dataclass no longer has training_mode."""
        from tscollection.models.convolutional.standard.tstcc.config import (
            TSTCCModelParameters,
        )

        assert 'training_mode' not in TSTCCModelParameters.__dataclass_fields__


class TestTSTCCFineTuningModule:
    """Verify TSTCC downstream via FineTuningModule works."""

    def test_finetuner_classification_shape(self) -> None:
        """make_tstcc_finetuner produces (B, num_classes) output."""
        backbone = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=4,
            num_classes=3,
        )
        module = make_tstcc_finetuner(
            backbone, num_classes=5, task='classification', freeze_backbone=False
        )
        # Verify module construction (head uses backbone.representation_dim)
        assert module._head._fc.in_features == backbone.representation_dim  # noqa: SLF001
        assert module._head._fc.out_features == 5  # noqa: SLF001

    def test_finetuner_training_step(self) -> None:
        """training_step returns scalar loss."""
        backbone = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=4,
            num_classes=3,
        )
        module = make_tstcc_finetuner(
            backbone, num_classes=5, task='classification', freeze_backbone=False
        )
        x = torch.randn(4, 2, 4)
        targets = torch.randint(0, 5, (4,))
        batch = (x, targets)
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
