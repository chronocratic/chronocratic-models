"""Tests for TS-TCC enum removal and downstream mode migration.

TSTCCTrainingMode was removed; model becomes single-purpose
(pretrain only). Downstream tasks move to SupervisedModule.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from chronocratic.models.convolutional.standard.tstcc.config import TSTCCModelParameters
from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.supervised import make_tstcc_supervised


class TestTSTCCEnumRemoved:
    """Verify TSTCCTrainingMode enum is no longer importable."""

    def test_enum_import_removed(self) -> None:
        """TSTCCTrainingMode is no longer importable from tstcc.enums."""
        with pytest.raises(ImportError):
            from chronocratic.models.convolutional.standard.tstcc.enums import (  # noqa: F401, PLC0415
                TSTCCTrainingMode,
            )

    def test_enum_not_in_barrel(self) -> None:
        """TSTCCTrainingMode is no longer in the tstcc package barrel."""
        with pytest.raises(ImportError):
            from chronocratic.models.convolutional.standard.tstcc import (  # noqa: F401, PLC0415
                TSTCCTrainingMode,
            )


class TestTSTCCModelCleaned:
    """Verify TSTCC model no longer accepts training_mode."""

    def test_model_no_training_mode_param(self) -> None:
        """TSTCC.__init__ no longer accepts training_mode."""
        sig = inspect.signature(TSTCC.__init__)
        assert "training_mode" not in sig.parameters

    def test_only_selfsupervised_contrastive(self) -> None:
        """TSTCC._compute_loss produces contrastive loss (no supervised branch)."""
        # Using L=256, stride=4, kernel=8 to get a large enough seq_len.
        # Encoder forward returns feature map (B, output_dims, L').
        seq_len = 256
        model = TSTCC(input_dims=2, conv_kernel_size=8, stride=4, output_dims=16)
        # Verify forward returns a single tensor (feature map), not a tuple
        test_x = torch.randn(1, 2, seq_len)
        features = model(test_x)
        assert isinstance(features, torch.Tensor)
        assert features.shape == (1, 16, features.shape[2])  # (B, output_dims, L')
        # Now run the contrastive loss
        x = torch.randn(4, 2, seq_len)
        labels = torch.randint(0, 3, (4,))
        batch = (x, labels)
        loss = model._compute_loss(batch)  # noqa: SLF001
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestTSTCCConfigCleaned:
    """Verify TSTCCModelParameters no longer has training_mode."""

    def test_config_no_training_mode_field(self) -> None:
        """TSTCCModelParameters dataclass no longer has training_mode."""
        assert "training_mode" not in TSTCCModelParameters.__dataclass_fields__


class TestTSTCCSupervisedModule:
    """Verify TSTCC downstream via SupervisedModule works."""

    def test_finetuner_classification_shape(self) -> None:
        """make_tstcc_supervised produces (B, num_outputs) classification logits."""
        backbone = TSTCC(input_dims=2, conv_kernel_size=8, stride=4, output_dims=16)
        module = make_tstcc_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=False
        )
        # Verify module construction (head uses backbone.representation_dim)
        assert module._head._fc.in_features == backbone.representation_dim  # noqa: SLF001
        assert module._head._fc.out_features == 5  # noqa: SLF001

    def test_finetuner_training_step(self) -> None:
        """training_step returns scalar loss."""
        backbone = TSTCC(input_dims=2, conv_kernel_size=8, stride=4, output_dims=16)
        module = make_tstcc_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=False
        )
        x = torch.randn(4, 2, 256)
        targets = torch.randint(0, 5, (4,))
        batch = (x, targets)
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_supervised_from_scratch_trains_backbone(self) -> None:
        """From-scratch mode (fresh backbone, freeze_backbone=False) sends grads to the backbone.

        This is the explicit replacement for the removed
        ``TSTCCTrainingMode.SUPERVISED``: an un-pretrained encoder trained
        end-to-end on labels.
        """
        backbone = TSTCC(input_dims=2, conv_kernel_size=8, stride=4, output_dims=16)
        module = make_tstcc_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=False
        )
        batch = (torch.randn(4, 2, 256), torch.randint(0, 5, (4,)))
        loss = module.training_step(batch, 0)
        loss.backward()
        backbone_grads = [p.grad for p in backbone.parameters() if p.requires_grad]
        assert backbone_grads, "backbone should have trainable params when freeze_backbone=False"
        assert any(g is not None for g in backbone_grads)


def test_tcc_encoder_accepts_btc_and_is_transpose_sensitive() -> None:
    """TCCEncoder must accept (B, T, C) input with T != C and return (B, output_dims, L').

    Regression test: without the transpose(1, 2) inside forward(), Conv1d
    sees T channels instead of input_dims and raises RuntimeError.
    """
    encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=1, output_dims=128)
    x = torch.randn(4, 50, 3)  # (B, T, C) with T=50 != C=3
    out = encoder(x)
    assert out.ndim == 3
    assert out.shape[0] == 4
    assert out.shape[1] == 128
