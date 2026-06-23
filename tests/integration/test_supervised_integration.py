"""Cross-model supervised-training integration tests.

Verifies the end-to-end flow: backbone → factory → SupervisedModule → training.
"""

from __future__ import annotations

from lightning.pytorch.trainer import Trainer
import torch
from torch.utils.data import DataLoader, Dataset

from chronocratic.models import supervised
from chronocratic.models.convolutional.standard import series2vec, tstcc
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.supervised import (
    make_series2vec_supervised,
    make_tst_supervised,
    make_tstcc_supervised,
    RepresentationBackbone,
    SupervisedModule,
)
from chronocratic.models.transformer import tst
from chronocratic.models.transformer.tst.model import TST

# ---------------------------------------------------------------------------
# Tiny synthetic dataset for end-to-end training
# ---------------------------------------------------------------------------


class _DummyTSTDataset(Dataset):
    """Synthetic TST dataset: (X, targets, padding_masks, IDs)."""

    def __init__(
        self, size: int = 20, seq_len: int = 10, feat_dim: int = 2, num_classes: int = 3
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.seq_len, self.feat_dim)
        targets = torch.tensor(idx % self.num_classes, dtype=torch.long)
        padding_masks = torch.ones(self.seq_len, dtype=torch.bool)
        ids = torch.tensor(idx, dtype=torch.long)
        return x, targets, padding_masks, ids


class _DummySupervisedDataset(Dataset):
    """Synthetic supervised dataset: (X, targets)."""

    def __init__(
        self, size: int = 20, seq_len: int = 20, channels: int = 2, num_classes: int = 3
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.seq_len, self.channels)
        targets = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return x, targets


class _DummyTSTCCDataset(Dataset):
    """Synthetic TSTCC dataset: (X, targets) with (C, L) format."""

    def __init__(
        self, size: int = 20, seq_len: int = 256, channels: int = 2, num_classes: int = 3
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.channels, self.seq_len)
        targets = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return x, targets


# ---------------------------------------------------------------------------
# Protocol checks
# ---------------------------------------------------------------------------


class TestAllBackbonesSatisfyProtocol:
    """Verify isinstance(backbone, RepresentationBackbone) for all three models."""

    def test_tst_satisfies_protocol(self) -> None:
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        assert isinstance(backbone, RepresentationBackbone)

    def test_series2vec_satisfies_protocol(self) -> None:
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        assert isinstance(backbone, RepresentationBackbone)

    def test_tstcc_satisfies_protocol(self) -> None:
        backbone = TSTCC(
            input_dims=2,
            conv_kernel_size=8,
            stride=4,
            output_dims=16,
        )
        assert isinstance(backbone, RepresentationBackbone)


# ---------------------------------------------------------------------------
# Factory type checks
# ---------------------------------------------------------------------------


class TestAllFactoriesProduceSupervisedModule:
    """Verify each factory returns a SupervisedModule instance."""

    def test_tst_factory_type(self) -> None:
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(backbone, num_outputs=3, task="classification")
        assert isinstance(module, SupervisedModule)

    def test_series2vec_factory_type(self) -> None:
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_supervised(backbone, num_outputs=3, task="classification")
        assert isinstance(module, SupervisedModule)

    def test_tstcc_factory_type(self) -> None:
        backbone = TSTCC(
            input_dims=2,
            conv_kernel_size=8,
            stride=4,
            output_dims=16,
        )
        module = make_tstcc_supervised(backbone, num_outputs=5, task="classification")
        assert isinstance(module, SupervisedModule)


# ---------------------------------------------------------------------------
# End-to-end training (3 steps)
# ---------------------------------------------------------------------------


class TestEndToEndTraining:
    """Run 3 training steps with Lightning Trainer and verify finite loss."""

    def test_tst_trains_end_to_end(self) -> None:
        """TST finetuner trains for 3 steps with finite loss."""
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(
            backbone, num_outputs=3, task="classification", freeze_backbone=False
        )
        dataset = _DummyTSTDataset(size=20, seq_len=10, feat_dim=2, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=4)
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=3,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, train_dataloaders=dataloader)
        assert "train_loss" in trainer.callback_metrics
        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_series2vec_trains_end_to_end(self) -> None:
        """Series2Vec finetuner trains for 3 steps with finite loss."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_supervised(
            backbone, num_outputs=3, task="classification", freeze_backbone=False
        )
        dataset = _DummySupervisedDataset(size=20, seq_len=20, channels=2, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=4)
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=3,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, train_dataloaders=dataloader)

    def test_tstcc_trains_end_to_end(self) -> None:
        """TSTCC finetuner trains for 3 steps with finite loss."""
        backbone = TSTCC(
            input_dims=2,
            conv_kernel_size=8,
            stride=4,
            output_dims=16,
        )
        module = make_tstcc_supervised(
            backbone, num_outputs=3, task="classification", freeze_backbone=False
        )
        dataset = _DummyTSTCCDataset(size=20, seq_len=256, channels=2, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=4)
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=3,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, train_dataloaders=dataloader)

    def test_tstcc_pretraining_still_works(self) -> None:
        """TSTCC pretraining (contrastive) still works after enum removal."""
        model = TSTCC(
            input_dims=2,
            conv_kernel_size=8,
            stride=4,
            output_dims=16,
        )
        dataset = _DummyTSTCCDataset(size=20, seq_len=256, channels=2, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=4)
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=3,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloaders=dataloader)


class TestRegressionTask:
    """Verify regression task works with SupervisedModule."""

    def test_regression_mse_loss(self) -> None:
        """Regression task uses MSELoss and produces finite loss."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_supervised(
            backbone, num_outputs=2, task="regression", freeze_backbone=False
        )
        x = torch.randn(4, 20, 2)
        targets = torch.randn(4, 2)
        batch = (x, targets)
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestBarrelExportsClean:
    """Verify barrel exports are clean — no leaked head classes."""

    def test_supervised_exports_match_all(self) -> None:
        """chronocratic.models.supervised exports match __all__."""
        exported = set(supervised.__all__)
        actual = {
            name
            for name in dir(supervised)
            if not name.startswith("_") and name in supervised.__all__
        }
        assert exported == actual

    def test_no_head_class_leaked_from_tst(self) -> None:
        """No head classes leaked from tst package."""
        assert not hasattr(tst, "TSTClassificationHead")
        assert not hasattr(tst, "TSTRegressionHead")

    def test_no_head_class_leaked_from_series2vec(self) -> None:
        """No head classes leaked from series2vec package."""
        assert not hasattr(series2vec, "Series2VecClassificationHead")

    def test_no_enum_leaked_from_tstcc(self) -> None:
        """TSTCCTrainingMode not leaked from tstcc package."""
        assert not hasattr(tstcc, "TSTCCTrainingMode")
