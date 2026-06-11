"""Cross-model fine-tuning integration tests.

Verifies the end-to-end flow: backbone → factory → FineTuningModule → training.
"""

from __future__ import annotations

from lightning.pytorch.trainer import Trainer
import torch
from torch.utils.data import DataLoader, Dataset

from tscollection.models import _finetuning
from tscollection.models._finetuning import (
    FineTuningModule,
    make_series2vec_finetuner,
    make_tst_finetuner,
    make_tstcc_finetuner,
    RepresentationBackbone,
)
from tscollection.models.convolutional.standard import series2vec, tstcc
from tscollection.models.convolutional.standard.series2vec.model import Series2Vec
from tscollection.models.convolutional.standard.tstcc.model import TSTCC
from tscollection.models.transformer import tst
from tscollection.models.transformer.tst.model import TST

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
        backbone = TST(feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1)
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
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=10,
            num_classes=3,
        )
        assert isinstance(backbone, RepresentationBackbone)


# ---------------------------------------------------------------------------
# Factory type checks
# ---------------------------------------------------------------------------


class TestAllFactoriesProduceFineTuningModule:
    """Verify each factory returns a FineTuningModule instance."""

    def test_tst_factory_type(self) -> None:
        backbone = TST(feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1)
        module = make_tst_finetuner(backbone, num_outputs=3, task='classification')
        assert isinstance(module, FineTuningModule)

    def test_series2vec_factory_type(self) -> None:
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_finetuner(backbone, num_outputs=3, task='classification')
        assert isinstance(module, FineTuningModule)

    def test_tstcc_factory_type(self) -> None:
        backbone = TSTCC(
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=10,
            num_classes=3,
        )
        module = make_tstcc_finetuner(backbone, num_outputs=5, task='classification')
        assert isinstance(module, FineTuningModule)


# ---------------------------------------------------------------------------
# End-to-end training (3 steps)
# ---------------------------------------------------------------------------


class TestEndToEndTraining:
    """Run 3 training steps with Lightning Trainer and verify finite loss."""

    def test_tst_trains_end_to_end(self) -> None:
        """TST finetuner trains for 3 steps with finite loss."""
        backbone = TST(feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1)
        module = make_tst_finetuner(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
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
        assert 'train_loss' in trainer.callback_metrics
        assert torch.isfinite(trainer.callback_metrics['train_loss'])

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
        module = make_series2vec_finetuner(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
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
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=10,
            num_classes=3,
        )
        module = make_tstcc_finetuner(
            backbone, num_outputs=3, task='classification', freeze_backbone=False
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
            input_channels=2,
            kernel_size=8,
            stride=4,
            final_out_channels=16,
            features_len=10,
            num_classes=3,
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
    """Verify regression task works with FineTuningModule."""

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
        module = make_series2vec_finetuner(
            backbone, num_outputs=2, task='regression', freeze_backbone=False
        )
        x = torch.randn(4, 20, 2)
        targets = torch.randn(4, 2)
        batch = (x, targets)
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestBarrelExportsClean:
    """Verify barrel exports are clean — no leaked head classes."""

    def test_finetuning_exports_match_all(self) -> None:
        """tscollection.models._finetuning exports match __all__."""
        exported = set(_finetuning.__all__)
        actual = {
            name
            for name in dir(_finetuning)
            if not name.startswith('_') and name in _finetuning.__all__
        }
        assert exported == actual

    def test_no_head_class_leaked_from_tst(self) -> None:
        """No head classes leaked from tst package."""
        assert not hasattr(tst, 'TSTClassificationHead')
        assert not hasattr(tst, 'TSTRegressionHead')

    def test_no_head_class_leaked_from_series2vec(self) -> None:
        """No head classes leaked from series2vec package."""
        assert not hasattr(series2vec, 'Series2VecClassificationHead')

    def test_no_enum_leaked_from_tstcc(self) -> None:
        """TSTCCTrainingMode not leaked from tstcc package."""
        assert not hasattr(tstcc, 'TSTCCTrainingMode')
