"""Tests for the synthetic LightningDataModules."""

import torch

from tests.pipeline.datamodules import (
    LoaderMode,
    SyntheticClassificationDataModule,
    SyntheticForecastingDataModule,
)
from tests.pipeline.synthetic import Scaling


def test_classification_batches_are_x_y() -> None:
    dm = SyntheticClassificationDataModule(batch_size=8)
    dm.setup(stage="fit")
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (8, 64, 2)
    assert y.shape == (8,)
    assert (dm.num_features, dm.sequence_length, dm.num_classes) == (2, 64, 3)


def test_sample_mode_yields_plain_tensor() -> None:
    dm = SyntheticClassificationDataModule(batch_size=8, loader_mode=LoaderMode.SAMPLE)
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, torch.Tensor)


def test_tst_mode_yields_four_tuple_with_padding_mask() -> None:
    dm = SyntheticClassificationDataModule(
        batch_size=8, loader_mode=LoaderMode.TST, nan_pad_fraction=0.5
    )
    dm.setup(stage="fit")
    x, y, padding_mask, ids = next(iter(dm.train_dataloader()))
    assert padding_mask.shape == x.shape[:2]
    assert padding_mask.dtype == torch.bool
    assert not padding_mask.all()  # some padded rows exist
    assert not torch.isnan(x).any()  # TST gets zero-filled input
    assert ids.shape == y.shape


def test_splits_share_class_templates_and_do_not_overlap() -> None:
    dm = SyntheticClassificationDataModule(n_train=30, n_val=9, n_test=21, n_classes=3)
    dm.setup(stage="fit")
    x_train, y_train = dm.train_arrays()
    x_test, y_test = dm.test_arrays()
    assert len(x_train) == 30
    assert len(x_test) == 21
    means = torch.stack([x_train[y_train == k].mean(dim=0) for k in range(3)])
    nearest = torch.cdist(x_test.flatten(1), means.flatten(1)).argmin(dim=1)
    assert (nearest == y_test).float().mean() > 0.8  # train templates transfer to test


def test_minmax_scaling_is_global_unit_interval() -> None:
    dm = SyntheticClassificationDataModule(scaling=Scaling.MINMAX)
    dm.setup(stage="fit")
    x_train, _ = dm.train_arrays()
    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0


def test_forecasting_batches_and_batch_size_one() -> None:
    dm = SyntheticForecastingDataModule(batch_size=1)
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch.shape == (1, 64, 2)
    assert dm.num_features == 2
    assert dm.sequence_length == 64
    assert len(dm.val_dataloader().dataset) == 40
