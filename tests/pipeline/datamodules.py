"""Synthetic LightningDataModules mirroring the tsdatasets interface."""

from enum import StrEnum

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.pipeline.synthetic import make_classification, make_forecasting_windows, scale, Scaling

__all__ = ["LoaderMode", "SyntheticClassificationDataModule", "SyntheticForecastingDataModule"]


class LoaderMode(StrEnum):
    """Shape of the batches a classification datamodule yields."""

    SAMPLE = "sample"  # plain x tensor
    SAMPLE_LABEL = "sample_label"  # (x, y)
    TST = "tst"  # (x, y, padding_mask, ids) with NaN zero-filled; mask True = valid


def _loader(
    *, tensors: tuple[torch.Tensor, ...], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    """Build a DataLoader; a single tensor is served as plain batches (no tuple)."""
    dataset = tensors[0] if len(tensors) == 1 else TensorDataset(*tensors)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


class SyntheticClassificationDataModule(pl.LightningDataModule):
    """UCR/UEA-like synthetic classification data.

    Args:
        n_train: Training samples.
        n_val: Validation samples.
        n_test: Test samples.
        seq_len: Series length.
        n_channels: Number of channels.
        n_classes: Number of classes.
        batch_size: Batch size for all loaders.
        scaling: Scaling applied to all splits jointly.
        nan_pad_fraction: Fraction of series with NaN-padded tails.
        loader_mode: Batch layout, see :class:`LoaderMode`.
        seed: Seed for generation and shuffling.
    """

    def __init__(
        self,
        *,
        n_train: int = 120,
        n_val: int = 40,
        n_test: int = 80,
        seq_len: int = 64,
        n_channels: int = 2,
        n_classes: int = 3,
        batch_size: int = 16,
        scaling: Scaling = Scaling.ZSCORE,
        nan_pad_fraction: float = 0.0,
        loader_mode: LoaderMode = LoaderMode.SAMPLE_LABEL,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._sizes = (n_train, n_val, n_test)
        self._seq_len = seq_len
        self._n_channels = n_channels
        self._n_classes = n_classes
        self.batch_size = batch_size
        self._scaling = scaling
        self._nan_pad_fraction = nan_pad_fraction
        self._loader_mode = LoaderMode(loader_mode)
        self._seed = seed
        self._splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def num_features(self) -> int:
        """Number of channels."""
        return self._n_channels

    @property
    def sequence_length(self) -> int:
        """Series length."""
        return self._seq_len

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self._n_classes

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Generate, scale and split the data once."""
        if self._splits:
            return
        x, y = make_classification(
            n_samples=sum(self._sizes),
            seq_len=self._seq_len,
            n_channels=self._n_channels,
            n_classes=self._n_classes,
            nan_pad_fraction=self._nan_pad_fraction,
            seed=self._seed,
        )
        x = scale(x=x, method=self._scaling)
        train_end, val_end = self._sizes[0], self._sizes[0] + self._sizes[1]
        self._splits = {
            "train": (x[:train_end], y[:train_end]),
            "val": (x[train_end:val_end], y[train_end:val_end]),
            "test": (x[val_end:], y[val_end:]),
        }

    def train_arrays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw ``(x, y)`` training tensors (NaN padding preserved)."""
        return self._splits["train"]

    def test_arrays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw ``(x, y)`` test tensors (NaN padding preserved)."""
        return self._splits["test"]

    def _batch_tensors(self, split: str) -> tuple[torch.Tensor, ...]:
        x, y = self._splits[split]
        if self._loader_mode is LoaderMode.SAMPLE:
            return (x,)
        if self._loader_mode is LoaderMode.SAMPLE_LABEL:
            return (x, y)
        padding_mask = ~torch.isnan(x).any(dim=-1)
        return (torch.nan_to_num(x, nan=0.0), y, padding_mask, torch.arange(len(x)))

    def _split_loader(self, *, split: str, shuffle: bool) -> DataLoader:
        return _loader(
            tensors=self._batch_tensors(split),
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self._seed,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training loader."""
        return self._split_loader(split="train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation loader."""
        return self._split_loader(split="val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader."""
        return self._split_loader(split="test", shuffle=False)


class SyntheticForecastingDataModule(pl.LightningDataModule):
    """ETT/electricity-like synthetic windows; batches are plain ``x`` tensors.

    Splits are chronological slices of one long series.

    Args:
        n_train: Training windows.
        n_val: Validation windows.
        n_test: Test windows.
        seq_len: Window length.
        n_channels: Number of channels.
        batch_size: Batch size for all loaders (1 reproduces the single-series case).
        scaling: Scaling applied to all windows jointly.
        seed: Seed for generation and shuffling.
    """

    def __init__(
        self,
        *,
        n_train: int = 160,
        n_val: int = 40,
        n_test: int = 40,
        seq_len: int = 64,
        n_channels: int = 2,
        batch_size: int = 16,
        scaling: Scaling = Scaling.ZSCORE,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._sizes = (n_train, n_val, n_test)
        self._seq_len = seq_len
        self._n_channels = n_channels
        self.batch_size = batch_size
        self._scaling = scaling
        self._seed = seed
        self._splits: dict[str, torch.Tensor] = {}

    @property
    def num_features(self) -> int:
        """Number of channels."""
        return self._n_channels

    @property
    def sequence_length(self) -> int:
        """Window length."""
        return self._seq_len

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Generate, scale and split the windows once."""
        if self._splits:
            return
        windows = make_forecasting_windows(
            n_windows=sum(self._sizes),
            seq_len=self._seq_len,
            n_channels=self._n_channels,
            seed=self._seed,
        )
        windows = scale(x=windows, method=self._scaling)
        train_end, val_end = self._sizes[0], self._sizes[0] + self._sizes[1]
        self._splits = {
            "train": windows[:train_end],
            "val": windows[train_end:val_end],
            "test": windows[val_end:],
        }

    def _split_loader(self, *, split: str, shuffle: bool) -> DataLoader:
        return _loader(
            tensors=(self._splits[split],),
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self._seed,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training loader."""
        return self._split_loader(split="train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation loader."""
        return self._split_loader(split="val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader."""
        return self._split_loader(split="test", shuffle=False)
