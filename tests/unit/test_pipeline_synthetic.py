"""Tests for synthetic data generators."""

import torch

from tests.pipeline.synthetic import make_classification, make_forecasting_windows, scale, Scaling


def test_classification_shapes_and_balance() -> None:
    x, y = make_classification(n_samples=60, seq_len=64, n_channels=2, n_classes=3, seed=0)
    assert x.shape == (60, 64, 2)
    assert x.dtype == torch.float32
    assert torch.bincount(y).tolist() == [20, 20, 20]


def test_classification_is_deterministic() -> None:
    kwargs = {"n_samples": 12, "seq_len": 32, "n_channels": 1, "n_classes": 2, "seed": 3}
    x1, y1 = make_classification(**kwargs)
    x2, y2 = make_classification(**kwargs)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_classes_are_separable_by_class_mean_template() -> None:
    """Nearest-class-mean must beat chance by a wide margin, else there is no signal."""
    x, y = make_classification(n_samples=300, seq_len=64, n_channels=2, n_classes=3, seed=1)
    means = torch.stack([x[y == k].mean(dim=0) for k in range(3)])
    distances = torch.cdist(x.flatten(1), means.flatten(1))
    assert (distances.argmin(dim=1) == y).float().mean() > 0.8


def test_nan_padding_only_at_the_tail() -> None:
    x, _ = make_classification(
        n_samples=40, seq_len=64, n_channels=2, n_classes=2, nan_pad_fraction=0.5, seed=0
    )
    nan_rows = torch.isnan(x).any(dim=(1, 2))
    assert 0 < nan_rows.sum() < 40
    for sample in x[nan_rows]:
        valid = ~torch.isnan(sample).any(dim=-1)
        first_nan = int((~valid).nonzero()[0])
        assert not valid[first_nan:].any()  # once NaN starts it never stops


def test_forecasting_windows_shape_and_signal() -> None:
    windows = make_forecasting_windows(n_windows=50, seq_len=64, n_channels=3, seed=0)
    assert windows.shape == (50, 64, 3)
    assert torch.isfinite(windows).all()
    assert windows.std(dim=1).mean() > 0.1  # not constant


def test_scale_zscore_is_nan_aware_and_per_series() -> None:
    x = torch.randn(4, 20, 2) * 5 + 3
    x[0, 15:] = float("nan")
    scaled = scale(x=x, method=Scaling.ZSCORE)
    assert torch.isnan(scaled[0, 15:]).all()
    assert torch.allclose(torch.nanmean(scaled[1], dim=0), torch.zeros(2), atol=1e-4)


def test_scale_minmax_maps_to_unit_interval_globally() -> None:
    x = torch.randn(4, 20, 2) * 5 + 3
    scaled = scale(x=x, method=Scaling.MINMAX)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_scale_none_is_identity() -> None:
    x = torch.randn(2, 10, 1)
    assert torch.equal(scale(x=x, method=Scaling.NONE), x)
