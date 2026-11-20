"""Tests for ``split_into_sections`` and its use in TS2Vec, CoST and AutoTCL."""

from collections.abc import Callable
import importlib

import numpy as np
import pytest
import torch

from chronocratic.models import AutoTCL, CoST, TS2Vec
from chronocratic.models.utils import split_into_sections


# --- Reference copies of the upstream helpers (TS2Vec utils.py, byte-identical in CoST and
# --- AutoTCL) and of the fit() preprocessing block (ts2vec.py:77-86).
def pad_nan_to_target(array, target_length, axis=0, both_side=False):  # noqa: ANN001, ANN201
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (pad_size // 2, pad_size - pad_size // 2) if both_side else (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):  # noqa: ANN001, ANN201
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    return [pad_nan_to_target(a, target_length, axis=axis) for a in arrs]


def centerize_vary_length_series(x):  # noqa: ANN001, ANN201
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def reference_preprocess(train_data: np.ndarray, max_train_length: int | None) -> np.ndarray:
    """Upstream ``fit()`` block, verbatim."""
    if max_train_length is not None:
        sections = train_data.shape[1] // max_train_length
        if sections >= 2:
            train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)
    return train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]


def _random(*, batch: int, length: int, channels: int = 3) -> torch.Tensor:
    return torch.randn(batch, length, channels, generator=torch.Generator().manual_seed(0))


def _with_padding(*, lead: int, trail: int) -> torch.Tensor:
    x = _random(batch=4, length=40)
    x[0, :lead] = torch.nan
    x[1, 40 - trail :] = torch.nan
    x[2, : lead + 3] = torch.nan
    x[2, 40 - trail :] = torch.nan
    return x


def _all_nan_row() -> torch.Tensor:
    x = _random(batch=3, length=40)
    x[1] = torch.nan
    return x


CASES: dict[str, tuple[Callable[[], torch.Tensor], int | None]] = {
    "divisible": (lambda: _random(batch=3, length=40), 10),
    "not_divisible": (lambda: _random(batch=2, length=47), 10),
    "leading_trailing_nan": (lambda: _with_padding(lead=5, trail=4), None),
    "leading_trailing_nan_split": (lambda: _with_padding(lead=5, trail=4), 10),
    "all_nan_row": (_all_nan_row, None),
    "all_nan_row_split": (_all_nan_row, 10),
    "max_length_none": (lambda: _random(batch=2, length=40), None),
    "fewer_than_two_sections": (lambda: _random(batch=2, length=19), 10),
}


@pytest.mark.parametrize("case", list(CASES))
def test_matches_upstream_reference(case: str) -> None:
    make, max_train_length = CASES[case]
    x = make()
    expected = reference_preprocess(x.numpy().copy(), max_train_length)
    actual = split_into_sections(x, max_train_length=max_train_length)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize("max_train_length", [None, 30])
def test_no_nan_no_split_returns_same_object(max_train_length: int | None) -> None:
    x = _random(batch=2, length=40)
    assert split_into_sections(x, max_train_length=max_train_length) is x


def test_all_rows_dropped_returns_input_unchanged() -> None:
    x = torch.full((2, 8, 3), torch.nan)
    assert split_into_sections(x, max_train_length=None) is x


def test_split_batch_is_piece_major() -> None:
    x = _random(batch=3, length=40)
    out = split_into_sections(x, max_train_length=10)
    assert out.shape == (12, 10, 3)
    torch.testing.assert_close(out[3:6], x[:, 10:20])


def _record_batch_sizes(monkeypatch: pytest.MonkeyPatch, *, module: str, name: str) -> list:
    """Wrap ``module.name`` to record the batch size of ``instance_1`` and its value."""
    mod = importlib.import_module(module)
    original = getattr(mod, name)
    calls: list[tuple[int, float]] = []

    def spy(*, instance_1: torch.Tensor, instance_2: torch.Tensor) -> torch.Tensor:
        value = original(instance_1=instance_1, instance_2=instance_2)
        calls.append((instance_1.size(0), float(value.detach())))
        return value

    monkeypatch.setattr(mod, name, spy)
    return calls


def _drive(model, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    """Run ``training_step`` without a Trainer by stubbing the Lightning hooks."""
    model.train()
    optimizers = model.configure_optimizers()
    model.optimizers = lambda: optimizers  # type: ignore[method-assign]
    model.log = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    model.manual_backward = lambda loss: loss.backward()  # type: ignore[method-assign]
    return model.training_step(x, 0)


def _ts2vec(max_train_length: int | None) -> TS2Vec:
    return TS2Vec(
        input_dim=3, hidden_dim=8, representation_dim=16, depth=2, max_train_length=max_train_length
    )


def _cost(max_train_length: int) -> CoST:
    return CoST(
        input_dim=3,
        sequence_length=64,
        hidden_dim=8,
        representation_dim=16,
        depth=2,
        max_train_length=max_train_length,
        queue_size=8,
    )


def _autotcl(max_train_length: int | None) -> AutoTCL:
    return AutoTCL(
        input_dim=3, hidden_dim=8, representation_dim=16, depth=2, max_train_length=max_train_length
    )


def test_ts2vec_single_series_becomes_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_batch_sizes(
        monkeypatch,
        module="chronocratic.models.losses.contrastive",
        name="instance_contrastive_loss",
    )
    _drive(_ts2vec(16), _random(batch=1, length=64))
    assert calls
    assert {size for size, _ in calls} == {4}
    assert any(value != 0.0 for _, value in calls)


def test_cost_single_series_becomes_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_batch_sizes(
        monkeypatch,
        module="chronocratic.models.convolutional.dilated.cost.model",
        name="instance_contrastive_loss",
    )
    _drive(_cost(16), _random(batch=1, length=64))
    assert calls
    assert {size for size, _ in calls} == {4}
    assert any(value != 0.0 for _, value in calls)


def test_autotcl_single_series_becomes_sections() -> None:
    model = _autotcl(16)
    sizes: list[int] = []
    model._encoder.register_forward_pre_hook(lambda _m, args: sizes.append(args[0].size(0)))
    loss = _drive(model, _random(batch=1, length=64))
    assert set(sizes) == {4}
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("module", "make"), [("ts2vec", lambda: _ts2vec(None)), ("autotcl", lambda: _autotcl(None))]
)
def test_max_train_length_none_is_identity(
    monkeypatch: pytest.MonkeyPatch, module: str, make: Callable[[], torch.nn.Module]
) -> None:
    x = _random(batch=2, length=40)
    torch.manual_seed(1)
    model = make()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(2)
    with_helper = _drive(model, x)

    mod = importlib.import_module(f"chronocratic.models.convolutional.dilated.{module}.model")
    monkeypatch.setattr(mod, "split_into_sections", lambda x, *, max_train_length: x)  # noqa: ARG005
    model.load_state_dict(state)
    torch.manual_seed(2)
    without_helper = _drive(model, x)
    torch.testing.assert_close(with_helper, without_helper, rtol=0, atol=0)
