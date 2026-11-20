"""Tests for ``num_covariate_channels`` across TimeNet, RecurrentAE, TimeVAE and TST."""

from collections.abc import Callable

import pytest
import torch

from chronocratic.models.generative.timevae.model import TimeVAE
from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder
from chronocratic.models.recurrent.timenet.model import TimeNet
from chronocratic.models.transformer.tst.model import TST
from chronocratic.models.utils import (
    masked_reconstruction_loss_mean,
    masked_reconstruction_loss_sum,
    validate_num_covariate_channels,
    zero_fill_padding,
)

B, T, C = 4, 16, 8
KS = (3, 7)


def _make_timenet(*, k: int) -> TimeNet:
    return TimeNet(input_dim=C, hidden_dim=8, depth=2, num_covariate_channels=k).eval()


def _make_recurrentae(*, k: int) -> RecurrentAutoEncoder:
    return RecurrentAutoEncoder(input_dim=C, layers=(8, 4), num_covariate_channels=k).eval()


def _make_timevae(*, k: int) -> TimeVAE:
    return TimeVAE(
        sequence_length=T, input_dim=C, hidden_layer_sizes=(8, 16), num_covariate_channels=k
    ).eval()


def _make_tst(*, k: int) -> TST:
    return TST(
        input_dim=C,
        sequence_length=T,
        hidden_dim=16,
        num_heads=2,
        depth=1,
        feedforward_dim=32,
        num_covariate_channels=k,
    ).eval()


def _data(*, seed: int = 0, nan_rows: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an input batch (optionally NaN-padded) and a fixed model output."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, C, generator=gen)
    output = torch.randn(B, T, C, generator=gen)
    if nan_rows:
        x[1, 10:] = float("nan")
    return x, output


def _perturb(*, x: torch.Tensor, channels: slice) -> torch.Tensor:
    out = x.clone()
    out[..., channels] += 5.0
    return out


RECURRENT_FACTORIES = [_make_timenet, _make_recurrentae]


@pytest.mark.parametrize("factory", RECURRENT_FACTORIES)
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("nan_rows", [False, True])
def test_recurrent_loss_ignores_covariates(
    factory: Callable[..., torch.nn.Module], k: int, nan_rows: bool
) -> None:
    model = factory(k=k)
    x, output = _data(nan_rows=nan_rows)
    model.forward = lambda _x: output
    base = model._compute_masked_loss(x)
    cov_changed = model._compute_masked_loss(_perturb(x=x, channels=slice(0, k)))
    raw_changed = model._compute_masked_loss(_perturb(x=x, channels=slice(k, None)))
    assert torch.equal(base, cov_changed)
    assert not torch.equal(base, raw_changed)
    assert torch.isfinite(base)


@pytest.mark.parametrize("factory", RECURRENT_FACTORIES)
@pytest.mark.parametrize("nan_rows", [False, True])
def test_recurrent_default_matches_reference(
    factory: Callable[..., torch.nn.Module], nan_rows: bool
) -> None:
    model = factory(k=0)
    x, output = _data(nan_rows=nan_rows)
    model.forward = lambda _x: output
    x_filled, keep = zero_fill_padding(x)
    per_element = (output - x_filled) ** 2
    expected = masked_reconstruction_loss_mean(per_element, keep)
    assert torch.equal(model._compute_masked_loss(x), expected)


def test_recurrentae_l1_branch_ignores_covariates() -> None:
    model = RecurrentAutoEncoder(
        input_dim=C, layers=(8, 4), loss="mae", num_covariate_channels=3
    ).eval()
    x, output = _data(nan_rows=True)
    model.forward = lambda _x: output
    base = model._compute_masked_loss(x)
    assert torch.equal(base, model._compute_masked_loss(_perturb(x=x, channels=slice(0, 3))))
    assert not torch.equal(base, model._compute_masked_loss(_perturb(x=x, channels=slice(3, None))))


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("nan_rows", [False, True])
def test_timevae_loss_ignores_covariates(k: int, nan_rows: bool) -> None:
    model = _make_timevae(k=k)
    x, recons = _data(nan_rows=nan_rows)
    x_filled, keep = zero_fill_padding(x)
    base = model._get_reconstruction_loss(x=x_filled, x_recons=recons, keep_mask=keep)
    for target in (x_filled, recons):
        changed = _perturb(x=target, channels=slice(0, k))
        kwargs = {"x": x_filled, "x_recons": recons}
        kwargs["x" if target is x_filled else "x_recons"] = changed
        assert torch.equal(model._get_reconstruction_loss(keep_mask=keep, **kwargs), base)
    raw = _perturb(x=x_filled, channels=slice(k, None))
    assert not torch.equal(
        model._get_reconstruction_loss(x=raw, x_recons=recons, keep_mask=keep), base
    )


@pytest.mark.parametrize("nan_rows", [False, True])
def test_timevae_default_matches_reference(nan_rows: bool) -> None:
    model = _make_timevae(k=0)
    x, recons = _data(nan_rows=nan_rows)
    x_filled, keep = zero_fill_padding(x)
    expected = masked_reconstruction_loss_sum(torch.pow(x_filled - recons, 2), keep)
    expected += torch.sum(torch.pow(x_filled.mean(dim=2) - recons.mean(dim=2), 2))
    got = model._get_reconstruction_loss(x=x_filled, x_recons=recons, keep_mask=keep)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("k", KS)
def test_tst_never_masks_or_scores_covariates(k: int) -> None:
    model = _make_tst(k=k)
    x, _ = _data(nan_rows=True)
    torch.manual_seed(0)
    masked_x, targets, target_masks, padding_masks = model._make_masked_inputs(x)
    assert not target_masks[..., :k].any()
    assert torch.equal(masked_x[..., :k], targets[..., :k])
    assert target_masks[..., k:].any()
    assert not padding_masks[1, 10:].any()

    predictions = torch.randn_like(targets)
    combined = target_masks & padding_masks.unsqueeze(-1)
    base = model._loss_fn(predictions, targets, combined)
    assert torch.equal(
        base, model._loss_fn(_perturb(x=predictions, channels=slice(0, k)), targets, combined)
    )
    assert not torch.equal(
        base, model._loss_fn(_perturb(x=predictions, channels=slice(k, None)), targets, combined)
    )


def test_tst_default_keep_matches_plain_draw() -> None:
    model = _make_tst(k=0)
    x, _ = _data()
    torch.manual_seed(7)
    masked_x, targets, target_masks, _ = model._make_masked_inputs(x)
    torch.manual_seed(7)
    keep = torch.rand(x.shape) >= model._masking_ratio
    assert torch.equal(target_masks, ~keep)
    assert torch.equal(masked_x, targets * keep)


def test_tst_rng_stream_unchanged_by_covariates() -> None:
    x, _ = _data()
    torch.manual_seed(3)
    _, _, masks_0, _ = _make_tst(k=0)._make_masked_inputs(x)
    torch.manual_seed(3)
    _, _, masks_k, _ = _make_tst(k=3)._make_masked_inputs(x)
    assert torch.equal(masks_0[..., 3:], masks_k[..., 3:])


@pytest.mark.parametrize("factory", [_make_timenet, _make_recurrentae, _make_timevae, _make_tst])
@pytest.mark.parametrize("bad", [-1, C])
def test_invalid_num_covariate_channels_raises(
    factory: Callable[..., torch.nn.Module], bad: int
) -> None:
    with pytest.raises(ValueError, match="num_covariate_channels"):
        factory(k=bad)


@pytest.mark.parametrize("bad", [-1, 5])
def test_validate_helper_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError, match="num_covariate_channels"):
        validate_num_covariate_channels(num_covariate_channels=bad, input_dim=5)


def test_validate_helper_returns_value() -> None:
    assert validate_num_covariate_channels(num_covariate_channels=4, input_dim=5) == 4
    assert validate_num_covariate_channels(num_covariate_channels=0, input_dim=1) == 0
