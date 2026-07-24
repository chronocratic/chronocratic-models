"""NaN-padded input defense for dilated encoding mixins.

Verifies that PoolingEncodingMixin (AutoTCL, TS2Vec) and
DecompositionEncodingMixin (CoST) produce finite representations when
encode(), encode_batch(), or sliding-window inference receive
variable-length series padded with trailing NaN timesteps.
"""

import pytest
import torch

from chronocratic.models.convolutional.dilated.autotcl.model import AutoTCL
from chronocratic.models.convolutional.dilated.cost.model import CoST
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec
from chronocratic.models.enums.encoding import EncodingOutputShape


def _make_nan_padded(shape: tuple[int, int, int]) -> torch.Tensor:
    """Create a tensor with NaN padding on the last few timesteps."""
    x = torch.randn(*shape)
    x[:, -3:, :] = float("nan")
    return x


@pytest.fixture
def autotcl() -> AutoTCL:
    """Small AutoTCL for fast testing."""
    return AutoTCL(
        input_dim=3,
        kernel_sizes=(1, 2, 4),
        hidden_dim=16,
        representation_dim=16,
        depth=2,
        conv_kernel_size=3,
    )


@pytest.fixture
def ts2vec() -> TS2Vec:
    """Small TS2Vec for fast testing."""
    return TS2Vec(input_dim=3, hidden_dim=16, representation_dim=16, depth=2, conv_kernel_size=3)


@pytest.fixture
def cost() -> CoST:
    """Small CoST for fast testing."""
    return CoST(
        input_dim=3,
        representation_dim=16,
        kernel_sizes=(1, 2, 4),
        sequence_length=32,
        hidden_dim=16,
        depth=2,
        conv_kernel_size=3,
    )


# ---------------------------------------------------------------------------
# AutoTCL — encode() / encode_batch() NaN tests
# ---------------------------------------------------------------------------


def test_autotcl_encode_nan_padded_vector(autotcl: AutoTCL) -> None:
    """AutoTCL encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((2, 32, 3))
    reps = autotcl.encode(data, batch_size=2, num_workers=0)
    assert torch.isfinite(reps).all(), "AutoTCL encode() VECTOR contains NaN/Inf"


def test_autotcl_encode_nan_padded_sequence(autotcl: AutoTCL) -> None:
    """AutoTCL encode() on NaN-padded input returns finite SEQUENCE."""
    data = _make_nan_padded((2, 32, 3))
    reps = autotcl.encode(data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE)
    assert torch.isfinite(reps).all(), "AutoTCL encode() SEQUENCE contains NaN/Inf"
    assert reps.ndim == 3


def test_autotcl_encode_batch_nan_padded(autotcl: AutoTCL) -> None:
    """AutoTCL encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((2, 32, 3))
    reps = autotcl.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "AutoTCL encode_batch() contains NaN/Inf"


# ---------------------------------------------------------------------------
# TS2Vec — encode() / encode_batch() NaN tests
# ---------------------------------------------------------------------------


def test_ts2vec_encode_nan_padded_vector(ts2vec: TS2Vec) -> None:
    """TS2Vec encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((2, 32, 3))
    reps = ts2vec.encode(data, batch_size=2, num_workers=0)
    assert torch.isfinite(reps).all(), "TS2Vec encode() VECTOR contains NaN/Inf"


def test_ts2vec_encode_nan_padded_sequence(ts2vec: TS2Vec) -> None:
    """TS2Vec encode() on NaN-padded input returns finite SEQUENCE."""
    data = _make_nan_padded((2, 32, 3))
    reps = ts2vec.encode(data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE)
    assert torch.isfinite(reps).all(), "TS2Vec encode() SEQUENCE contains NaN/Inf"
    assert reps.ndim == 3


def test_ts2vec_encode_batch_nan_padded(ts2vec: TS2Vec) -> None:
    """TS2Vec encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((2, 32, 3))
    reps = ts2vec.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "TS2Vec encode_batch() contains NaN/Inf"


# ---------------------------------------------------------------------------
# CoST — encode() / encode_batch() NaN tests
# ---------------------------------------------------------------------------


def test_cost_encode_nan_padded_vector(cost: CoST) -> None:
    """CoST encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((2, 32, 3))
    reps = cost.encode(data, batch_size=2, num_workers=0)
    assert torch.isfinite(reps).all(), "CoST encode() VECTOR contains NaN/Inf"


def test_cost_encode_nan_padded_sequence(cost: CoST) -> None:
    """CoST encode() on NaN-padded input returns finite SEQUENCE."""
    data = _make_nan_padded((2, 32, 3))
    reps = cost.encode(data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE)
    assert torch.isfinite(reps).all(), "CoST encode() SEQUENCE contains NaN/Inf"
    assert reps.ndim == 3


def test_cost_encode_batch_nan_padded(cost: CoST) -> None:
    """CoST encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((2, 32, 3))
    reps = cost.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "CoST encode_batch() contains NaN/Inf"


# ---------------------------------------------------------------------------
# Cross-model parametrized — all 3 dilated models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["autotcl", "ts2vec", "cost"])
def test_dilated_models_encode_nan_padded_vector(
    request: pytest.FixtureRequest, model_fixture: str
) -> None:
    """All dilated mixin models return finite encode() for NaN-padded input."""
    model = request.getfixturevalue(model_fixture)
    data = _make_nan_padded((2, 32, 3))
    reps = model.encode(data, batch_size=2, num_workers=0)
    assert torch.isfinite(reps).all(), f"{model_fixture} encode() VECTOR contains NaN/Inf"


@pytest.mark.parametrize("model_fixture", ["autotcl", "ts2vec", "cost"])
def test_dilated_models_encode_nan_padded_sequence(
    request: pytest.FixtureRequest, model_fixture: str
) -> None:
    """All dilated mixin models return finite SEQUENCE for NaN-padded input."""
    model = request.getfixturevalue(model_fixture)
    if EncodingOutputShape.SEQUENCE not in model.supported_outputs:
        pytest.skip(f"{model_fixture} does not support SEQUENCE output")
    data = _make_nan_padded((2, 32, 3))
    reps = model.encode(data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE)
    assert torch.isfinite(reps).all(), f"{model_fixture} encode() SEQUENCE contains NaN/Inf"
    assert reps.ndim == 3, f"{model_fixture} SEQUENCE should be 3-D, got {reps.ndim}-D"


# ---------------------------------------------------------------------------
# Sliding-window inference with NaN padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["autotcl", "ts2vec"])
def test_dilated_sliding_window_nan_padded(
    request: pytest.FixtureRequest, model_fixture: str
) -> None:
    """Sliding-window encode() on NaN-padded input returns finite SEQUENCE."""
    model = request.getfixturevalue(model_fixture)
    data = _make_nan_padded((1, 64, 3))
    reps = model.encode(
        data,
        batch_size=1,
        num_workers=0,
        sliding_length=32,
        sliding_padding=0,
        output=EncodingOutputShape.SEQUENCE,
    )
    assert torch.isfinite(reps).all(), f"{model_fixture} sliding encode() contains NaN/Inf"
