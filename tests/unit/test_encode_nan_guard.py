"""NaN-padded input defense for BasicEncodingMixin encode() paths.

Verifies that all BasicEncodingMixin models produce finite representations
when encode(), encode_batch(), forward(), or predict() receive
variable-length series padded with trailing NaN timesteps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chronocratic.models.convolutional.standard.mcl.model import MCL
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.generative.timevae.model import TimeVAE
from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder
from chronocratic.models.recurrent.timenet.model import TimeNet
from chronocratic.models.transformer.tst.model import TST

# --------------------------------------------------------------------------- #
# Fixtures — tiny models for fast testing
# --------------------------------------------------------------------------- #


@pytest.fixture
def timevae() -> TimeVAE:
    """Small TimeVAE for fast testing."""
    return TimeVAE(
        sequence_length=32,
        input_dim=3,
        latent_dim=8,
        hidden_layer_sizes=(16, 32),
        conv_kernel_size=3,
        conv_stride=2,
    )


@pytest.fixture
def mcl() -> MCL:
    """Small MCL for fast testing."""
    return MCL(
        input_dim=3,
        representation_dim=8,
        encoder_channels=(16, 32),
        encoder_kernels=(3, 3),
        encoder_dilations=(1, 1),
        projection_dim=8,
    )


@pytest.fixture
def tstcc() -> TSTCC:
    """Small TSTCC for fast testing."""
    return TSTCC(
        input_dim=3,
        sequence_length=32,
        representation_dim=8,
        encoder_channels=(16, 32),
        encoder_inner_kernels=(3, 3),
        temporal_contrast_hidden_dim=8,
        temporal_contrast_timesteps=3,
        conv_kernel_size=3,
        stride=1,
    )


@pytest.fixture
def timenet() -> TimeNet:
    """Small TimeNet for fast testing."""
    return TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)


@pytest.fixture
def recurrentae() -> RecurrentAutoEncoder:
    """Small RecurrentAE for fast testing."""
    return RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0)


@pytest.fixture
def tst() -> TST:
    """Small TST for fast testing."""
    return TST(
        input_dim=3,
        sequence_length=32,
        hidden_dim=16,
        num_heads=4,
        depth=1,
        feedforward_dim=32,
        dropout_rate=0.0,
    )


@pytest.fixture
def series2vec() -> Series2Vec:
    """Small Series2Vec for fast testing."""
    return Series2Vec(
        input_dim=3,
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=32,
        representation_dim=8,
        dropout_rate=0.0,
        sequence_length=32,
    )


# --------------------------------------------------------------------------- #
# NaN-padded input test helpers
# --------------------------------------------------------------------------- #


def _make_nan_padded(shape: tuple[int, int, int]) -> torch.Tensor:
    """Create a tensor with NaN padding on the last few timesteps."""
    x = torch.randn(*shape)
    x[:, -3:, :] = float("nan")
    return x


def _make_all_nan_row(shape: tuple[int, int, int]) -> torch.Tensor:
    """Create a tensor where one row is entirely NaN."""
    x = torch.randn(*shape)
    x[0, :, :] = float("nan")
    return x


# --------------------------------------------------------------------------- #
# TimeVAE — encode(), encode_batch(), forward(), predict() NaN tests
# --------------------------------------------------------------------------- #


def test_timevae_encode_nan_padded_vector(timevae: TimeVAE) -> None:
    """TimeVAE encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = timevae.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "TimeVAE encode() VECTOR contains NaN/Inf"


def test_timevae_encode_batch_nan_padded(timevae: TimeVAE) -> None:
    """TimeVAE encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = timevae.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "TimeVAE encode_batch() contains NaN/Inf"


def test_timevae_forward_nan_padded(timevae: TimeVAE) -> None:
    """TimeVAE forward() on NaN-padded input returns finite reconstruction."""
    x = _make_nan_padded((4, 32, 3))
    out = timevae(x)
    assert torch.isfinite(out).all(), "TimeVAE forward() contains NaN/Inf"


def test_timevae_predict_nan_padded(timevae: TimeVAE) -> None:
    """TimeVAE predict() on NaN-padded NumPy input returns finite reconstruction."""
    x = _make_nan_padded((4, 32, 3))
    x_np = x.numpy()
    out = timevae.predict(x_np)
    assert np.isfinite(out).all(), "TimeVAE predict() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# MCL — encode(), encode_batch() NaN tests
# --------------------------------------------------------------------------- #


def test_mcl_encode_nan_padded_vector(mcl: MCL) -> None:
    """MCL encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = mcl.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "MCL encode() VECTOR contains NaN/Inf"


def test_mcl_encode_batch_nan_padded(mcl: MCL) -> None:
    """MCL encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = mcl.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "MCL encode_batch() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# TSTCC — encode(), encode_batch() NaN tests
# --------------------------------------------------------------------------- #


def test_tstcc_encode_nan_padded_vector(tstcc: TSTCC) -> None:
    """TSTCC encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = tstcc.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "TSTCC encode() VECTOR contains NaN/Inf"


def test_tstcc_encode_batch_nan_padded(tstcc: TSTCC) -> None:
    """TSTCC encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = tstcc.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "TSTCC encode_batch() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# TimeNet — encode(), encode_batch() NaN tests
# --------------------------------------------------------------------------- #


def test_timenet_encode_nan_padded_vector(timenet: TimeNet) -> None:
    """TimeNet encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = timenet.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "TimeNet encode() VECTOR contains NaN/Inf"


def test_timenet_encode_batch_nan_padded(timenet: TimeNet) -> None:
    """TimeNet encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = timenet.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "TimeNet encode_batch() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# RecurrentAE — encode(), encode_batch() NaN tests
# --------------------------------------------------------------------------- #


def test_recurrentae_encode_nan_padded_vector(recurrentae: RecurrentAutoEncoder) -> None:
    """RecurrentAE encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = recurrentae.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "RecurrentAE encode() VECTOR contains NaN/Inf"


def test_recurrentae_encode_batch_nan_padded(recurrentae: RecurrentAutoEncoder) -> None:
    """RecurrentAE encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = recurrentae.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "RecurrentAE encode_batch() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# TST — encode() NaN tests (already guarded via _split_padding)
# --------------------------------------------------------------------------- #


def test_tst_encode_nan_padded_vector(tst: TST) -> None:
    """TST encode() on NaN-padded input returns finite VECTOR (masked pooling)."""
    data = _make_nan_padded((4, 32, 3))
    reps = tst.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "TST encode() VECTOR contains NaN/Inf"


def test_tst_encode_batch_nan_padded(tst: TST) -> None:
    """TST encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = tst.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "TST encode_batch() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# Series2Vec — encode() NaN tests (already guarded via network-level NaN defense)
# --------------------------------------------------------------------------- #


def test_series2vec_encode_nan_padded_vector(series2vec: Series2Vec) -> None:
    """Series2Vec encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = series2vec.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "Series2Vec encode() VECTOR contains NaN/Inf"


# --------------------------------------------------------------------------- #
# Cross-model parametrized test — all 7 models
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model_fixture", "seq_len", "input_dim"),
    [
        ("timevae", 32, 3),
        ("mcl", 32, 3),
        ("tstcc", 32, 3),
        ("timenet", 32, 3),
        ("recurrentae", 32, 3),
        ("tst", 32, 3),
        ("series2vec", 32, 3),
    ],
)
def test_all_models_encode_nan_padded(
    request: pytest.FixtureRequest, model_fixture: str, seq_len: int, input_dim: int
) -> None:
    """All BasicEncodingMixin models return finite encode() for NaN-padded input."""
    model = request.getfixturevalue(model_fixture)
    data = _make_nan_padded((2, seq_len, input_dim))
    reps = model.encode(data, batch_size=2)
    assert torch.isfinite(reps).all(), f"{model_fixture} encode() VECTOR contains NaN/Inf"


# --------------------------------------------------------------------------- #
# Fully-NaN row degenerate test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model_fixture", "seq_len", "input_dim"),
    [
        ("timevae", 32, 3),
        ("mcl", 32, 3),
        ("tstcc", 32, 3),
        ("timenet", 32, 3),
        ("recurrentae", 32, 3),
    ],
)
def test_fully_nan_row_no_crash(
    request: pytest.FixtureRequest, model_fixture: str, seq_len: int, input_dim: int
) -> None:
    """One fully-NaN row in the batch does not crash or produce NaN output.

    TST is excluded because it raises ValueError on zero-valid-timestep rows
    (by design — see test_tst_batch_contract.py::TestPaddingMask).
    """
    model = request.getfixturevalue(model_fixture)
    data = _make_all_nan_row((4, seq_len, input_dim))
    reps = model.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), (
        f"{model_fixture} encode() contains NaN/Inf for fully-NaN row"
    )


# --------------------------------------------------------------------------- #
# SEQUENCE output test — models that support it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model_fixture", "seq_len", "input_dim"),
    [("mcl", 32, 3), ("tstcc", 32, 3), ("timenet", 32, 3), ("recurrentae", 32, 3), ("tst", 32, 3)],
)
def test_sequence_output_nan_padded(
    request: pytest.FixtureRequest, model_fixture: str, seq_len: int, input_dim: int
) -> None:
    """Models supporting SEQUENCE output return finite results for NaN-padded input.

    TimeVAE is excluded: its SEQUENCE output is a fallback (unsqueeze-1).
    Series2Vec is excluded: it only supports VECTOR output.
    """
    model = request.getfixturevalue(model_fixture)
    if EncodingOutputShape.SEQUENCE not in model.supported_outputs:
        pytest.skip(f"{model_fixture} does not support SEQUENCE output")
    data = _make_nan_padded((2, seq_len, input_dim))
    reps = model.encode(data, batch_size=2, output=EncodingOutputShape.SEQUENCE)
    assert torch.isfinite(reps).all(), f"{model_fixture} encode() SEQUENCE contains NaN/Inf"
    assert reps.ndim == 3, f"{model_fixture} SEQUENCE should be 3-D, got {reps.ndim}-D"


# --------------------------------------------------------------------------- #
# Guard-placement regression — the NaN guard belongs in _encode_batch, never
# in BasicEncodingMixin. TST derives its padding mask from x.isnan(), so a
# nan_to_num applied above _encode_batch yields an all-True mask and silently
# pools padding into the VECTOR representation. The output stays finite, so
# every isfinite assertion in this file passes while the numbers are wrong.
# These two tests are the only thing standing between that refactor and main.
# --------------------------------------------------------------------------- #

_VALID = 20


def _tst_padded_batch() -> torch.Tensor:
    """A (2, 32, 3) batch whose first sample is NaN-padded from t=_VALID onward."""
    x = torch.randn(2, 32, 3)
    x[0, _VALID:, :] = float("nan")
    return x


def test_encode_batch_preserves_nan_for_mask_derivation(tst: TST) -> None:
    """encode_batch() must hand NaN to _encode_batch intact.

    Fails if a NaN guard is added to BasicEncodingMixin.encode_batch().
    """
    x = _tst_padded_batch()
    tst.eval()
    with torch.no_grad():
        vector = tst.encode_batch(x)
        sequence = tst.encode_batch(x, output=EncodingOutputShape.SEQUENCE)
    # VECTOR is the mean over real timesteps only ...
    assert torch.allclose(vector[0], sequence[0, :_VALID].mean(dim=0), atol=1e-6)
    # ... and is NOT the mean over all timesteps (guards against a vacuous test
    # should _VALID ever reach sequence_length).
    assert not torch.allclose(vector[0], sequence[0].mean(dim=0), atol=1e-6)


def test_encode_preserves_nan_for_mask_derivation(tst: TST) -> None:
    """encode() must hand NaN to _encode_batch intact.

    Fails if a NaN guard is added to either BasicEncodingMixin.encode() or
    .encode_batch(); the sibling test localizes which.
    """
    x = _tst_padded_batch()
    vector = tst.encode(x, batch_size=2)
    sequence = tst.encode(x, batch_size=2, output=EncodingOutputShape.SEQUENCE)
    assert torch.allclose(vector[0], sequence[0, :_VALID].mean(dim=0), atol=1e-6)
    assert not torch.allclose(vector[0], sequence[0].mean(dim=0), atol=1e-6)
