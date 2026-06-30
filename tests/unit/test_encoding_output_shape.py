"""Comprehensive tests for the encoding output-shape contract across all 10 models.

Verifies spec §10 requirements:
- All 10 models return 2-D with VECTOR (default).
- Tier A/B models return 3-D with SEQUENCE.
- Tier C models return (N, 1, D) with SEQUENCE plus a once-per-class warning.
- supported_outputs is correct per model.
- Gradient flows through both output paths.
- Producer-side rank assert fires on wrong rank.
- encode() and encode_batch() agree on output rank.
- Dilated encoding_window precedence is preserved.
- Tier C warning dedup works across batches.

Model tier classification
    Tier A (native-reduce, {VECTOR, SEQUENCE}):
        TSTCC, TimeNet, RecurrentAutoEncoder (Basic mixin),
        CoST, TS2Vec, AutoTCL (Dilated mixin).
    Tier B (native 3-D, {VECTOR, SEQUENCE}): TST (Basic mixin).
    Tier C (flat-permissive, {VECTOR}):
        Series2Vec, MCL, TimeVAE (Basic mixin).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from chronocratic.models import (
    AutoTCL,
    CoST,
    MCL,
    RecurrentAutoEncoder,
    Series2Vec,
    TS2Vec,
    TST,
    TSTCC,
    TimeNet,
    TimeVAE,
)
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.utils.helpers import _warned_sequence_fallback


# ---------------------------------------------------------------------------
# Model fixtures — lightweight instances that fit in CPU memory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def timevae_model() -> TimeVAE:
    """TimeVAE (Tier C, Basic mixin)."""
    return TimeVAE(sequence_length=32, input_dims=3, latent_dim=8)


@pytest.fixture(scope="module")
def timenet_model() -> TimeNet:
    """TimeNet (Tier A, Basic mixin)."""
    return TimeNet(hidden_dims=16, depth=1, input_dims=3)


@pytest.fixture(scope="module")
def recurrentae_model() -> RecurrentAutoEncoder:
    """RecurrentAutoEncoder (Tier A, Basic mixin)."""
    return RecurrentAutoEncoder(input_dims=3, layers=(16,))


@pytest.fixture(scope="module")
def mcl_model() -> MCL:
    """MCL (Tier C, Basic mixin)."""
    return MCL(input_dims=3, output_dims=16)


@pytest.fixture(scope="module")
def tstcc_model() -> TSTCC:
    """TSTCC (Tier A, Basic mixin)."""
    return TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)


@pytest.fixture(scope="module")
def tst_model() -> TST:
    """TST (Tier B, Basic mixin)."""
    return TST(
        input_dims=3,
        sequence_length=32,
        hidden_dims=16,
        num_heads=4,
        depth=1,
        feedforward_dims=32,
    )


@pytest.fixture(scope="module")
def series2vec_model() -> Series2Vec:
    """Series2Vec (Tier C, Basic mixin)."""
    return Series2Vec(input_dims=3, representation_dims=8)


@pytest.fixture(scope="module")
def cost_model() -> CoST:
    """CoST (Tier A, Dilated mixin)."""
    return CoST(
        input_dims=3,
        sequence_length=64,
        hidden_dims=16,
        output_dims=32,
        depth=2,
    )


@pytest.fixture(scope="module")
def ts2vec_model() -> TS2Vec:
    """TS2Vec (Tier A, Dilated mixin)."""
    return TS2Vec(input_dims=3, hidden_dims=16, output_dims=32, depth=2)


@pytest.fixture(scope="module")
def autotcl_model() -> AutoTCL:
    """AutoTCL (Tier A, Dilated mixin)."""
    return AutoTCL(input_dims=3, hidden_dims=16, output_dims=32, depth=2)


# Mapping: human-readable name -> pytest fixture name
MODEL_FIXTURES: dict[str, str] = {
    "TimeVAE": "timevae_model",
    "TimeNet": "timenet_model",
    "RecurrentAutoEncoder": "recurrentae_model",
    "MCL": "mcl_model",
    "TSTCC": "tstcc_model",
    "TST": "tst_model",
    "Series2Vec": "series2vec_model",
    "CoST": "cost_model",
    "TS2Vec": "ts2vec_model",
    "AutoTCL": "autotcl_model",
}

TIER_A_BASIC = {"TSTCC", "TimeNet", "RecurrentAutoEncoder"}
TIER_B_BASIC = {"TST"}
TIER_C_BASIC = {"Series2Vec", "MCL", "TimeVAE"}
TIER_A_DILATED = {"CoST", "TS2Vec", "AutoTCL"}

ALL_BASIC = TIER_A_BASIC | TIER_B_BASIC | TIER_C_BASIC

ALL_MODEL_NAMES = sorted(MODEL_FIXTURES.keys())
TIER_AB_NAMES = sorted(TIER_A_BASIC | TIER_B_BASIC | TIER_A_DILATED)
TIER_C_NAMES = sorted(TIER_C_BASIC)
DILATED_NAMES = sorted(TIER_A_DILATED)
ALL_BASIC_NAMES = sorted(ALL_BASIC)


def _get_fixture_name(model_class_name: str) -> str:
    """Return the pytest fixture name for a model class name."""
    return MODEL_FIXTURES[model_class_name]


# Input data shapes per model
_MODEL_DATA_SHAPES: dict[str, tuple[int, ...]] = {
    "TimeVAE": (4, 32, 3),
    "TimeNet": (4, 20, 3),
    "RecurrentAutoEncoder": (4, 20, 3),
    "MCL": (4, 50, 3),
    "TSTCC": (4, 64, 3),
    "TST": (4, 32, 3),
    "Series2Vec": (4, 64, 3),
    "CoST": (4, 64, 3),
    "TS2Vec": (4, 64, 3),
    "AutoTCL": (4, 64, 3),
}


def _make_data(model_name: str) -> torch.Tensor:
    """Create input data for the named model."""
    return torch.randn(*_MODEL_DATA_SHAPES[model_name])


# ---------------------------------------------------------------------------
# Helper: resolve model from class name via request.getfixturevalue
# Each test uses `model_name` param and resolves the fixture itself.
# ---------------------------------------------------------------------------


def _resolve_model(model_name: str, request: pytest.FixtureRequest) -> nn.Module:
    """Look up the model instance for *model_name* via the fixture cache."""
    return request.getfixturevalue(_get_fixture_name(model_name))


# ---------------------------------------------------------------------------
# Test 1: All 10 models return 2-D with VECTOR default (encode)
# ---------------------------------------------------------------------------


class TestDefaultVectorShape:
    """All 10 models return 2-D tensors when using the default output=VECTOR."""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_encode_default_returns_2d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode() with default output returns (N, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        result = model.encode(data, batch_size=2, num_workers=0)
        assert result.ndim == 2, (
            f"{model_name} encode() default should be 2-D, got {result.ndim}-D"
        )
        assert result.shape[0] == data.shape[0]

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_encode_batch_default_returns_2d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch() with default output returns (N, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        result = model.encode_batch(data)
        assert result.ndim == 2, (
            f"{model_name} encode_batch() default should be 2-D, got {result.ndim}-D"
        )


# ---------------------------------------------------------------------------
# Test 2: All 10 models return 2-D with explicit VECTOR (encode_batch)
# ---------------------------------------------------------------------------


class TestExplicitVector:
    """All 10 models return 2-D tensors with explicit output=VECTOR."""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_encode_batch_vector_returns_2d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=VECTOR) returns (N, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        result = model.encode_batch(data, output=EncodingOutputShape.VECTOR)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# Test 3: Tier A/B models return 3-D with SEQUENCE
# ---------------------------------------------------------------------------


class TestSequenceShape:
    """Tier A/B models return 3-D (N, T, D) with output=SEQUENCE."""

    @pytest.mark.parametrize("model_name", TIER_AB_NAMES)
    def test_encode_batch_sequence_returns_3d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=SEQUENCE) returns (N, T, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        result = model.encode_batch(data, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3, (
            f"{model_name} SEQUENCE should be 3-D, got {result.ndim}-D"
        )
        assert result.shape[0] == data.shape[0]

    @pytest.mark.parametrize("model_name", TIER_AB_NAMES)
    def test_encode_sequence_returns_3d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode(output=SEQUENCE) returns (N, T, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3


# ---------------------------------------------------------------------------
# Test 4: Tier C models return (N, 1, D) with SEQUENCE + warn
# ---------------------------------------------------------------------------


class TestTierCSequence:
    """Tier C models return (N, 1, D) with SEQUENCE and emit a warning."""

    @pytest.mark.parametrize("model_name", TIER_C_NAMES)
    def test_encode_batch_sequence_returns_3d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=SEQUENCE) returns (N, 1, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        result = model.encode_batch(data, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3
        assert result.shape[1] == 1, f"{model_name} SEQUENCE should have T=1"

    @pytest.mark.parametrize("model_name", TIER_C_NAMES)
    def test_encode_sequence_returns_3d(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode(output=SEQUENCE) returns (N, 1, D)."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3
        assert result.shape[1] == 1


# ---------------------------------------------------------------------------
# Test 5: supported_outputs correct per model
# ---------------------------------------------------------------------------


class TestSupportedOutputs:
    """Each model declares the correct supported_outputs frozenset."""

    @pytest.mark.parametrize("model_name", TIER_AB_NAMES)
    def test_tier_ab_supports_both(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Tier A/B models support both VECTOR and SEQUENCE."""
        model = _resolve_model(model_name, request)
        assert model.supported_outputs == frozenset(
            {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
        ), f"{model_name} should support {{VECTOR, SEQUENCE}}"

    @pytest.mark.parametrize("model_name", TIER_C_NAMES)
    def test_tier_c_supports_vector_only(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Tier C models support only VECTOR."""
        model = _resolve_model(model_name, request)
        assert model.supported_outputs == frozenset(
            {EncodingOutputShape.VECTOR}
        ), f"{model_name} should support {{VECTOR}} only"


# ---------------------------------------------------------------------------
# Test 6: Gradient flows through both output paths
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Autograd graph is preserved through both VECTOR and SEQUENCE paths."""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_encode_batch_vector_gradient(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=VECTOR) preserves gradient to input."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        x = data.requires_grad_(True)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        result = model.encode_batch(x, output=EncodingOutputShape.VECTOR)
        loss = result.pow(2).sum()
        loss.backward()
        assert x.grad is not None, f"{model_name} VECTOR gradient did not flow"

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_encode_batch_sequence_gradient(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=SEQUENCE) preserves gradient to input."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        x = data.requires_grad_(True)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        result = model.encode_batch(x, output=EncodingOutputShape.SEQUENCE)
        loss = result.pow(2).sum()
        loss.backward()
        assert x.grad is not None, f"{model_name} SEQUENCE gradient did not flow"

    @pytest.mark.parametrize("model_name", ALL_BASIC_NAMES)
    def test_encode_vector_gradient_enabled(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode(output=VECTOR, gradient_enabled=True) preserves autograd."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        x = data.requires_grad_(True)
        result = model.encode(
            x,
            batch_size=2,
            num_workers=0,
            output=EncodingOutputShape.VECTOR,
            gradient_enabled=True,
        )
        assert result.requires_grad, (
            f"{model_name} VECTOR encode should require grad"
        )

    @pytest.mark.parametrize("model_name", sorted(TIER_A_BASIC | TIER_B_BASIC | TIER_C_BASIC))
    def test_encode_sequence_gradient_enabled(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode(output=SEQUENCE, gradient_enabled=True) preserves autograd."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        x = data.requires_grad_(True)
        result = model.encode(
            x,
            batch_size=2,
            num_workers=0,
            output=EncodingOutputShape.SEQUENCE,
            gradient_enabled=True,
        )
        assert result.requires_grad, (
            f"{model_name} SEQUENCE encode should require grad"
        )


# ---------------------------------------------------------------------------
# Test 7: Producer-side rank assert fires on wrong rank
# ---------------------------------------------------------------------------


class TestRankAssert:
    """Producer-side rank assert catches mismatched output rank."""

    def test_vector_assert_fires_on_3d_output(self) -> None:
        """encode(output=VECTOR) raises AssertionError if _encode_batch returns 3-D."""

        class _BrokenModel(MCL):
            """MCL subclass that always returns 3-D."""

            def _encode_batch(
                self,
                encoder: nn.Module,
                batch_x: torch.Tensor,
                *,
                output: EncodingOutputShape = EncodingOutputShape.VECTOR,
            ) -> torch.Tensor:
                # Always return 3-D regardless of output
                return encoder(batch_x).unsqueeze(1)

        model = _BrokenModel(input_dims=3, output_dims=16)
        data = torch.randn(4, 50, 3)
        with pytest.raises(AssertionError, match=r"Expected 2D, got 3D"):
            model.encode(
                data, batch_size=2, num_workers=0, output=EncodingOutputShape.VECTOR
            )

    def test_sequence_assert_fires_on_2d_output(self) -> None:
        """encode(output=SEQUENCE) raises AssertionError if _encode_batch returns 2-D."""

        class _BrokenModel(TST):
            """TST subclass that always returns 2-D."""

            def _encode_batch(
                self,
                encoder: nn.Module,
                batch_x: torch.Tensor,
                *,
                output: EncodingOutputShape = EncodingOutputShape.VECTOR,
            ) -> torch.Tensor:
                # Always return 2-D regardless of output
                padding_masks = torch.ones(
                    batch_x.shape[:2], dtype=torch.bool, device=batch_x.device
                )
                seq = encoder.encode_representations(batch_x, padding_masks)
                return seq.mean(dim=1)  # (B, H) -- 2-D

        model = _BrokenModel(
            input_dims=3,
            sequence_length=32,
            hidden_dims=16,
            num_heads=4,
            depth=1,
            feedforward_dims=32,
        )
        data = torch.randn(4, 32, 3)
        with pytest.raises(AssertionError, match=r"Expected 3D, got 2D"):
            model.encode(
                data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
            )


# ---------------------------------------------------------------------------
# Test 8: encode / encode_batch rank agreement
# ---------------------------------------------------------------------------


class TestEncodeEncodeBatchAgreement:
    """encode() and encode_batch() return the same rank for the same output."""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_vector_rank_agreement(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode(output=VECTOR) and encode_batch(output=VECTOR) agree on rank."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        encode_result = model.encode(
            data,
            batch_size=data.shape[0],
            num_workers=0,
            output=EncodingOutputShape.VECTOR,
        )

        data2 = _make_data(model_name)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        batch_result = model.encode_batch(data2, output=EncodingOutputShape.VECTOR)

        assert encode_result.ndim == batch_result.ndim, (
            f"{model_name} VECTOR rank mismatch: "
            f"encode={encode_result.ndim}D, encode_batch={batch_result.ndim}D"
        )

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_sequence_rank_agreement(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode(output=SEQUENCE) and encode_batch(output=SEQUENCE) agree on rank."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        encode_result = model.encode(
            data,
            batch_size=data.shape[0],
            num_workers=0,
            output=EncodingOutputShape.SEQUENCE,
        )

        data2 = _make_data(model_name)
        if model_name in TIER_A_DILATED:
            encoder = model._get_encoder()
            encoder.eval()
        batch_result = model.encode_batch(data2, output=EncodingOutputShape.SEQUENCE)

        assert encode_result.ndim == batch_result.ndim, (
            f"{model_name} SEQUENCE rank mismatch: "
            f"encode={encode_result.ndim}D, encode_batch={batch_result.ndim}D"
        )


# ---------------------------------------------------------------------------
# Test 9: Dilated encoding_window precedence preserved
# ---------------------------------------------------------------------------


class TestEncodingWindowPrecedence:
    """Explicit encoding_window overrides output for dilated models."""

    @pytest.mark.parametrize("model_name", DILATED_NAMES)
    def test_explicit_full_series_wins_over_sequence(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=SEQUENCE, encoding_window='full_series') returns 2-D."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        encoder = model._get_encoder()
        encoder.eval()
        result = model.encode_batch(
            data, output=EncodingOutputShape.SEQUENCE, encoding_window="full_series"
        )
        assert result.ndim == 2, (
            f"{model_name}: explicit encoding_window='full_series' "
            f"should produce 2-D despite output=SEQUENCE, got {result.ndim}-D"
        )

    @pytest.mark.parametrize("model_name", DILATED_NAMES)
    def test_explicit_none_wins_over_vector(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=VECTOR, encoding_window=None) returns 3-D."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        encoder = model._get_encoder()
        encoder.eval()
        result = model.encode_batch(
            data, output=EncodingOutputShape.VECTOR, encoding_window=None
        )
        assert result.ndim == 3, (
            f"{model_name}: explicit encoding_window=None "
            f"should produce 3-D despite output=VECTOR, got {result.ndim}-D"
        )


# ---------------------------------------------------------------------------
# Test 10: Tier C warning dedup
# ---------------------------------------------------------------------------


class TestTierCWarning:
    """Tier C models emit exactly one warning per class (dedup works)."""

    @pytest.mark.parametrize("model_name", TIER_C_NAMES)
    def test_warning_emitted_on_sequence(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=SEQUENCE) emits UserWarning."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.encode_batch(data, output=EncodingOutputShape.SEQUENCE)
            seq_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "SEQUENCE" in str(x.message)
            ]
            assert len(seq_warnings) >= 1, f"{model_name} should emit SEQUENCE warning"

    @pytest.mark.parametrize("model_name", TIER_C_NAMES)
    def test_warning_dedup_across_batches(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode() with multiple batches emits only one SEQUENCE warning."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # batch_size=2, 4 samples -> 2 batches
            model.encode(
                data,
                batch_size=2,
                num_workers=0,
                output=EncodingOutputShape.SEQUENCE,
            )
            seq_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "SEQUENCE" in str(x.message)
            ]
            assert len(seq_warnings) == 1, (
                f"{model_name} should emit exactly 1 SEQUENCE warning "
                f"(dedup across batches), got {len(seq_warnings)}"
            )

    @pytest.mark.parametrize("model_name", TIER_C_NAMES)
    def test_no_warning_on_vector(
        self, model_name: str, request: pytest.FixtureRequest
    ) -> None:
        """encode_batch(output=VECTOR) does not emit SEQUENCE warning."""
        _warned_sequence_fallback.clear()
        model = _resolve_model(model_name, request)
        data = _make_data(model_name)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.encode_batch(data, output=EncodingOutputShape.VECTOR)
            seq_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "SEQUENCE" in str(x.message)
            ]
            assert len(seq_warnings) == 0
