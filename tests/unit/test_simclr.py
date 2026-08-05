"""Unit tests for the SimCLR model.

Covers the config-to-model contract, encoder shapes and length robustness,
the encoding-output-shape contract, NaN-padding defense, augmentation
injection, and the numerical equivalences that license porting the
reference's 2-D ResNet as a 1-D one.

All shapes are asymmetric (``T != C``) so a dropped transpose crashes rather
than silently passing.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from types import SimpleNamespace
import warnings

import lightning.pytorch as pl
import pytest
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair
from chronocratic.models.augmentation.primitives import Jitter, JitterParameters
from chronocratic.models.augmentation.producers import FullOverlapProducer, IndependentPairProducer
from chronocratic.models.convolutional.standard.simclr.augmentations import _default_simclr_pair
from chronocratic.models.convolutional.standard.simclr.config import SimCLRModelParameters
from chronocratic.models.convolutional.standard.simclr.encoder import Conv1dResNetEncoder
from chronocratic.models.convolutional.standard.simclr.layers import (
    Conv1dBasicBlock,
    Conv1dBottleneckBlock,
)
from chronocratic.models.convolutional.standard.simclr.model import SimCLR
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.losses import NTXentLoss
from chronocratic.models.protocols import HasEncoder
from chronocratic.models.utils import ensure_pairable_batch
from chronocratic.models.utils.helpers import reset_warning_cache

# Asymmetric on purpose: T=50, C=3.
BATCH, TIME, CHANNELS = 4, 50, 3

# The model's own default, restated so the schedule tests assert against a
# named peak rather than a literal.
LEARNING_RATE = 1e-3

# Tiny architecture so the suite stays fast. The library defaults build a
# ResNet-50 (~23M parameters), which is correct but not something to
# instantiate dozens of times in unit tests.
SMALL = {
    "encoder_stage_channels": (8, 16),
    "encoder_stage_depths": (1, 1),
    "encoder_stage_strides": (1, 2),
    "stem_conv_channels": 8,
    "projection_hidden_dim": 16,
    "projection_dim": 8,
}

# Encoder-only kwargs for constructing Conv1dResNetEncoder directly.
_SMALL_ENCODER_KWARGS = {
    "encoder_stage_channels": SMALL["encoder_stage_channels"],
    "encoder_stage_depths": SMALL["encoder_stage_depths"],
    "encoder_stage_strides": SMALL["encoder_stage_strides"],
    "stem_conv_channels": SMALL["stem_conv_channels"],
}


def _small_model(**overrides: object) -> SimCLR:
    """Build a small SimCLR instance, overriding any keyword."""
    kwargs = {"input_dim": CHANNELS} | SMALL | overrides
    return SimCLR(**kwargs)  # type: ignore[arg-type]


def _data(batch: int = BATCH, time: int = TIME, channels: int = CHANNELS) -> torch.Tensor:
    """Return a random ``(B, T, C)`` batch."""
    return torch.randn(batch, time, channels)


def _structured_batch(
    sample_count: int = BATCH, seq_len: int = 128, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Build a batch with three distinguishable pattern families.

    Uses sinusoids of varying frequency/phase, square waves, and ramps with
    light additive noise. Structured input is required because white noise has
    no between-sample structure, so any encoder clusters it and similarity
    statistics say nothing about the encoder.

    Args:
        sample_count: Number of series to generate.
        seq_len: Sequence length for each series.
        generator: Seeded RNG for reproducibility.

    Returns:
        Tensor of shape ``(sample_count, seq_len, 1)``.
    """
    if generator is None:
        generator = torch.Generator()
    time = torch.linspace(0, 4 * math.pi, seq_len)
    frequency = torch.randint(1, 6, (sample_count,), generator=generator).float()
    phase = torch.rand(sample_count, generator=generator) * math.pi
    family = torch.randint(0, 3, (sample_count,), generator=generator)
    sine = torch.stack([torch.sin(frequency[i] * time + phase[i]) for i in range(sample_count)])
    ramp = (time / time[-1]).expand(sample_count, seq_len) * frequency[:, None] / 5
    series = torch.where(
        family[:, None] == 0, sine, torch.where(family[:, None] == 1, torch.sign(sine), ramp)
    )
    noise = 0.05 * torch.randn(sample_count, seq_len, generator=generator)
    return (series + noise).unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Config-to-model contract
# --------------------------------------------------------------------------- #


class TestConfigContract:
    """SimCLRModelParameters and SimCLR.__init__ must mirror each other."""

    def test_config_splat_instantiates(self) -> None:
        """Model(**vars(ModelParameters(...))) must not raise."""
        config = SimCLRModelParameters(input_dim=CHANNELS, **SMALL)  # type: ignore[arg-type]
        assert isinstance(SimCLR(**vars(config)), SimCLR)

    def test_config_splat_with_defaults_only(self) -> None:
        """Partial config instantiation works — every optional field has a default."""
        config = SimCLRModelParameters(input_dim=1)
        assert isinstance(SimCLR(**vars(config)), SimCLR)

    def test_field_names_match_init_parameters(self) -> None:
        """Every config field has an __init__ parameter of the same name."""
        init_params = set(inspect.signature(SimCLR.__init__).parameters) - {"self"}
        field_names = {f.name for f in dataclasses.fields(SimCLRModelParameters)}
        assert field_names == init_params

    def test_defaults_match_init_defaults(self) -> None:
        """Every config default equals the matching __init__ default."""
        init_params = inspect.signature(SimCLR.__init__).parameters
        for field in dataclasses.fields(SimCLRModelParameters):
            if field.default is dataclasses.MISSING:
                continue
            assert field.default == init_params[field.name].default, (
                f"default mismatch for {field.name}"
            )

    def test_init_is_keyword_only(self) -> None:
        """No positional parameters besides self."""
        params = list(inspect.signature(SimCLR.__init__).parameters.values())[1:]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)

    def test_sequence_hyperparameters_are_tuples(self) -> None:
        """Sequence-typed hyperparameters default to tuples, not lists."""
        config = SimCLRModelParameters(input_dim=1)
        assert isinstance(config.encoder_stage_channels, tuple)
        assert isinstance(config.encoder_stage_depths, tuple)
        assert isinstance(config.encoder_stage_strides, tuple)

    def test_hyperparameters_stored_privately(self) -> None:
        """Hyperparameters land on self._{name} attributes."""
        model = _small_model()
        for name in ("input_dim", "projection_dim", "temperature", "learning_rate"):
            assert hasattr(model, f"_{name}")

    def test_augmentation_excluded_from_saved_hyperparameters(self) -> None:
        """save_hyperparameters(ignore=["augmentation"]) keeps the producer out."""
        model = _small_model(augmentation=IndependentPairProducer(aug=Jitter()))
        assert "augmentation" not in model.hparams

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_dim": 0},
            {"input_dim": 1, "conv_kernel_size": 0},
            {"input_dim": 1, "stem_conv_channels": 0},
            {"input_dim": 1, "projection_dim": 0},
            {"input_dim": 1, "projection_hidden_dim": 0},
            {"input_dim": 1, "temperature": 0.0},
            {"input_dim": 1, "max_train_length": 0},
        ],
    )
    def test_config_rejects_invalid_values(self, kwargs: dict) -> None:
        """__post_init__ validates numeric constraints."""
        with pytest.raises(ValueError):  # noqa: PT011
            SimCLRModelParameters(**kwargs)


# --------------------------------------------------------------------------- #
# Reference fidelity — the equivalences that license the 1-D port
# --------------------------------------------------------------------------- #


class TestReferenceEquivalence:
    """The reference's 2-D stack computes what this 1-D port computes."""

    def test_conv2d_on_width_one_equals_conv1d_centre_column(self) -> None:
        """Conv2d(k=3, pad=1) over (B, C, T, 1) == Conv1d(k=3, pad=1), centre column.

        The reference unsqueezes the series to width 1 and runs Conv2d over it.
        The outer kernel columns then only ever see zero padding, so they are
        dead weights and the stack collapses to a Conv1d. This is the claim the
        whole port rests on, so it is asserted rather than argued.
        """
        for stride in (1, 2):
            conv2d = nn.Conv2d(CHANNELS, 8, kernel_size=3, stride=stride, padding=1, bias=False)
            conv1d = nn.Conv1d(CHANNELS, 8, kernel_size=3, stride=stride, padding=1, bias=False)
            with torch.no_grad():
                conv1d.weight.copy_(conv2d.weight[:, :, :, 1])
            x = torch.randn(BATCH, CHANNELS, TIME, 1)
            assert torch.equal(conv2d(x).squeeze(-1), conv1d(x.squeeze(-1)))

    def test_batchnorm2d_on_width_one_equals_batchnorm1d(self) -> None:
        """BatchNorm2d over (B, C, T, 1) == BatchNorm1d over (B, C, T).

        The normalization counterpart of the convolution equivalence above,
        and the claim that lets ``BATCH`` call itself a reproduction of the
        reference: BatchNorm2d accumulates per-channel statistics over
        ``B * T * 1`` elements and BatchNorm1d over ``B * T`` -- the same
        values, hence the same mean, variance and affine output. Checked in
        both modes because only training mode updates running statistics.

        Compared with a tolerance rather than exactly: the two kernels reduce
        in different orders, so the last bits depend on the input's memory
        layout. A transposed (non-contiguous) view is enough to make them
        differ in float32. The equality is mathematical, not bitwise.
        """
        batch_norm_2d = nn.BatchNorm2d(CHANNELS)
        batch_norm_1d = nn.BatchNorm1d(CHANNELS)
        with torch.no_grad():
            batch_norm_2d.weight.normal_()
            batch_norm_2d.bias.normal_()
            batch_norm_1d.weight.copy_(batch_norm_2d.weight)
            batch_norm_1d.bias.copy_(batch_norm_2d.bias)

        for training in (True, False):
            batch_norm_2d.train(training)
            batch_norm_1d.train(training)
            x = torch.randn(BATCH, CHANNELS, TIME)
            assert torch.allclose(
                batch_norm_2d(x.unsqueeze(-1)).squeeze(-1), batch_norm_1d(x), atol=1e-6
            )
            assert torch.allclose(batch_norm_2d.running_mean, batch_norm_1d.running_mean, atol=1e-6)
            assert torch.allclose(batch_norm_2d.running_var, batch_norm_1d.running_var, atol=1e-6)

    def test_reference_pooling_equals_mean_over_time(self) -> None:
        """repeat-to-square then avg_pool2d == a plain mean over the time axis."""
        features = torch.randn(BATCH, 16, 7, 1)
        tiled = features.repeat(1, 1, 1, features.size(2))
        reference = functional.avg_pool2d(tiled, tiled.size(2)).flatten(1)
        assert torch.allclose(reference, features.squeeze(-1).mean(dim=-1), atol=1e-5)

    def test_ntxent_equals_the_reference_contrastive_loss(self) -> None:
        """NT-Xent reproduces the reference's ``contrastive_loss`` exactly.

        The reference builds the loss a different way: it exponentiates the
        whole 2N x 2N similarity matrix including the diagonal, then subtracts
        ``exp(1 / tau)`` from each row sum to remove the self-term. That is
        only valid because its projections are unit-norm, which makes every
        diagonal entry exactly ``exp(1 / tau)``. Transcribed here and checked
        against the library's masked implementation, because a denominator
        that is off by one term still trains and still looks plausible -- it
        just optimizes a slightly different objective.
        """

        def reference_loss(xi: torch.Tensor, xj: torch.Tensor, tau: float) -> torch.Tensor:
            x = torch.cat((xi, xj), dim=0)
            similarity = torch.exp(torch.mm(x, x.T) / tau)
            positives = torch.exp(torch.sum(xi * xj, dim=-1) / tau)
            positives = torch.cat((positives, positives), dim=0)
            self_term = torch.exp(torch.ones(x.size(0)) / tau)
            return torch.mean(-torch.log(positives / (similarity.sum(dim=-1) - self_term)))

        torch.manual_seed(0)
        for batch, temperature in [(8, 0.5), (32, 0.5), (16, 0.1)]:
            first = functional.normalize(torch.randn(batch, 128), dim=-1)
            second = functional.normalize(torch.randn(batch, 128), dim=-1)
            ours = NTXentLoss(temperature=temperature, use_cosine_similarity=True)(first, second)
            # Loose at 1e-3 because the reference formulation is the numerically
            # worse one: it exponentiates before subtracting, so at tau=0.1 it
            # forms exp(1 / tau) = e^10 and loses float32 digits that the
            # masked implementation never computes. The gap is precision, not
            # a difference of objective -- it shrinks with temperature.
            assert ours == pytest.approx(reference_loss(first, second, temperature), abs=1e-3)

    def test_ntxent_at_chance_is_log_2n_minus_1(self) -> None:
        """Uninformative projections score ``ln(2N - 1)``, not ``ln(2N)``.

        Pins the negative count: the denominator runs over the ``2N - 1``
        candidates that are not the anchor itself. This value is also the
        diagnostic signature of a view pair that shares no information -- an
        observed loss sitting exactly here means training is learning nothing.
        """
        batch = 16
        identical = functional.normalize(torch.ones(batch, 32), dim=-1)
        loss = NTXentLoss(temperature=0.5, use_cosine_similarity=True)(identical, identical)
        assert loss == pytest.approx(math.log(2 * batch - 1), abs=1e-4)

    def test_loss_is_ntxent(self) -> None:
        """SimCLR uses NT-Xent — the loss the reference's contrastive_loss computes."""
        assert isinstance(_small_model().criterion, NTXentLoss)

    def test_default_augmentation_matches_reference_constants(self) -> None:
        """The default pair carries ULTS DataTransform's constants.

        Scaling is the documented exception; see the sigma test below.
        """
        producer = _default_simclr_pair()
        assert producer._first._params.mean == pytest.approx(1.0)
        permutation, jitter = producer._second._augmentations
        assert permutation._params.max_segments == 5
        assert jitter._params.sigma == pytest.approx(0.8)

    def test_default_scaling_sigma_cannot_flip_the_signal(self) -> None:
        """Scaling must not invert whole series, which breaks the view pair.

        The factor is drawn from ``N(mean, sigma)``, so the reference's
        ``sigma=1.1`` around ``mean=1.0`` sends 18.2% of draws negative. That
        leaves NT-Xent at exactly ``ln(2N - 1)`` — pure chance — on GunPoint.
        Three sigma inside the positive half-line keeps sign flips negligible.
        """
        params = _default_simclr_pair()._first._params
        assert params.mean - 3 * params.sigma > 0.0
        assert params.sigma == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #


class TestEncoder:
    """Conv1dResNetEncoder shapes, widths, and configuration validation."""

    def test_forward_returns_unpooled_feature_map(self) -> None:
        """Encoder returns (B, representation_dim, T') — pooling is the caller's job."""
        encoder = Conv1dResNetEncoder(
            input_dim=CHANNELS,
            encoder_stage_channels=(8, 16),
            encoder_stage_depths=(1, 1),
            encoder_stage_strides=(1, 2),
            stem_conv_channels=8,
        )
        result = encoder(_data())
        assert result.ndim == 3
        assert result.shape[:2] == (BATCH, encoder.representation_dim)
        assert result.shape[2] > 1, "the temporal axis must survive the encoder"

    @pytest.mark.parametrize(
        ("block_type", "expected"),
        [(ResidualBlockType.BASIC, 16), (ResidualBlockType.BOTTLENECK, 64)],
    )
    def test_representation_dim_follows_expansion(
        self, block_type: ResidualBlockType, expected: int
    ) -> None:
        """representation_dim == encoder_stage_channels[-1] * block.expansion."""
        encoder = Conv1dResNetEncoder(
            input_dim=CHANNELS,
            encoder_stage_channels=(8, 16),
            encoder_stage_depths=(1, 1),
            encoder_stage_strides=(1, 2),
            stem_conv_channels=8,
            residual_block_type=block_type,
        )
        assert encoder.representation_dim == expected

    def test_block_expansions(self) -> None:
        """Expansion factors match the ResNet paper."""
        assert Conv1dBasicBlock.expansion == 1
        assert Conv1dBottleneckBlock.expansion == 4

    def test_fourth_stage_honours_its_own_block_count(self) -> None:
        """Unlike the reference, stage 4 uses encoder_stage_depths[3], not encoder_stage_depths[2].

        The reference builds its stages with ``layers[2]`` twice, so being
        handed (3, 4, 6, 3) it builds (3, 4, 6, 6).
        """
        encoder = Conv1dResNetEncoder(
            input_dim=CHANNELS,
            encoder_stage_channels=(4, 4, 4, 4),
            encoder_stage_depths=(1, 1, 2, 1),
            encoder_stage_strides=(1, 1, 1, 1),
            stem_conv_channels=4,
            residual_block_type=ResidualBlockType.BASIC,
        )
        assert [len(stage) for stage in encoder._stages] == [1, 1, 2, 1]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"encoder_stage_channels": ()},
            {"encoder_stage_depths": (1,)},
            {"encoder_stage_strides": (1,)},
        ],
    )
    def test_rejects_mismatched_stage_tuples(self, kwargs: dict) -> None:
        """Stage tuples must be the same length."""
        base = {
            "input_dim": CHANNELS,
            "encoder_stage_channels": (8, 16),
            "encoder_stage_depths": (1, 1),
            "encoder_stage_strides": (1, 2),
            "stem_conv_channels": 8,
        }
        with pytest.raises(ValueError):  # noqa: PT011
            Conv1dResNetEncoder(**{**base, **kwargs})

    @pytest.mark.parametrize("norm", [NormalizationLayerType.CHANNEL, NormalizationLayerType.BATCH])
    def test_both_normalization_types_run(self, norm: NormalizationLayerType) -> None:
        """CHANNEL (default) and BATCH (the reference's choice) both work."""
        model = _small_model(normalization_layer_type=norm)
        assert torch.isfinite(model.encode_batch(_data())).all()


# --------------------------------------------------------------------------- #
# Encoding output shapes
# --------------------------------------------------------------------------- #


class TestEncodingOutput:
    """The encode() contract: rank, width, and the SEQUENCE fallback."""

    def test_is_basic_encoding_mixin(self) -> None:
        """SimCLR uses the fixed-length mixin."""
        assert isinstance(_small_model(), BasicEncodingMixin)

    def test_satisfies_has_encoder(self) -> None:
        """The encoder property is publicly reachable."""
        assert isinstance(_small_model(), HasEncoder)

    def test_supported_outputs_declares_both(self) -> None:
        """supported_outputs is a class-level frozenset naming both shapes."""
        assert SimCLR.supported_outputs == frozenset(
            {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
        )

    def test_encode_vector_is_2d(self) -> None:
        """VECTOR returns (N, D)."""
        model = _small_model()
        result = model.encode(_data(), batch_size=2)
        assert result.ndim == 2
        assert result.shape == (BATCH, model.representation_dim)

    def test_encode_sequence_has_real_temporal_axis(self) -> None:
        """SEQUENCE returns (N, T', D) with a genuine temporal axis and no warning."""
        model = _small_model()
        result = model.encode(_data(), batch_size=2, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3
        assert result.shape[0] == BATCH
        assert result.shape[2] == model.representation_dim
        assert result.shape[1] > 1, "SEQUENCE must not be a length-1 placeholder"

    def test_encode_sequence_does_not_warn(self) -> None:
        """The length-1 fallback warning is obsolete — SEQUENCE is now real."""
        reset_warning_cache()
        model = _small_model()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="torch")
            warnings.filterwarnings("error", category=UserWarning, module="chronocratic")
            model.encode(_data(), batch_size=2, output=EncodingOutputShape.SEQUENCE)

    def test_encode_rejects_unsupported_output(self) -> None:
        """An unrecognized output shape raises rather than guessing."""
        model = _small_model()
        with pytest.raises(ValueError, match="does not support output"):
            model._encode_batch(model.encoder, _data(), output="triangular")  # type: ignore[arg-type]

    def test_encode_returns_backbone_features_not_projections(self) -> None:
        """encode() must bypass the projection head, as SimCLR prescribes."""
        model = _small_model()
        assert model.representation_dim != model._projection_dim
        assert model.encode_batch(_data()).shape[1] == model.representation_dim

    def test_encode_batch_preserves_gradients(self) -> None:
        """encode_batch is differentiable — the adversarial path depends on it."""
        model = _small_model()
        x = _data().requires_grad_(requires_grad=True)
        result = model.encode_batch(x)
        assert result.requires_grad
        result.sum().backward()
        assert x.grad is not None

    def test_forward_returns_unit_norm_projections(self) -> None:
        """forward() gives the projection space the loss operates in, per timestep."""
        model = _small_model()
        reduced_len = model.encoder(_data()).shape[-1]
        z = model(_data())
        assert z.shape == (BATCH, reduced_len * model._projection_dim)
        assert torch.allclose(z.norm(dim=-1), torch.ones(BATCH), atol=1e-5)


# --------------------------------------------------------------------------- #
# NaN padding
# --------------------------------------------------------------------------- #


class TestNaNHandling:
    """Variable-length batches arrive NaN-padded; nothing may return NaN."""

    def test_trailing_nan_batch_encodes_finite(self) -> None:
        """Partial padding on some samples still yields finite representations."""
        model = _small_model()
        x = _data()
        x[0, 30:, :] = float("nan")
        x[1, 45:, :] = float("nan")
        assert torch.isfinite(model.encode(x, batch_size=2)).all()

    def test_all_nan_sample_encodes_finite(self) -> None:
        """A wholly-padded sample yields a finite representation."""
        model = _small_model()
        x = _data()
        x[2, :, :] = float("nan")
        assert torch.isfinite(model.encode(x, batch_size=2)).all()

    def test_clean_input_regression(self) -> None:
        """The NaN guard leaves clean inputs untouched."""
        model = _small_model()
        model.eval()
        x = _data()
        with torch.no_grad():
            guarded = model.encode_batch(x)
            direct = model.encoder(x).mean(dim=-1)  # (B, C, T') -> (B, C)
        assert torch.allclose(guarded, direct, atol=1e-6)

    def test_training_step_on_nan_padded_batch(self) -> None:
        """The training path zero-fills before augmenting, so the loss stays finite."""
        model = _small_model()
        x = _data()
        x[0, 30:, :] = float("nan")
        assert math.isfinite(model._compute_loss(x).item())


# --------------------------------------------------------------------------- #
# Input length robustness
# --------------------------------------------------------------------------- #


class TestInputLengthRobustness:
    """One model instance must encode every length it is shown."""

    def test_one_instance_encodes_several_lengths(self) -> None:
        """Shorter and longer than nominal, through a single instance."""
        model = _small_model()
        for length in (16, TIME, 137, 400):
            result = model.encode(_data(time=length), batch_size=2)
            assert result.shape == (BATCH, model.representation_dim)

    def test_gradient_flow_at_non_nominal_length(self) -> None:
        """encode_batch stays differentiable away from the nominal length."""
        model = _small_model()
        x = _data(time=137).requires_grad_(requires_grad=True)
        model.encode_batch(x).sum().backward()
        assert x.grad is not None

    def test_long_input_training_step(self) -> None:
        """A batch many times the nominal length yields a finite scalar loss."""
        model = _small_model()
        loss = model._compute_loss(_data(time=TIME * 20))
        assert loss.ndim == 0
        assert loss.requires_grad
        assert math.isfinite(loss.item())

    def test_max_train_length_crops_training_only(self) -> None:
        """max_train_length bounds the training path and leaves encode() alone."""
        model = _small_model(max_train_length=32)
        assert math.isfinite(model._compute_loss(_data(time=400)).item())
        assert model.encode(_data(time=400), batch_size=2).shape[0] == BATCH

    def test_max_train_length_defaults_to_no_op(self) -> None:
        """The default must not perturb a working configuration."""
        assert SimCLRModelParameters(input_dim=1).max_train_length is None


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


class TestAugmentation:
    """The producer contract is SimCLR's view-generation hook."""

    def test_default_producer_returns_view_pair(self) -> None:
        """With augmentation=None a default pair is built at init."""
        model = _small_model()
        assert model._augmentation is not None
        assert isinstance(model._augmentation.produce(_data()), ViewPair)

    def test_default_pair_views_differ(self) -> None:
        """Two views of the same batch must not be identical."""
        pair = _default_simclr_pair().produce(_data())
        assert not torch.equal(pair.first, pair.second)

    def test_injected_producer_passes_through(self) -> None:
        """A custom producer is stored verbatim."""
        producer = IndependentPairProducer(aug=Jitter(JitterParameters(sigma=0.2)))
        assert _small_model(augmentation=producer)._augmentation is producer

    def test_injected_producer_is_used(self) -> None:
        """The training path calls produce() exactly once per step."""

        class CountingProducer:
            def __init__(self) -> None:
                self.calls = 0
                self._inner = IndependentPairProducer(aug=Jitter())

            def produce(self, x: torch.Tensor) -> ViewPair:
                self.calls += 1
                return self._inner.produce(x)

        producer = CountingProducer()
        model = _small_model(augmentation=producer)
        model._compute_loss(_data())
        model._compute_loss(_data())
        assert producer.calls == 2

    def test_accepts_aligned_pair_producer_by_covariance(self) -> None:
        """AlignedPair is-a ViewPair, so an AlignedPair producer fits the slot."""
        model = _small_model(augmentation=FullOverlapProducer(aug=Jitter()))
        assert math.isfinite(model._compute_loss(_data()).item())

    def test_producer_satisfies_protocol(self) -> None:
        """The default producer conforms to AugmentationProducer[ViewPair]."""
        producer: AugmentationProducer[ViewPair] = _default_simclr_pair()
        assert isinstance(producer.produce(_data()), ViewPair)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class TestTraining:
    """End-to-end Lightning training."""

    @pytest.mark.parametrize("with_labels", [False, True])
    def test_trains_with_finite_losses(self, with_labels: bool) -> None:
        """Five steps, bare-tensor and (data, labels) batches alike."""
        num_steps = 5
        model = _small_model()
        data = _data(batch=BATCH * num_steps)
        dataset = (
            TensorDataset(data, torch.zeros(len(data), dtype=torch.long))
            if with_labels
            else TensorDataset(data)
        )

        collected: list[torch.Tensor] = []
        original_step = model.training_step

        def patched(batch: object, batch_idx: int) -> torch.Tensor:
            loss = original_step(batch, batch_idx)  # type: ignore[arg-type]
            collected.append(loss.detach())
            return loss

        model.training_step = patched  # type: ignore[method-assign]
        trainer = pl.Trainer(
            accelerator="cpu",
            max_steps=num_steps,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=BATCH))

        assert len(collected) == num_steps
        assert all(loss.ndim == 0 and math.isfinite(loss.item()) for loss in collected)

    def test_loss_decreases_on_a_separable_batch(self) -> None:
        """Overfitting one batch drives NT-Xent down — the objective is wired up."""
        torch.manual_seed(0)
        jitter = Jitter(JitterParameters(sigma=0.05))
        model = _small_model(augmentation=IndependentPairProducer(aug=jitter))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = _data(batch=8)
        first = model._compute_loss(x)
        for _ in range(30):
            optimizer.zero_grad()
            loss = model._compute_loss(x)
            loss.backward()
            optimizer.step()
        assert loss.item() < first.item()

    def test_configure_optimizers_returns_adam_without_a_trainer(self) -> None:
        """A single Adam over every parameter, as the reference uses.

        Outside a Trainer there is no epoch budget to anneal over, so the
        scheduler is dropped rather than guessed at.
        """
        model = _small_model()
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-6)

    def test_configure_optimizers_respects_disabled_scheduler(self) -> None:
        """``use_lr_scheduler=False`` trains at a constant learning rate."""
        model = _small_model(use_lr_scheduler=False)
        model._trainer = SimpleNamespace(max_epochs=100)
        assert isinstance(model.configure_optimizers(), torch.optim.Adam)

    @pytest.mark.parametrize("max_epochs", [None, -1, 0])
    def test_configure_optimizers_needs_a_finite_epoch_budget(self, max_epochs: int | None) -> None:
        """Step-bounded and open-ended runs have no horizon to anneal over."""
        model = _small_model()
        model._trainer = SimpleNamespace(max_epochs=max_epochs)
        assert isinstance(model.configure_optimizers(), torch.optim.Adam)

    def test_validation_step_runs(self) -> None:
        """The validation path produces a finite loss under a real Trainer."""
        model = _small_model()
        loader = DataLoader(TensorDataset(_data(batch=BATCH * 2)), batch_size=BATCH)
        trainer = pl.Trainer(
            accelerator="cpu",
            limit_val_batches=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        (results,) = trainer.validate(model, dataloaders=loader, verbose=False)
        assert math.isfinite(results["val_loss_epoch"])


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #


class TestLearningRateSchedule:
    """Linear warmup then cosine annealing, per google-research/simclr."""

    @staticmethod
    def _trace(*, max_epochs: int, warmup_epochs: int = 10) -> list[float]:
        """Return the per-epoch learning rates a full run would see."""
        model = _small_model(warmup_epochs=warmup_epochs)
        model._trainer = SimpleNamespace(max_epochs=max_epochs)
        config = model.configure_optimizers()
        optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]
        rates = []
        for _ in range(max_epochs):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return rates

    def test_scheduler_steps_once_per_epoch(self) -> None:
        """Cosine annealing is defined over epochs, not optimizer steps."""
        model = _small_model()
        model._trainer = SimpleNamespace(max_epochs=100)
        assert model.configure_optimizers()["lr_scheduler"]["interval"] == "epoch"

    def test_warmup_ramps_up_and_peaks_at_the_base_rate(self) -> None:
        """The rate climbs from ~0 and tops out at ``learning_rate``."""
        rates = self._trace(max_epochs=100)
        assert rates[0] < LEARNING_RATE / 100
        assert rates[:10] == sorted(rates[:10])
        assert max(rates) == pytest.approx(LEARNING_RATE)
        assert rates.index(max(rates)) == 10

    def test_annealing_reaches_zero_by_the_final_epoch(self) -> None:
        """The last epochs barely move the weights, so training settles.

        This is what makes a run reproducible: with a constant rate the final
        weights are wherever the optimizer happened to be when the budget ran
        out, and arbitrarily small numeric differences compound into different
        models. See ``configure_optimizers``.
        """
        rates = self._trace(max_epochs=100)
        assert rates[11:] == sorted(rates[11:], reverse=True)
        assert rates[-1] < LEARNING_RATE / 1000

    def test_warmup_is_capped_at_half_the_budget(self) -> None:
        """A run shorter than ``warmup_epochs`` still spends half of it annealing."""
        rates = self._trace(max_epochs=4, warmup_epochs=10)
        assert rates.index(max(rates)) == 2
        assert rates[-1] < max(rates)

    def test_zero_warmup_starts_at_the_base_rate(self) -> None:
        """``warmup_epochs=0`` anneals from the first epoch."""
        rates = self._trace(max_epochs=20, warmup_epochs=0)
        assert rates[0] == pytest.approx(LEARNING_RATE)
        assert rates == sorted(rates, reverse=True)

    def test_negative_warmup_is_rejected(self) -> None:
        """Config validation catches the nonsensical case at construction."""
        with pytest.raises(ValueError, match="warmup_epochs must be non-negative"):
            SimCLRModelParameters(input_dim=CHANNELS, warmup_epochs=-1)


# --------------------------------------------------------------------------- #
# Singleton-batch guard
# --------------------------------------------------------------------------- #


class TestSingletonBatch:
    """NT-Xent is batch-relative, so ``B == 1`` must not silently no-op."""

    def test_loss_is_nonzero_at_batch_size_one(self) -> None:
        """A singleton batch still produces a real loss.

        Without the split the only candidate is the anchor's own partner, the
        softmax is over a single logit, and the loss collapses to a constant
        with no gradient — training runs and learns nothing.
        """
        torch.manual_seed(0)
        model = _small_model()
        loss = model._compute_loss(_data(batch=1))
        assert math.isfinite(loss.item())
        assert loss.item() > 0.0

    def test_gradients_flow_at_batch_size_one(self) -> None:
        """The encoder receives non-zero gradient from a singleton batch."""
        torch.manual_seed(0)
        model = _small_model()
        model._compute_loss(_data(batch=1)).backward()
        grads = [p.grad for p in model._encoder.parameters() if p.grad is not None]
        assert grads, "encoder received no gradient at all"
        assert any(g.abs().sum().item() > 0.0 for g in grads)

    def test_split_is_a_no_op_above_batch_size_one(self) -> None:
        """Batches that already have negatives are passed through untouched."""
        batch = _data(batch=BATCH)
        assert torch.equal(ensure_pairable_batch(batch, split_count=3), batch)

    def test_short_series_is_left_alone(self) -> None:
        """Too short to split is returned unchanged, not padded into fake windows.

        ``min_window_len`` is the convolution kernel size, below which a window
        cannot be convolved at all.
        """
        short = _data(batch=1, time=2)
        assert torch.equal(ensure_pairable_batch(short, split_count=3, min_window_len=3), short)

    @pytest.mark.parametrize("split_count", [0, 1, -1])
    def test_config_rejects_degenerate_split_count(self, split_count: int) -> None:
        """Fewer than two windows cannot produce a negative pair."""
        with pytest.raises(ValueError, match="singleton_split_count must be >= 2"):
            SimCLRModelParameters(input_dim=CHANNELS, singleton_split_count=split_count)

    def test_singleton_batch_still_produces_gradients(self) -> None:
        """ensure_pairable_batch's window split must survive the shape change."""
        model = _small_model()
        model.train()
        loss = model._compute_loss(_data()[:1])
        loss.backward()
        stem_grad = model.encoder._stem[0].weight.grad
        assert stem_grad is not None
        assert torch.isfinite(stem_grad).all()
        assert stem_grad.abs().sum().item() > 0.0


# --------------------------------------------------------------------------- #
# Angular-diversity and training-regression guards
# --------------------------------------------------------------------------- #


class TestPoolingDecoupling:
    """Guarantees that decoupling pooling from the encoder holds."""

    def test_feature_map_preserves_angular_diversity(self) -> None:
        """The unpooled feature map must not collapse to a single direction.

        NT-Xent compares angles only, so a feature map whose flattened vectors
        all point the same way gives the loss nothing to learn from. Measured on
        structured input, never on ``torch.randn`` — white noise has no
        between-sample structure, so any encoder clusters it and the statistic
        says nothing about the encoder.
        """
        encoder = Conv1dResNetEncoder(input_dim=1, **_SMALL_ENCODER_KWARGS)
        encoder.eval()
        with torch.no_grad():
            flat = encoder(_structured_batch(sample_count=64)).flatten(1)

        unit = functional.normalize(flat, dim=-1)
        similarity = unit @ unit.T
        off_diagonal = similarity[~torch.eye(len(flat), dtype=torch.bool)]
        assert off_diagonal.mean().item() < 0.95

        _, singular_values, _ = torch.linalg.svd(flat)
        assert (singular_values[0] / singular_values[1]).item() < 20.0

    def test_contrastive_loss_escapes_the_uniform_baseline(self) -> None:
        """Loss must fall meaningfully below log(K) — the flat-logit floor.

        A contrastive loss pinned at ``log(K)`` means every logit is equal:
        the encoder emits one direction for every sample and NT-Xent has no
        gradient. This is the defect that pooling inside the encoder caused,
        and it is invisible to a shape assertion, so it is asserted here
        directly. Kept short (80 steps, tiny backbone) so it stays a unit test.
        """
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(1)
        model = _small_model(input_dim=1)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for _ in range(80):
            loss = model._compute_loss(_structured_batch(sample_count=16, generator=generator))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        uniform_baseline = math.log(2 * 16 - 1)
        assert losses[0] <= uniform_baseline + 0.05
        assert min(losses[-10:]) < uniform_baseline - 0.2, (
            f"loss did not escape the uniform baseline: "
            f"start={losses[0]:.4f} end={losses[-1]:.4f} floor={uniform_baseline:.4f}"
        )

    def test_vector_output_equals_the_previous_pooled_encoder(self) -> None:
        """VECTOR must be bit-comparable to the old in-encoder GAP.

        ``AdaptiveAvgPool1d(1)`` followed by ``flatten(1)`` is arithmetically a
        mean over the last axis. Moving that mean from the encoder into
        ``_encode_batch`` must not change a single downstream value; probes and
        checkpoints depend on it.
        """
        model = _small_model()
        model.eval()
        batch = _data()
        with torch.no_grad():
            feature_map = model.encoder(batch)
            legacy = nn.AdaptiveAvgPool1d(1)(feature_map).flatten(1)
            current = model.encode_batch(batch)
        assert current.shape == (BATCH, model.representation_dim)
        assert torch.allclose(current, legacy, atol=1e-6)

    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    def test_forward_accepts_any_sequence_length(self, seq_len: int) -> None:
        """No parameter shape depends on T', so any length works untouched."""
        model = _small_model()
        batch = torch.randn(BATCH, seq_len, CHANNELS)
        z = model(batch)
        assert z.shape[0] == BATCH
        assert torch.allclose(z.norm(dim=-1), torch.ones(BATCH), atol=1e-5)
        assert model.encode_batch(batch).shape == (BATCH, model.representation_dim)
