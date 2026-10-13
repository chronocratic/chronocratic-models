"""Registry describing how to build, feed and judge each model in the pipeline tests.

Adding a model to ``chronocratic.models`` requires adding a ``ModelSpec`` here;
``test_completeness.py`` fails otherwise.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial

import lightning.pytorch as pl

from chronocratic.models import (
    AutoTCL,
    CoST,
    MCL,
    RecurrentAutoEncoder,
    Series2Vec,
    SimCLR,
    TimeNet,
    TimeVAE,
    TS2Vec,
    TST,
    TSTCC,
)
from chronocratic.models.supervised import (
    make_series2vec_supervised,
    make_tst_supervised,
    make_tstcc_supervised,
)
from tests.pipeline.datamodules import (
    LoaderMode,
    SyntheticClassificationDataModule,
    SyntheticForecastingDataModule,
)
from tests.pipeline.synthetic import Scaling

__all__ = ["MODEL_SPECS", "ModelSpec", "Task", "spec_id"]


class Task(StrEnum):
    """Which learn check applies."""

    CONTRASTIVE = "contrastive"  # loss drop + gradients + linear probe
    GENERATIVE = "generative"  # loss drop + gradients + reconstruction baseline
    SUPERVISED = "supervised"  # loss drop + gradients + accuracy above chance


@dataclass(frozen=True)
class ModelSpec:
    """How to build, feed and judge one model.

    Pass/fail thresholds are shared by every spec (see ``test_pipeline.py``); the
    pipeline tests are a canary, so models differ only in how long they train.
    """

    name: str
    model_class: type[pl.LightningModule]
    task: Task
    build: Callable[[pl.LightningDataModule], pl.LightningModule]
    make_datamodule: Callable[[], pl.LightningDataModule]
    # Per-model knobs are epochs and the datamodule; any override needs a comment
    # with the measured reason. Thresholds are shared (test_pipeline.py).
    epochs: int = 10
    frozen_groups: frozenset[str] = frozenset()
    encode_kwargs: Mapping[str, object] = field(default_factory=dict)
    reconstruct: Callable[[pl.LightningModule, object], object] | None = None
    loss_key: str | None = None
    xfail: str | None = None


def spec_id(spec: ModelSpec) -> str:
    """pytest id for a spec."""
    return spec.name


def _classification(
    *, loader_mode: LoaderMode = LoaderMode.SAMPLE_LABEL, **overrides: object
) -> Callable[[], SyntheticClassificationDataModule]:
    return partial(SyntheticClassificationDataModule, loader_mode=loader_mode, **overrides)


def _reconstruct(model: pl.LightningModule, x: object) -> object:
    return model(x)


# --- builders: each takes the datamodule and returns a fresh, small model ----------


def _ts2vec(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return TS2Vec(input_dim=dm.num_features, hidden_dim=16, representation_dim=32, depth=3)


def _autotcl(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return AutoTCL(
        input_dim=dm.num_features,
        kernel_sizes=(1, 2, 4, 8),
        hidden_dim=8,
        representation_dim=16,
        depth=2,
    )


def _cost(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return CoST(
        input_dim=dm.num_features,
        sequence_length=dm.sequence_length,
        kernel_sizes=(1, 2, 4, 8),
        max_train_length=dm.sequence_length,
        hidden_dim=16,
        representation_dim=32,
        depth=3,
        queue_size=64,
    )


def _mcl(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return MCL(
        input_dim=dm.num_features,
        representation_dim=32,
        encoder_channels=(16, 32, 16),
        projection_dim=16,
    )


def _series2vec(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return Series2Vec(
        input_dim=dm.num_features,
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=32,
        representation_dim=32,
        sequence_length=dm.sequence_length,
    )


def _simclr(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return SimCLR(
        input_dim=dm.num_features,
        stem_conv_channels=8,
        encoder_stage_channels=(8, 16),
        encoder_stage_depths=(1, 1),
        encoder_stage_strides=(1, 2),
        projection_hidden_dim=32,
        projection_dim=16,
        use_lr_scheduler=False,
    )


def _tstcc(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return TSTCC(
        input_dim=dm.num_features,
        sequence_length=dm.sequence_length,
        conv_kernel_size=8,
        stride=1,
        representation_dim=32,
        encoder_channels=(16, 32),
        temporal_contrast_hidden_dim=32,
        temporal_contrast_timesteps=4,
    )


def _tst(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return TST(
        input_dim=dm.num_features,
        sequence_length=dm.sequence_length,
        hidden_dim=16,
        num_heads=2,
        depth=1,
        feedforward_dim=32,
    )


def _timevae(dm: SyntheticForecastingDataModule) -> pl.LightningModule:
    return TimeVAE(
        sequence_length=dm.sequence_length,
        input_dim=dm.num_features,
        latent_dim=8,
        hidden_layer_sizes=(16, 32),
    )


def _recurrent_ae(dm: SyntheticForecastingDataModule) -> pl.LightningModule:
    return RecurrentAutoEncoder(
        input_dim=dm.num_features, layers=(32, 16), dropout=0.0, learning_rate=1e-2
    )


def _timenet(dm: SyntheticForecastingDataModule) -> pl.LightningModule:
    return TimeNet(
        input_dim=dm.num_features, hidden_dim=32, depth=1, dropout_rate=0.0, learning_rate=1e-2
    )


def _tst_supervised(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return make_tst_supervised(
        _tst(dm), num_outputs=dm.num_classes, freeze_backbone=False, learning_rate=3e-3
    )


def _series2vec_supervised(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return make_series2vec_supervised(
        _series2vec(dm), num_outputs=dm.num_classes, freeze_backbone=False, learning_rate=3e-3
    )


def _tstcc_supervised(dm: SyntheticClassificationDataModule) -> pl.LightningModule:
    return make_tstcc_supervised(
        _tstcc(dm), num_outputs=dm.num_classes, freeze_backbone=False, learning_rate=3e-3
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="ts2vec",
        frozen_groups=frozenset({"_averaged_encoder"}),  # SWA copy, updated outside backprop
        model_class=TS2Vec,
        task=Task.CONTRASTIVE,
        build=_ts2vec,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="autotcl",
        frozen_groups=frozenset({"_averaged_encoder"}),  # SWA copy, updated outside backprop
        model_class=AutoTCL,
        task=Task.CONTRASTIVE,
        build=_autotcl,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="cost",
        epochs=30,
        model_class=CoST,
        task=Task.CONTRASTIVE,
        build=_cost,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="mcl",
        epochs=20,
        model_class=MCL,
        task=Task.CONTRASTIVE,
        build=_mcl,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE),
    ),
    ModelSpec(
        name="series2vec",
        model_class=Series2Vec,
        task=Task.CONTRASTIVE,
        build=_series2vec,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE),
    ),
    ModelSpec(
        name="simclr",
        epochs=30,
        model_class=SimCLR,
        task=Task.CONTRASTIVE,
        build=_simclr,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="tstcc",
        epochs=20,
        model_class=TSTCC,
        task=Task.CONTRASTIVE,
        build=_tstcc,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="tstcc_minmax",
        # Guards the instance_normalize=True default (#100): with it off, TS-TCC does not
        # learn on raw MINMAX data, so this fails if the default is ever flipped.
        epochs=20,
        model_class=TSTCC,
        task=Task.CONTRASTIVE,
        build=_tstcc,
        make_datamodule=_classification(
            loader_mode=LoaderMode.SAMPLE_LABEL, scaling=Scaling.MINMAX
        ),
    ),
    ModelSpec(
        name="tst",
        epochs=30,
        model_class=TST,
        task=Task.CONTRASTIVE,
        build=_tst,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="tst_nan_padded",
        epochs=30,
        model_class=TST,
        task=Task.CONTRASTIVE,
        build=_tst,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL, nan_pad_fraction=0.4),
    ),
    ModelSpec(
        name="timevae",
        model_class=TimeVAE,
        task=Task.GENERATIVE,
        build=_timevae,
        make_datamodule=SyntheticForecastingDataModule,
        reconstruct=_reconstruct,
    ),
    ModelSpec(
        name="recurrent_ae",
        model_class=RecurrentAutoEncoder,
        task=Task.GENERATIVE,
        build=_recurrent_ae,
        make_datamodule=partial(SyntheticForecastingDataModule, seq_len=16),
        reconstruct=_reconstruct,
    ),
    ModelSpec(
        name="timenet",
        model_class=TimeNet,
        task=Task.GENERATIVE,
        build=_timenet,
        make_datamodule=partial(SyntheticForecastingDataModule, seq_len=16),
        reconstruct=_reconstruct,
    ),
    ModelSpec(
        name="tst_supervised",
        model_class=TST,
        task=Task.SUPERVISED,
        build=_tst_supervised,
        make_datamodule=_classification(loader_mode=LoaderMode.TST),
    ),
    ModelSpec(
        name="series2vec_supervised",
        model_class=Series2Vec,
        task=Task.SUPERVISED,
        build=_series2vec_supervised,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
    ModelSpec(
        name="tstcc_supervised",
        model_class=TSTCC,
        task=Task.SUPERVISED,
        build=_tstcc_supervised,
        make_datamodule=_classification(loader_mode=LoaderMode.SAMPLE_LABEL),
    ),
)
