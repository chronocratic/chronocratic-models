"""End-to-end: train each model through a DataModule and assert it learns.

Slow. Disable with ``pytest --skip-pipeline`` (or ``SKIP_PIPELINE_TESTS=1``).
"""

import lightning.pytorch as pl
import pytest

from tests.pipeline.checks import (
    gradients_flow,
    loss_dropped,
    probe_beats_baselines,
    recon_beats_baseline,
    supervised_accuracy,
)
from tests.pipeline.harness import fit_model
from tests.pipeline.specs import MODEL_SPECS, ModelSpec, spec_id, Task

pytestmark = pytest.mark.pipeline

_SEED = 0


def _marked(spec: ModelSpec) -> object:
    marks = [pytest.mark.xfail(reason=spec.xfail, strict=True)] if spec.xfail else []
    return pytest.param(spec, id=spec_id(spec), marks=marks)


@pytest.mark.parametrize("spec", [_marked(spec) for spec in MODEL_SPECS])
def test_model_learns_on_synthetic_data(spec: ModelSpec) -> None:
    datamodule = spec.make_datamodule()
    datamodule.setup(stage="fit")

    pl.seed_everything(_SEED, workers=True)
    random_init = spec.build(datamodule) if spec.task is Task.CONTRASTIVE else None
    pl.seed_everything(_SEED, workers=True)
    model = spec.build(datamodule)

    record = fit_model(
        model=model, datamodule=datamodule, epochs=spec.epochs, loss_key=spec.loss_key
    )

    loss_dropped(record=record, max_ratio=spec.max_loss_ratio)
    gradients_flow(record=record)
    if spec.task is Task.SUPERVISED:
        supervised_accuracy(model=model, datamodule=datamodule, margin=spec.probe_margin)
        return
    if spec.task is Task.GENERATIVE:
        assert spec.reconstruct is not None, f"{spec.name}: generative spec needs `reconstruct`"
        recon_beats_baseline(model=model, datamodule=datamodule, reconstruct=spec.reconstruct)
        return
    probe_beats_baselines(
        trained=model,
        random_init=random_init,
        datamodule=datamodule,
        encode_kwargs=spec.encode_kwargs,
        margin=spec.probe_margin,
        random_tolerance=spec.random_tolerance,
    )
