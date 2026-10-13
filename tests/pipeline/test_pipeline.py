"""End-to-end: train each model through a DataModule and assert it learns.

Slow. Disable with ``pytest --skip-pipeline`` (or ``SKIP_PIPELINE_TESTS=1``).
"""

import lightning.pytorch as pl
import pytest

from tests.pipeline.checks import (
    gradients_flow,
    loss_dropped,
    probe_beats_chance,
    recon_beats_baseline,
    supervised_accuracy,
)
from tests.pipeline.harness import fit_model
from tests.pipeline.specs import MODEL_SPECS, ModelSpec, spec_id, Task

pytestmark = pytest.mark.pipeline

_SEED = 0

# Canary thresholds, shared by every spec. Loose on purpose: they catch a model that
# does not learn at all (flat loss, dead gradients, collapsed or untrained features),
# not one that learns slowly. Tune epochs in the spec, never these per model.
_MAX_LOSS_RATIO = 0.97  # late-epoch loss must sit >= 3% below early-epoch loss
_PROBE_MARGIN = 0.1  # contrastive: linear probe above chance by this much
_ACCURACY_MARGIN = 0.25  # supervised: untrained wrappers reach ~chance + 0.1
_RECON_MAX_RATIO = 0.9  # generative: untrained models sit at ~1.0 x mean baseline


def _marked(spec: ModelSpec) -> object:
    marks = [pytest.mark.xfail(reason=spec.xfail, strict=True)] if spec.xfail else []
    return pytest.param(spec, id=spec_id(spec), marks=marks)


@pytest.mark.parametrize("spec", [_marked(spec) for spec in MODEL_SPECS])
def test_model_learns_on_synthetic_data(spec: ModelSpec) -> None:
    datamodule = spec.make_datamodule()
    datamodule.setup(stage="fit")

    pl.seed_everything(_SEED, workers=True)
    model = spec.build(datamodule)

    record = fit_model(
        model=model, datamodule=datamodule, epochs=spec.epochs, loss_key=spec.loss_key
    )

    loss_dropped(record=record, max_ratio=_MAX_LOSS_RATIO)
    gradients_flow(record=record, frozen_groups=spec.frozen_groups)
    if spec.task is Task.SUPERVISED:
        supervised_accuracy(model=model, datamodule=datamodule, margin=_ACCURACY_MARGIN)
        return
    if spec.task is Task.GENERATIVE:
        assert spec.reconstruct is not None, f"{spec.name}: generative spec needs `reconstruct`"
        recon_beats_baseline(
            model=model,
            datamodule=datamodule,
            reconstruct=spec.reconstruct,
            max_ratio=_RECON_MAX_RATIO,
        )
        return
    probe_beats_chance(
        model=model, datamodule=datamodule, encode_kwargs=spec.encode_kwargs, margin=_PROBE_MARGIN
    )
