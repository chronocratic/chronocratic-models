"""Every exported model must have a pipeline spec. Never skipped by --skip-pipeline."""

import inspect

import lightning.pytorch as pl
import pytest

from chronocratic import models
from tests.pipeline.specs import MODEL_SPECS


def _exported_lightning_modules() -> set[type[pl.LightningModule]]:
    exported = (getattr(models, name) for name in models.__all__)
    return {obj for obj in exported if inspect.isclass(obj) and issubclass(obj, pl.LightningModule)}


def test_every_exported_model_has_a_pipeline_spec() -> None:
    covered = {spec.model_class for spec in MODEL_SPECS}
    missing = sorted(cls.__name__ for cls in _exported_lightning_modules() - covered)
    assert not missing, (
        f"Models without a pipeline spec: {missing}. Add a ModelSpec to "
        "tests/pipeline/specs.py (see docs/contributing.md, 'Adding a model')."
    )


def test_every_spec_has_a_unique_name() -> None:
    names = [spec.name for spec in MODEL_SPECS]
    assert len(names) == len(set(names))


def test_completeness_check_detects_a_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NewModel(pl.LightningModule):
        pass

    monkeypatch.setattr(models, "__all__", [*models.__all__, "_NewModel"])
    monkeypatch.setattr(models, "_NewModel", _NewModel, raising=False)
    covered = {spec.model_class for spec in MODEL_SPECS}
    assert _NewModel in _exported_lightning_modules() - covered
