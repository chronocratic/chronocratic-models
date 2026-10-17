"""Training randomness must follow the global seed set by ``seed_everything``."""

from collections.abc import Callable
import io
from pathlib import Path
import re
import tokenize

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
import pytest
import torch

from chronocratic.models.augmentation.decorators import Seeded
from chronocratic.models.convolutional.dilated.encoders.masking import (
    generate_binomial_mask,
    generate_continuous_mask,
)
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer
from chronocratic.models.utils.utils import process_sample_length
from tests.pipeline.specs import MODEL_SPECS, ModelSpec, spec_id


def _crop(*, sample: torch.Tensor) -> torch.Tensor:
    return process_sample_length(sample=sample, max_sample_length=16)


def _binomial_mask(*, sample: torch.Tensor) -> torch.Tensor:
    return generate_binomial_mask(batch_size=sample.size(0), seq_length=sample.size(1))


def _continuous_mask(*, sample: torch.Tensor) -> torch.Tensor:
    return generate_continuous_mask(batch_size=sample.size(0), seq_length=sample.size(1))


def _crop_shift(*, sample: torch.Tensor) -> torch.Tensor:
    # A fresh producer per draw, as a fresh training run would build one.
    pair = CropShiftProducer().produce(sample)
    return torch.cat([pair.first, pair.second], dim=1)


@pytest.mark.parametrize("draw", [_crop, _binomial_mask, _continuous_mask, _crop_shift])
def test_draw_repeats_under_same_seed(draw: Callable[..., torch.Tensor]) -> None:
    """Two draws after the same seed are identical; a different seed changes them."""
    sample = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256, 1)

    def seeded_draws(*, seed: int) -> list[torch.Tensor]:
        seed_everything(seed=seed, verbose=False)
        return [draw(sample=sample) for _ in range(3)]

    first = seeded_draws(seed=42)
    second = seeded_draws(seed=42)
    other = seeded_draws(seed=7)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not all(
        a.shape == c.shape and torch.equal(a, c) for a, c in zip(first, other, strict=True)
    )


def test_seeded_producer_ignores_outer_state() -> None:
    """``Seeded`` repeats its views whatever the outer torch state is."""
    producer = Seeded(inner=CropShiftProducer(), seed=3)
    sample = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64, 1)

    torch.manual_seed(0)
    first = producer.produce(sample)
    torch.manual_seed(1)
    second = producer.produce(sample)

    assert first.overlap_length == second.overlap_length
    assert torch.equal(first.first, second.first)
    assert torch.equal(first.second, second.second)


def _fitted_state(*, spec: ModelSpec) -> dict[str, torch.Tensor]:
    """Seed, build and train ``spec`` on two batches; return its weights and buffers."""
    seed_everything(seed=42, workers=True, verbose=False)
    datamodule = spec.make_datamodule()
    model = spec.build(datamodule)
    pl.Trainer(
        # A batch limit, not ``max_steps``: models that step their optimizers outside
        # Lightning (TS-TCC) never advance ``global_step``, so ``max_steps`` never stops.
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="cpu",
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(model=model, datamodule=datamodule)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=spec_id)
def test_training_repeats_under_same_seed(spec: ModelSpec) -> None:
    """Every model ends two short fits under the same seed with identical state.

    ``MODEL_SPECS`` must list every exported model (``test_completeness.py``), so a
    new model joins this test automatically.
    """
    first = _fitted_state(spec=spec)
    second = _fitted_state(spec=spec)

    assert first.keys() == second.keys()
    differing = [name for name in first if not torch.equal(first[name], second[name])]
    assert not differing, f"{spec.name}: state differs between seeded runs: {differing}"


# Generator APIs that the global torch seed does not reach.
_PRIVATE_GENERATOR_PATTERNS = (
    re.compile(r"\bnp\s*\.\s*random\b"),
    re.compile(r"\bnumpy\s*\.\s*random\b"),
    re.compile(r"\bdefault_rng\b"),
    re.compile(r"\bimport\s+random\b"),
    re.compile(r"\bfrom\s+random\s+import\b"),
    re.compile(r"\bfrom\s+numpy\s+import\s+random\b"),
    re.compile(r"\btorch\s*\.\s*Generator\s*\("),
)
_IGNORED_TOKENS = frozenset({tokenize.COMMENT, tokenize.STRING, tokenize.FSTRING_MIDDLE})
_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _private_generator_lines(*, source: str) -> list[int]:
    """Return the lines of ``source`` whose code (not comments or strings) uses a banned API."""
    code_by_line: dict[int, list[str]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _IGNORED_TOKENS:
            code_by_line.setdefault(token.start[0], []).append(token.string)
    return [
        line
        for line, parts in code_by_line.items()
        if any(pattern.search(" ".join(parts)) for pattern in _PRIVATE_GENERATOR_PATTERNS)
    ]


def test_no_private_random_generators_in_src() -> None:
    """All randomness in ``src`` must come from the global torch generator."""
    offenders = [
        f"{path.relative_to(_SOURCE_ROOT)}:{line}"
        for path in sorted(_SOURCE_ROOT.rglob("*.py"))
        for line in _private_generator_lines(source=path.read_text())
    ]
    assert not offenders, (
        f"Private random generators found: {offenders}. Draw with torch.randint, "
        "torch.rand and similar instead (see docs/contributing.md, 'Randomness and Seeding')."
    )


@pytest.mark.parametrize(
    "code",
    [
        "rng = np.random.default_rng()",
        "import random",
        "from numpy import random",
        "generator = torch.Generator()",
    ],
)
def test_scan_detects_private_generator(code: str) -> None:
    """The scan flags each banned API in code."""
    assert _private_generator_lines(source=f"x = 1\n{code}\n") == [2]


def test_scan_ignores_comments_and_strings() -> None:
    """Mentions in comments and docstrings are not code."""
    source = '"""The reference draws with np.random."""\n# import random\nx = 1\n'
    assert _private_generator_lines(source=source) == []
