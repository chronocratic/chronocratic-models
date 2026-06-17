# Coding Conventions

**Analysis Date:** 2026-06-17

## Naming Patterns

**Files:**
- `snake_case` for all Python modules: `config.py`, `supervised.py`, `encoding.py`
- Private/internal modules prefixed with underscore: `_adapters.py`, `_callbacks.py`, `_utils.py`
- Test files prefixed with `test_`: `test_config.py`, `test_augmentation.py`, `test_smoke.py`
- Barrel files (`__init__.py`) use explicit `__all__` exports with named imports (not star imports)

**Classes:**
- `PascalCase`: `TS2Vec`, `SupervisedModule`, `CropShiftProducer`
- Config/parameter dataclasses suffixed with `Parameters`: `TS2VecModelParameters`, `CropShiftAugmentationParameters`, `JitterParameters`
- Mixin classes suffixed with `Mixin`: `PoolingEncodingMixin`, `DecompositionEncodingMixin`
- ABCs use descriptive names: `AugmentationTrainingStrategy`, `TrainableAugmentationProducer`
- Protocol classes use descriptive names: `Augmentation`, `AugmentationProducer[V]`, `RepresentationBackbone`, `BatchAdapter`
- Head classes suffixed with `Head`: `FlattenLinearHead`
- Test classes prefixed with `Test`: `TestTS2VecModelParameters`, `TestSupervisedModule`
- Test helper classes prefixed with underscore: `_DummyEncoder`, `_DummyBackbone`, `_PoolingTestModel`
- View-set dataclasses: `SingleView`, `ViewPair`, `AlignedPair`

**Functions:**
- `snake_case`: `training_step`, `configure_optimizers`, `extract_features_from_batch`
- Private methods prefixed with single underscore: `_shared_step`, `_encode_augmented_views`, `_validate_task`
- Factory functions use `make_*` prefix: `make_tst_supervised`, `make_series2vec_supervised`, `make_tstcc_supervised`
- Representation functions suffixed with `*_representations`: `tst_representations`, `series2vec_representations`
- Batch adapters suffixed with `*_batch_adapter`: `tst_batch_adapter`, `supervised_batch_adapter`
- Internal helpers prefixed with underscore: `_normalize_dim`, `_should_apply`, `_min_permute_len`

**Variables:**
- `snake_case`: `batch_size`, `learning_rate`, `train_loss`
- Module-level constants: `UPPER_SNAKE_CASE`: `_VALID_TASKS`, `_EXPECTED_INPUT_DIMS`
- Private instance attributes: leading underscore: `_backbone`, `_head`, `_encoder`, `_augmentation`
- Module logger: `_logger = logging`

**Types:**
- `PascalCase` for type aliases, protocols, and dataclasses
- Enums: `PascalCase` (`MaskMode`)
- Python 3.12 native union syntax: `torch.Tensor | None`, `int | None`
- Built-in generics: `list[int]`, `tuple[torch.Tensor, ...]`
- PEP 695 type parameters for variance: `class AugmentationProducer[V](Protocol)`
- `collections.abc.Callable` imported via `TYPE_CHECKING` block

## Code Style

**Formatter:** Ruff (`ruff.toml`)
- Line length: 100
- Quote style: **double quotes**
- Indentation: spaces (4 per level)
- Skip magic trailing comma: `true`
- Target version: `py312`

**Linter:** Ruff (`ruff.toml`)
- Select: `ALL` rules
- Key ignored rules:
  - `D107` -- Allow missing docstring in `__init__` when class docstring is present
  - `D100` -- Allow missing module docstring
  - `D101` -- Allow missing class docstring
  - `RET504` -- Allow unnecessary variable assignment in return (readability in AI code)
  - `PLR0913` -- Allow too many arguments in functions (readability in AI code)
  - `COM812` -- Skip trailing comma enforcement (formatter conflict)
  - `Q000` -- Quote-style lint silenced (formatter enforces double quotes)
  - `D212` -- First-line summary style disabled in favor of D213
  - `INP001` -- Allow implicit namespace packages
- Per-file ignores:
  - `__init__.py`: `D104`, `F401`, `F403`, `F405` (barrel re-export files)
  - `tests/**/*.py`: All `D` rules, `PLR2004`, `S101` (no docstrings, magic numbers, assertions allowed)
  - `notebooks/**/*.ipynb`: All `D` rules, `E402`, `T201`
- Excluded from linting: `dependencies/`, `experiments_output/`, `src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py`

**Import sorting:** isort via ruff
- `combine-as-imports = true`
- `force-sort-within-sections = true`
- `order-by-type = false`
- `split-on-trailing-comma = false`

**Type checker:** `ty` (dev dependency in `pyproject.toml`)

## Import Organization

**Order observed in source files:**

1. `from __future__ import annotations` (when needed for forward refs; not universal)
2. `__all__` exports list (always first code line after imports)
3. Standard library imports (`abc`, `dataclasses`, `typing`, `math`, `tempfile`, `os`)
4. Third-party imports (`torch`, `lightning.pytorch`, `numpy`, `einops`)
5. Local imports (`from chronocratic.models ...`)

**`TYPE_CHECKING` pattern:** Used to avoid circular imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from chronocratic.models.transformer.tst.model import TST
```

See: `src/chronocratic/models/supervised/supervised.py`, `src/chronocratic/models/supervised/factory.py`, `src/chronocratic/models/augmentation/primitives.py`

**Lazy imports:** Used inside methods to break circular dependencies:

```python
def __init__(self, ...):
    if augmentation is None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (  # noqa: PLC0415
            CropShiftProducer,
        )
        self._augmentation = CropShiftProducer()
```

**`from __future__ import annotations`:** Used selectively (25 of 87 source files). Present in barrel files, factory/callback files, model files with circular deps, and protocol definitions. Absent in standalone config files like `ts2vec/config.py`, `autotcl/config.py`.

## Type Annotations

**All public functions have type hints**, including return types.

- Function parameters: explicit types (`input_dims: int`, `learning_rate: float`)
- Return types: always present (`-> None`, `-> torch.Tensor`, `-> SupervisedModule`)
- Union types: use `|` syntax: `torch.Tensor | None`, `int | None`

**Dataclass patterns (config classes):**

```python
@dataclass(kw_only=True)
class TS2VecModelParameters:
    """Configuration for the TS2Vec model.

    Args:
        input_dims: Number of input features (channels).
        hidden_dims: Number of hidden units in each encoder layer.
    """
    input_dims: int
    hidden_dims: int = 64
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
```

- Always `kw_only=True` for config dataclasses
- Use `field(default_factory=lambda: ...)` for mutable defaults (lists)
- Frozen dataclasses for view-set types: `@dataclass(frozen=True)` on `SingleView`, `ViewPair`, `AlignedPair`

**Protocol pattern (structural typing):**

```python
@runtime_checkable
class Augmentation(Protocol):
    """Structural protocol for model-agnostic augmentation primitives."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

**Generic protocol pattern (PEP 695):**

```python
class AugmentationProducer[V](Protocol):
    """Assembles the view set a model's loss requires from a batch."""

    def produce(self, x: torch.Tensor) -> V:
        ...
```

See: `src/chronocratic/models/augmentation/base.py`

**`@override` decorator:** Used for mixin method overrides. See `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py`

**`cast()` from typing:** Used for type narrowing when static checkers need help:

```python
optimizer = cast('torch.optim.Optimizer', self.optimizers())
```

## `__all__` Exports

**Every module declares `__all__`** at the top of the file, listing public symbols:

```python
__all__ = [
    "AlignedPair",
    "Augmentation",
    "AugmentationProducer",
    "AugmentationTrainingStrategy",
    "Reseedable",
    "SingleView",
    "TrainableAugmentationProducer",
    "ViewPair",
]
```

**Barrel files** use named imports (not star imports) for clean re-exports:

```python
from .convolutional import (
    AutoTCL,
    CoST,
    TS2Vec,
    ...
)

__all__ = ['FCN', 'TST', 'TSTCC', 'AutoTCL', ...]
```

See: `src/chronocratic/models/__init__.py`

## Error Handling

**Validation errors:** Raise `ValueError` with descriptive messages stored in `msg` variable:

```python
def _validate_task(task: str) -> None:
    if task not in _VALID_TASKS:
        msg = f"task must be 'classification' or 'regression', got '{task}'"
        raise ValueError(msg)
```

See: `src/chronocratic/models/supervised/factory.py`

**Input validation:** Raise `ValueError` for invalid parameter values:

```python
msg = f"Unsupported batch format; {type(batch)}"
raise ValueError(msg)
```

See: `src/chronocratic/models/utils.py`

**Type errors:** Let Python raise `TypeError` naturally (e.g., missing required args). Tests verify this:

```python
def test_missing_input_dims_raises(self) -> None:
    with pytest.raises(TypeError):
        CoSTModelParameters(sequence_length=100)  # type: ignore[call-arg]
```

**ABC enforcement:** Abstract classes cannot be instantiated; `with pytest.raises(TypeError)` verifies this.

**Frozen dataclasses:** Immutable view-set types raise `FrozenInstanceError` on mutation.

**No try/except for control flow.** Exceptions propagate up from boundaries.

## Logging

**Framework:** Python standard library `logging` via private alias.

**Module-level:** `_logger = logging` (see `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py`)

**Lightning metrics:** Use `self.log()` for training/validation metrics:

```python
self.log(
    "train_loss",
    loss,
    on_step=True,
    on_epoch=True,
    prog_bar=True,
    sync_dist=self._sync_dist,
)
```

See: `src/chronocratic/models/supervised/supervised.py`

**No `print()` statements** in production code.

## Docstrings

**Style:** Google (configured via `ruff.toml`: `convention = "google"`)

**Module docstrings:** Multi-line description with scope and exported symbols list:

```python
"""Abstract base classes for augmentation strategies.

This module defines the shared augmentation hierarchy...

Exported symbols:
    - ``Augmentation``: Structural protocol for primitive transforms.
    - ``AugmentationProducer[V]``: Protocol for typed view-set production.
"""
```

**Class docstrings:** Description followed by `Args:` section:

```python
class SupervisedModule(pl.LightningModule):
    """Generic supervised-training wrapper for labeled downstream tasks.

    Args:
        backbone: A (possibly pretrained) model exposing the representation fn.
        head: Maps a representation tensor to ``(B, num_outputs)``.
        representation_fn: ``(backbone, *encoder_inputs) -> Tensor``.
    """
```

**Function docstrings:** Description, `Args:`, `Returns:`:

```python
def forward(self, *encoder_inputs: torch.Tensor) -> torch.Tensor:
    """Run representations through the head.

    Args:
        encoder_inputs: Model-specific tensors passed to ``representation_fn``.

    Returns:
        Predictions of shape ``(B, num_outputs)``.
    """
```

**Cross-references:** Use `:class:\`Name\`` and `:meth:\`method_name\`` for RST-style Sphinx links.

**Dataclass docstrings:** Document fields in `Args:` section:

```python
@dataclass
class JitterParameters:
    """Parameters for :class:`Jitter`.

    Args:
        sigma: Std of the additive Gaussian noise.
        p: Probability of applying the transform. ``1.0`` means always.
    """
```

## Function Design

**Keyword-only arguments:** Constructors and factories enforce keyword arguments via `*,`:

```python
def __init__(
    self,
    *,
    input_dims: int,
    augmentation: AugmentationProducer[ViewPair] | None = None,
    hidden_dims: int = 64,
    ...
) -> None:
```

See: `src/chronocratic/models/convolutional/dilated/ts2vec/model.py`

Factory functions:

```python
def make_tst_supervised(
    backbone: TST,
    *,
    num_outputs: int,
    task: str = "classification",
    freeze_backbone: bool = True,
    ...
) -> SupervisedModule:
```

See: `src/chronocratic/models/supervised/factory.py`

**All function calls use keyword arguments** for clarity and flexibility.

## Module Design

**Private module convention:** Modules prefixed with `_` are internal:
- `src/chronocratic/models/_mixin/encoding.py` -- Encoding mixins
- `src/chronocratic/models/supervised/_adapters.py` -- Batch adapters
- `src/chronocratic/models/supervised/_callbacks.py` -- Training callbacks
- `src/chronocratic/models/supervised/_utils.py` -- Loss utilities

**Design patterns:**
- **Strategy:** `AugmentationTrainingStrategy` for swappable augmentation network training
- **Protocol / Structural typing:** `Augmentation`, `AugmentationProducer[V]`, `RepresentationBackbone`, `BatchAdapter`
- **Factory:** `make_tst_supervised`, `make_series2vec_supervised`, `make_tstcc_supervised`
- **Mixin:** `PoolingEncodingMixin`, `DecompositionEncodingMixin`, `BasicEncodingMixin`
- **Template Method:** `_shared_step` in `SupervisedModule`
- **Decorator:** `Seeded` wraps producers for deterministic RNG injection
- **Composition:** `ComposeAugmentation` chains primitives

**Standard file layout:**
```python
"""Module docstring with description and exported symbols."""

from __future__ import annotations  # if needed

__all__ = [...]
# Standard library
# Third-party
# Local (via TYPE_CHECKING when needed)
```

**Section dividers:** ASCII art comments group logical sections:
```python
# --------------------------------------------------------------------------- #
# Augmentation Protocol (primitive, model-agnostic)
# --------------------------------------------------------------------------- #
```

## Comments

- Inline comments for tensor shapes: `# Shape: (batch, time, channels)`
- Section dividers: `# --------------------------------------------------------------------------- #`
- Verification IDs in tests: `(VER-01 through VER-07)` for traceability
- `# noqa: <code>` for rule suppressions on specific lines
- `# type: ignore[<reason>]` for known type limitations

---

*Convention analysis: 2026-06-17*
