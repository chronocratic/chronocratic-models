---
phase: 01-augmentation-producer-contract
reviewed: "2026-06-12T19:00:00Z"
depth: deep
files_reviewed: 15
files_reviewed_list:
  - src/tscollection/models/augmentation/__init__.py
  - src/tscollection/models/augmentation/base.py
  - src/tscollection/models/augmentation/decorators.py
  - src/tscollection/models/augmentation/primitives.py
  - src/tscollection/models/augmentation/producers.py
  - src/tscollection/models/augmentation/trainable_support.py
  - src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py
  - src/tscollection/models/convolutional/dilated/autotcl/augmentation/training.py
  - src/tscollection/models/convolutional/dilated/autotcl/model.py
  - src/tscollection/models/convolutional/dilated/autotcl/utils.py
  - src/tscollection/models/convolutional/dilated/cost/augmentation.py
  - src/tscollection/models/convolutional/dilated/cost/model.py
  - src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py
  - src/tscollection/models/convolutional/dilated/ts2vec/model.py
  - src/tscollection/models/convolutional/standard/tstcc/augmentations.py
findings:
  critical: 1
  warning: 5
  info: 7
  total: 13
status: issues_found
---

# Phase 01: Augmentation Producer Contract -- Code Review Report

**Reviewed:** 2026-06-12T19:00:00Z
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Deep review of the augmentation producer contract refactoring across 15 source files. The architecture is sound: clean separation between base abstractions (`Augmentation`, `AugmentationProducer[V]`, ViewSets), shared primitives (`Jitter`, `Scaling`, `Permutation`), producer combinators (`SingleViewProducer`, `IndependentPair`, `RolePair`, `FullOverlapPair`), and per-model concretions. Import hygiene is maintained (no circular dependencies). All four models (TS2Vec, CoST, AutoTCL, TSTCC) are wired to the new contract.

However, one barrel export gap will cause `import *` and docs-build failures, `FullOverlapPair` silently produces incorrect metadata for non-(B,T,C) layouts, `Permutation` crashes on short sequences, and two SPEC-design-doc references leak into production code. A total of 13 findings are documented below.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `CropShiftProducer` imported but missing from barrel `__all__`

**File:** `src/tscollection/models/augmentation/__init__.py:82-85` (import), `:88-122` (__all__)

**Issue:** `CropShiftProducer` is imported into the barrel namespace (line 84) but is absent from the `__all__` list (lines 88-122). Its companion `CropShiftAugmentationParameters` IS present in `__all__` (line 121). This means:

1. `from tscollection.models.augmentation import *` does NOT export `CropShiftProducer`.
2. Sphinx autodoc and IDE autocomplete that rely on `__all__` will not surface it from the barrel.
3. This is inconsistent with every other per-model concretion re-exported in the barrel.

**Fix:**
```python
# In __all__, add 'CropShiftProducer' alongside 'CropShiftAugmentationParameters':
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentationParameters',
    'CropShiftProducer',
    'RIPTrainingStrategy',
```

## Warnings

### WR-01: `FullOverlapPair.produce()` assumes (batch, time, channels) layout

**File:** `src/tscollection/models/augmentation/producers.py:168-172`

**Issue:** `overlap_length=x.size(1)` hard-codes the time dimension as index 1, i.e., (B, T, C) layout. TSTCC uses (B, C, T) layout. While TSTCC currently uses `RolePair`, cross-model reuse is a stated goal of this refactoring. If `FullOverlapPair` is injected into a TSTCC pipeline, `overlap_length` would be set to the channel count instead of the sequence length, producing silently incorrect alignment metadata.

**Fix:** Accept a `time_dim` parameter:

```python
class FullOverlapPair:
    def __init__(self, *, aug: Augmentation, time_dim: int = 1) -> None:
        self._aug = aug
        self._time_dim = time_dim

    def produce(self, x: torch.Tensor) -> AlignedPair:
        return AlignedPair(
            first=self._aug(x),
            second=self._aug(x),
            overlap_length=x.size(self._time_dim),
        )
```

### WR-02: `Permutation` crashes on sequences shorter than 3 time steps

**File:** `src/tscollection/models/augmentation/primitives.py:197`

**Issue:** When `seq_len < 3`, `torch.randperm(seq_len - 2)` receives a negative argument. `torch.randperm(-N)` raises `RuntimeError`. With `max_segments=5` (default), `num_segments` can be drawn from [1, 5), so `num_segments > 1` is possible. Short sequences (length 1 or 2) will crash rather than being handled gracefully.

**Fix:** Add a bounds check:

```python
def __call__(self, x: torch.Tensor) -> torch.Tensor:
    t_dim = _normalize_dim(x, self._params.time_dim)
    seq_len = x.size(t_dim)
    if seq_len < 3:
        return x.clone()  # Cannot meaningfully permute segments on short sequences
    batch_size = x.size(0)
    # ... rest of method
```

### WR-03: `maybe_train_augmentation` mode management is fragile

**File:** `src/tscollection/models/augmentation/trainable_support.py:70-74`

**Issue:** The helper sets `encoder.eval()` (line 70) and `augmentation.train()` (line 71), then `augmentation.eval()` (line 73) after training. If `augmentation.train_step()` returns `None` (permitted by the abstract signature `-> torch.Tensor | None`), the function returns `None` at line 74 with the encoder still in eval mode. The current AutoTCL implementation always returns a loss tensor, but subclasses or future changes could return `None` mid-path, leaving the encoder in eval mode for Phase 2 training -- silently disabling gradients through BatchNorm and dropout.

**Fix:** Guard the encoder mode restoration:

```python
if not isinstance(augmentation, TrainableAugmentationProducer):
    return None
if not augmentation.should_train_augmentation(epoch=epoch, batch_idx=batch_idx):
    return None
encoder.eval()
augmentation.train()
try:
    loss = augmentation.train_step(x=x, encoder=encoder, batch_idx=batch_idx)
finally:
    augmentation.eval()
    encoder.train()  # Always restore encoder to train mode
return loss
```

### WR-04: Inconsistent return types for backward-compat `augment()` aliases

**File:** `src/tscollection/models/convolutional/dilated/cost/augmentation.py:124-140`, `src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py:131-145`

**Issue:** Two backward-compat `augment()` shims exist with different return types:
- `CosTRandomFunctionAugmentation.augment()` returns raw `torch.Tensor`
- `AutoTCLNeuralNetworkAugmentation.augment()` returns `SingleView`

If external callers use `.augment()` polymorphically, the mismatch causes crashes. Both are explicitly marked as backward-compat, but having divergent contracts from the same "interface" is error-prone.

**Fix:** Remove both aliases (the old `TrainingViews` contract is deleted) or add deprecation warnings:

```python
import warnings

def augment(self, data: torch.Tensor, **kwargs: Any) -> SingleView:
    warnings.warn(
        "augment() is deprecated; use produce() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.produce(data)
```

### WR-05: `CosTRandomFunctionAugmentation` constructor ambiguity

**File:** `src/tscollection/models/convolutional/dilated/cost/augmentation.py:71-88`

**Issue:** The constructor accepts three parameter sources with ambiguous interaction:
1. `params: CosTRandomFunctionAugmentationParameters` dataclass
2. `params: dict[str, Any]`
3. `sigma: float` convenience kwarg

If both `params={'sigma': 0.1}` and `sigma=0.2` are passed, `sigma=0.2` is silently ignored (line 71: `if params is None and sigma is not None`). A caller would reasonably expect the explicit `sigma` kwarg to override or raise.

**Fix:** Add validation that both are not provided simultaneously:

```python
if params is not None and sigma is not None:
    msg = "Cannot specify both 'params' and 'sigma'. Use one or the other."
    raise ValueError(msg)
if params is None and sigma is not None:
    params = {'sigma': sigma}
```

## Info

### IN-01: Stale Sphinx cross-refs to deleted `TrainingViews`

**File:** `src/tscollection/models/augmentation/primitives.py:6`, `src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py:6`

**Issue:** Two module docstrings reference `:class:\`TrainingViews\``, a deleted symbol. Sphinx will fail to resolve these cross-references in nitpicky mode:
- `primitives.py:6`: `no :class:\`TrainingViews\` wrapping`
- `ts2vec/augmentation.py:6`: `Returns :class:\`AlignedPair\` instead of :class:\`TrainingViews\``

**Fix:** Replace with plain-text references or remove entirely:
```python
# primitives.py:6
of the same shape -- no TrainingViews wrapping.

# ts2vec/augmentation.py:6
Returns :class:`AlignedPair` instead of the old untyped metadata pattern,
```

### IN-02: SPEC design-doc reference in ts2vec/augmentation.py

**File:** `src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py:7`

**Issue:** Module docstring contains `(SPEC §2.3 root cause #1, §4.7, §5)`. This references internal planning documents that are not available to end users consuming this library.

**Fix:** Remove the parenthetical. The remaining docstring is self-sufficient.

### IN-03: SPEC references in autotcl/model.py diagnostic comments

**File:** `src/tscollection/models/convolutional/dilated/autotcl/model.py:204,213`

**Issue:** Two inline comments reference `SPEC §4.5.1 exception` to justify direct `isinstance(TrainableAugmentationProducer)` checks in diagnostic methods. While these document intentional design decisions, they reference internal artifacts. End users reading the source code or error messages will not have access to the SPEC document.

**Fix:** Replace with self-contained rationale:

```python
# Exception: diagnostic methods may branch on TrainableAugmentationProducer
# directly to manage eval/train mode during measurement.
```

### IN-04: Stale `TrainableAugmentation` + `TrainingViews` references in methods.py docstring

**File:** `src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py:8`

**Issue:** Module docstring references deleted symbols: `TrainableAugmentation` and `augment() -> TrainingViews`. These cannot be resolved by readers who only have the current codebase.

**Fix:**
```python
"""AutoTCL augmentation methods.

Contains ``AutoTCLNeuralNetworkAugmentation`` and its parameters dataclass.

Implements ``TrainableAugmentationProducer`` (nominal ABC + nn.Module)
with a ``produce() -> SingleView`` contract.
"""
```

### IN-05: Stale `CropShiftAugmentation` name in ts2vec/augmentation.py

**File:** `src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py:4`

**Issue:** Docstring mentions "reshaped from ``CropShiftAugmentation``", referencing a class that was deleted in this refactor.

**Fix:** Remove the parenthetical:

```python
"""TS2Vec augmentation: crop-and-shift producer.

Contains the ``CropShiftProducer`` class and its
``CropShiftAugmentationParameters`` dataclass.
"""
```

### IN-06: `extract_subsequences_per_row` docstring says "2D tensor" but works for N-D

**File:** `src/tscollection/models/convolutional/dilated/ts2vec/utils.py:10-11`

**Issue:** The docstring says "The input 2D tensor" but the function is called with 3D `(batch, time, channels)` tensors from `CropShiftProducer`. The advanced indexing works for both, so this is purely a documentation inaccuracy.

**Fix:**
```python
"""Extract subsequences from each row of a tensor based on starting
indices and subsequence length. Works for 2D and N-D tensors by
indexing the first two dimensions and preserving remaining axes.
"""
```

### IN-07: `backward compatibility` phrasing in parameter docstrings

**File:** `src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py:92,95`, `src/tscollection/models/convolutional/dilated/cost/augmentation.py:66`

**Issue:** Three parameter docstrings mention "backward compatibility":
- `methods.py:92`: `a dict with encoder kwargs for backward compatibility`
- `methods.py:95`: `backward compatibility with factory-based instantiation`
- `cost/augmentation.py:66`: `keys for backward compatibility`

These are legitimate shim comments that document why dict-acceptance exists. No fix required, but they should be converted to deprecation warnings eventually if these shims are intended to be removed.

**Fix:** Low priority. Add `DeprecationWarning` when dict params are used, with a timeline for removal.

## Cross-File Analysis

### Import Graph (verified acyclic)

```
augmentation/base.py              -- no upstream imports from augmentation/
augmentation/primitives.py        --> base.py (TYPE_CHECKING only)
augmentation/producers.py         --> base.py (runtime + TYPE_CHECKING)
augmentation/decorators.py        --> base.py (runtime)
augmentation/trainable_support.py --> base.py (runtime)
augmentation/__init__.py          --> base, primitives, producers, decorators, trainable_support
                                    --> autotcl/augmentation/methods, autotcl/augmentation/training
                                    --> cost/augmentation, ts2vec/augmentation
autotcl/model.py                  --> base.py, trainable_support.py (lazy import of methods.py)
autotcl/utils.py                  --> base.py
autotcl/augmentation/training.py  --> base.py (lazy import of utils.py)
cost/model.py                     --> base.py (lazy import of producers.py, cost/augmentation.py)
ts2vec/model.py                   --> base.py (lazy import of ts2vec/augmentation.py)
tstcc/model.py                    --> base.py (lazy import of tstcc/augmentations.py)
tstcc/augmentations.py            --> base.py, primitives.py, producers.py
```

No circular dependencies detected. Lazy imports in model files (`PLC0415`) correctly break potential cycles through the barrel.

### Type Consistency at Boundaries

- `AugmentationProducer[V]` is covariant (PEP 695); `AlignedPair` extends `ViewPair`, so `AugmentationProducer[AlignedPair]` is assignable to `AugmentationProducer[ViewPair]`. Verified by runtime `issubclass` check.
- `TrainableAugmentationProducer.produce()` returns `SingleView`, matching its structural satisfaction of `AugmentationProducer[SingleView]`.
- `maybe_train_augmentation` accepts `AugmentationProducer[Any]` and dispatches on `isinstance(TrainableAugmentationProducer)`. The `Any` type erasure is intentional and correct for the null-object pattern.

### Error Propagation

- `CropShiftProducer.produce()` raises `ValueError` when `min_crop_length >= total_length`. Propagates correctly.
- `calculate_regular_consistency` raises `ValueError` when `time_steps <= 3`. Propagates correctly.
- `Seeded.__init__` raises `TypeError` when wrapping trainable producers. Propagates correctly.
- `Permutation.__call__()` crashes on `seq_len < 3` without a descriptive error (WR-02).
- `maybe_train_augmentation()` returns `None` for early-exit paths correctly; mode management is fragile (WR-03).

---

_Reviewed: 2026-06-12T19:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
