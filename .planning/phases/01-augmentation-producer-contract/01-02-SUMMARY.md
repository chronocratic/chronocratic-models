---
phase: 01-augmentation-producer-contract
plan: 02
subsystem: augmentation
tags: [torch, pytest, TDD, Protocol, dataclass, augmentation-primitives]

# Dependency graph
requires:
  - phase: 01-augmentation-producer-contract
    provides: augmentation/base.py (AugmentationMethod ABC, TrainingViews)
provides:
  - Augmentation Protocol in base.py (callable, Tensor -> Tensor)
  - Shared primitives: Jitter, Scaling, Permutation, ComposeAugmentation in primitives.py
  - Parameter dataclasses: JitterParameters, ScalingParameters, PermutationParameters
  - 16 passing tests for all primitives
affects:
  - 01-03 (producers), 01-04 (decorators), model migrations (Phase B)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Augmentation Protocol (structural, @runtime_checkable, callable)
    - Primitive -> Tensor interface (no TrainingViews wrapping)
    - TYPE_CHECKING import for Protocol to avoid runtime coupling

key-files:
  created:
    - src/tscollection/models/augmentation/primitives.py
    - tests/test_aug_primitives.py
  modified:
    - src/tscollection/models/augmentation/base.py

key-decisions:
  - "Augmentation Protocol uses __call__ (not augment()) for primitive interface"
  - "Augmentation Protocol placed in base.py to share with all augmentation modules"
  - "Import Augmentation via TYPE_CHECKING block in primitives.py (from __future__ import annotations defers evaluation)"

requirements-completed: [G1]

# Metrics
duration: 5min
completed: 2026-06-12
---

# Phase 1 Plan 2: Augmentation Primitives Summary

**Augmentation Protocol and shared primitives (Jitter, Scaling, Permutation, ComposeAugmentation) extracted from tstcc/augmentations.py, reshaped to callable Tensor->Tensor interface with 16 passing TDD tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-12T11:15:39Z
- **Completed:** 2026-06-12T11:21:31Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments

- Added `Augmentation` Protocol to `base.py` — structural, `@runtime_checkable`, callable interface
- Created `primitives.py` with 4 shared primitives: Jitter, Scaling, Permutation, ComposeAugmentation
- Extracted parameter dataclasses with correct defaults matching original `tstcc/augmentations.py`
- All primitives return bare `torch.Tensor` (no `TrainingViews` wrapping) — primitive-to-producer separation
- 16 passing tests covering shape preservation, parameter defaults, protocol compliance, and chaining

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for primitives (RED)** - `25437c2` (test)
2. **Task 1: Implement primitives + Augmentation Protocol (GREEN)** - `6deccb0` (feat)

_Note: TDD task has two commits (test -> feat)_

## Files Created/Modified

- `src/tscollection/models/augmentation/primitives.py` — Shared primitives: Jitter, Scaling, Permutation, ComposeAugmentation + parameter dataclasses
- `src/tscollection/models/augmentation/base.py` — Added Augmentation Protocol (callable, Tensor -> Tensor)
- `tests/test_aug_primitives.py` — 16 tests: shape, noise, probability gating, per-sample scaling, permutation reordering, compose chaining, callable protocol, parameter defaults

## Decisions Made

- **Augmentation Protocol in base.py**: The `Augmentation` Protocol was added to `base.py` alongside existing ABCs. This follows SPEC §4.9 — shared module that all augmentation code imports from. It was not part of the plan but was required as a blocking dependency (Rule 3).
- **`__call__` over `augment()`**: Primitives use `__call__(x: Tensor) -> Tensor` instead of `augment(data, **kwargs) -> TrainingViews`. This is the primitive-to-producer separation — primitives are bare transforms, producers handle view assembly.
- **TYPE_CHECKING import**: `Augmentation` is imported via `TYPE_CHECKING` in `primitives.py` because `from __future__ import annotations` defers type annotation evaluation. This satisfies TC001 (type-checking block) ruff rule.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added Augmentation Protocol to base.py**
- **Found during:** Task 1 (primitives implementation)
- **Issue:** Plan expects `from tscollection.models.augmentation.base import Augmentation` but the Protocol did not exist in `base.py`. Without it, primitives cannot satisfy the planned interface.
- **Fix:** Added `@runtime_checkable class Augmentation(Protocol)` to `base.py` with `__call__(self, x: torch.Tensor) -> torch.Tensor`. Updated `__all__` to export it.
- **Files modified:** `src/tscollection/models/augmentation/base.py`
- **Verification:** Type check passes, existing tests still pass, primitives import successfully.
- **Committed in:** `6deccb0` (part of GREEN commit)

**2. [Rule 1 - Bug] Fixed flaky Permutation reordering test**
- **Found during:** Task 1 (GREEN phase test run)
- **Issue:** `torch.randint(1, max_segments)` can return `1`, causing the permutation to be a no-op. Original test used seed 123 which happened to produce this.
- **Fix:** Changed test to iterate over 10 seeds and assert that at least one produces reordering.
- **Files modified:** `tests/test_aug_primitives.py`
- **Verification:** Test passes consistently across runs.
- **Committed in:** `6deccb0` (part of GREEN commit)

**3. [Rule 1 - Bug] Fixed unused imports (lint)**
- **Found during:** Task 1 (GREEN phase ruff check)
- **Issue:** `fields` and `pytest` imported but unused in test file; `Augmentation` not in TYPE_CHECKING block.
- **Fix:** Removed unused imports; moved `Augmentation` to TYPE_CHECKING block.
- **Files modified:** `tests/test_aug_primitives.py`, `src/tscollection/models/augmentation/primitives.py`
- **Verification:** `ruff check` passes with zero errors.
- **Committed in:** `6deccb0` (part of GREEN commit)

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 bugs)
**Impact on plan:** Augmentation Protocol addition was necessary to unblock the planned import pattern. Test and lint fixes are correctness improvements.

## Issues Encountered

- **Worktree venv lacks numba**: The worktree creates an isolated `.venv` that doesn't include numba (not a declared dependency). Resolved by using the main repo's `.venv` directly with absolute PYTHONPATH. This is a pre-existing environment issue — existing tests also fail in the worktree's own venv.

## Verification

- `pytest tests/test_aug_primitives.py -v` — 16 passed
- `ty check src/tscollection/models/augmentation/primitives.py` — All checks passed
- `ruff check` — All checks passed
- `pytest tests/test_augmentation_base.py -v` — 7 passed (regression check)
- Import verification: `from tscollection.models.augmentation.primitives import Jitter, Scaling, Permutation, ComposeAugmentation` — succeeds

## Next Phase Readiness

- Primitives are ready for consumption by producers (plan 01-03)
- `Augmentation` Protocol in `base.py` is available for structural typing
- No breaking changes to existing code — old `AugmentationMethod` path remains intact

---
*Phase: 01-augmentation-producer-contract*
*Completed: 2026-06-12*
