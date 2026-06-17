---
phase: 01-augmentation-producer-contract
plan: 01
subsystem: augmentation
tags: [protocol, dataclass, abc, torch, typing]

requires: []
provides:
  - Augmentation Protocol for primitive transforms (Tensor -> Tensor)
  - Typed ViewSet dataclasses: SingleView, ViewPair, AlignedPair
  - AugmentationProducer[V] covariant Protocol for view assembly
  - TrainableAugmentationProducer nominal ABC + nn.Module
  - Backward-compatible exports: TrainingViews, AugmentationMethod, TrainableAugmentation, AugmentationTrainingStrategy
affects: [augmentation-refactor, model-migration, primitives, producers]

tech-stack:
  added: [numba>=0.65.1]
  patterns: [Protocol structural typing, covariant TypeVar, frozen dataclass, nominal ABC + nn.Module]

key-files:
  created: [tests/test_aug_contract.py]
  modified: [src/tscollection/models/augmentation/base.py, pyproject.toml]

key-decisions:
  - "Augmentation Protocol is structural (not @runtime_checkable) — verified via method presence tests"
  - "AugmentationProducer[V] uses covariant TypeVar for Liskov substitution (AlignedPair <: ViewPair)"
  - "TrainableAugmentationProducer is a nominal ABC (not Protocol) for isinstance runtime gating"
  - "Existing symbols (TrainingViews, AugmentationMethod, TrainableAugmentation) retained for backward compat (D-05)"

requirements-completed: [G2, G6]

duration: 12min
completed: 2026-06-12
---

# Phase 1 Plan 1: Augmentation Contract Types Summary

**Augmentation Protocol, typed ViewSet dataclasses, covariant AugmentationProducer, and TrainableAugmentationProducer ABC added to base.py with 25 TDD-verified tests**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-12T11:15:03Z
- **Completed:** 2026-06-12T11:27:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments

- Augmentation Protocol: structural typing for primitive transforms (`__call__: Tensor -> Tensor`)
- ViewSet hierarchy: `SingleView`, `ViewPair`, `AlignedPair` as frozen dataclasses with `AlignedPair <: ViewPair`
- `AugmentationProducer[V]` covariant Protocol enabling Liskov substitution across view set types
- `TrainableAugmentationProducer` nominal ABC combining `nn.Module` lifecycle with augmentation training strategy delegation
- All 25 tests pass; `ty check` zero errors; existing augmentation tests unaffected

## Task Commits

1. **Task 1: New contract types (TDD)** — 3 commits:
   - `aa7b063` (test): add failing tests for new augmentation contract types
   - `db40ce5` (feat): implement new augmentation contract types in base.py
   - `3fac49b` (chore): add numba dependency to unblock test imports

## Files Created/Modified

- `tests/test_aug_contract.py` — 25 tests covering all new contract types, Protocol structural conformance, dataclass immutability, ABC enforcement, backward compatibility
- `src/tscollection/models/augmentation/base.py` — Added Augmentation Protocol, SingleView/ViewPair/AlignedPair dataclasses, V TypeVar, AugmentationProducer[V] Protocol, TrainableAugmentationProducer ABC; updated __all__; existing symbols retained unchanged
- `pyproject.toml` — Added `numba>=0.65.1` production dependency (pre-existing import blocker in soft_dtw_cuda.py)

## Decisions Made

- Augmentation and AugmentationProducer are NOT `@runtime_checkable` — structural conformance verified via method presence tests instead of `issubclass` checks, per plan specification
- `AlignedPair` subclasses `ViewPair` directly (Liskov substitution: `issubclass(AlignedPair, ViewPair)`)
- `TrainableAugmentationProducer` inherits from `nn.Module, ABC` (nominal, not Protocol) to enable `isinstance()` runtime gating
- `__all__` updated to include new symbols alongside existing ones (alphabetical order)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing numba dependency**
- **Found during:** Task 1 RED phase (test import)
- **Issue:** `soft_dtw_cuda.py` imports `numba` unconditionally, but `numba` was not declared in `pyproject.toml`. The barrel import chain (`tscollection.models.__init__` -> `series2vec` -> `soft_dtw` -> `numba`) blocks all test imports.
- **Fix:** Added `numba>=0.65.1` as a production dependency via `uv add numba`.
- **Files modified:** `pyproject.toml`, `uv.lock`
- **Verification:** Tests collect and run after the fix.
- **Committed in:** `3fac49b`

**2. [Rule 1 - Bug] Protocol tests use `issubclass` on non-`@runtime_checkable` Protocols**
- **Found during:** Task 1 GREEN phase (test execution)
- **Issue:** Tests used `issubclass(ConcreteClass, Augmentation)` and `issubclass(ConcreteClass, AugmentationProducer[SingleView])`, but non-`@runtime_checkable` Protocols cannot be used with `issubclass`.
- **Fix:** Replaced `issubclass` checks with structural verification (method presence + runtime invocation).
- **Files modified:** `tests/test_aug_contract.py`
- **Verification:** All 25 tests pass.
- **Committed in:** `db40ce5`

**3. [Rule 1 - Bug] AdamW optimizer receives empty parameter list**
- **Found during:** Task 1 GREEN phase (test execution)
- **Issue:** `_ConcreteProducer` test double had no learnable parameters, causing `AdamW(self.parameters(), lr=lr)` to raise `ValueError: optimizer got an empty parameter list`.
- **Fix:** Added `self._dummy_param = nn.Linear(4, 4)` to concrete test doubles.
- **Files modified:** `tests/test_aug_contract.py`
- **Verification:** `test_configure_optimizer_returns_adamw` passes.
- **Committed in:** `db40ce5`

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 bugs)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered

- Worktree import chain triggers full barrel (`tscollection.models.__init__`), which imports `series2vec` -> `soft_dtw` -> `numba`. Resolved by adding numba as a production dependency.

## Known Stubs

None — all implemented types have complete, functional definitions.

## Threat Flags

None — contract types are pure abstractions (Protocols, dataclasses, ABCs) with no network, auth, or file access surfaces.

## Next Phase Readiness

- New contract types are importable and tested
- `ty check` passes with zero errors
- Existing tests unaffected (backward compatibility verified)
- Ready for primitives.py (plan 02) and producers.py (plan 03)

---
*Phase: 01-augmentation-producer-contract*
*Completed: 2026-06-12*
