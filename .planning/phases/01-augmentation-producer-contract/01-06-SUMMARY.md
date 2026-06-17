---
phase: 01-augmentation-producer-contract
plan: 06
type: execute
subsystem: cost-model
tags:
  - producer-contract
  - tdd-green
  - cost
  - view-pair
  - independent-pair
dependency_graph:
  requires:
    - 01-01 (Augmentation Protocol, ViewSet dataclasses, AugmentationProducer)
    - 01-02 (Augmentation primitives)
    - 01-03 (Producer combinators: IndependentPair)
    - 01-04 (Seeded decorator, maybe_* helpers)
    - 01-05 (TS2Vec migration — first model wired)
  provides:
    - CosTRandomFunctionAugmentation reshaped to Augmentation Protocol
    - CoST wired to AugmentationProducer[ViewPair] with IndependentPair default
  affects: []
tech_stack:
  added: []
  patterns:
    - Augmentation Protocol (__call__: Tensor -> Tensor)
    - IndependentPair producer combinator
    - Backward-compatible AugmentationMethod.augment() wrapper
key_files:
  created: []
  modified:
    - src/tscollection/models/convolutional/dilated/cost/augmentation.py
    - src/tscollection/models/convolutional/dilated/cost/model.py
    - tests/test_cost_producer.py
decisions:
  - "CosTRandomFunctionAugmentation stays in cost/augmentation.py (D-06: model-specific)"
  - "Backward-compat augment() -> TrainingViews retained (D-05)"
  - "CoST wraps plain Augmentation in IndependentPair for backward compat"
metrics:
  duration_minutes: 8
  tasks_completed: 1
  tests_passing: 11
  completed_date: "2026-06-12"
requirements-completed: [G3, G6]
---

# Phase 1 Plan 06: Wire CoST to Producer Contract

Reshaped `CosTRandomFunctionAugmentation` to implement the `Augmentation` Protocol (`__call__: Tensor -> Tensor`) and wired `CoST` to accept `AugmentationProducer[ViewPair] | None` with `IndependentPair` as default, replacing the old double-`.augment(x).views[0]` pattern.

## Completed Tasks

| Task | Name | Commit | Done Criteria |
|------|------|--------|---------------|
| RED (prev) | Failing tests for CoST producer integration | `d21630e` | 11 tests covering protocol, integration, training, seeded equiv |
| 1 | Reshape + Wire (GREEN) | `84b3fb0` | CosTRandomFunctionAugmentation callable, CoST uses IndependentPair, 11/11 tests pass, ty check clean |

## Key Changes

### `cost/augmentation.py` — CosTRandomFunctionAugmentation
- Inherits from `Augmentation` Protocol (structural) instead of `AugmentationMethod` (nominal ABC)
- New `__call__(self, x: torch.Tensor) -> torch.Tensor`: returns bare tensor (jitter/shift/scale)
- Retains `augment(data, **kwargs) -> TrainingViews` as backward-compat wrapper that calls `self(data)`
- Added `sigma: float | None = None` keyword convenience arg to `__init__`
- Import updated: `Augmentation` instead of `AugmentationMethod` from `base.py`

### `cost/model.py` — CoST
- Constructor param: `augmentation: AugmentationProducer[ViewPair] | None = None` (was `AugmentationMethod | None`)
- Default: `IndependentPair(aug=CosTRandomFunctionAugmentation())` (lazy import)
- Backward compat: wraps plain `Augmentation` instances in `IndependentPair` automatically
- `training_step`: single `pair = self._augmentation.produce(x)` call, accesses `pair.first`, `pair.second`
- `validation_step`: same pattern (single `.produce()` call)
- No more `.augment()` or `.views[0]` access in model code

### `tests/test_cost_producer.py` — 11 tests
- Protocol conformance: callable, isinstance Augmentation, shape preservation, sigma=0 identity, p=0 identity
- Integration: accepts IndependentPair, default is IndependentPair, produce returns ViewPair
- Training: 5 steps with producer (finite loss), 5 steps with default (finite loss)
- Seeded equivalence: two CoST models with same seed produce identical loss sequences (SC-7)
- Fix: also seed numpy RNG in `_train_steps` (CosTRandomFunctionAugmentation uses `np.random`)

## Verification

- [x] `CosTRandomFunctionAugmentation` implements `Augmentation` Protocol (`__call__: Tensor -> Tensor`)
- [x] `CoST` accepts `AugmentationProducer[ViewPair] | None`
- [x] CoST uses `IndependentPair` as default, single `.produce()` call
- [x] No `.augment()` or `.views[0]` access in CoST model code
- [x] Seeded comparison test verifies numerical equivalence (SC-7)
- [x] 11/11 tests pass
- [x] `ty check` passes (zero errors)

## Deviations from Plan

**1. [Rule 1 - Bug] Seeded equivalence test needed numpy RNG seeding**
- **Found during:** GREEN phase test execution
- **Issue:** `CosTRandomFunctionAugmentation` uses `np.random.random()` for its probability gates. Seeding only `torch.manual_seed()` was insufficient for deterministic output across runs.
- **Fix:** Added `np.random.seed(seed)` alongside `torch.manual_seed(seed)` in the `_train_steps` helper.
- **Files modified:** `tests/test_cost_producer.py`
- **Commit:** `84b3fb0`

## Threat Flags

None — changes are refactoring-only (contract migration). No new network endpoints, auth paths, or file access surfaces.

## Known Stubs

None.
