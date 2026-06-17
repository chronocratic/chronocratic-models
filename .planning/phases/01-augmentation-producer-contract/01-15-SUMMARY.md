---
phase: 01-augmentation-producer-contract
plan: 15
subsystem: test infrastructure
tags: [refactor, producer-tests, conftest, shared-fixtures]
depends_on: [01-10, 01-12, 01-13]
dependency_graph:
  requires: []
  provides: [shared-conftest-fixtures-for-producer-tests]
  affects: [test_ts2vec_producer.py, test_cost_producer.py, test_autotcl_producer.py]
tech_stack:
  added: []
  patterns: [pytest-fixture-sharing, DRY-test-refactoring]
key_files:
  created: []
  modified:
    - tests/test_ts2vec_producer.py
    - tests/test_cost_producer.py
    - tests/test_autotcl_producer.py
    - src/tscollection/models/convolutional/dilated/cost/augmentation.py
decisions:
  - Replaced manual loss-assertion loops with finite_losses fixture for consistency
  - Simplified cost augment() to return Tensor directly instead of deleted TrainingViews
metrics:
  duration: ~5m
  tasks_completed: 1
  tasks_total: 1
  completed_date: 2026-06-12
---

# Phase 1 Plan 15: Refactor per-model producer tests to use shared conftest fixtures

Removed local `_train_steps()` helpers from 3 per-model producer test files (ts2vec, cost, autotcl), replacing all calls with the shared `train_steps` and `finite_losses` fixtures from `tests/conftest.py`. Eliminated ~174 lines of duplicated boilerplate. Also fixed `cost/augmentation.py` which still imported the deleted `TrainingViews` symbol from plan 01-11.

## Changes

- **tests/test_ts2vec_producer.py**: Removed `_train_steps()` (35 lines) and unused imports (`math`, `lightning.pytorch`, `DataLoader`, `TensorDataset`). Updated 2 training tests to use `train_steps` + `finite_losses` fixtures.
- **tests/test_cost_producer.py**: Removed `_train_steps()` (49 lines) and unused imports (`math`, `numpy`). Updated 3 training tests to use fixtures.
- **tests/test_autotcl_producer.py**: Removed `_train_steps()` (43 lines) and unused imports (`math`, `numpy`). Updated 3 training tests to use fixtures.
- **src/tscollection/models/convolutional/dilated/cost/augmentation.py**: Dropped `TrainingViews` import (deleted in plan 01-11). Simplified `augment()` to return `torch.Tensor` directly instead of wrapping in the now-deleted `TrainingViews`.
- **tests/test_tstcc_producer.py**: Already used conftest fixtures — no changes needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed cost/augmentation.py broken TrainingViews import**
- **Found during:** Task 1 verification
- **Issue:** `cost/augmentation.py` imports `TrainingViews` from `base.py`, but `TrainingViews` was deleted in plan 01-11. This blocked collection of `test_cost_producer.py` (and transitively all CoST tests).
- **Fix:** Removed `TrainingViews` import; simplified `augment()` to return `torch.Tensor` directly instead of `TrainingViews(views=(result,), metadata={})`.
- **Files modified:** `src/tscollection/models/convolutional/dilated/cost/augmentation.py`
- **Commit:** `e1fdea6`

## Verification

- `uv run pytest tests/test_*producer*.py -v --tb=short`: **62 passed, 2 skipped** (same pre-existing skips)
- No `_train_steps` defined locally in any per-model test file
- All producer tests use `train_steps` fixture from conftest
- All finite loss assertions use `finite_losses` fixture
- Net line reduction: ~174 lines of duplicated boilerplate removed

## Self-Check

- [x] `tests/test_ts2vec_producer.py` — no local `_train_steps`
- [x] `tests/test_cost_producer.py` — no local `_train_steps`
- [x] `tests/test_autotcl_producer.py` — no local `_train_steps`
- [x] `tests/test_tstcc_producer.py` — no local `_train_steps` (already clean)
- [x] `e1fdea6` commit exists in git log
- [x] Full producer test suite passes
