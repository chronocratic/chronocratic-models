---
phase: 01-augmentation-producer-contract
plan: 11
type: execute
subsystem: augmentation
tags:
  - refactoring
  - deletion
  - contract-migration
dependency_graph:
  requires: [01-12]
  provides: []
  affects: [augmentation/base.py, augmentation/__init__.py, augmentation/dual.py]
tech_stack:
  added: []
  patterns: []
key_files:
  created: []
  modified:
    - src/tscollection/models/augmentation/base.py
    - src/tscollection/models/augmentation/__init__.py
  deleted:
    - src/tscollection/models/augmentation/dual.py
decisions:
  - Dropped per-model Cost re-exports from barrel (cost/augmentation.py still imports deleted TrainingViews; plan 13 will fix)
  - Preserved TrainableAugmentationProducer; removed only old TrainableAugmentation
metrics:
  duration_seconds: 900
  completed_date: "2026-06-12"
---

# Phase 01 Plan 11: Core Deletion Summary

Removed `TrainingViews`, `AugmentationMethod`, `TrainableAugmentation` from `base.py`, deleted `dual.py`, and cleaned the augmentation barrel of all legacy exports.

## Changes

### base.py
- Deleted `TrainingViews` dataclass (L43-L57)
- Deleted `AugmentationMethod` ABC (L96-L117)
- Deleted `TrainableAugmentation` ABC (L183-L260)
- Removed unused `Any` import
- Updated module docstring to reflect new symbol list
- Kept `AugmentationTrainingStrategy` and `TrainableAugmentationProducer` (new contract)

### dual.py
- Deleted entirely (`DualAugmentation` class removed)

### augmentation/__init__.py
- Removed legacy import block (TrainingViews, AugmentationMethod, TrainableAugmentation, DualAugmentation)
- Removed Cost per-model re-exports (CosTRandomFunctionAugmentation, CosTRandomFunctionAugmentationParameters) because cost/augmentation.py still imports deleted TrainingViews
- Restored safe per-model re-exports (autotcl, ts2vec)
- Added AugmentationTrainingStrategy to barrel exports
- Updated module docstring

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Dropped Cost per-model barrel re-exports**

- **Found during:** Task 1 — barrel import chain crashed at runtime
- **Issue:** `cost/augmentation.py` still imports `TrainingViews` from `base.py`. When `__init__.py` re-exports Cost symbols, the import chain crashes, blocking the entire package from being imported.
- **Fix:** Removed Cost re-exports from barrel `__init__.py`. Restored autotcl/ts2vec re-exports (they don't depend on deleted symbols).
- **Files modified:** `src/tscollection/models/augmentation/__init__.py`
- **Commit:** `990cf3a`
- **Deferred to:** Plan 13 (per-model cleanup) will fix `cost/augmentation.py` and restore barrel re-exports

## Verification

- [x] `ty check src/tscollection/models/augmentation/` passes with zero errors
- [x] `ls src/tscollection/models/augmentation/dual.py` returns "No such file"
- [x] New symbols importable: `AugmentationProducer`, `SingleView`, `ViewPair`, `AlignedPair`, `Seeded`, `AugmentationTrainingStrategy`, `TrainableAugmentationProducer`
- [x] Old symbols not in base.py or `__init__.py`: `TrainingViews`, `AugmentationMethod`, `TrainableAugmentation`, `DualAugmentation`
- [x] Full test suite: 5 collection errors remain (all in per-model files or tests that still import deleted symbols — deferred to plan 13)

## Deferred Items

- `cost/augmentation.py` still imports `TrainingViews` from `base.py` — plan 13
- Tests importing `AugmentationMethod`, `TrainingViews`, `TrainableAugmentation` from barrel — plan 13
- `test_aug_contract.py::TestBackwardCompatibility` tests expect deleted symbols — plan 13
- `test_smoke.py` imports `AugmentationMethod`, `TrainingViews`, `CosTRandomFunctionAugmentationParameters` — plan 13

## Self-Check: PASSED
