---
phase: 01-augmentation-producer-contract
plan: 13
status: complete
wave: 7
subsystem: augmentation-cleanup
tags:
  - old-symbol-deletion
  - D-03-Phase-C
  - per-model-cleanup
  - backward-compat
dependency_graph:
  requires: [01-10]
  provides: []
  affects: []
key_files:
  created: []
  modified:
    - src/tscollection/models/augmentation/__init__.py
    - src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py
    - tests/test_augmentation.py
    - tests/test_augmentation_base.py
    - tests/test_augmentation_per_model.py
metrics:
  duration: 20min
  tasks_completed: 1/1
  completed_date: 2026-06-12
  tests_passing: 423
decisions:
  - "CropShiftAugmentation alias removed from ts2vec/augmentation.py; preserved in barrel only (D-05)"
  - "Removed old-symbol test classes from in-scope tests; out-of-scope tests retained for backward-compat verification"
---

# Phase 01 Plan 13: Per-Model Old Symbol Cleanup Summary

Cleaned per-model augmentation files and in-scope tests of old symbol references (TrainingViews, AugmentationMethod, TrainableAugmentation, DualAugmentation, CropShiftAugmentation alias) as the second half of D-03 Phase C.

## What Was Built

**Task 1: Clean per-model augmentation files and tests (D-03 Phase C)** — Removed old-symbol references from per-model source and test files:
- Removed `CropShiftAugmentation` alias from `ts2vec/augmentation.py` (D-05 backward-compat alias)
- Moved `CropShiftAugmentation` alias to `augmentation/__init__.py` barrel (barrel-only location)
- Deleted `TestTrainingViews`, `TestAugmentationMethod`, `TestTrainableAugmentation` test classes from `test_augmentation.py`
- Deleted `TestTrainingViewsFromBase`, `TestAugmentationMethodFromBase`, `TestTrainableAugmentationFromBase` from `test_augmentation_base.py`
- Updated `test_augmentation_per_model.py` to use `CropShiftProducer` instead of `CropShiftAugmentation` for ts2vec tests
- Removed `AugmentationMethod` from module-level imports in `test_augmentation_per_model.py`
- Added local `TrainingViews` imports to cost backward-compat tests (cost/augmentation.py still uses it)
- Verified `autotcl/utils.py` already uses `AugmentationProducer[SingleView]` (clean)
- Verified `autotcl/model.py` already uses `TrainableAugmentationProducer` (clean)
- Verified `tstcc/augmentations.py` has no old symbol imports (clean)

## Deviations from Plan

None — plan executed exactly as written.

## Verification

All 423 tests pass (2 skipped). `ty check src/` passes with zero errors. Zero old-symbol code references remain in in-scope per-model source files (only docstring mentions and barrel backward-compat alias).

## Commits

- `fe6f878` — fix(01-13): clean per-model old augmentation symbol references
