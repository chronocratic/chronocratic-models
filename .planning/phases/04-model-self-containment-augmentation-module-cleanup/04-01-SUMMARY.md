---
phase: 04
plan: 01
subsystem: augmentation
tags:
  - abc-hierarchy
  - module-extraction
  - refactoring
dependency_graph:
  requires: []
  provides:
    - augmentation/base.py (TrainingViews, AugmentationMethod, AugmentationTrainingStrategy, TrainableAugmentation)
  affects:
    - augmentation/strategies.py (still contains all symbols; dedup pending Plan 02)
tech_stack:
  added: []
  patterns:
    - Abstract Base Class (ABC)
    - Strategy Pattern
    - Composition (TrainableAugmentation composes AugmentationTrainingStrategy)
key_files:
  created:
    - src/tscollection/models/augmentation/base.py
    - tests/test_augmentation_base.py
  modified: []
decisions:
  - No circular imports: base.py does not import from augmentation/__init__.py
  - Lazy loss imports (D-19): base.py does not import loss functions; concrete strategies import lazily
  - __all__ export list is alphabetical
  - from __future__ import annotations used to avoid forward reference issues with nn.Module + ABC MRO
metrics:
  duration_seconds: 300
  completed_date: "2026-05-22"
---

# Phase 4 Plan 1: Augmentation Base Module Extraction Summary

Extract `TrainingViews`, `AugmentationMethod`, `AugmentationTrainingStrategy`, and `TrainableAugmentation` from the monolithic `strategies.py` into a new `base.py` module (~235 lines) containing only abstract base classes and the views dataclass.

## What Was Built

Created `src/tscollection/models/augmentation/base.py` with four exported symbols:

1. **TrainingViews** — dataclass for augmentation output views (tuple of tensors) and metadata (dict)
2. **AugmentationMethod** — ABC defining the `augment()` contract for all time-series augmentations
3. **AugmentationTrainingStrategy** — ABC defining `compute_loss()` and `should_train()` for trainable augmentation optimization
4. **TrainableAugmentation** — ABC combining `AugmentationMethod` + `nn.Module`, composing an `AugmentationTrainingStrategy` for loss computation

## TDD Gate Compliance

- **RED:** `test_augmentation_base.py` committed (7 tests, all failing — `base.py` did not exist)
- **GREEN:** `base.py` committed (7 tests pass, 18 existing tests still pass)

## Verification

- All 4 classes present in `base.py`
- All 4 symbols importable: `from tscollection.models.augmentation.base import ...`
- No concrete strategies leaked in (`RIPTrainingStrategy`, `AdversarialTrainingStrategy`, etc. are absent)
- No circular imports
- Ruff linting passes (0 errors)
- Full test suite passes (25/25 tests green)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. This is a refactoring-only plan; no new network endpoints, auth paths, or trust boundaries introduced.
