---
phase: 01-augmentation-producer-contract
plan: 10
status: complete
wave: 6
subsystem: augmentation-test-suite
tags:
  - test-migration
  - producer-contract
  - backward-compat
  - D-05
dependency_graph:
  requires: [01-09]
  provides: []
  affects: []
key_files:
  created: []
  modified:
    - tests/test_smoke.py
    - tests/test_augmentation.py
    - tests/test_augmentation_base.py
    - tests/test_augmentation_per_model.py
    - tests/test_aug_config.py
    - tests/test_from_config.py
metrics:
  duration: 25min
  tasks_completed: 2/2
  completed_date: 2026-06-12
  tests_added: 38
  tests_modified: 12
  total_tests: 137
  test_duration: 2.53s
decisions:
  - "CropShiftAugmentation tests updated to .produce() -> AlignedPair (alias is CropShiftProducer, D-05)"
  - "CosTRandomFunctionAugmentation no longer subclass of AugmentationMethod; test updated to check Augmentation Protocol"
  - "Added TSTCC instantiation tests with _default_tstcc_pair() to test_from_config.py"
  - "Added primitive parameter tests (Jitter, Scaling, Permutation) to test_aug_config.py"
  - "Backward compat tests kept for old symbol imports (D-05)"
---

# Phase 01 Plan 10: Migrate Test Files to New Contract Summary

Migrated all 6 existing test files to use new producer contract types (AugmentationProducer[ViewSet]) while maintaining backward compatibility with old symbols (D-05). All 137 tests pass with both old and new types coexisting.

## What Was Built

**Task 1: test_smoke.py migration** — Updated all model training smoke tests to use producer-based augmentations:
- TS2Vec: `CropShiftProducer()` (AlignedPair producer)
- CoST: `IndependentPair(CosTRandomFunctionAugmentation())` (ViewPair producer)
- AutoTCL: `AutoTCLNeuralNetworkAugmentation()` (SingleView trainable producer)
- TSTCC: `_default_tstcc_pair()` (ViewPair producer via RolePair)
- New identity producer test (AlignedPair, structural typing)
- New custom SingleView noise producer test for AutoTCL
- Kept backward-compat identity AugmentationMethod test

**Task 2: Remaining test file migrations** — Updated 5 test files:
- `test_augmentation.py`: CropShift tests use `.produce()` -> AlignedPair; added TestViewSetTypes, TestAugmentationProtocol, TestSingleViewProducer
- `test_augmentation_base.py`: Added ViewSet dataclass tests, Augmentation Protocol tests, TrainableAugmentationProducer ABC tests
- `test_augmentation_per_model.py`: Updated CropShift/CoST tests for new contract; added backward compat documentation
- `test_aug_config.py`: Added JitterParameters, ScalingParameters, PermutationParameters tests
- `test_from_config.py`: Updated CoST to IndependentPair; added TSTCCInstantiation tests

## Deviations from Plan

None — plan executed exactly as written. All 6 test files updated, both tasks committed.

## Verification

All 137 tests pass in 2.53s:
- `test_smoke.py`: 8 tests (TS2Vec, CoST, AutoTCL training smoke + extensibility)
- `test_augmentation.py`: 36 tests (old + new contract types)
- `test_augmentation_base.py`: 22 tests (ABC + Protocol + ViewSet types)
- `test_augmentation_per_model.py`: 47 tests (per-model augmentations)
- `test_aug_config.py`: 23 tests (all parameter dataclasses)
- `test_from_config.py`: 21 tests (model construction)

## Commits

- `fa74309` — feat(01-10): migrate test_smoke.py to producer-based augmentations
- `645c84d` — feat(01-10): migrate remaining test files to new producer contract
