---
phase: 01-augmentation-producer-contract
plan: 14
subsystem: testing
tags: [cleanup, docstrings, backward-compat, refactoring]

# Dependency graph
requires:
  - phase: 01-augmentation-producer-contract
    provides: All prior plans (01-01 through 01-13) that introduced D-XX markers and backward compat aliases
provides:
  - Clean source code with zero D-XX references
  - Zero plan number references in source/test code
  - CropShiftAugmentation alias removed from barrel, tests updated to use CropShiftProducer
affects: [verifier, future-cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns: [final-hygiene-pass, alias-removal]

key-files:
  created: []
  modified:
    - src/tscollection/models/augmentation/__init__.py
    - src/tscollection/models/augmentation/producers.py
    - src/tscollection/models/augmentation/trainable_support.py
    - src/tscollection/models/convolutional/dilated/cost/augmentation.py
    - src/tscollection/models/convolutional/standard/tstcc/augmentations.py
    - src/tscollection/models/supervised/factory.py
    - tests/test_aug_contract.py
    - tests/test_aug_cross_model.py
    - tests/test_augmentation.py
    - tests/test_augmentation_base.py
    - tests/test_augmentation_per_model.py
    - tests/test_from_config.py
    - tests/unit/test_tstcc_supervised.py

key-decisions:
  - "Removed CropShiftAugmentation alias entirely (not just D-XX tag) to meet success criteria"
  - "Updated all test imports/references from CropShiftAugmentation to CropShiftProducer"
  - "Preserved historical note in ts2vec/augmentation.py docstring ('reshaped from CropShiftAugmentation') as it is factual, not a D-XX ref"

requirements-completed: []

# Metrics
duration: 15min
completed: 2026-06-12
---

# Phase 1 Plan 14: Final Hygiene Pass Summary

**Removed all D-XX planning markers, plan number references, and the CropShiftAugmentation backward compat alias from source and test code**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-06-12T13:17:00Z
- **Completed:** 2026-06-12T13:32:00Z
- **Tasks:** 1
- **Files modified:** 13

## Accomplishments
- Stripped 14 D-XX references from 8 source files and 5 test files
- Stripped 5 plan number references from 4 test files and 1 source file
- Removed `CropShiftAugmentation = CropShiftProducer` alias from `augmentation/__init__.py` barrel
- Updated all test code that imported `CropShiftAugmentation` to use `CropShiftProducer` directly
- Renamed `TestCropShiftAugmentation` to `TestCropShiftProducer` for accuracy

## Task Commits

1. **Task 1: Audit and clean source code** - `4baa373` (chore)

## Files Created/Modified
- `src/tscollection/models/augmentation/__init__.py` - Removed D-05 tags, plan 01-11 ref, CropShiftAugmentation alias
- `src/tscollection/models/augmentation/producers.py` - Removed D-01 ref from module docstring
- `src/tscollection/models/augmentation/trainable_support.py` - Removed 3x D-02 refs from docstrings
- `src/tscollection/models/convolutional/dilated/cost/augmentation.py` - Removed 2x D-05 refs from docstrings
- `src/tscollection/models/convolutional/standard/tstcc/augmentations.py` - Removed D-06 ref from comment
- `src/tscollection/models/supervised/factory.py` - Removed 2x D-01 refs from docstrings
- `tests/test_aug_contract.py` - Removed D-05 ref from docstring
- `tests/test_aug_cross_model.py` - Removed plan 04 ref from docstring
- `tests/test_augmentation.py` - Removed D-05 and 2x plan 01-13 refs, updated CropShiftAugmentation->CropShiftProducer
- `tests/test_augmentation_base.py` - Removed 2x plan 01-13 refs
- `tests/test_augmentation_per_model.py` - Updated CropShiftAugmentation->CropShiftProducer in barrel test and docstring
- `tests/test_from_config.py` - Removed D-05 ref, updated CropShiftAugmentation->CropShiftProducer
- `tests/unit/test_tstcc_supervised.py` - Removed D-03 ref from module docstring

## Decisions Made
- Removed the CropShiftAugmentation alias entirely rather than keeping it with a cleaned docstring, since the plan's success criteria explicitly requires "No backward compat aliases remaining" and `CropShiftAugmentation` was the only alias (not a legacy class being preserved for plan 01-11)
- Historical notes in `ts2vec/augmentation.py` ("reshaped from CropShiftAugmentation") were kept because they describe origin, not a D-XX planning marker

## Deviations from Plan

**1. [Rule 2 - Missing Critical] Updated additional test files beyond plan scope**
- **Found during:** Task 1 (audit)
- **Issue:** Plan listed specific test files but `test_from_config.py`, `test_augmentation_per_model.py`, and `tests/unit/test_tstcc_supervised.py` also contained D-XX refs and CropShiftAugmentation imports that would crash after alias removal
- **Fix:** Cleaned D-XX refs from all 3 files and updated CropShiftAugmentation imports to CropShiftProducer
- **Files modified:** tests/test_from_config.py, tests/test_augmentation_per_model.py, tests/unit/test_tstcc_supervised.py
- **Committed in:** 4baa373

**2. [Rule 1 - Bug] Removed CropShiftAugmentation from __all__ barrel**
- **Found during:** Task 1 (alias removal)
- **Issue:** Multiple tests imported `CropShiftAugmentation` from the barrel; removing the alias without updating tests would cause ImportError
- **Fix:** Updated all test imports to use `CropShiftProducer` directly
- **Files modified:** tests/test_augmentation.py, tests/test_from_config.py, tests/test_augmentation_per_model.py
- **Committed in:** 4baa373

---

**Total deviations:** 2 auto-fixed (1 Rule 2, 1 Rule 1)
**Impact on plan:** Both auto-fixes were necessary for correctness — the alias removal would have crashed tests without updating imports.

## Verification

- **grep -rn "D-[0-9]" src/ tests/**: Zero hits (excluding .planning/)
- **grep -rn "plan [0-9]" src/ tests/**: Zero hits (excluding .planning/)
- **CropShiftAugmentation alias**: Removed from source barrel; all tests updated to CropShiftProducer
- **uv run pytest** (242 tests): All passed
- **ty check src/**: All checks passed
- **ruff check src/**: Only pre-existing errors remain (I001 import sorting in __init__.py, TC001 in tstcc/augmentations.py)

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Source code is clean of planning artifacts
- Plan 01-11 (core deletion) can now proceed without confusion about which references are D-XX markers vs. real code

---
*Phase: 01-augmentation-producer-contract*
*Completed: 2026-06-12*
