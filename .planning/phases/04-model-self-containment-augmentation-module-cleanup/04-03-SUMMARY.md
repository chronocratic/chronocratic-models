---
phase: 04-model-self-containment-augmentation-module-cleanup
plan: 03
subsystem: augmentation
tags: [per-model-augmentation, module-restructuring, lazy-imports, circular-dependency-fix]

requires:
  - phase: 04-model-self-containment-augmentation-module-cleanup
    provides: Base ABCs in augmentation/base.py (plan 04-01), per-model config hierarchy (plan 04-02)
provides:
  - Per-model augmentation modules: ts2vec/augmentation.py, cost/augmentation.py, autotcl/augmentation/
  - Inline parameter dataclasses in per-model augmentation files
  - Lazy __getattr__ re-exports in barrel modules for backward compatibility
  - Model files importing from augmentation/base.py (not barrel)
affects: [phase-04, augmentation-cleanup, import-restructuring]

tech-stack:
  added: []
  patterns:
    - Lazy __getattr__ for barrel re-exports (circular import avoidance)
    - Direct base.py imports in model files (not barrel)
    - Per-model self-containment for augmentation + params

key-files:
  created:
    - src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py
    - src/tscollection/models/convolutional/dilated/cost/augmentation.py
    - src/tscollection/models/convolutional/dilated/autotcl/augmentation/__init__.py
    - src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py
    - src/tscollection/models/convolutional/dilated/autotcl/augmentation/training.py
    - tests/test_augmentation_per_model.py
  modified:
    - src/tscollection/models/augmentation/__init__.py
    - src/tscollection/models/augmentation/config.py
    - src/tscollection/models/augmentation/strategies.py
    - src/tscollection/models/convolutional/dilated/ts2vec/model.py
    - src/tscollection/models/convolutional/dilated/cost/model.py
    - src/tscollection/models/convolutional/dilated/autotcl/model.py
    - src/tscollection/models/convolutional/dilated/autotcl/utils.py

key-decisions:
  - "Model files import from augmentation/base.py directly to avoid circular imports through barrel"
  - "Barrel __init__.py, config.py, strategies.py use lazy __getattr__ for concrete class re-exports"
  - "Parameter dataclasses moved inline with augmentation classes in per-model modules"
  - "Backward compatibility preserved: old import paths still work via lazy __getattr__"

requirements-completed: []

duration: 10min
completed: 2026-05-22
---

# Phase 4 Plan 3: Per-Model Augmentation Modules Summary

**Concrete augmentations and training strategies moved from shared strategies.py into per-model directories with inline parameter dataclasses; barrel modules use lazy __getattr__ re-exports to preserve backward compatibility and avoid circular imports.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-22T12:57:43Z
- **Completed:** 2026-05-22T13:07:24Z
- **Tasks:** 2 (combined TDD cycle)
- **Files modified:** 17 (5 new, 12 modified)

## Accomplishments
- Created 5 per-model augmentation files: ts2vec/augmentation.py, cost/augmentation.py, autotcl/augmentation/ (3 files)
- Moved inline parameter dataclasses (CropShiftAugmentationParameters, CosTRandomFunctionAugmentationParameters, AutoTCLNeuralNetworkAugmentationParameters) alongside their augmentation classes
- Updated model files to import ABCs from augmentation/base.py (not barrel)
- Rewrote barrel modules (__init__.py, config.py, strategies.py) to use lazy __getattr__ re-exports, breaking circular import chains
- Full backward compatibility: all old import paths continue working

## Task Commits

Each task was committed atomically:

1. **Task 1+2: TDD for per-model augmentation modules** - `4cd0fe3` (test) + `9168809` (feat) + `4fccc6f` (style)
   - TDD: RED phase (failing tests) committed, then GREEN phase (implementation)
   - Ruff formatting applied to new modules

**Plan metadata:** Pending final commit (docs: complete plan)

## Files Created/Modified

### Created
- `src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py` - CropShiftAugmentation + CropShiftAugmentationParameters
- `src/tscollection/models/convolutional/dilated/cost/augmentation.py` - CosTRandomFunctionAugmentation + CosTRandomFunctionAugmentationParameters
- `src/tscollection/models/convolutional/dilated/autotcl/augmentation/__init__.py` - Barrel re-export for autotcl augmentation
- `src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py` - AutoTCLNeuralNetworkAugmentation + AutoTCLNeuralNetworkAugmentationParameters
- `src/tscollection/models/convolutional/dilated/autotcl/augmentation/training.py` - RIPTrainingStrategy, AdversarialTrainingStrategy
- `tests/test_augmentation_per_model.py` - 39 tests for per-model modules and backward compatibility

### Modified
- `src/tscollection/models/augmentation/__init__.py` - Rewritten with lazy __getattr__ re-exports
- `src/tscollection/models/augmentation/config.py` - Rewritten with lazy __getattr__ re-exports
- `src/tscollection/models/augmentation/strategies.py` - Rewritten with lazy __getattr__ re-exports
- `src/tscollection/models/convolutional/dilated/ts2vec/model.py` - Import from base.py instead of barrel
- `src/tscollection/models/convolutional/dilated/cost/model.py` - Import from base.py instead of barrel
- `src/tscollection/models/convolutional/dilated/autotcl/model.py` - Import from base.py instead of barrel
- `src/tscollection/models/convolutional/dilated/autotcl/utils.py` - Import from base.py instead of barrel

## Decisions Made
- Model files import directly from `augmentation/base.py` to break the circular import chain triggered by per-model `__init__.py` barrels
- Barrel modules (`__init__.py`, `config.py`, `strategies.py`) use `__getattr__` for lazy re-exports of concrete classes; ABCs from `base.py` are imported eagerly since they have no circular dependencies
- `from __future__ import annotations` added to barrel modules for forward reference safety

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Circular import chain resolved with lazy __getattr__**
- **Found during:** Task 1 implementation
- **Issue:** Moving concrete augmentations to per-model directories triggered a circular import: `augmentation/__init__.py` -> `config.py` -> `cost.augmentation` -> `cost/__init__.py` -> `cost/model.py` -> `augmentation` barrel (partially initialized). The barrel's eager `from .config import *` and `from .strategies import *` blocked the cycle resolution.
- **Fix:** Rewrote barrel modules to use lazy `__getattr__` for concrete class re-exports. Updated model files to import ABCs from `augmentation/base.py` directly. ABCs are imported eagerly in barrels (no circular dependency); concrete classes are lazy-loaded on first access.
- **Files modified:** `augmentation/__init__.py`, `augmentation/config.py`, `augmentation/strategies.py`, `ts2vec/model.py`, `cost/model.py`, `autotcl/model.py`, `autotcl/utils.py`
- **Verification:** All 166 tests pass. Ruff linting clean.
- **Committed in:** `9168809` (part of feat commit)

## Issues Encountered
- Ruff initially flagged stale `noqa` directives after removing `ANN401` and `ARG002` from return-type lines; corrected by moving noqa to the parameter lines where the issues originate
- `F822` (undefined name in `__all__`) required explicit `# noqa: F822` on `__all__` lists in barrel modules since lazy exports aren't module-level variables

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All concrete augmentations are now self-contained in per-model directories
- Parameter dataclasses live inline with their augmentation classes
- Backward compatibility is preserved via lazy barrel re-exports
- Existing test suite (166 tests) passes without regressions
- Ready for any plan that removes old classes from `strategies.py`/`config.py` entirely or changes model import paths

---
*Phase: 04-model-self-containment-augmentation-module-cleanup*
*Completed: 2026-05-22*
