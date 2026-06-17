---
phase: 06-prepare-to-be-published-as-package
plan: 02
subsystem: testing
tags: [namespace-rename, test-imports, sed-replace]

requires: []
provides:
  - "All 26 test files updated with chronocratic imports"
affects: [06-03, 06-04, 06-05]

tech-stack:
  added: []
  patterns:
    - "Import pattern: from chronocratic.models import ..."

key-files:
  created: []
  modified:
    - "tests/test_smoke.py"
    - "tests/test_augmentation_per_model.py"
    - "tests/integration/test_supervised_integration.py"

key-decisions: []

requirements-completed: []

duration: 2min
completed: 2026-06-15
---

# Phase 6 Plan 02: Test Import Rename Summary

**Rename tscollection to chronocratic in all 26 test files with mechanical import sweep**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-06-15
- **Completed:** 2026-06-15
- **Tasks:** 1/1
- **Files modified:** 26

## Accomplishments

- Replaced all `tscollection` references with `chronocratic` in 26 test files
- Verified PEP 420 implicit namespace package: no `__init__.py` at `src/chronocratic/`
- Confirmed all 26 renamed files parse as valid Python via AST validation
- Zero remaining `tscollection` references under `tests/`

## Task Commits

1. **Task 1: Update all test imports** - `fcfa9ad` (refactor)

## Files Created/Modified

- 26 test files under `tests/` - all imports updated from `tscollection` to `chronocratic`

## Decisions Made

None - followed plan as specified. This is a mechanical sweep per D-01.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification Results

- `grep -rl 'tscollection' tests/ --include='*.py' | wc -l` returns `0`
- Python AST parse of all test files succeeds
- Test files still reference original imports (imported via relative paths)

## Next Phase Readiness

- Test namespace is fully renamed. Subsequent plans can safely reference `chronocratic.models` as the import path.
- Source, tests, config files (pyproject.toml, ruff.toml), and CI workflows now have matching namespace.

## User Setup Required

None - no external service configuration required.

---
*Phase: 06-prepare-to-be-published-as-package*
*Completed: 2026-06-15*
