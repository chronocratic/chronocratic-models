---
phase: 06-prepare-to-be-published-as-package
plan: 03
subsystem: build-packaging
tags: [pyproject, ruff, namespace-rename, tscollection, chronocratic]

requires:
  - phase: 06-prepare-to-be-published-as-package
    provides: Source directory renamed from tscollection to chronocratic (06-01)
provides:
  - pyproject.toml with package name chronocratic-models
  - ruff.toml with updated exclude path referencing src/chronocratic/
affects: [06-04, 06-05, 06-06, 06-07, 06-08, 06-09]

tech-stack:
  added: []
  patterns: [Namespace rename config sweep]

key-files:
  created: []
  modified:
    - pyproject.toml
    - ruff.toml

key-decisions:
  - "Only rename package name in pyproject.toml; dependency/metadata changes deferred to 06-04"
  - "Single exclude path update in ruff.toml; no other tscollection references found"

requirements-completed: []

duration: 1min
completed: 2026-06-15
---

# Phase 6 Plan 03: Config Namespace Rename Summary

**Update pyproject.toml package name and ruff.toml exclude path from tscollection to chronocratic per D-01**

## Performance

- **Duration:** 1 min
- **Started:** 2026-06-15T12:07:51Z
- **Completed:** 2026-06-15T12:08:24Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- pyproject.toml renamed from tscollection-models to chronocratic-models
- ruff.toml exclude path updated to reference src/chronocratic/ instead of src/tscollection/

## Task Commits

1. **Task 1: Update pyproject.toml package name** - `7e76b66` (feat)
2. **Task 2: Update ruff.toml exclude path** - `987a861` (feat)

## Files Created/Modified
- `pyproject.toml` - Package name changed from "tscollection-models" to "chronocratic-models" (line 5)
- `ruff.toml` - Exclude path changed from "src/tscollection/..." to "src/chronocratic/..." (line 8)

## Decisions Made
- Only the package name was changed in pyproject.toml. Dependencies, metadata, setuptools_scm, and towncrier are deferred to plan 06-04 per plan scope.
- No other tscollection references exist in either config file after this plan.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Config files now reference the chronocratic namespace.
- Plan 06-04 (pyproject.toml metadata + setuptools_scm) can proceed.

---
*Phase: 06-prepare-to-be-published-as-package*
*Completed: 2026-06-15*
