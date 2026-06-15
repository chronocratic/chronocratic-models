---
phase: 06-prepare-to-be-published-as-package
plan: 08
subsystem: ci
tags: [github-actions, pytest-cov, sphinx, uv]
requires:
  - phase: 06-prepare-to-be-published-as-package
    plan: 06-01
    provides: "Namespace rename (tscollection -> chronocratic)"
  - phase: 06-prepare-to-be-published-as-package
    plan: 06-03
    provides: "Config namespace rename"
  - phase: 06-prepare-to-be-published-as-package
    plan: 06-04
    provides: "pyproject.toml overhaul with setuptools_scm, towncrier, docs extras"
provides:
  - "Fixed CI workflow coverage path (src/chronocratic/models)"
  - "Fixed docs job to use uv sync --extra docs"
affects:
  - "CI pipeline"
  - "docs build"
tech-stack:
  added: []
  patterns:
    - "uv-based docs installation in CI (consistent with test/lint jobs)"
key-files:
  created: []
  modified:
    - ".github/workflows/build-and-test.yml"
key-decisions:
  - "Coverage path targets src/chronocratic/models (not datasets)"
  - "Docs job uses uv sync --extra docs instead of pip install -e ."
requirements-completed: []
duration: 0min (previously completed)
completed: 2026-06-15
---

# Phase 6 Plan 08: CI Workflow Coverage Path and Docs Job Fix Summary

**Fixed build-and-test.yml coverage path to src/chronocratic/models and docs job to use uv sync --extra docs**

## Performance

- **Duration:** 0 min (task already completed by prior wave agent)
- **Started:** 2026-06-15T12:35:33Z
- **Completed:** 2026-06-15T12:35:33Z
- **Tasks:** 1/1 (previously committed)
- **Files modified:** 1

## Accomplishments

- Coverage path corrected from `src/chronocratic/datasets` to `src/chronocratic/models`
- Docs job updated from `pip install -e ".[docs]"` to `uv sync --extra docs`
- Verified no `tscollection` or `chronocratic/datasets` references remain in CI workflows

## Task Commits

1. **Task 1: Fix build-and-test.yml coverage and docs job** - `f461d77` (fix) - PREVIOUSLY COMMITTED by prior wave agent

## Files Modified

- `.github/workflows/build-and-test.yml` - Coverage path and docs job fix

## Decisions Made

None - followed plan as specified. Changes were already made by a prior wave agent.

## Deviations from Plan

None - plan executed exactly as written. The task was already completed in commit `f461d77` by a prior wave agent. All acceptance criteria verified:

- `build-and-test.yml` coverage uses `--cov=src/chronocratic/models`
- No references to `tscollection` or `chronocratic/datasets` remain in CI workflows
- `pypi-publish.yml` is unchanged (already correct)

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

CI workflows are correctly configured for the renamed package. No blockers.

---

*Phase: 06-prepare-to-be-published-as-package*
*Completed: 2026-06-15*
