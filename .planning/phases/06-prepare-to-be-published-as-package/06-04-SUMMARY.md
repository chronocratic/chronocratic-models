---
phase: 06-prepare-to-be-published-as-package
plan: 04
subsystem: packaging
tags: [pyproject.toml, setuptools-scm, towncrier, dependency-pruning, tqdm, version-export]

requires:
  - phase: 06-prepare-to-be-published-as-package
    provides: "Namespace rename (06-01), pyproject.toml baseline (06-03)"
provides:
  - "Complete pyproject.toml with setuptools_scm dynamic versioning"
  - "Pruned runtime dependencies (7 packages, 7 removed)"
  - "tqdm added as runtime dependency (bug fix)"
  - "towncrier changelog configuration with 6 type directories"
  - "Sphinx docs optional-dependencies"
  - "__version__ export from chronocratic.models"
affects: [06-05, 06-06, 06-07, CI workflows, Sphinx docs]

tech-stack:
  added: []
  patterns:
    - "setuptools_scm dynamic versioning from git tags"
    - "towncrier fragment-based changelog"
    - "try/except ImportError guard for __version__"

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/chronocratic/models/__init__.py

key-decisions:
  - "All pyproject.toml changes already applied by prior plans (06-01, 06-03)"
  - "All __version__ export already applied by prior plans"
  - "Zero code changes needed -- verified target state matches"

patterns-established:
  - "Dynamic versioning via setuptools_scm with no-local-version scheme"
  - "towncrier changelog fragments with 6 type directories"

requirements-completed: []

duration: 2min
completed: 2026-06-15
---

# Phase 6 Plan 04: Overhaul pyproject.toml with setuptools_scm, towncrier, and dependency pruning Summary

**Verified complete pyproject.toml with setuptools_scm dynamic versioning, towncrier changelog config, pruned 7 unused deps, added missing tqdm, and __version__ export from prior plan commits**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-15T14:27:00Z
- **Completed:** 2026-06-15T14:29:00Z
- **Tasks:** 2 verified (0 changes needed)
- **Files modified:** 0

## Accomplishments

- Verified pyproject.toml matches target state: setuptools-scm>=8 in build-system, dynamic versioning, 7 pruned runtime deps, tqdm bug fix, complete metadata, towncrier config, docs optional-dependencies
- Verified __version__ export in chronocratic.models/__init__.py with ImportError guard
- Zero deltas between current state and plan target -- all work was completed by prior plans (06-01, 06-03)

## Task Commits

Both tasks were verified as already complete. No new commits for task changes.

1. **Task 1: Overhaul pyproject.toml** -- Already complete (prior plans 06-01, 06-03). All acceptance criteria verified:
   - `setuptools-scm>=8` in build-system requires
   - `dynamic = ["version"]`
   - 7 runtime dependencies (torchvision, torchaudio, joblib, openpyxl, h5py, pandas, scikit-learn removed; tqdm>=4.66.0 added)
   - Complete metadata: description, readme, license, keywords, authors, classifiers
   - `[project.optional-dependencies] docs` with sphinx, pydata-sphinx-theme, myst-parser
   - `[tool.setuptools_scm]` with version_file = src/chronocratic/models/_version.py
   - `[tool.towncrier]` with 6 type directories
   - `requires-python = ">=3.12"` (no upper bound)
   - Project URLs

2. **Task 2: Add __version__ export to models/__init__.py** -- Already complete (prior plans). Verified:
   - `from ._version import __version__` with try/except ImportError guard
   - `"__version__"` is first entry in `__all__`

## Files Created/Modified

No files modified -- target state already achieved by prior plans.

## Decisions Made

- Verified all 14 must-have truths from plan against current state
- Confirmed zero deltas between plan target and actual state

## Deviations from Plan

None -- plan executed exactly as written. All tasks were pre-completed by prior plans.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- pyproject.toml is fully configured for publishing
- __version__ export is available for Sphinx docs integration
- towncrier changelog infrastructure is in place
- Ready for CI workflow fixes and docs build

---
*Phase: 06-prepare-to-be-published-as-package*
*Completed: 2026-06-15*
