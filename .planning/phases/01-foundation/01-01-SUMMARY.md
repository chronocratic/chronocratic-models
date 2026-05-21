---
phase: 01-foundation
plan: 01
subsystem: testing
tags: [dataclasses, config, typing, pytest, MaskMode]

# Dependency graph
requires: []
provides:
  - ModelParameters ABC base class for typed model configs
  - DilatedCNNModelParameters shared fields for dilated CNN models
  - TS2VecModelParameters config matching TS2Vec __init__ signature
  - CoSTModelParameters config matching CoST __init__ signature (inherits from ModelParameters)
  - AutoTCLModelParameters config matching AutoTCL __init__ signature
affects: [01-02, 01-03, 01-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ABC dataclass with __new__ guard for uninstantiable base"
    - "field(default_factory=list) for mutable default collections"
    - "TDD cycle: failing tests -> minimal impl -> type checker refactor"

key-files:
  created:
    - src/tscollection/models/config.py
    - tests/test_config.py
    - tests/conftest.py
  modified: []

key-decisions:
  - "CoSTModelParameters inherits from ModelParameters directly, not DilatedCNNModelParameters (D-03)"
  - "Augmentation fields excluded from all configs (D-01), deferred to Phase 3"
  - "Runner artifacts excluded from all configs (D-02)"
  - "ModelParameters blocks instantiation via __new__ guard with TypeError"

patterns-established:
  - "Config dataclass __all__ exports sorted alphabetically"
  - "Config fields derived 1:1 from model __init__ signatures"

requirements-completed: [CFG-01, CFG-03]

# Metrics
duration: ~5min
completed: 2026-05-21
---

# Phase 1 Plan 1: Config Dataclass Hierarchy Summary

**Typed config dataclasses for TS2Vec, CoST, and AutoTCL with full test coverage and static type checking**

## Performance

- **Duration:** ~5 min
- **Tasks:** 1 (TDD: RED + GREEN + REFACTOR)
- **Files modified:** 3
- **Tests:** 51 passing

## Accomplishments
- 5 dataclasses: ModelParameters (ABC), DilatedCNNModelParameters, TS2VecModelParameters, CoSTModelParameters, AutoTCLModelParameters
- CoSTModelParameters inherits from ModelParameters directly (not DilatedCNNModelParameters) per D-03
- All field defaults derived from model __init__ signatures; augmentation fields excluded per D-01
- 51 pytest tests covering instantiation, inheritance, defaults, vars() unpacking, mutable default isolation
- Static type checking clean (ty check passes)

## Task Commits

Each task was committed atomically using TDD cycle:

1. **Task 1: Config dataclass hierarchy with tests** - TDD cycle
   - `4616042` (test) — add failing tests for config dataclass hierarchy
   - `d4c6507` (feat) — implement typed config dataclass hierarchy
   - `480ee81` (refactor) — use object.__new__ in ModelParameters for type checker compliance

## Files Created/Modified
- `src/tscollection/models/config.py` — 5 dataclasses with proper inheritance, all fields typed, __all__ exports
- `tests/test_config.py` — 51 tests across 6 test classes covering all behavior criteria
- `tests/conftest.py` — Shared pytest fixtures for config tests

## Decisions Made
- Used `__new__` guard to prevent ModelParameters instantiation (TypeError) instead of adding a dummy abstract method that all subclasses would need to implement
- `field(default_factory=list)` for `kernel_sizes` in both CoSTModelParameters and AutoTCLModelParameters to ensure mutable default isolation
- All docstrings follow Google style with Args section matching model __init__ parameter names

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ModelParameters ABC allows instantiation without abstract methods**
- **Found during:** Task 1 (RED → GREEN)
- **Issue:** `abc.ABC` does not prevent instantiation when no abstract methods exist. `ModelParameters()` succeeded instead of raising TypeError.
- **Fix:** Added `__new__` override that checks `cls is ModelParameters` and raises TypeError for direct instantiation, allows subclasses through via `object.__new__(cls)`.
- **Files modified:** src/tscollection/models/config.py
- **Verification:** `pytest tests/test_config.py::TestModelParametersBase::test_cannot_instantiate` passes
- **Committed in:** d4c6507 (part of GREEN commit)

**2. [Rule 1 - Bug] ty rejects super().__new__(cls) with invalid-super-argument**
- **Found during:** Task 1 (REFACTOR)
- **Issue:** `super().__new__(cls)` in `ModelParameters.__new__` triggers `ty` error[invalid-super-argument] because `type` is not a subclass of ModelParameters.
- **Fix:** Replaced `super().__new__(cls)` with `object.__new__(cls)` (canonical pattern for `__new__` methods).
- **Files modified:** src/tscollection/models/config.py
- **Verification:** `uv run ty check src/tscollection/models/config.py` exits 0
- **Committed in:** 480ee81 (refactor commit)

---

**Total deviations:** 2 auto-fixed (Rule 1 bugs)
**Impact on plan:** Both fixes required for correctness. No scope creep.

## TDD Gate Compliance

- [x] RED gate: `test(01-01)` commit (4616042) — failing tests before implementation
- [x] GREEN gate: `feat(01-01)` commit (d4c6507) — minimal code to pass all tests
- [x] REFACTOR gate: `refactor(01-01)` commit (480ee81) — type checker compliance fix

## Issues Encountered
- None beyond the two auto-fixed deviations documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Config dataclasses are ready for `from_config()` classmethod implementation (plans 01-02 through 01-04)
- Test infrastructure (tests/conftest.py) established for subsequent plans
- No blockers

---
*Phase: 01-foundation*
*Completed: 2026-05-21*
