---
phase: 01-foundation
plan: 02
subsystem: testing
tags: [mixin, abc, polymorphism, encoding, pytorch, pytest, tdd]

# Dependency graph
requires: []
provides:
  - BaseEncodingMixin(ABC) with shared encode() entry point
  - PoolingEncodingMixin for TS2Vec/AutoTCL-style encoding
  - DecompositionEncodingMixin for CoST-style encoding
  - Polymorphic _get_encoder/_get_eval_method/_get_slice dispatch
  - Bug fixes: persistent_workers=num_workers > 0, sliding window transpose
affects:
  - 01-03 (model inheritance update)
  - 01-04 (config dataclasses)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mixin hierarchy with @override decorators"
    - "TYPE_CHECKING guard for MaskMode to avoid circular imports"
    - "Polymorphic dispatch instead of hasattr branching"

key-files:
  created:
    - tests/test_mixin.py
    - tests/conftest.py
  modified:
    - src/tscollection/models/_abstract/encoding_functionality_mixin.py
    - src/tscollection/models/_abstract/__init__.py

key-decisions:
  - "Split single EncodingFunctionalityMixin into 3-class ABC hierarchy"
  - "Dropped encoder is None guard (D-05: encoders always set by constructors)"
  - "Use polymorphic dispatch via _get_encoder()/_get_eval_method() instead of instance state mutation"
  - "Fix persistent_workers crash: use num_workers > 0 condition"
  - "Fix sliding window full_series shape: use .transpose(1, 2).contiguous()"

patternsestablished:
  - "Template method pattern via _get_encoder, _get_eval_method, _get_slice"
  - "@override decorator on all subclass method overrides"
  - "Module-level _EXPECTED_INPUT_DIMS constant for D103 compliance"

requirements-completed: [MIX-01, MIX-02, MIX-03, MIX-04]

# Metrics
duration: 20min
completed: 2026-05-21
---

# Phase 1 Plan 2: Mixin Hierarchy Refactor Summary

**3-class ABC mixin hierarchy replacing hasattr branching with polymorphic dispatch, including persistent_workers and sliding window bug fixes**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-21T11:53:00Z
- **Completed:** 2026-05-21T12:13:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Created 24 passing tests covering importability, hierarchy structure, polymorphic dispatch, encode behavior, bug fixes, decomposition validation, and source compliance
- Replaced single `EncodingFunctionalityMixin` with `BaseEncodingMixin(ABC)`, `PoolingEncodingMixin`, `DecompositionEncodingMixin`
- Eliminated all `hasattr()` branching in favor of compile-time polymorphism
- Fixed `persistent_workers=True` crash when `num_workers=0`
- Fixed sliding window `full_series` shape regression with `.transpose(1, 2).contiguous()`
- Dropped `encoder is None` guard per D-05 (encoders always set by constructors)
- Updated `__init__.py` barrel exports for the 3 new classes

## Task Commits

Each task was committed atomically (TDD cycle):

1. **Task 1: Mixin hierarchy with polymorphic dispatch and tests (RED)** - `b61070b` (test)
2. **Task 1: Implement 3-class mixin hierarchy (GREEN)** - `8dc8854` (feat)

## Files Created/Modified
- `tests/test_mixin.py` - 24 unit tests covering all behavior, structure, and compliance requirements
- `tests/conftest.py` - Shared test fixtures (minimal)
- `src/tscollection/models/_abstract/encoding_functionality_mixin.py` - Complete rewrite: 3-class hierarchy with polymorphic dispatch
- `src/tscollection/models/_abstract/__init__.py` - Updated barrel exports for BaseEncodingMixin, PoolingEncodingMixin, DecompositionEncodingMixin

## Decisions Made
- Used `@override` decorator (typing.override, native in Python 3.12) on all method overrides in PoolingEncodingMixin and DecompositionEncodingMixin
- Placed `MaskMode` import inside `TYPE_CHECKING` guard to avoid circular imports
- Used `_logger` with `%s` formatting (not f-strings) per project conventions
- Made `causal`, `sliding_length`, `sliding_padding` keyword-only after `*` in `encode()` signature
- Extracted `_EXPECTED_INPUT_DIMS = 3` constant for clarity and D103 compliance
- Used explicit import list in `__init__.py` (not star import) for clarity

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed type checker error: all_outputs variable reassigned**
- **Found during:** Task 1 (GREEN phase, ty check)
- **Issue:** `ty check` flagged `all_outputs` being reassigned from `list[Tensor]` to `Tensor` via `torch.cat`
- **Fix:** Renamed the concatenated result to `result` to avoid type reassignment
- **Files modified:** src/tscollection/models/_abstract/encoding_functionality_mixin.py
- **Verification:** `ty check` passes cleanly
- **Committed in:** `8dc8854` (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Type checker fix required for compliance. No behavioral change.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Mixin hierarchy is complete and tested
- Plan 01-03 (model inheritance update) can now switch TS2Vec/AutoTCL to `PoolingEncodingMixin` and CoST to `DecompositionEncodingMixin`
- Old `EncodingFunctionalityMixin` name is removed from exports; downstream plans must update imports

---
*Phase: 01-foundation*
*Completed: 2026-05-21*
