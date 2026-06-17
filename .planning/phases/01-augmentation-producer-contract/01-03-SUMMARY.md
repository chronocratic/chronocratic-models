---
phase: 01-augmentation-producer-contract
plan: 03
subsystem: augmentation
tags: [torch, pytest, TDD, Protocol, dataclass, producer-combinators]

requires:
  - 01-01 (Augmentation Protocol, AugmentationProducer[V], SingleView, ViewPair, AlignedPair in base.py)
  - 01-02 (Shared primitives: Jitter, Scaling, Permutation, ComposeAugmentation)
provides:
  - SingleViewProducer: wraps one Augmentation, returns SingleView
  - IndependentPair: applies one Augmentation twice, returns ViewPair
  - RolePair: applies two Augmentations, returns ViewPair
  - FullOverlapPair: applies one Augmentation twice, returns AlignedPair
  - 15 passing TDD tests for all producer combinators
affects:
  - 01-04 (decorators), 01-05 (trainable_support), model migrations (Phase B)

tech-stack:
  added: []
  patterns:
    - AugmentationProducer[V] structural Protocol compliance
    - Keyword-only constructors (CONVENTIONS.md pattern)
    - TYPE_CHECKING-only torch import (from __future__ import annotations)

key-files:
  created:
    - src/tscollection/models/augmentation/producers.py
    - tests/test_aug_producers.py
  modified: []

key-decisions:
  - "All four producers satisfy AugmentationProducer[V] structurally (no inheritance)"
  - "producers.py imports only from base.py — no model-specific or primitive dependencies"
  - "Keyword-only constructors (*, aug=...) enforce named arguments per CONVENTIONS.md"

requirements-completed: [G1, G2]

duration: 4min
completed: 2026-06-12
---

# Phase 1 Plan 3: Producer Combinators Summary

**Shared producer combinators (SingleViewProducer, IndependentPair, RolePair, FullOverlapPair) implemented with 15 TDD-verified tests, zero model-specific imports, full type check compliance**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-06-12T11:26:39Z
- **Completed:** 2026-06-12T11:31:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files created:** 2

## Accomplishments

- `SingleViewProducer` — wraps one `Augmentation`, returns `SingleView(view=aug(x))`
- `IndependentPair` — applies one `Augmentation` twice, returns `ViewPair(first, second)` with independent stochastic draws
- `RolePair` — applies two different `Augmentation`s, returns `ViewPair` with role-specific transforms
- `FullOverlapPair` — applies one `Augmentation` twice, returns `AlignedPair` with `overlap_length=x.size(1)`
- All four satisfy `AugmentationProducer[V]` structurally via `produce(x) -> V`
- Keyword-only constructors prevent positional argument misuse
- 15 passing tests: 7 behavioral (VER-01 through VER-07), 4 protocol compliance, 4 keyword-only enforcement
- `ty check` passes with zero errors; `ruff check` passes with zero errors
- No model-specific imports — generic combinators per SPEC §4.4

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for producer combinators (RED)** — `60fbe71` (test)
2. **Task 1: Implement producer combinators (GREEN)** — `d05a515` (feat)

## Files Created

- `src/tscollection/models/augmentation/producers.py` — 4 producer classes + `__all__` exports. Imports only from `base.py` (Augmentation, AlignedPair, SingleView, ViewPair). No model-specific dependencies.
- `tests/test_aug_producers.py` — 15 tests: SingleViewProducer produces SingleView with .view tensor, IndependentPair produces ViewPair with independent draws, RolePair produces ViewPair with two different transforms, FullOverlapPair produces AlignedPair with overlap_length == T, protocol compliance, keyword-only constructors.

## Decisions Made

- **TYPE_CHECKING-only torch import**: `from __future__ import annotations` defers type evaluation, so `torch` lives in the TYPE_CHECKING block. Runtime code uses `x.size(1)` which doesn't reference `torch`.
- **No primitives dependency**: producers.py does not import from primitives.py — it only references the `Augmentation` Protocol from base.py. This satisfies the N+M decoupling: producers are generic combinators that accept any conforming primitive.
- **`__all__` alphabetical order**: Follows existing project convention.

## Deviations from Plan

None — plan executed exactly as written. All seven behavioral tests, protocol compliance tests, and keyword-only tests match the plan specification.

## Verification

- `uv run pytest tests/test_aug_producers.py -v` — 15 passed
- `uv run ty check src/tscollection/models/augmentation/producers.py` — All checks passed
- `uv run ruff check src/tscollection/models/augmentation/producers.py tests/test_aug_producers.py` — All checks passed
- Regression: `uv run pytest tests/test_aug_contract.py tests/test_aug_primitives.py -v` — 41 passed
- Import check: `producers.py` imports only from `base.py` (no primitives, no model-specific code)

## Known Stubs

None — all four producers have complete, functional implementations.

## Threat Flags

None — producer combinators are pure functional wrappers with no network, auth, or file access surfaces.

---
*Phase: 01-augmentation-producer-contract*
*Completed: 2026-06-12*
