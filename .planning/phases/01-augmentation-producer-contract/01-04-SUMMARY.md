---
phase: 01-augmentation-producer-contract
plan: 04
subsystem: augmentation
tags: [torch, pytest, TDD, decorator, trainable-support]

requires:
  - 01-01 (AugmentationProducer[V], TrainableAugmentationProducer in base.py)
  - 01-02 (Shared primitives)
  - 01-03 (Producer combinators)
provides:
  - Seeded[V]: Deterministic wrapper for stateless AugmentationProducer
  - maybe_train_augmentation(): Centralized trainable gate
  - maybe_configure_augmentation_optimizer(): Centralized optimizer gate
affects:
  - 01-05 (TS2Vec wiring), 01-07 (AutoTCL wiring)

tech-stack:
  added: []
  patterns:
    - PEP 695 inline type parameters (class Seeded[V])
    - Null-Object Pattern for maybe_* helpers
    - isinstance gate on TrainableAugmentationProducer (D-02, only place)

key-files:
  created:
    - src/tscollection/models/augmentation/decorators.py
    - src/tscollection/models/augmentation/trainable_support.py
    - tests/test_aug_decorators.py
    - tests/test_aug_trainable_support.py
  modified: []

key-decisions:
  - "Seeded uses PEP 695 inline type params: class Seeded[V]"
  - "trainable_support.py is the ONLY place with isinstance(..., TrainableAugmentationProducer)"
  - "Seeded rejects TrainableAugmentationProducer at runtime (SPEC §4.6)"

requirements-completed: [G1, G2]

duration: 3min
completed: 2026-06-12
---

# Phase 1 Plan 4: Decorators + Trainable Helpers Summary

**Seeded[V] decorator and centralized maybe_* trainable helpers implemented with 20 TDD-verified tests, PEP 695 inline type parameters, zero TypeVar usage**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-12T13:29:00Z
- **Completed:** 2026-06-12T13:33:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files created:** 4

## Accomplishments

- `Seeded[V]` — Deterministic wrapper for stateless `AugmentationProducer`. Uses `torch.random.fork_rng()` + `torch.manual_seed()`. Rejects `TrainableAugmentationProducer` at `__init__` (SPEC §4.6). PEP 695 inline type params — no `TypeVar`.
- `maybe_train_augmentation()` — Returns `None` for stateless producers, delegates to `train_step()` for trainable. Only `isinstance(..., TrainableAugmentationProducer)` gate in codebase (D-02).
- `maybe_configure_augmentation_optimizer()` — Returns `None` for stateless producers, delegates to `configure_optimizer()` for trainable.
- 20 passing tests: 8 Seeded behavioral + guard tests, 12 maybe_* helper tests
- `ty check` zero errors, `ruff check` zero errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Seeded decorator + maybe_* helpers (RED + GREEN)** — `34012c8` (test), `bef4ccb` (feat)

## Files Created

- `src/tscollection/models/augmentation/decorators.py` — `Seeded[V]` class with `torch.random.fork_rng()`. Rejects trainable producers. PEP 695 type params.
- `src/tscollection/models/augmentation/trainable_support.py` — `maybe_train_augmentation()` and `maybe_configure_augmentation_optimizer()`. Centralized `isinstance` gates per D-02.
- `tests/test_aug_decorators.py` — 8 tests: Seeded produces deterministic output, preserves ViewSet type, rejects TrainableAugmentationProducer, isinstance gate verification.
- `tests/test_aug_trainable_support.py` — 12 tests: maybe_train_augmentation returns None for stateless, delegates for trainable, respects should_train_augmentation, optimizer configuration.

## Decisions Made

- **PEP 695 inline type params**: `class Seeded[V]` instead of `V = TypeVar("V")`.
- **Centralized isinstance**: `trainable_support.py` is the only file with `isinstance(..., TrainableAugmentationProducer)` — D-02.
- **TYPE_CHECKING imports**: `torch` imported in `TYPE_CHECKING` block where possible.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `uv run pytest tests/test_aug_decorators.py tests/test_aug_trainable_support.py -v` — 20 passed
- `uv run ty check src/tscollection/models/augmentation/` — All checks passed
- `uv run ruff check src/tscollection/models/augmentation/decorators.py src/tscollection/models/augmentation/trainable_support.py` — All checks passed

## Known Stubs

None — both modules have complete, functional implementations.

## Threat Flags

None — decorator and helpers are pure functional wrappers with no network, auth, or file access surfaces.

---
*Phase: 01-augmentation-producer-contract*
*Completed: 2026-06-12*
