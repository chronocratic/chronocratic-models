---
phase: 01-augmentation-producer-contract
plan: 05
type: execute
subsystem: ts2vec-model
tags:
  - producer-contract
  - tdd-green
  - ts2vec
  - aligned-pair
  - crop-shift-producer
dependency_graph:
  requires:
    - 01-01 (AugmentationMethod ABC)
    - 01-02 (Augmentation primitives)
    - 01-03 (Producer combinators: FullOverlapPair, Seeded)
    - 01-04 (Seeded decorator, maybe_* helpers)
  provides:
    - CropShiftProducer (AlignedPair-returning producer)
    - TS2Vec wired to AugmentationProducer[AlignedPair]
  affects: []
tech_stack:
  added: []
  patterns:
    - PEP 695 class type params (AugmentationProducer[V])
    - Structural protocol satisfaction (AugmentationProducer[AlignedPair])
key_files:
  created:
    - src/tscollection/models/convolutional/dilated/ts2vec/augmentation.py
  modified:
    - src/tscollection/models/convolutional/dilated/ts2vec/model.py
decisions:
  - CropShiftProducer returns AlignedPair (not TrainingViews), eliminating metadata dict dependency
  - produce(x) takes NO temporal_unit kwarg — baked into __init__ via CropShiftAugmentationParameters
  - TS2Vec constructor accepts AugmentationProducer[AlignedPair] | None (not AugmentationMethod)
  - CropShiftAugmentation kept as backward-compat alias (D-05)
metrics:
  duration_minutes: 25
  tasks_completed: 5
  tests_passing: 10
  completed_date: "2026-06-12"
---

# Phase 1 Plan 05: Wire TS2Vec to Producer Contract

GREEN phase: Reshaped `CropShiftAugmentation` to `CropShiftProducer` (returns `AlignedPair` instead of `TrainingViews`), wired `TS2Vec` to accept `AugmentationProducer[AlignedPair] | None` and use `.produce()` / `.first` / `.second` / `.overlap_length`.

## Completed Tasks

| Task | Name | Commit | Done Criteria |
|------|------|--------|---------------|
| RED (prev) | Failing tests for TS2Vec producer integration | `3f07892` | 10 tests, all failing |
| 1 | Reshape CropShiftAugmentation → CropShiftProducer | `4dac659` | Returns AlignedPair, has produce(x), backward compat alias |
| 2 | Wire TS2Vec model to AugmentationProducer[AlignedPair] | `4dac659` | Constructor typed, _encode_augmented_views uses .produce(), no metadata access |
| 3 | Run tests | `4dac659` | 10/10 passing |
| 4 | Type check | `4dac659` | ty check clean |

## Key Changes

### `augmentation.py` — CropShiftProducer
- Renamed class `CropShiftAugmentation` → `CropShiftProducer`
- Renamed `augment(data, **kwargs) -> TrainingViews` → `produce(x: torch.Tensor) -> AlignedPair`
- `temporal_unit` is NO longer a per-call kwarg — baked into `__init__` via `CropShiftAugmentationParameters`
- Returns `AlignedPair(first=..., second=..., overlap_length=crop_length)` instead of `TrainingViews(views=(...), metadata={'crop_length': ...})`
- Kept `CropShiftAugmentation = CropShiftProducer` alias (D-05: backward compat until final delete)
- Imports `AlignedPair` from `augmentation/base.py` (not barrel) to avoid circular deps

### `model.py` — TS2Vec
- Constructor: `augmentation: AugmentationProducer[AlignedPair] | None = None` (was `AugmentationMethod | None`)
- Default lazy import: `CropShiftProducer()` (was `CropShiftAugmentation()`)
- `_encode_augmented_views`: `pair = self._augmentation.produce(x)` (no `temporal_unit` kwarg)
- `encoder(pair.first)[:, -pair.overlap_length:]` (was `encoder(views.views[0])[:, -crop_length:]`)
- `encoder(pair.second)[:, :pair.overlap_length]` (was `encoder(views.views[1])[:, :crop_length]`)
- Removed `views.metadata['crop_length']` access

## Tests

10/10 passing (verified by `uv run pytest tests/test_ts2vec_producer.py -v`):
- CropShiftProducer produces AlignedPair ✓
- overlap_length in valid range ✓
- first/second shapes correct ✓
- Per-sample crop offsets ✓
- TS2Vec accepts CropShiftProducer ✓
- TS2Vec accepts FullOverlapPair(Jitter) ✓
- Default is CropShiftProducer ✓
- Train 5 steps with CropShiftProducer (finite loss) ✓
- Train 5 steps with FullOverlapPair(Jitter) (finite loss) ✓
- Seeded producer produces identical embeddings (SC-7 determinism) ✓

## Type Check

`ty check src/tscollection/models/convolutional/dilated/ts2vec/` — All checks passed!

## Verification

- [x] `CropShiftProducer` returns `AlignedPair` (not `TrainingViews`)
- [x] `TS2Vec` accepts `AugmentationProducer[AlignedPair] | None`
- [x] `TS2Vec._encode_augmented_views` uses `.produce()` and reads `.first`, `.second`, `.overlap_length`
- [x] No `views.metadata['crop_length']` access in TS2Vec
- [x] TS2Vec trains 5 steps with CropShiftProducer (finite loss)
- [x] Seeded TS2Vec produces identical embeddings (SC-7)
- [x] FullOverlapPair(Jitter) trains with finite loss
- [x] `ty check` passes

## Deviations from Plan

None — executed exactly as written. The augmentation.py was already reshaped from the RED phase context; model.py was updated during GREEN to wire to the new contract.

## Success Criteria

All met. See Verification checklist above.
