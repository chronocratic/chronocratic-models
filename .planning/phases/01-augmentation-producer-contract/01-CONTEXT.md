# Phase 01: Augmentation Producer Contract - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

## Phase Boundary

Replace untyped `TrainingViews` augmentation bag with typed, capability-segregated contract. Decouple augmentation *primitives* from model-specific *view assembly*. Eliminate `metadata` dict, `DualAugmentation` ABC, and all runtime branching on augmentation type inside model bodies. N×M coupling → N+M.

**In scope:** TS2Vec, CoST, AutoTCL, TS-TCC only. Augmentation contract, producers, primitives, decorators, trainable support, test suite.

**Out of scope:** MCL, Series2Vec, TimeVAE, TimeNet, TST. Model architectures, losses, hyperparameters. Name→producer registry. Functional/HOF producer style.

## Requirements (locked via SPEC.md)

**8 requirements are locked.** See `01-SPEC.md` for full requirements, boundaries, and acceptance criteria.

Downstream agents MUST read `01-SPEC.md` before planning or implementing. Requirements are not duplicated here.

**In scope (from SPEC.md):** Core contract, primitive reshapes, producer combinators, 4 model migrations, test suite.
**Out of scope (from SPEC.md):** Non-consuming models, model architectures/losses/hyperparameters, registry, functional/HOF style, capability-forwarding decorators.

## Implementation Decisions

### D-01: File Layout (from SPEC §4.9)

```
augmentation/
  base.py              # Augmentation Protocol, AugmentationProducer Protocol,
                       #   SingleView / ViewPair / AlignedPair, TrainableAugmentationProducer,
                       #   AugmentationTrainingStrategy (retained unchanged)
  primitives.py        # NEW — shared agnostic: Jitter, Scaling, Permutation, ComposeAugmentation
  producers.py         # NEW — shared agnostic combinators: SingleViewProducer, IndependentPair,
                       #   RolePair, FullOverlapPair
  decorators.py        # NEW — Seeded[V] (stateless-only)
  trainable_support.py # NEW — maybe_train_augmentation, maybe_configure_augmentation_optimizer

convolutional/dilated/ts2vec/augmentation.py   # CropShiftProducer (model-specific)
convolutional/dilated/cost/augmentation.py     # CosTRandomFunctionAugmentation (model-specific primitive)
convolutional/dilated/autotcl/augmentation/    # AutoTCLNeuralNetworkAugmentation (trainable)
convolutional/standard/tstcc/augmentations.py  # _default_tstcc_pair() (per-model wiring)
```

**Rules:** Shared modules import nothing model-specific. No model imports augmentation code from another model.

### D-02: Trainable Gate — Centralized Null-Object Helpers

`AugmentationProducer[V]` — Protocol, static-only. Never `@runtime_checkable`, never `isinstance`.
`TrainableAugmentationProducer` — nominal ABC + `nn.Module`. Only place for `isinstance`.

Two helpers in `augmentation/trainable_support.py` are the **only** augmentation-type branches in the codebase. Models call helpers; path is branchless on their side:
- `maybe_train_augmentation()` → `Tensor | None`
- `maybe_configure_augmentation_optimizer()` → `Optimizer | None`

AutoTCL param: `AugmentationProducer[SingleView] | None`. Static augs (e.g., `SingleViewProducer(Jitter(...))`) drop in — helpers return `None`, encoder training runs identically.

### D-03: Migration Sequencing — Bottom-Up

**Phase A:** Build new files (base.py, primitives.py, producers.py, decorators.py, trainable_support.py). No deletes. New code green alongside old.
**Phase B:** Wire 4 models one at a time (TS2Vec → CoST → AutoTCL → TS-TCC). Each model commit atomic, tests pass.
**Phase C:** Delete old symbols (TrainingViews, DualAugmentation, TSTCCDualAugmentation, AugmentationMethod old ABC) + `augmentation/dual.py`. Final cleanup.

### D-04: Test Strategy — TDD Per Task Wave

Write tests for contract types, primitives, producers as each is built. Model migration: update existing tests to new contract per model. Final wave: cross-model reuse + covariance tests. Numerical equivalence: seeded comparison of old vs new for each model.

### D-05: Import Transition — Keep Old Until Final Delete

`augmentation/__init__.py` keeps old names (`AugmentationMethod`, `TrainingViews`) alongside new exports during migration. Old tests, old model code still import. Final delete commit removes both symbols and exports. No breakage mid-migration.

### D-06: Primitive Extraction — In Scope

`Jitter`, `Scaling`, `Permutation`, `ComposeAugmentation` move from `tstcc/augmentations.py` to `augmentation/primitives.py`, reshaped to `Augmentation` Protocol (`__call__: Tensor → Tensor`). `tstcc/augmentations.py` imports them. Required for N+M decoupling at import layer.

`CosTRandomFunctionAugmentation` stays in `cost/augmentation.py` — model-specific function pool, not shared.

### D-07: AugmentationTrainingStrategy — Retained Unchanged

Loss strategy ABC in `base.py`. Not a producer — naming convention (`*Producer`) doesn't apply. Signature unchanged. `TrainableAugmentationProducer` still takes it in `__init__`.

### Claude's Discretion

- Exact task ordering within Phase A/B (dependency-driven).
- Primitive parameter dataclass co-location (inline in `primitives.py` or separate section).
- Barrel `__init__.py` export order and grouping.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spec & Design
- `.planning/phases/01-augmentation-producer-contract/01-SPEC.md` — Locked requirements, design, file layout, success criteria
- `.planning/PROJECT.md` — Core value: subclass-to-add, library-only scope
- `.planning/ROADMAP.md` — Phase 1-4 context (prior milestone)

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Augmentation layer, data flow, key abstractions
- `.planning/codebase/CONVENTIONS.md` — Naming, typing, kw-only, Protocol patterns
- `.planning/codebase/TESTING.md` — Test structure, smoke tests, augmentation extensibility tests

### Source Files (Current State)
- `src/tscollection/models/augmentation/base.py` — Current ABCs: AugmentationMethod, TrainableAugmentation, TrainingViews
- `src/tscollection/models/augmentation/dual.py` — Current DualAugmentation ABC (to delete)
- `src/tscollection/models/convolutional/dilated/ts2vec/model.py` — TS2Vec model (consumes TrainingViews)
- `src/tscollection/models/convolutional/dilated/cost/model.py` — CoST model (calls augment twice)
- `src/tscollection/models/convolutional/dilated/autotcl/model.py` — AutoTCL model (trainable gate)
- `src/tscollection/models/convolutional/standard/tstcc/augmentations.py` — Current Jitter, Scaling, Permutation, ComposeAugmentation (extract to primitives.py)
- `src/tscollection/models/convolutional/standard/tstcc/model.py` — TS-TCC model (DualAugmentation consumer)

## Existing Code Insights

### Reusable Assets
- Protocol patterns already used: `BatchAdapter`, `RepresentationBackbone` in `supervised/supervised.py` — same `@runtime_checkable` + structural typing approach
- Frozen dataclasses: existing project convention for immutable configs
- `AugmentationTrainingStrategy`: retains unchanged — existing `should_train(epoch, batch_idx)` interface
- Test infrastructure: `_train_steps()` helper in `test_smoke.py`, existing smoke tests for all 4 models

### Established Patterns
- Keyword-only constructors (`*, input_dims: int, ...`) — project convention
- Google-style docstrings — project convention
- Full type hints on all public functions — project convention
- Lazy imports for circular dependency avoidance (e.g., TS2Vec `__init__` imports CropShiftAugmentation)
- `__all__` exports in every module

### Integration Points
- `augmentation/__init__.py` — barrel re-export, update with new symbols
- Each model's `__init__` — constructor param changes: `AugmentationProducer[ViewSet] | None`
- Each model's training loop — `.augment()` → `.produce()`, result access changes from `.views[0]` to `.first`/`.second`
- AutoTCL `configure_optimizers()` — add `maybe_configure_augmentation_optimizer()`
- AutoTCL `training_step()` — add `maybe_train_augmentation()`
- All existing augmentation tests — `TrainingViews` → ViewSet types, `AugmentationMethod` → `Augmentation`/`AugmentationProducer`

## Specific Ideas

- SPEC §4.1 reframe: `overlap_length` as universal question, not crop property. Every pair-producer answers. Model has one code path.
- SPEC §4.8 mental model: model depends on `AugmentationProducer[ItsViewSet]`, producer assembles from primitives. Liskov substitution + covariance makes rich producers usable in narrow slots.

## Deferred Ideas

- **Registry + self-registration** (SPEC §10) — config-string selection
- **Functional/HOF producer style** (SPEC §10) — revisit if producer set grows
- **Two-phase plan()/produce()** (SPEC §10) — for pre-apply parameter queries
- **Capability-forwarding decorators** (SPEC §10) — wrap trainable producers
- **One generic Views[Meta] container** (SPEC §10) — rejected, reintroduces index access

---

*Phase: 01-Augmentation Producer Contract*
*Context gathered: 2026-06-12*
