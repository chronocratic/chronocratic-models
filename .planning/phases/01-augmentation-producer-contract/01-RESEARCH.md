# Phase 01: Augmentation Producer Contract — Research

**Date:** 2026-06-12
**Phase:** 01
**Domain:** Augmentation refactoring (producer contract, primitives, producers)

## Summary

This research maps the current augmentation landscape across 4 models (TS2Vec, CoST, AutoTCL, TSTCC) to identify exact code locations, consumption patterns, import dependencies, and test coverage. The goal is to produce executable tasks for the planner to replace the monolithic `AugmentationMethod` contract with a typed, capability-segregated producer system.

**Current state:** All augmentations share a single `AugmentationMethod` base class. Models call `.augment()` and get a `TrainingViews` dataclass (views tuple + metadata dict). TSTCC adds a `DualAugmentation` wrapper for weak/strong view pairs. AutoTCL adds a `TrainableAugmentation` subclass for neural-network augmentations. The codebase has 955 knowledge graph nodes and 1763 edges; the graph is fresh (1h old, same commit).

**Primary recommendation:** Replace `AugmentationMethod.augment() -> TrainingViews` with `AugmentationProducer.produce() -> ViewSet`. Extract primitives (Jitter, Scaling, Permutation) into shared modules. Keep model-specific augmentations (CropShift, CosTRandom, AutoTCLNeuralNetwork) alongside their models. Eliminate `DualAugmentation` in favor of typed view sets.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **File layout:** `augmentation/` package with `base.py` (ABCs), `producers.py` (shared), `primitives.py` (shared); model-specific augmentations stay in their directories (D-01)
- **Augmentation training strategy:** `AugmentationTrainingStrategy` remains unchanged in `base.py`; not a producer (D-07)
- **Primitive extraction:** `Jitter`, `Scaling`, `Permutation`, `ComposeAugmentation` move from `tstcc/augmentations.py` to `augmentation/primitives.py` (D-06)
- **CosTRandomFunctionAugmentation:** Stays in `cost/augmentation.py` — model-specific primitive (D-06)
- **Migration sequencing:** Phase A (build new files) → Phase B (wire 4 models one at a time) → Phase C (delete old symbols, `augmentation/dual.py`) (D-03)
- **Test strategy:** TDD per task wave; numerical equivalence via seeded comparison (D-04)
- **Import transition:** Keep old names in barrel during migration; delete in final commit (D-05)
- **Scope:** TS2Vec, CoST, AutoTCL, TSTCC only. MCL, Series2Vec, TimeVAE, TimeNet, TST out of scope.

### Claude's Discretion
- Exact task ordering within Phase A/B (dependency-driven)
- Primitive parameter dataclass co-location (inline in `primitives.py` or separate section)
- Barrel `__init__.py` export order and grouping

### Deferred Ideas (OUT OF SCOPE)
- Registry + self-selection (SPEC §10)
- Functional/HOF producer style (SPEC §10)
- Two-phase plan()/produce() (SPEC §10)
- Capability-forwarding decorators (SPEC §10)
- One generic Views[Meta] container (SPEC §10) — rejected, reintroduces index access

## Current Code Inventory

### Core Augmentation Module (`augmentation/`)

| File | Lines | Key Symbols | Purpose |
|------|-------|-------------|---------|
| `base.py` | 223 | `TrainingViews`, `AugmentationMethod`, `AugmentationTrainingStrategy`, `TrainableAugmentation` | ABCs and output dataclass |
| `dual.py` | 53 | `DualAugmentation` | Two-view wrapper (to be deleted in Phase C) |
| `__init__.py` | 60 | Re-exports all symbols from `base.py`, `dual.py`, and per-model modules | Barrel |

### Per-Model Augmentations

| File | Lines | Key Symbols | Model |
|------|-------|-------------|-------|
| `ts2vec/augmentation.py` | 133 | `CropShiftAugmentation`, `CropShiftAugmentationParameters` | TS2Vec |
| `cost/augmentation.py` | 111 | `CosTRandomFunctionAugmentation`, `CosTRandomFunctionAugmentationParameters` | CoST |
| `autotcl/augmentation/methods.py` | 170 | `AutoTCLNeuralNetworkAugmentation`, `AutoTCLNeuralNetworkAugmentationParameters` | AutoTCL |
| `autotcl/augmentation/training.py` | 128 | `RIPTrainingStrategy`, `AdversarialTrainingStrategy` | AutoTCL |
| `autotcl/augmentation/__init__.py` | 20 | Re-exports | AutoTCL |
| `tstcc/augmentations.py` | 256 | `Jitter`, `Scaling`, `Permutation`, `ComposeAugmentation`, `TSTCCDualAugmentation` + parameter dataclasses | TSTCC |

**Total augmentation code:** ~1,154 lines across 9 files

### Model Files

| File | Lines | Augmentation Usage |
|------|-------|-------------------|
| `ts2vec/model.py` | 151 | Single `.augment()` call, reads `views[0]`, `views[1]`, `metadata['crop_length']` |
| `cost/model.py` | 312 | Double `.augment()` call (query/key), reads `views[0]` each time |
| `autotcl/model.py` | 236 | Single `.augment()` call, reads `views[0]`, `isinstance(TrainableAugmentation)` checks |
| `tstcc/model.py` | 206 | Single `.augment()` call, reads `views[0]`, `views[1]`, uses `DualAugmentation` type |

## Model Consumption Patterns

### TS2Vec (single call, two views + metadata)

```python
# ts2vec/model.py:93-98
views = self._augmentation.augment(x, temporal_unit=self._temporal_unit)
crop_length = views.metadata['crop_length']
emb_1 = encoder(views.views[0])[:, -crop_length:]
emb_2 = encoder(views.views[1])[:, :crop_length]
```

- **Calls:** 1× `.augment()`
- **Views:** 2 (overlapping crops)
- **Metadata:** `crop_length` (int) — used for embedding slicing
- **Temp:** Passes `temporal_unit` kwarg
- **Default:** `CropShiftAugmentation()` (lazy import)

### CoST (double call, single view each)

```python
# cost/model.py:269-270
query = self._augmentation.augment(x).views[0]
key = self._augmentation.augment(x).views[0]
```

- **Calls:** 2× `.augment()` (independent stochastic transforms)
- **Views:** 1 per call
- **Metadata:** Not used
- **Default:** `CosTRandomFunctionAugmentation()` (lazy import)

### AutoTCL (single call, single view + trainable branch)

```python
# autotcl/model.py:148-149
views = self._augmentation.augment(x)
aug_x = views.views[0]
```

- **Calls:** 1× `.augment()`
- **Views:** 1
- **Metadata:** Not used
- **Trainable branch:** `isinstance(self._augmentation, TrainableAugmentation)` at lines 91, 124, 145, 195, 203
- **Default:** `AutoTCLNeuralNetworkAugmentation()` (lazy import)

### TSTCC (single call, two views via DualAugmentation)

```python
# tstcc/model.py:119-120
views = self._augmentation.augment(data)
aug1, aug2 = views.views[0], views.views[1]
```

- **Calls:** 1× `.augment()`
- **Views:** 2 (weak/strong)
- **Metadata:** Not used
- **Type:** `DualAugmentation` (lazy import)
- **Default:** `TSTCCDualAugmentation()` (lazy import)

## Import Graph Analysis

### Who imports from `augmentation/`

| Importer | Symbols | Path |
|----------|---------|------|
| `dual.py` | `AugmentationMethod`, `TrainingViews` | Direct from `base.py` |
| `tstcc/augmentations.py` | `AugmentationMethod`, `TrainingViews`, `DualAugmentation` | Direct from `base.py`, `dual.py` |
| `tstcc/model.py` | `DualAugmentation` | TYPE_CHECKING from `dual.py` |
| `ts2vec/model.py` | `AugmentationMethod` | Direct from `base.py` |
| `ts2vec/augmentation.py` | `AugmentationMethod`, `TrainingViews` | Direct from `base.py` |
| `cost/model.py` | `AugmentationMethod` | Direct from `base.py` |
| `cost/augmentation.py` | `AugmentationMethod`, `TrainingViews` | Direct from `base.py` |
| `autotcl/model.py` | `AugmentationMethod`, `TrainableAugmentation` | Direct from `base.py` |
| `autotcl/augmentation/methods.py` | `AugmentationTrainingStrategy`, `TrainableAugmentation`, `TrainingViews` | Direct from `base.py` |
| `autotcl/augmentation/training.py` | `AugmentationTrainingStrategy` | Direct from `base.py` |
| `autotcl/utils.py` | `AugmentationMethod` | Direct from `base.py` |

### Cross-model import paths

**No model imports augmentation code from another model.** Each model's augmentation lives in its own directory. The only cross-dependency is `DualAugmentation` (used by TSTCC) which inherits from `AugmentationMethod`.

### `augmentation/__init__.py` barrel imports

The barrel re-exports from:
- `base.py` (4 symbols)
- `dual.py` (1 symbol)
- `ts2vec/augmentation.py` (2 symbols)
- `cost/augmentation.py` (2 symbols)
- `autotcl/augmentation/` (4 symbols)

Total: 13 symbols in `__all__`

### Tests that import augmentation symbols

| Test File | Lines | Imports |
|-----------|-------|---------|
| `test_augmentation.py` | 332 | Barrel imports (10 symbols) |
| `test_augmentation_base.py` | 82 | Direct from `base.py` (5 symbols) |
| `test_augmentation_per_model.py` | 416 | Direct from `base.py` (4 symbols) |
| `test_aug_config.py` | 128 | Barrel imports (3 symbols) |
| `test_smoke.py` | 252 | Barrel imports (4 symbols), direct model imports (6 symbols) |
| `test_from_config.py` | 130 | Barrel imports (6 symbols) |

**Total augmentation test code:** ~1,340 lines

### Tests referencing old symbols

| Symbol | Test Files |
|--------|-----------|
| `TrainingViews` | `test_augmentation.py`, `test_augmentation_base.py`, `test_augmentation_per_model.py`, `test_smoke.py` |
| `AugmentationMethod` | `test_augmentation.py`, `test_augmentation_base.py`, `test_augmentation_per_model.py`, `test_smoke.py` |
| `DualAugmentation` | Not directly tested (TSTCC integration only) |
| `TrainableAugmentation` | `test_augmentation.py`, `test_augmentation_base.py`, `test_augmentation_per_model.py` |
| `AugmentationTrainingStrategy` | `test_augmentation.py`, `test_augmentation_base.py`, `test_augmentation_per_model.py` |

## Risk Assessment

### High Risk

1. **`TrainingViews` is the shared bag.** Every model accesses `.views[i]` and `.metadata[key]`. Changing the return type breaks all 4 models. Must migrate in lockstep.

2. **AutoTCL `isinstance(TrainableAugmentation)` checks** at 5 locations. These are runtime type checks that couple the model to the augmentation hierarchy. Must be replaced with duck-typed or capability-based access.

3. **TS2Vec `metadata['crop_length']` dependency.** The crop length is embedded in `TrainingViews.metadata` — a string-keyed dict. This is a runtime contract with no type safety.

4. **`DualAugmentation.augment()` implementation** (line 51-53) accesses `.views[0]` on sub-augmentation results. This couples the dual wrapper to the internal structure of `TrainingViews`.

### Medium Risk

5. **Barrel imports in tests.** Tests import from `augmentation/` barrel. If old symbols are removed before tests are migrated, all tests crash.

6. **`augmentation/__init__.py` re-exports per-model augmentations.** This creates a central import path that tests depend on. Must maintain during migration.

7. **`TSTCCDualAugmentation` inherits from `DualAugmentation` which inherits from `AugmentationMethod`.** Three-level inheritance chain. Breaking any level requires coordinated changes.

### Low Risk

8. **Primitive parameters dataclasses.** `JitterParameters`, `ScalingParameters`, `PermutationParameters` are self-contained. Moving them is a file rename + import update.

9. **`ComposeAugmentation`** only used by `TSTCCDualAugmentation._default_strong()`. Single consumer.

10. **`CropShiftAugmentation.lazy import`** of `extract_subsequences_per_row` is already a pattern. No structural change needed.

## Protocol Patterns in Codebase

The project uses `Protocol` (from `typing`) for structural typing:

- `RepresentationBackbone` — `@runtime_checkable` protocol with `representation_dim` property
- `BatchAdapter` — callable protocol with `__call__` signature

Both are in `src/tscollection/models/supervised/supervised.py` (lines 20-47). The pattern is:
1. Import `Protocol, runtime_checkable` from `typing`
2. Define structural interface
3. Use `@runtime_checkable` for `isinstance` support

**Recommendation:** New augmentation contract should use `Protocol` for `Augmentation` (primitive transform) and `AugmentationProducer` (view assembler). Use `@runtime_checkable` only if runtime inspection is needed. For `TrainableAugmentation`, continue using `nn.Module` + nominal inheritance as the existing pattern.

## Type Checking

- **Type checker:** `ty` (dev dependency, version >=0.0.28)
- **Type hints:** All public functions have type hints including return types
- **`from __future__ import annotations`:** Used selectively for circular import avoidance
- **Python 3.12 native union syntax:** `torch.Tensor | None` (not `Optional`)
- **Built-in generics:** `list[int]`, `tuple[torch.Tensor, ...]` (not `typing.List`)

## Dependencies

No external packages are required for this phase. All dependencies are already in the project:
- `torch` (PyTorch)
- `lightning.pytorch` (PyTorch Lightning)
- `numpy` (used by CropShift, CosT augmentations)
- Standard library: `abc`, `dataclasses`, `typing`

## Environment

- **Python:** 3.12
- **PyTorch:** >=2.4, <3.0
- **Lightning:** >=2.5, <3.0
- **OS:** macOS (Darwin 25.5.0)
- **Package manager:** uv

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `AugmentationTrainingStrategy` interface remains unchanged | User Constraints | Low — locked decision D-07 |
| A2 | TSTCC has no smoke test in `test_smoke.py` | Test Inventory | Medium — if TSTCC is missing from smoke tests, the migration plan may not have a regression guard for it |
| A3 | `DualAugmentation` is only consumed by TSTCC | Import Graph | Low — verified by grep |
| A4 | No model imports augmentation code from another model | Import Graph | Low — verified by grep |

## Open Questions

1. **ViewSet types:** Should `SingleView`, `ViewPair`, `AlignedPair` be frozen dataclasses or named tuples? (Claude's discretion)
2. **Primitive `__call__` vs `augment`:** The SPEC suggests `__call__: Tensor → Tensor` for primitives. This changes the method name from `augment()` to `__call__()`. Tests that call `.augment()` directly will break. (Phase C timing)
3. **`metadata` replacement:** TS2Vec needs `crop_length`. Should this be a property on the view set or a separate method? (Claude's discretion)
4. **Seeded decorator:** SPEC mentions `decorators.py` with `Seeded[V]`. Implementation scope not detailed. (Deferred)

## Sources

### Primary (HIGH confidence)
- Source code files (read directly)
- Knowledge graph (graphify-out/graph.json, fresh — 1h old, same commit)

### Secondary (MEDIUM confidence)
- CONTEXT.md (locked decisions)
- SPEC.md (requirements)
- ARCHITECTURE.md, CONVENTIONS.md, TESTING.md (project maps)

### Tertiary (LOW confidence)
- Web search results for Python Protocol patterns
- Lightning manual optimization documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all imports verified via grep, no external dependencies
- Architecture: HIGH — file inventory complete, import graph mapped
- Pitfalls: HIGH — runtime consumption patterns verified by reading model source

**Research date:** 2026-06-12
**Valid until:** 2026-07-12 (stable codebase, no active changes on this branch)
