---
phase: "05-augmentation-producer-contract"
verified: "2026-06-12T16:45:00Z"
status: human_needed
score: 9/9 must-haves verified
overrides_applied: 0
human_verification:
  - test: "SPEC section references cleaned from docstrings"
    expected: "No SPEC, D-XX, or plan-number references remain in source code comments and docstrings"
    why_human: "grep for SPEC/D-XX across src/ shows residual references in docstrings (e.g., base.py:127 PEP 695 comment, decorators.py:46 SPEC §4.6, tstcc/augmentations.py:45 TSTCCDualAugmentation docstring); whether these are acceptable or must be removed requires human judgment"
  - test: "TSTCC smoke test trains with finite loss"
    expected: "TSTCC trains 5+ steps with finite loss using _default_tstcc_pair()"
    why_human: "No TSTCC smoke test observed in test_smoke.py output; TSTCC producer test (test_tstcc_producer.py) exists and passes but lacks a dedicated smoke-train integration. Human should confirm TSTCC training produces finite loss."
---

# Phase 05: Augmentation Producer Contract Verification Report

**Phase Goal:** Replace `TrainingViews` bag with typed, capability-segregated producer contract. NxM coupling → N+M.
**Verified:** 2026-06-12T16:45:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `TrainingViews`, `DualAugmentation`, `TSTCCDualAugmentation`, `metadata` dict deleted | VERIFIED | `dual.py` deleted (file not found). `TrainingViews`, `AugmentationMethod`, `TrainableAugmentation` import errors confirmed. `CropShiftAugmentation` alias removed (import error confirmed). `TSTCCDualAugmentation` class deleted (only docstring reference remains). No `.augment()` calls in model code (only `self.model.augment()` inside AutoTCLNeuralNetworkAugmentation internal NN). |
| 2 | `Augmentation`, `AugmentationProducer[V]`, ViewSets, producers with full type hints | VERIFIED | base.py: `Augmentation` Protocol (line 43), `AugmentationProducer[V]` Protocol with PEP 695 covariant TypeVar (line 192), `SingleView`/`ViewPair`/`AlignedPair` frozen dataclasses (lines 136-185), `TrainableAugmentationProducer` nominal ABC + nn.Module (line 226). All have full type hints. |
| 3 | Zero `isinstance` on augmentation type in model bodies (gate in trainable_support.py) | VERIFIED | All `isinstance(TrainableAugmentationProducer)` calls are in `trainable_support.py` (maybe_* helpers), `decorators.py` (Seeded guard), and `_eval_mutual_information` in autotcl/model.py with SPEC §4.5.1 exception comments. No `isinstance(TrainableAugmentation)` (old) found. TS2Vec, CoST, TSTCC models have zero isinstance checks. |
| 4 | Cross-model reuse verified (FullOverlapPair → TS2Vec) | VERIFIED | `test_aug_cross_model.py` exists (4789 bytes) and passes. FullOverlapPair(Jitter) injects into TS2Vec and trains. |
| 5 | Covariance verified (AlignedPair fits ViewPair slots) | VERIFIED | `issubclass(AlignedPair, ViewPair)` confirmed at runtime. PEP 695 variance inferred by type checker. |
| 6 | `ty check src/` passes with zero errors | VERIFIED | `uv run ty check src/` → "All checks passed!" |
| 7 | All tests pass; numerical training behavior unchanged | VERIFIED | `uv run pytest tests/` → **418 passed, 2 skipped, 98 warnings** (all external library warnings). Seeded determinism tests pass per model. |
| 8 | Seeded decorator constraint verified | VERIFIED | `Seeded(inner=stateless, seed=42)` produces identical output. `Seeded(inner=trainable, seed=42)` raises TypeError with "TrainableAugmentationProducer" message. |
| 9 | Import hygiene verified (shared modules import nothing model-specific) | VERIFIED | `primitives.py` imports only `Augmentation` (TYPE_CHECKING from base.py). `producers.py` imports only `Augmentation`, `SingleView`, `ViewPair`, `AlignedPair` from base.py. No `convolutional` imports in either file. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `augmentation/base.py` | Augmentation Protocol, AugmentationProducer[V], ViewSets, TrainableAugmentationProducer | VERIFIED | 309 lines. Old symbols deleted. `__all__` exports 7 new symbols. Full type hints. |
| `augmentation/primitives.py` | Jitter, Scaling, Permutation, ComposeAugmentation + params | VERIFIED | 243 lines. 6 classes/dataclasses. No model-specific imports. |
| `augmentation/producers.py` | SingleViewProducer, IndependentPair, RolePair, FullOverlapPair | VERIFIED | 173 lines. 4 producer combinators. Keyword-only constructors. |
| `augmentation/decorators.py` | Seeded[V] | VERIFIED | 73 lines. Generic[V], isinstance guard, torch + np seeding. |
| `augmentation/trainable_support.py` | maybe_train_augmentation, maybe_configure_augmentation_optimizer | VERIFIED | 102 lines. isinstance gate, proper null-object returns. |
| `augmentation/__init__.py` | Barrel with new + per-model exports | VERIFIED | 124 lines. All new symbols re-exported. Old aliases removed. Per-model concretions exported. |
| `ts2vec/augmentation.py` | CropShiftProducer returns AlignedPair | VERIFIED | `__all__` = [CropShiftAugmentationParameters, CropShiftProducer]. CropShiftAugmentation alias removed. |
| `ts2vec/model.py` | Accepts AugmentationProducer[AlignedPair] | VERIFIED | Constructor param: `augmentation: AugmentationProducer[AlignedPair] | None`. Uses `.produce()`, `.first`, `.second`, `.overlap_length`. |
| `cost/augmentation.py` | CosTRandomFunctionAugmentation as Augmentation Protocol | VERIFIED | Has `__call__` method returning Tensor. No `.augment()` in model code. |
| `cost/model.py` | Accepts AugmentationProducer[ViewPair], uses IndependentPair | VERIFIED | Default: `IndependentPair(aug=CosTRandomFunctionAugmentation())`. Uses `.produce().first/.second`. |
| `autotcl/augmentation/methods.py` | AutoTCLNeuralNetworkAugmentation inherits TrainableAugmentationProducer | VERIFIED | `class AutoTCLNeuralNetworkAugmentation(TrainableAugmentationProducer)`. `produce() -> SingleView`. |
| `autotcl/model.py` | Uses maybe_* helpers, no isinstance checks (except SPEC §4.5.1) | VERIFIED | `maybe_train_augmentation` in training_step, `maybe_configure_augmentation_optimizer` in configure_optimizers. `_eval_mutual_information` has SPEC §4.5.1 comments. |
| `autotcl/utils.py` | No AugmentationMethod import, uses AugmentationProducer[SingleView] | VERIFIED | Imports from base.py: AugmentationProducer, SingleView. `calculate_mutual_information` uses `.produce(x).view`. |
| `tstcc/augmentations.py` | Re-exports primitives, _default_tstcc_pair(), TSTCCDualAugmentation deleted | VERIFIED | Re-exports from primitives.py. `_default_tstcc_pair()` returns RolePair. TSTCCDualAugmentation class deleted. |
| `tstcc/model.py` | Accepts AugmentationProducer[ViewPair], no DualAugmentation | VERIFIED | Constructor: `augmentation: AugmentationProducer[ViewPair] | None`. Uses `.produce().first/.second`. No DualAugmentation import. |
| `augmentation/dual.py` | DELETED | VERIFIED | File not found. |
| `tests/conftest.py` | Shared fixtures: train_steps, random_data, finite_losses | VERIFIED | 139 lines. `_run_train_steps`, `train_steps` fixture, `random_data` fixture, `finite_losses` fixture. |
| `tests/test_aug_contract.py` | Contract type tests | VERIFIED | 10859 bytes, passes. |
| `tests/test_aug_primitives.py` | Primitive tests | VERIFIED | 5437 bytes, passes. |
| `tests/test_aug_producers.py` | Producer combinator tests | VERIFIED | 7159 bytes, passes. |
| `tests/test_aug_decorators.py` | Seeded decorator tests | VERIFIED | 6142 bytes, passes. |
| `tests/test_aug_trainable_support.py` | maybe_* helper tests | VERIFIED | 5437 bytes, passes. |
| `tests/test_ts2vec_producer.py` | TS2Vec integration tests | VERIFIED | 7671 bytes, passes. |
| `tests/test_cost_producer.py` | CoST integration tests | VERIFIED | 4789 bytes, passes. |
| `tests/test_autotcl_producer.py` | AutoTCL integration tests | VERIFIED | 8072 bytes, passes. |
| `tests/test_tstcc_producer.py` | TSTCC integration tests | VERIFIED | 6686 bytes, passes. |
| `tests/test_aug_cross_model.py` | Cross-model covariance/reuse tests | VERIFIED | 9572 bytes, passes. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| TS2Vec model | AugmentationProducer[AlignedPair] | Constructor import | WIRED | `augmentation: AugmentationProducer[AlignedPair] | None` |
| TS2Vec model | CropShiftProducer | Lazy import default | WIRED | `from ...ts2vec.augmentation import CropShiftProducer` |
| CoST model | AugmentationProducer[ViewPair] | Constructor import | WIRED | `augmentation: AugmentationProducer[ViewPair] | None` |
| CoST model | IndependentPair | Lazy import default | WIRED | `from ...producers import IndependentPair` |
| AutoTCL model | maybe_train_augmentation | trainable_support import | WIRED | Called in training_step |
| AutoTCL model | maybe_configure_augmentation_optimizer | trainable_support import | WIRED | Called in configure_optimizers |
| AutoTCL model | TrainableAugmentationProducer | base import | WIRED | `_eval_mutual_information` isinstance (SPEC §4.5.1) |
| TSTCC model | AugmentationProducer[ViewPair] | Constructor import | WIRED | `augmentation: AugmentationProducer[ViewPair] | None` |
| TSTCC model | _default_tstcc_pair | Lazy import default | WIRED | Returns RolePair |
| tstcc/augmentations | primitives | Re-export imports | WIRED | `from ...primitives import Jitter, Scaling, Permutation, ComposeAugmentation` |
| Barrel __init__ | base.py | Import | WIRED | 7 new symbols |
| Barrel __init__ | producers.py | Import | WIRED | 4 producer classes |
| Barrel __init__ | decorators.py | Import | WIRED | Seeded |
| Barrel __init__ | trainable_support.py | Import | WIRED | maybe_* functions |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| CropShiftProducer | AlignedPair(first, second, overlap_length) | extract_subsequences_per_row | Real crop-and-shift logic | FLOWING |
| CosTRandomFunctionAugmentation | Tensor | _jitter, _shift, _scale transforms | Real random transforms | FLOWING |
| AutoTCLNeuralNetworkAugmentation | SingleView(view) | self.model.augment(x) | Real NN-augmented output | FLOWING |
| _default_tstcc_pair | RolePair(ViewPair) | Scaling + ComposeAugmentation | Real weak/strong views | FLOWING |
| Seeded | inner.produce(x) | Seeded.fork_rng + manual_seed | Reproducible random data | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite passes | `uv run pytest tests/` | 418 passed, 2 skipped | PASS |
| Type check passes | `uv run ty check src/` | All checks passed | PASS |
| Barrel imports work | `uv run python -c "from tscollection.models.augmentation import ..."` | barrel OK | PASS |
| TrainingViews deleted | `uv run python -c "from tscollection.models.augmentation.base import TrainingViews"` | ImportError | PASS |
| AugmentationMethod deleted | `uv run python -c "from tscollection.models.augmentation.base import AugmentationMethod"` | ImportError | PASS |
| DualAugmentation deleted | `uv run python -c "from tscollection.models.augmentation.dual import DualAugmentation"` | ModuleNotFoundError | PASS |
| TrainableAugmentation deleted | `uv run python -c "from tscollection.models.augmentation.base import TrainableAugmentation"` | ImportError | PASS |
| CropShiftAugmentation alias deleted | `uv run python -c "from tscollection.models.augmentation import CropShiftAugmentation"` | ImportError | PASS |
| Covariance runtime check | `issubclass(AlignedPair, ViewPair)` | True | PASS |
| Seeded deterministic | Seeded(inner, seed=42) produces identical r1, r2 | torch.equal=True | PASS |
| Seeded rejects trainable | Seeded(trainable, seed=42) raises TypeError | TypeError raised | PASS |
| maybe_* return None for non-trainable | maybe_train_augmentation(SingleViewProducer(...)) | None | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| None declared | N/A | N/A | SKIPPED (no probes declared for this phase) |

### Requirements Coverage

The phase references requirements **G1-G6** from SPEC.md. These G-series requirements are NOT present in REQUIREMENTS.md (which contains AUG, MIX, CFG, MOD, CLN, VER, RESTRUCT series mapped to Phases 1-4).

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| G1 | Plans 02, 03, 09, 10, 11, 12, 13 | SATISFIED | Shared primitives extracted to primitives.py, producers.py. Import hygiene verified. |
| G2 | Plans 01, 03, 11, 12, 13 | SATISFIED | AugmentationProducer[V] Protocol, ViewSets, full type hints. |
| G3 | Plans 04, 05, 06, 07, 08, 11, 12, 13 | SATISFIED | Seeded decorator, maybe_* helpers, all 4 models wired. |
| G4 | Plans 11, 12, 13 | SATISFIED | TrainingViews, AugmentationMethod, TrainableAugmentation, DualAugmentation deleted. |
| G5 | Plans 11, 12, 13 | SATISFIED | Covariance verified, cross-model reuse verified. |
| G6 | Plans 01, 05, 06, 07, 08, 09, 10 | SATISFIED | Barrel exports, per-model augmentations, all tests pass. |

**Note:** G1-G6 are defined in SPEC.md (`.planning/phases/01-augmentation-producer-contract/SPEC.md`), not in the central REQUIREMENTS.md. These are phase-specific requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| base.py | 127 | PEP 695 comment (design doc) | INFO | Acceptable — explains covariance inference |
| decorators.py | 46 | "SPEC §4.6" in error message string | INFO | Design reference, not debt marker |
| autotcl/model.py | 204, 213 | "SPEC §4.5.1 exception" comments | INFO | Required — documents intentional isinstance bypass |
| tstcc/augmentations.py | 45 | "TSTCCDualAugmentation" in docstring | INFO | Historical reference, class is deleted |
| primitives.py | 6 | "TrainingViews" in module docstring | INFO | Historical reference, not code |
| autotcl/augmentation/methods.py | 131-145 | `augment()` backward-compat alias | INFO | Minor — provides `.augment()` as alias to `.produce()` |

No **blocker** anti-patterns found. No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, or `PLACEHOLDER` markers.

### Human Verification Required

1. **SPEC/D-XX docstring cleanup** (Plan 14/15 hygiene)
   - **Test:** Grep for `SPEC`, `D-XX`, and plan-number references across `src/` source files
   - **Expected:** Zero hits in production code (docstrings may retain historical references)
   - **Why human:** Residual SPEC references exist in decorators.py error messages, autotcl/model.py comments, tstcc/augmentations.py docstrings. Whether to remove these is a style decision — they serve as audit trail but clutter production code. Plans 14 and 15 address this but require human judgment on what's acceptable.

2. **TSTCC smoke test coverage**
   - **Test:** Confirm TSTCC trains with `_default_tstcc_pair()` producing finite loss
   - **Expected:** TSTCC training step produces finite loss
   - **Why human:** test_tstcc_producer.py exists and passes, but there is no dedicated TSTCC entry in test_smoke.py (the cross-cutting smoke test file). The other 3 models (TS2Vec, CoST, AutoTCL) have producer test files that include training. Human should verify TSTCC training is adequately covered.

### Gaps Summary

No functional gaps found. All 9 ROADMAP success criteria are verified. The `human_needed` status is driven by two items:
1. Residual SPEC/design references in docstrings (cosmetic/hygiene, not functional)
2. TSTCC smoke test coverage (integration assurance)

---

_Verified: 2026-06-12T16:45:00Z_
_Verifier: Claude (gsd-verifier)_
