# Roadmap: tsmodels

**Created:** 2026-05-21
**Mode:** yolo
**Granularity:** fine

---

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Foundation | 14/15 | In Progress|  |
| 2 | Directory Restructure | Move to cnn/dilated layout, collocate encoders/layers/augmentations | RESTRUCT-01..04 | 5 |
| 3 | Augmentation Refactor | 3/3 | Complete   | 2026-05-22 |
| 4 | Cleanup and Verification | 8/8 | Complete    | 2026-06-02 |
| 5 | Augmentation Producer Contract | 13/13 plans | In planning | 2026-06-12 |

---

### Phase 1: Foundation

**Goal:** Split encoding mixin into hierarchy and add config dataclasses

**Requirements:** MIX-01, MIX-02, MIX-03, MIX-04, CFG-01, CFG-03

**Plans:** 14/15 plans executed

**Success Criteria:**

1. TS2Vec and AutoTCL inherit `PoolingEncodingMixin`; CoST inherits `DecompositionEncodingMixin`
2. `encode()` behavior identical to pre-refactor for all 3 models (no regression)
3. `TS2VecModelParameters`, `CoSTModelParameters`, `AutoTCLModelParameters` dataclasses exist with `from_config()` classmethod
4. `ty check src/` passes with zero errors

**Canonical refs:** `_sources/rbspaper/src/rbspaper/models/abstract/encoding_functionality_mixin.py`, `_sources/rbspaper/src/rbspaper/models/config.py`

**Plan list:**

- [x] 01-01-PLAN.md — Config dataclasses (ModelParameters hierarchy, D-03, D-07)
- [x] 01-02-PLAN.md — Mixin hierarchy (Base/Pooling/Decomposition, D-05, D-06)
- [x] 01-03-PLAN.md — Model inheritance + from_config() factory (D-04)
- [x] 01-04-PLAN.md — Type checking + lint + test verification

---

### Phase 2: Directory Restructure

**Goal:** Move models to cnn/dilated layout, collocate encoders, layers, and augmentations

**Requirements:** RESTRUCT-01, RESTRUCT-02, RESTRUCT-03, RESTRUCT-04

**Success Criteria:**

1. Models live at `cnn/dilated/{ts2vec,cost,autotcl}/model.py`; all imports resolved
2. Encoders at `cnn/dilated/encoders/`, layers at `cnn/dilated/layers/convolutions/`
3. Model-specific augmentations colocated: `ts2vec/augmentation.py`, `cost/augmentation.py`, `autotcl/augmentation.py`
4. Shared layers (`BandedFourierLayer`) remain at `models/layers/general.py`
5. `ty check src/` passes with zero errors

**Canonical refs:** `.planning/phases/01-foundation/01-CONTEXT.md` D-08, D-09

---

### Phase 3: Augmentation Refactor

**Goal:** Replace enum-based augmentation dispatch with polymorphic ABC strategy system

**Requirements:** AUG-01, AUG-02, AUG-03, AUG-04, AUG-05, AUG-06, CFG-02, MOD-01, MOD-02, MOD-03, MOD-04, MOD-05

**Success Criteria:**

1. All 3 models accept `augmentation: AugmentationMethod` in constructor; no enum imports remain
2. AutoTCL training_step delegates augmentation loss to `augmentation.train_step()` — no internal dispatch methods
3. `RIPTrainingStrategy` and `AdversarialTrainingStrategy` produce identical loss values as pre-refactor
4. Custom identity `AugmentationMethod` subclass injects into any model without errors
5. `ruff check src/` passes; no references to `enums.py` or `factories.py`

**Canonical refs:** `.planning/todos/augmentation-refactor.md`

**Plans:** 3/3 plans complete

**Plan list:**

- [x] 03-01-PLAN.md — Rename _augmentation/ to augmentation/, build ABC hierarchy (TrainingViews, TrainableAugmentation, strategies), config dataclasses
- [x] 03-02-PLAN.md — Refactor TS2Vec, CoST, AutoTCL to accept AugmentationMethod polymorphically; update tests
- [x] 03-03-PLAN.md — Delete enums.py/factories.py, full test/lint/type verification

---

### Phase 4: Model Self-Containment & Augmentation Module Cleanup

**Goal:** Collocate augmentations, training strategies, configs, and losses with their models. Split the monolithic `augmentation/strategies.py` into properly named modules. Fix AutoTCL training schedule regression.

**Requirements:** CLN-01, CLN-02, CLN-03, CLN-04, VER-01, VER-02, VER-03, VER-04, VER-05

**Success Criteria:**

1. Concrete augmentations live in model directories: `ts2vec/augmentation.py`, `cost/augmentation.py`, `autotcl/augmentation.py`
2. Training strategies co-located: `autotcl/training.py` (RIPTrainingStrategy, AdversarialTrainingStrategy)
3. Augmentation param dataclasses co-located with their augmentation classes (not in a central config.py)
4. Model loss functions extracted to per-model modules: `cost/loss.py`, `autotcl/loss.py`, `ts2vec/loss.py`
5. Config hierarchy matches directory structure — CoST configs inherit from dilated model base
6. Shared `augmentation/` module contains only ABCs: `AugmentationMethod`, `TrainableAugmentation`, `TrainingViews`, `AugmentationTrainingStrategy`
7. AutoTCL `training_ratio_step` epoch-gating restored (regression fix: aug-network trains every N epochs, not every step)
8. Smoke tests pass: TS2Vec, CoST, AutoTCL each train 5 steps with finite loss
9. `ty check src/` and `ruff format --check src/` pass with zero errors

**Detailed context:**

The current `augmentation/strategies.py` (524 lines) mixes:

- General ABCs (AugmentationMethod, TrainableAugmentation, AugmentationTrainingStrategy)
- TrainingViews dataclass
- Training strategies (RIPTrainingStrategy, AdversarialTrainingStrategy) — AutoTCL-specific
- Concrete augmentations — each model-specific

Additionally, `augmentation/config.py` centralizes all param dataclasses. The `augmentation/__init__.py` barrel imports everything with star imports.

**Migration plan:**

1. Split `strategies.py` to `views.py` (TrainingViews), `base.py` (ABCs), `augmentations.py` (concrete classes + co-located params). Delete `config.py`.
2. Move `augmentations.py` contents into per-model directories:
   - `CropShiftAugmentation` + `CropShiftAugmentationParameters` to `ts2vec/augmentation.py`
   - `CosTRandomFunctionAugmentation` + `CosTRandomFunctionAugmentationParameters` to `cost/augmentation.py`
   - `AutoTCLNeuralNetworkAugmentation` + `AutoTCLNeuralNetworkAugmentationParameters` to `autotcl/augmentation.py`
   - `RIPTrainingStrategy`, `AdversarialTrainingStrategy` to `autotcl/training.py`
3. Update barrel `augmentation/__init__.py` to re-export from model directories for backward compat
4. Fix `RIPTrainingStrategy` and `AdversarialTrainingStrategy`: add `training_ratio_step` param, override `should_train()` with `epoch % ratio == 0`
5. Extract model losses: `cost/loss.py`, `autotcl/loss.py`, `ts2vec/loss.py`
6. Fix config inheritance for CoST

**Canonical refs:** Current branch `gsd/phase-03-augmentation-refactor`; main branch model files for original logic comparison.

**Plans:** 8/8 plans complete

**Plan list:**

- [x] 04-01-PLAN.md — Create augmentation/base.py with ABC hierarchy (D-01)
- [x] 04-02-PLAN.md — Create per-model config hierarchy + update from_config docstrings (D-08–D-13, D-18)
- [x] 04-03-PLAN.md — Create per-model augmentation files (D-04–D-06)
- [x] 04-04-PLAN.md — configs/ re-export package (created then removed, not needed)
- [x] 04-05-PLAN.md — Update per-model __init__.py barrel exports (CLN-03)
- [x] 04-06-PLAN.md — Fix 3 failing tests in test_aug_config.py (CLN-04)
- [x] 04-07-PLAN.md — Create smoke tests for VER-01 through VER-05 (TDD)
- [x] 04-08-PLAN.md — Make augmentation optional, delete from_config and merge_config_kwargs (CLN-04)

---

### Phase 5: Augmentation Producer Contract

**Goal:** Replace `TrainingViews` bag with typed, capability-segregated producer contract. N×M coupling → N+M.

**Requirements:** G1-G6 (from SPEC.md)

**Plans:** 13/13 plans ready

**Success Criteria:**

1. `TrainingViews`, `DualAugmentation`, `TSTCCDualAugmentation`, `metadata` dict deleted
2. `Augmentation`, `AugmentationProducer[V]`, ViewSets, producers with full type hints
3. Zero `isinstance` on augmentation type in model bodies (gate in `trainable_support.py`)
4. Cross-model reuse verified (FullOverlapPair → TS2Vec)
5. Covariance verified (AlignedPair fits ViewPair slots)
6. `ty check src/` passes with zero errors
7. All tests pass; numerical training behavior unchanged
8. Seeded decorator constraint verified
9. Import hygiene verified (shared modules import nothing model-specific)

**Canonical refs:** `.planning/phases/01-augmentation-producer-contract/SPEC.md`

**Plan list:**

- [x] 01-01-PLAN.md — Contract types: Augmentation, AugmentationProducer[V], ViewSets, TrainableAugmentationProducer
- [x] 01-02-PLAN.md — Shared primitives: Jitter, Scaling, Permutation, ComposeAugmentation
- [x] 01-03-PLAN.md — Producer combinators: SingleViewProducer, IndependentPair, RolePair, FullOverlapPair
- [x] 01-04-PLAN.md — Seeded decorator + trainable_support helpers
- [x] 01-05-PLAN.md — Wire TS2Vec: CropShiftProducer, AugmentationProducer[AlignedPair]
- [x] 01-06-PLAN.md — Wire CoST: IndependentPair, reshape CosTRandomFunctionAugmentation
- [x] 01-07-PLAN.md — Wire AutoTCL: TrainableAugmentationProducer, maybe_* helpers, utils.py
- [x] 01-08-PLAN.md — Wire TS-TCC: RolePair, _default_tstcc_pair(), delete DualAugmentation
- [x] 01-09-PLAN.md — Barrel updates: augmentation/__init__.py + autotcl barrel
- [x] 01-10-PLAN.md — Test migrations: update to new contract
- [x] 01-11-PLAN.md — Core deletion: remove TrainingViews, AugmentationMethod, dual.py
- [x] 01-12-PLAN.md — Cross-model verification: covariance, reuse, import hygiene
- [x] 01-13-PLAN.md — Per-model cleanup: remove old-symbol references from model files

### Phase 6: prepare to be published as package

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 5
**Plans:** 8/9 plans complete

Plans:

- [x] TBD (run /gsd-plan-phase 6 to break down) (completed 2026-06-15)

### Phase 7: consistent-parameters-and-defaults

**Goal:** Standardize hyperparameter naming across all 9 models, fill missing defaults from reference repositories, and extract hardcoded architecture constants into config dataclasses.

**Requirements:** D-01 through D-06 (from CONTEXT.md)

**Depends on:** Phase 6

**Plans:** 7 plans

Plans:

- [ ] 07-01-PLAN.md — TST: rename 8 config fields, update model and TSTransformerEncoder (Wave 1)
- [ ] 07-02-PLAN.md — MCL: rename input field, de-harden encoder, fix sync_dist bug (Wave 1)
- [ ] 07-03-PLAN.md — Series2Vec: add 5 missing defaults from reference repo (Wave 1)
- [ ] 07-04-PLAN.md — TSTCC: rename 8 config fields, de-harden encoder inner blocks (Wave 1)
- [ ] 07-05-PLAN.md — TimeNet: rename 3 config fields to canonical names (Wave 1)
- [ ] 07-06-PLAN.md — TimeVAE: rename config fields, cascade to base class, list->tuple (Wave 1)
- [ ] 07-07-PLAN.md — Update all test files, add config tests, full verification (Wave 2)

### Phase 8: fix-models-supervision

**Goal:** Remove `num_classes` and supervision-related code from model cores (TSTCC), delegating to `SupervisedModule` wrapper

**Depends on:** Phase 7
**Plans:** 3/3 plans complete (2026-06-23)

Plans:

- [x] 08-01-PLAN.md — Strip num_classes/features_len from config, encoder, model; update adapter and factory docstrings (D-01 through D-08)
- [x] 08-02-PLAN.md — Update all 7 TSTCC test files: remove deleted params, fix assertions, update mock (D-06, D-09)
- [x] 08-03-PLAN.md — Verification: full test suite, lint, type check (D-09)

---

### Phase 9: Fixes and Updates

**Goal:** Fix blocking runtime bugs, unify input shape contract, harden cross-device compatibility, and refactor encoding mixin.

**Depends on:** Phase 8

**Plans:** 3/3 plans ready

**Workstreams (3 independent):**

1. **Encoding Mixin Refactor + Gradient Enabled** — Collapse `BasicEncodingMixin` from 4-hook (`_get_encoder` + `_get_encoder_module` + `_prepare_inputs` + `_postprocess`) to 2-hook (`_get_encoder` → `nn.Module` + `_encode_batch`). Add `gradient_enabled: bool = False` kwarg to `encode()` on both mixin families.
   - **Spec:** `.planning/todos/specs/encoding-mixin-refactor-and-gradient-encode.md`

2. **Input Shape Convention + Conv1D Fix** — Audit confirmed: all 9 models receive `(B, T, C)` at entry points. 8 models handle internal shape conversion correctly. 2 models (MCL/FCN, TSTCC) crash — Conv1d receives `(B, T, C)` instead of `(B, C, T)`. Fix: "encoder owns the transpose" (Option D) — add `transpose(1,2)` as first line in Conv1d-based encoders. Document `(B, T, C)` as library-wide convention in `contributing.md`.
   - **Spec:** `.planning/todos/specs/conv1d-shape-transpose-fix-spec.md`

3. **GPU/MPS Device Hardening** — Crash fixes (Series2Vec `lfilter` via `np.asarray()` on MPS), tensor device mismatches (AutoTCL losses `torch.zeros` without device), and defensive convention: `.cpu().numpy()` before all host-side ops, `.to(device)` after, create tensors on `x.device` not CPU. Include lint guard and smoke test.
   - **Spec:** `.planning/todos/specs/gpu-mps-device-hardening.md`

Plans:

- [ ] 09-01-PLAN.md — Conv1D encoder transpose fix + augmentation axes + convention doc (D-01)
- [ ] 09-02-PLAN.md — Encoding mixin refactor (4-hook → 2-hook) + gradient_enabled (D-02, D-03)
- [ ] 09-03-PLAN.md — GPU/MPS device hardening + lint guard + smoke tests (D-04)

---

*Last updated: 2026-06-24 — Phase 9 plans finalized*
