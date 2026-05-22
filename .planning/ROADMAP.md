# Roadmap: tsmodels

**Created:** 2026-05-21
**Mode:** yolo
**Granularity:** fine

---

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Foundation | 4/4 | Complete   | 2026-05-21 |
| 2 | Directory Restructure | Move to cnn/dilated layout, collocate encoders/layers/augmentations | RESTRUCT-01..04 | 5 |
| 3 | Augmentation Refactor | 3/3 | Complete   | 2026-05-22 |
| 4 | Cleanup and Verification | 2/8 | In Progress|  |

---

### Phase 1: Foundation

**Goal:** Split encoding mixin into hierarchy and add config dataclasses

**Requirements:** MIX-01, MIX-02, MIX-03, MIX-04, CFG-01, CFG-03

**Plans:** 4/4 plans complete

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

**Canonical refs:** `.planning/phases/01-foundation/01-CONTEXT.md` §D-08, D-09

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
1. Split `strategies.py` → `views.py` (TrainingViews), `base.py` (ABCs), `augmentations.py` (concrete classes + co-located params). Delete `config.py`.
2. Move `augmentations.py` contents into per-model directories:
   - `CropShiftAugmentation` + `CropShiftAugmentationParameters` → `ts2vec/augmentation.py`
   - `CosTRandomFunctionAugmentation` + `CosTRandomFunctionAugmentationParameters` → `cost/augmentation.py`
   - `AutoTCLNeuralNetworkAugmentation` + `AutoTCLNeuralNetworkAugmentationParameters` → `autotcl/augmentation.py`
   - `RIPTrainingStrategy`, `AdversarialTrainingStrategy` → `autotcl/training.py`
3. Update barrel `augmentation/__init__.py` to re-export from model directories for backward compat
4. Fix `RIPTrainingStrategy` and `AdversarialTrainingStrategy`: add `training_ratio_step` param, override `should_train()` with `epoch % ratio == 0`
5. Extract model losses: `cost/loss.py`, `autotcl/loss.py`, `ts2vec/loss.py`
6. Fix config inheritance for CoST

**Canonical refs:** Current branch `gsd/phase-03-augmentation-refactor`; main branch model files for original logic comparison.

**Plans:** 2/8 plans executed

**Plan list:**
- [x] 04-01-PLAN.md — Create augmentation/base.py with ABC hierarchy (D-01)
- [x] 04-02-PLAN.md — Create per-model config hierarchy + update from_config docstrings (D-08–D-13, D-18)
- [ ] 04-03-PLAN.md — Create per-model augmentation files (D-04–D-06)
- [ ] 04-04-PLAN.md — Create configs/ re-export package (D-14–D-15)
- [ ] 04-05-PLAN.md — Rewire barrel files and root config.py (D-02, D-08, D-16)
- [ ] 04-06-PLAN.md — Update test imports, fix stale cnn paths (CLN-04)
- [ ] 04-07-PLAN.md — Create smoke tests for VER-01 through VER-05
- [ ] 04-08-PLAN.md — Delete old strategies.py and config.py, full verification (CLN-01, CLN-04)

---

*Last updated: 2026-05-22*
