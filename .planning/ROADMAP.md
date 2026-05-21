# Roadmap: tsmodels

**Created:** 2026-05-21
**Mode:** yolo
**Granularity:** fine

---

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Foundation | Split mixin hierarchy + add config layer | MIX-01..04, CFG-01, CFG-03 | 4 |
| 2 | Directory Restructure | Move to cnn/dilated layout, collocate encoders/layers/augmentations | RESTRUCT-01..04 | 4 |
| 3 | Augmentation Refactor | Replace enums/factories with ABC strategy system, collocate per-model | AUG-01..06, CFG-02, MOD-01..05 | 5 |
| 4 | Cleanup and Verification | Remove dead code, polish imports, smoke tests | CLN-01..04, VER-01..05 | 5 |

---

### Phase 1: Foundation

**Goal:** Split encoding mixin into hierarchy and add config dataclasses

**Requirements:** MIX-01, MIX-02, MIX-03, MIX-04, CFG-01, CFG-03

**Plans:** 4 plans

**Success Criteria:**
1. TS2Vec and AutoTCL inherit `PoolingEncodingMixin`; CoST inherits `DecompositionEncodingMixin`
2. `encode()` behavior identical to pre-refactor for all 3 models (no regression)
3. `TS2VecModelParameters`, `CoSTModelParameters`, `AutoTCLModelParameters` dataclasses exist with `from_config()` classmethod
4. `ty check src/` passes with zero errors

**Canonical refs:** `_sources/rbspaper/src/rbspaper/models/abstract/encoding_functionality_mixin.py`, `_sources/rbspaper/src/rbspaper/models/config.py`

**Plan list:**
- [ ] 01-01-PLAN.md — Config dataclasses (ModelParameters hierarchy, D-03, D-07)
- [ ] 01-02-PLAN.md — Mixin hierarchy (Base/Pooling/Decomposition, D-05, D-06)
- [ ] 01-03-PLAN.md — Model inheritance + from_config() factory (D-04)
- [ ] 01-04-PLAN.md — Type checking + lint + test verification

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

---

### Phase 4: Cleanup and Verification

**Goal:** Remove dead code, polish public API, verify all models train correctly

**Requirements:** CLN-01, CLN-02, CLN-03, CLN-04, VER-01, VER-02, VER-03, VER-04, VER-05

**Success Criteria:**
1. `enums.py`, `factories.py`, `_sources/rbspaper/` deleted; no import errors remain
2. Public imports work: `from tscollection.models import TS2Vec`, `from tscollection.cnn.dilated.ts2vec.augmentation import CropShiftAugmentation`
3. Smoke tests pass: TS2Vec, CoST, AutoTCL each train 5 steps with finite loss
4. Checkpoint round-trip: save → reload → encoder weights match for all 3 models
5. `ty check src/` and `ruff format --check src/` pass with zero errors

---

*Last updated: 2026-05-21 after plan-phase*
