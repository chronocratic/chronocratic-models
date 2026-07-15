# Requirements: tsmodels

**Defined:** 2026-05-21
**Core Value:** Users can add new augmentation methods and training strategies by subclassing — zero library modification required

## v1 Requirements

### Mixin Architecture

- [ ] **MIX-01**: User inherits `PoolingEncodingMixin` for models that use pooling-based encoding
- [ ] **MIX-02**: User inherits `DecompositionEncodingMixin` for models that use decomposition-based encoding
- [ ] **MIX-03**: `BaseEncodingMixin` provides shared `encode()` entry point used by all model types
- [ ] **MIX-04**: Existing TS2Vec, CoST, and AutoTCL encoding behavior is preserved after mixin split

### Augmentation Interface

- [ ] **AUG-01**: User creates custom augmentation by subclassing `AugmentationMethod`
- [ ] **AUG-02**: User creates trainable augmentation by subclassing `TrainableAugmentation`
- [ ] **AUG-03**: User creates training strategy by subclassing `AugmentationTrainingStrategy`
- [ ] **AUG-04**: All pre-made augmentations live in `tscollection.augmentation` single import path
- [ ] **AUG-05**: Model accepts `augmentation: AugmentationMethod` in constructor (no enums, no factories)
- [ ] **AUG-06**: Model calls `augmentation.train_step()` and `augmentation.configure_optimizer()` polymorphically

### Model Refactors

- [ ] **MOD-01**: TS2Vec accepts `augmentation` parameter, drops `augmentation_mode` enum
- [ ] **MOD-02**: CoST accepts `augmentation` parameter, drops `augmentation_mode` enum
- [ ] **MOD-03**: AutoTCL drops all internal training dispatch methods (`_exec_training_step_function`, `_calculate_augmentation_loss_*`)
- [ ] **MOD-04**: AutoTCL delegates augmentation training to `TrainableAugmentation` + strategy
- [ ] **MOD-05**: All models use `save_hyperparameters(ignore=['augmentation'])`

### Config Layer

- [ ] **CFG-01**: Model parameters expressed as typed dataclasses (`TS2VecModelParameters`, `CoSTModelParameters`, `AutoTCLModelParameters`)
- [ ] **CFG-02**: Augmentation parameters expressed as typed dataclasses (`CropShiftAugmentationParameters`, etc.)
- [ ] **CFG-03**: Config dataclasses provide IDE autocompletion and type checking

### Cleanup

- [x] **CLN-01**: `enums.py` and `factories.py` removed from augmentation module
- [x] **CLN-02**: `_sources/rbspaper/` directory removed after all merges complete
- [x] **CLN-03**: Public module names use clean paths (no underscore prefixes on `abstract/`, `augmentation/`)
- [x] **CLN-04**: All imports resolve without warnings; `ty check` passes on full source

### Verification

- [x] **VER-01**: Smoke test: TS2Vec + CropShiftAugmentation trains 5 steps, loss finite
- [x] **VER-02**: Smoke test: CoST + CosTRandomFunctionAugmentation trains 5 steps, loss finite
- [x] **VER-03**: Smoke test: AutoTCL + AutoTCLNeuralNetworkAugmentation(RIPTrainingStrategy) trains 5 steps
- [x] **VER-04**: Extension test: trivial identity `AugmentationMethod` subclass works with any model
- [x] **VER-05**: Checkpoint round-trip: save → reload → encoder weights match

## v2 Requirements

### Model Extensibility

- **MIX-05**: User adds new model by subclassing `BaseEncodingMixin` + `LightningModule` without modifying existing code
- **AUG-06**: Cross-model augmentation reuse — same augmentation works with multiple model types

## Out of Scope

| Feature | Reason |
|---------|--------|
| Experiment runners | Library-only scope, users compose their own training loops |
| CLI tools | Not needed for library consumption |
| Data modules | Domain-specific, user responsibility |
| Evaluation pipelines | Outside core scope |
| Model registry | Direct imports are clearer for library users |
| Pretrained model hub | Infrastructure concern, not library concern |

### Directory Restructure

- [ ] **RESTRUCT-01**: All dilated CNN models moved to `cnn/dilated/{ts2vec,cost,autotcl}/`
- [ ] **RESTRUCT-02**: Encoders, layers, masking colocated at `cnn/dilated/encoders/`, `cnn/dilated/layers/`
- [ ] **RESTRUCT-03**: Model-specific augmentations colocated at `cnn/dilated/{model}/augmentation.py`
- [ ] **RESTRUCT-04**: Shared layers (`BandedFourierLayer`) remain at `models/layers/general.py`

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MIX-01 | Phase 1 | Pending |
| MIX-02 | Phase 1 | Pending |
| MIX-03 | Phase 1 | Pending |
| MIX-04 | Phase 1 | Pending |
| CFG-01 | Phase 1 | Pending |
| CFG-02 | Phase 3 | Pending |
| CFG-03 | Phase 1 | Pending |
| RESTRUCT-01 | Phase 2 | Pending |
| RESTRUCT-02 | Phase 2 | Pending |
| RESTRUCT-03 | Phase 2 | Pending |
| RESTRUCT-04 | Phase 2 | Pending |
| AUG-01 | Phase 3 | Pending |
| AUG-02 | Phase 3 | Pending |
| AUG-03 | Phase 3 | Pending |
| AUG-04 | Phase 3 | Pending |
| AUG-05 | Phase 3 | Pending |
| AUG-06 | Phase 3 | Pending |
| MOD-01 | Phase 3 | Pending |
| MOD-02 | Phase 3 | Pending |
| MOD-03 | Phase 3 | Pending |
| MOD-04 | Phase 3 | Pending |
| MOD-05 | Phase 3 | Pending |
| CLN-01 | Phase 4 | Complete |
| CLN-02 | Phase 4 | Complete |
| CLN-03 | Phase 4 | Complete |
| CLN-04 | Phase 4 | Complete |
| VER-01 | Phase 4 | Complete |
| VER-02 | Phase 4 | Complete |
| VER-03 | Phase 4 | Complete |
| VER-04 | Phase 4 | Complete |
| VER-05 | Phase 4 | Complete |

**Coverage:**

- v1 requirements: 26 total
- Plus RESTRUCT: 4 total (30 requirements)
- Mapped to phases: 30
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-21*
*Last updated: 2026-05-21 after initial definition*
