# Phase 1: Foundation - Context

**Gathered:** 2026-05-21
**Status:** Ready for planning

## Phase Boundary

Split encoding mixin into hierarchy (BaseEncodingMixin → Pooling, Decomposition) and add config dataclasses for type-safe model parameters. All 3 models inherit new mixins. `encode()` behavior identical to pre-refactor.

## Implementation Decisions

### Config Dataclass Scope
- **D-01:** Model-only configs in Phase 1. CFG-02 (aug params) moves to Phase 3.
- **D-02:** Strip runner artifacts: `model_name`, `set_input_dims`, `set_sequence_length` removed. Configs = type-safe containers only.
- **D-03:** Config hierarchy: `ModelParameters (ABC) → DilatedCNNModelParameters → TS2VecModelParameters, CoSTModelParameters, AutoTCLModelParameters`
- **D-04:** `from_config()` classmethod on each model. `__init__` stays flat for backward compat. Config = recommended new path.

### Mixin Adaptation
- **D-05:** Selective port from rbspaper — take polymorphism (`_get_encoder()`, `_get_eval_method()`, `_get_slice()`), sliding-window fixes, drop runner-specific guards (`encoder is None` check).
- **D-06:** TS2Vec + AutoTCL → `PoolingEncodingMixin`. CoST → `DecompositionEncodingMixin`. Both extend `BaseEncodingMixin (ABC)`.

### Module Organization
- **D-07:** Configs live at `src/tscollection/models/config.py` for now.
- **D-08:** Folder restructure (`cnn/dilated/` layout with colocated encoders, layers, augmentations) deferred to Phase 2. No file moves in Phase 1.
- **D-09:** Model-specific augmentations (CropShift, CoSTRandomFunction, AutoTCLNeuralNetwork) → colocated in each model folder (ts2vec/augmentation.py, cost/augmentation.py, autotcl/augmentation.py). Only `AugmentationMethod` ABC stays shared at `_augmentation/strategies.py`.

### Roadmap Changes
- **D-10:** Phase 2 = directory restructure (cnn/dilated layout). Phase 3 = augmentation refactor (was Phase 2). Phase 4 = cleanup and verification (was Phase 3).

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Requirements
- `.planning/ROADMAP.md` — Phase 1 goal, success criteria
- `.planning/REQUIREMENTS.md` — MIX-01..04, CFG-01, CFG-03
- `.planning/PROJECT.md` — library-only scope, key decisions, context

### Rbspaper Reference (read-only)
- `_sources/rbspaper/src/rbspaper/models/abstract/encoding_functionality_mixin.py` — split mixin to adapt
- `_sources/rbspaper/src/rbspaper/models/config.py` — model config dataclasses to adapt
- `_sources/rbspaper/src/rbspaper/models/augmentation/config.py` — aug config (deferred to Phase 3)

### Current Code
- `src/tscollection/models/_abstract/encoding_functionality_mixin.py` — current single-class mixin with hasattr branching
- `src/tscollection/models/ts2vec/model.py` — current TS2Vec constructor
- `src/tscollection/models/cost/model.py` — current CoST constructor
- `src/tscollection/models/autotcl/model.py` — current AutoTCL constructor
- `src/tscollection/models/_augmentation/strategies.py` — current AugmentationMethod ABC

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — component responsibilities, data flows
- `.planning/codebase/STRUCTURE.md` — directory layout, naming conventions
- `.planning/codebase/CONVENTIONS.md` — import order, error handling, docstring style

## Existing Code Insights

### Reusable Assets
- `AugmentationMethod` ABC (`_augmentation/strategies.py:39`) — base class, no changes needed
- Shared utils (`utils.py`) — `extract_features_from_batch`, `process_sample_length` — used by all models
- `AveragedModel` usage in TS2Vec, AutoTCL — encoders always set, no None check needed

### Established Patterns
- All modules declare `__all__` at top, use barrel `__init__.py` re-exports
- Error pattern: build `msg` variable, then `raise ValueError(msg)`
- LightningModule + mixin via multiple inheritance
- Manual optimization (`automatic_optimization = False`)
- `save_hyperparameters()` for checkpoint reproducibility

### Integration Points
- Mixin split touches 3 model files + 1 mixin file
- Config dataclasses need to map 1:1 to each model's current __init__ params
- `from_config()` reads dataclass, unpacks to __init__ kwargs

## Specific Ideas

- Config hierarchy mirrors mixin hierarchy (ModelParameters → DilatedCNNModelParameters → per-model)
- Folder structure targets `cnn/dilated/<model>/augmentation.py` for model-specific augmentations
- Shared layers (`BandedFourierLayer`) stays at `models/layers/general.py`
- Dilated-specific layers (`Conv1dDilatedEncoder`, `same_pad`) move into `cnn/dilated/layers/`

## Deferred Ideas

- CFG-02 (aug config dataclasses) → Phase 3
- Full directory restructure (cnn/dilated/) → Phase 2
- enums.py, factories.py deletion → Phase 4
- Smoke tests (VER-01..05) → Phase 4

---

*Phase: 1-Foundation*
*Context gathered: 2026-05-21*
