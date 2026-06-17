# Codebase Structure

**Analysis Date:** 2026-06-17

## Directory Layout

```
tsmodels/
├── pyproject.toml              # Project metadata, deps, tool config (setuptools, pytest, ruff, towncrier)
├── CHANGELOG.md                # Version history (towncrier-managed)
├── README.md                   # Project description
├── LICENSE                     # BSD-3-Clause
├── changelog.d/                # Towncrier fragment directory
├── docs/                       # Sphinx documentation source
│   ├── _build/                 # Generated HTML docs
│   ├── api/                    # API reference pages
│   └── _static/                # Static assets
├── graphify-out/               # Knowledge graph (AST-based indexing)
├── src/
│   └── chronocratic/
│       └── models/             # Main package (src layout)
│           ├── __init__.py     # Public API barrel (re-exports all models + configs)
│           ├── _version.py     # Auto-generated version (setuptools_scm)
│           ├── _mixin/         # Shared encoding mixin hierarchy
│           ├── augmentation/   # Augmentation framework (protocols, producers, primitives)
│           ├── convolutional/  # Convolutional model family
│           │   ├── dilated/    #   Dilated conv models (TS2Vec, AutoTCL, CoST)
│           │   └── standard/   #   Standard conv models (TSTCC, Series2Vec, MCL)
│           ├── distances/      # Distance metrics (Soft-DTW)
│           ├── generative/     # Generative model family (TimeVAE)
│           ├── layers/         # Shared neural layer building blocks
│           ├── losses/         # Shared loss functions
│           ├── recurrent/      # Recurrent model family (TimeNet)
│           ├── supervised/     # Downstream supervised training framework
│           ├── transformer/    # Transformer model family (TST)
│           └── utils.py        # Shared utilities (pooling, batch extraction, sliding windows)
├── tests/                      # Test suite (pytest)
│   ├── conftest.py             # Shared fixtures
│   ├── test_*.py               # Top-level unit tests
│   ├── integration/            # Integration tests
│   └── unit/                   # Organized unit tests
├── .claude/                    # Claude project config
├── .agents/                    # Agent skill definitions
└── .github/workflows/          # CI configuration
```

## Directory Purposes

### `src/chronocratic/models/`

**Purpose:** Main package root. All public symbols are re-exported from `__init__.py`.

**Structure:** Flat barrel at the top level delegates to sub-package barrels:
- `from .transformer import TST, TSTModelParameters`
- `from .convolutional import TS2Vec, TSTCC, Series2Vec, ...`
- `from .generative import TimeVAE, TimeVAEModelParameters`
- `from .recurrent import TimeNet, TimeNetModelParameters`

**Package name changed:** Previously `tscollection`; now `chronocratic`. All internal imports updated.

### `src/chronocratic/models/_mixin/`

**Purpose:** Shared encoding mixin hierarchy for standard (non-dilated) models.

**Contains:**
- `encoding.py` — `BasicEncodingMixin` (ABC with template method pattern)

**Key files:**
- `_mixin/encoding.py`: `BasicEncodingMixin` with hooks `_get_encoder`, `_get_encoder_module`, `_prepare_inputs`, `_postprocess`

### `src/chronocratic/models/augmentation/`

**Purpose:** Model-agnostic augmentation framework. Three-layer architecture.

**Contains:**
- `base.py` — Protocols (`Augmentation`, `AugmentationProducer[V]`, `Reseedable`), frozen dataclasses (`SingleView`, `ViewPair`, `AlignedPair`), ABCs (`AugmentationTrainingStrategy`, `TrainableAugmentationProducer`)
- `producers.py` — Generic combinators (`SingleViewProducer`, `IndependentPair`, `RolePair`, `FullOverlapPair`)
- `primitives.py` — Model-agnostic transforms (`Jitter`, `Scaling`, `Permutation`, `ComposeAugmentation`) with typed config dataclasses
- `decorators.py` — `@Seeded` decorator for RNG determinism
- `trainable_support.py` — `maybe_train_augmentation`, `maybe_configure_augmentation_optimizer`

**Redesigned from previous version:** Replaced nominal `AugmentationMethod`/`DualAugmentation` ABCs with structural `AugmentationProducer[V]` Protocol and typed view-set dataclasses.

### `src/chronocratic/models/convolutional/`

**Purpose:** Barrel for all convolutional models (delegates to `standard/` and `dilated/`).

#### `convolutional/standard/`

**Purpose:** Standard convolution models with fixed-length sequence encoding (`BasicEncodingMixin`).

**Subpackages (one per model):**
- `tstcc/` — TS-TCC (temporal contextual contrastive learning)
  - `config.py`: `TSTCCModelParameters`
  - `model.py`: `TSTCC` (LightningModule + BasicEncodingMixin)
  - `encoder.py`: `TCCEncoder` (conv backbone + logits head)
  - `losses.py`: `NTXentLoss`
  - `temporal_contrast.py`: `TemporalContrast`
  - `augmentations.py`: Model-specific producer (`_default_tstcc_pair`)
- `series2vec/` — Series2Vec
  - `config.py`: `Series2VecModelParameters`
  - `model.py`: `Series2Vec` (LightningModule + BasicEncodingMixin)
  - `encoder.py`: Series2Vec encoder
  - `network.py`: `Series2VecNetwork` (temporal + frequency branches)
  - `filters.py`: Signal processing filters
  - `losses.py`: Model-specific losses
- `mcl/` — MCL/FCN
  - `config.py`: `MCLModelParameters`
  - `model.py`: `FCN` (LightningModule + BasicEncodingMixin)
  - `encoder.py`: FCN encoder
  - `losses.py`: Model-specific losses

#### `convolutional/dilated/`

**Purpose:** Dilated convolution models with sliding-window inference and multi-scale pooling.

**Shared components:**
- `layers/` — Dilated conv layers (`dilated.py`, `same_pad.py`)
- `encoders/` — Time series encoder building blocks (`encoders.py`, `masking.py`)
- `_mixin/` — Encoding mixin hierarchy:
  - `encoding.py`: `BaseEncodingMixin`, `PoolingEncodingMixin`, `DecompositionEncodingMixin`

**Model subpackages:**
- `ts2vec/` — TS2Vec
  - `config.py`: `TS2VecModelParameters`
  - `model.py`: `TS2Vec` (LightningModule + PoolingEncodingMixin)
  - `augmentation.py`: `CropShiftProducer`
  - `utils.py`: TS2Vec-specific helpers
- `autotcl/` — AutoTCL
  - `config.py`: `AutoTCLModelParameters`
  - `model.py`: `AutoTCL` (LightningModule + PoolingEncodingMixin)
  - `losses.py`: AutoTCL losses
  - `utils.py`: Helpers
  - `augmentation/`: Neural augmentation network + training strategies
- `cost/` — CoST (trend-seasonality decomposition)
  - `config.py`: `CoSTModelParameters`
  - `model.py`: `CoST` (LightningModule + DecompositionEncodingMixin)
  - `augmentation.py`: CoST-specific augmentation
  - `utils.py`: Helpers

### `src/chronocratic/models/transformer/`

**Purpose:** Transformer-based models. Currently only TST.

**Contains:**
- `tst/` — Time Series Transformer
  - `config.py`: `TSTModelParameters`
  - `model.py`: `TST` (LightningModule + BasicEncodingMixin)
  - `ts_transformer.py`: `TSTransformerEncoder` (pure nn.Module)
  - `loss.py`: `MaskedMSELoss`

### `src/chronocratic/models/recurrent/`

**Purpose:** Recurrent (RNN/GRU-based) models.

**Contains:**
- `timenet/` — TimeNet (GRU encoder-decoder)
  - `config.py`: `TimeNetModelParameters`
  - `model.py`: `TimeNet` (LightningModule + BasicEncodingMixin), `GRUWrapper`

### `src/chronocratic/models/generative/`

**Purpose:** Generative models (VAE-based).

**Contains:**
- `timevae/` — TimeVAE
  - `config.py`: `TimeVAEModelParameters`
  - `model.py`: `TimeVAE`, `TimeVAEEncoder`, `TimeVAEDecoder`
  - `vae_base.py`: `BaseVariationalAutoencoder` (ABC), `Sampling`

### `src/chronocratic/models/supervised/`

**Purpose:** Downstream supervised training framework (classification/regression).

**Contains:**
- `supervised.py`: `SupervisedModule`, `FlattenLinearHead`, `RepresentationBackbone`, `BatchAdapter`
- `factory.py`: `make_tst_supervised`, `make_series2vec_supervised`, `make_tstcc_supervised`
- `_adapters.py`: Batch adapters and representation functions per backbone
- `_callbacks.py`: `BackboneUnfreeze` (gradual unfreeze callback)
- `_utils.py`: `classification_loss`, `regression_loss`

### `src/chronocratic/models/losses/`

**Purpose:** Shared contrastive loss functions used across models. Moved from flat `losses.py` to package.

**Contains:**
- `contrastive.py`: `hierarchical_contrastive_loss`, `instance_contrastive_loss`, `temporal_contrastive_loss`

### `src/chronocratic/models/layers/`

**Purpose:** Shared neural layer building blocks (VAE-specific).

**Contains:**
- `general.py`: `BandedFourierLayer`, `LevelModel`, `ResidualConnection`, `SeasonalLayer`, `TrendLayer`

### `src/chronocratic/models/distances/`

**Purpose:** Distance metric implementations.

**Contains:**
- `soft_dtw/soft_dtw_cuda.py`: `SoftDTW` (CUDA-enabled soft dynamic time warping)

### `tests/`

**Purpose:** Pytest test suite.

**Structure:**
- `conftest.py`: Shared fixtures
- `test_*.py`: Top-level unit tests (augmentation, config, mixins, smoke tests, model-specific producer tests)
- `integration/`: Integration tests for supervised training
- `unit/`: Organized unit tests for supervised module per backbone

**Key test files:**
- `test_augmentation.py`: Augmentation framework tests
- `test_aug_contract.py`: Producer protocol contract tests
- `test_mixin.py`: Encoding mixin tests
- `test_config.py`, `test_config_hierarchy.py`: Config dataclass tests
- `test_from_config.py`: Model construction from config tests
- `test_smoke.py`: Model smoke tests
- `integration/test_supervised_integration.py`: End-to-end supervised training integration
- `unit/test_backbone_representation_dim.py`: Backbone dim correctness
- `unit/test_*_supervised.py`: Per-backbone supervised tests (Series2Vec, TST, TSTCC)

## Module Organization

### Model Package Pattern

Each model follows a consistent internal structure:
1. `__init__.py` — Re-exports model class + config dataclass
2. `config.py` — `*ModelParameters` kw-only dataclass
3. `model.py` — `LightningModule` + encoding mixin (if applicable)
4. `encoder.py` — Pure `nn.Module` encoder (optional, when encoder is complex)
5. `losses.py` — Model-specific loss functions (optional)
6. `augmentation.py` or `augmentation/` — Model-specific augmentation (optional)
7. `utils.py` — Model-specific helpers (optional)

### Import Patterns

**Public API:**
```python
from chronocratic.models import TST, TSTModelParameters
from chronocratic.models import TS2Vec, TS2VecModelParameters
```

**Supervised training:**
```python
from chronocratic.models.supervised import make_tst_supervised, SupervisedModule
```

**Augmentation:**
```python
from chronocratic.models.augmentation import RolePair, Jitter, Scaling
```

**Internal imports:**
- All internal modules use absolute imports: `from chronocratic.models._mixin import BasicEncodingMixin`
- Type-checking imports use `if TYPE_CHECKING:` guard to avoid circular deps
- Model-specific augmentations use lazy imports: `from .augmentation import ...` only when needed
- `from __future__ import annotations` used selectively (only when needed for forward references)

### Namespace Organization

- **`chronocratic.models`** — Root package with public API barrel
- **`chronocratic.models._mixin`** — Private infrastructure (leading underscore)
- **`chronocratic.models.augmentation`** — Public augmentation framework
- **`chronocratic.models.supervised`** — Public supervised training framework
- **`chronocratic.models.{family}`** — Model families (transformer, convolutional, etc.)
- **`chronocratic.models.{family}.{subfamily}.{model}`** — Specific model packages
- All `__all__` lists are explicit; no implicit symbol leakage

## Package Structure

**Build system:** setuptools + setuptools_scm (git-tag-based versioning)

**Package layout:** src layout (`src/chronocratic/models`)

**Version:** Dynamic via `setuptools_scm` -> written to `src/chronocratic/models/_version.py`

**Python requirement:** 3.12+

**Core dependencies:** numpy, scipy, lightning, torch, einops, numba, tqdm

## Test Structure

**Runner:** pytest (configured in `pyproject.toml`)

**Location:** `tests/` (top-level, not src-layout)

**Organization:**
- `conftest.py` — Shared fixtures
- `test_*.py` — Top-level tests (augmentation, config, mixins, smoke)
- `integration/` — Integration tests (`__init__.py` + `test_supervised_integration.py`)
- `unit/` — Unit tests (`__init__.py` + per-backbone supervised tests)

**Coverage tools:** pytest-cov

## Naming Conventions

**Files:**
- `snake_case` for all Python files: `model.py`, `config.py`, `losses.py`, `ts_transformer.py`
- Private modules use underscore prefix: `_mixin/`, `_adapters.py`, `_callbacks.py`, `_utils.py`
- Test files prefixed with `test_`: `test_config.py`, `test_augmentation.py`

**Classes:**
- `PascalCase` for all classes: `TST`, `TS2Vec`, `AugmentationProducer`, `SupervisedModule`
- Model config dataclasses: `*ModelParameters` suffix: `TSTModelParameters`, `TS2VecModelParameters`

**Functions:**
- `snake_case` for functions: `extract_features_from_batch`, `make_tst_supervised`, `hierarchical_contrastive_loss`

**Directories:**
- `snake_case` for all directories: `convolutional/`, `transformer/`, `supervised/`
- Private internal directories use underscore prefix: `_mixin/`

## Where to Add New Code

### New Model (existing family):
- Implementation: `src/chronocratic/models/{family}/{subfamily}/{model_name}/model.py`
- Config: `src/chronocratic/models/{family}/{subfamily}/{model_name}/config.py`
- Encoder: `src/chronocratic/models/{family}/{subfamily}/{model_name}/encoder.py`
- Barrel: Add re-export to `src/chronocratic/models/{family}/{subfamily}/__init__.py`
- Public API: Add re-export to `src/chronocratic/models/__init__.py`
- Tests: `tests/test_{model_name}_producer.py` or `tests/unit/test_{model_name}_supervised.py`

### New Model (new family):
- Package: `src/chronocratic/models/{family_name}/` with `__init__.py`
- Model: `src/chronocratic/models/{family_name}/{model_name}/model.py`
- Barrel: `src/chronocratic/models/{family_name}/__init__.py`
- Public API: Add import to `src/chronocratic/models/__init__.py`

### New shared component:
- Layers: `src/chronocratic/models/layers/general.py`
- Losses: `src/chronocratic/models/losses/contrastive.py`
- Utilities: `src/chronocratic/models/utils.py`
- Augmentation primitives: `src/chronocratic/models/augmentation/primitives.py`
- Augmentation producers: `src/chronocratic/models/augmentation/producers.py`

### New supervised factory:
- Factory: `src/chronocratic/models/supervised/factory.py` (add `make_{model}_supervised`)
- Adapter: `src/chronocratic/models/supervised/_adapters.py` (add batch_adapter + representation_fn)
- Barrel: `src/chronocratic/models/supervised/__init__.py` (add to `__all__`)

### New encoding mixin:
- Standard models: Extend `BasicEncodingMixin` in `src/chronocratic/models/_mixin/encoding.py`
- Dilated models: Extend `BaseEncodingMixin` in `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py`

## Special Directories

**`graphify-out/`:**
- Purpose: Knowledge graph for codebase navigation
- Generated: Yes (AST-based, no API cost)
- Committed: Yes

**`changelog.d/`:**
- Purpose: Towncrier changelog fragments
- Generated: No (manually written per PR)
- Committed: Yes

**`docs/_build/`:**
- Purpose: Generated Sphinx HTML documentation
- Generated: Yes
- Committed: No (should be in .gitignore)

**`.planning/`:**
- Purpose: GSD planning documents and analysis
- Generated: Yes (by GSD commands)
- Committed: Yes

---

*Structure analysis: 2026-06-17*
