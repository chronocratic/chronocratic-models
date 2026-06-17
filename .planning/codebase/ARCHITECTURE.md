<!-- refreshed: 2026-06-17 -->
# Architecture

**Analysis Date:** 2026-06-17

## System Overview

Chronocratic-models is a **model library** (not an application). It provides ready-to-use self-supervised and supervised time-series models as `LightningModule` subclasses, each encapsulating encoder, training loop, and inference. The architecture follows a **composable mixin pattern** — models inherit from `LightningModule` plus encoding mixins, and inject augmentation producers rather than hard-coding strategies.

```text
+---------------------------------------------------------------+
|                   Public API Layer                             |
|  chronocratic.models (barrel imports all model classes)        |
+----------------------------+----------------------------------+
                             |
+----------------------------+----------------------------------+
|                    Model Families                              |
|  transformer/tst     convolutional/standard    convolutional/  |
|  (TST)               (TSTCC, Series2Vec, MCL)  dilated/        |
|                                          (TS2Vec, AutoTCL, CoST)|
|  recurrent/timenet   generative/timevae                              |
|  (TimeNet)           (TimeVAE)                                       |
+----------------------------+----------------------------------+
                             |
+----------------------------+----------------------------------+
|               Shared Infrastructure                             |
|  _mixin/encoding      augmentation/      supervised/            |
|  (BasicEncodingMixin) (producer contracts) (downstream wrapper) |
+----------------------------+----------------------------------+
                             |
+----------------------------+----------------------------------+
|                 Common Components                               |
|  layers/           losses/           distances/         utils/  |
|  (VAE building)    (contrastive)     (Soft-DTW)         (pooling)|
+---------------------------------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **TST** | Transformer masked-reconstruction pretraining | `src/chronocratic/models/transformer/tst/model.py` |
| **TS2Vec** | Dilated convolutional hierarchical contrastive pretraining | `src/chronocratic/models/convolutional/dilated/ts2vec/model.py` |
| **CoST** | Decomposition-based (trend+seasonality) contrastive pretraining | `src/chronocratic/models/convolutional/dilated/cost/model.py` |
| **AutoTCL** | Trainable augmentation neural network with InfoNCE | `src/chronocratic/models/convolutional/dilated/autotcl/model.py` |
| **MCL (FCN)** | Multi-scale contrastive learning with standard convolutions | `src/chronocratic/models/convolutional/standard/mcl/model.py` |
| **Series2Vec** | Series alignment via temporal + frequency distance matrices | `src/chronocratic/models/convolutional/standard/series2vec/model.py` |
| **TSTCC** | Temporal-contextual contrastive pretraining with dual views | `src/chronocratic/models/convolutional/standard/tstcc/model.py` |
| **TimeNet** | GRU-based sequence reconstruction | `src/chronocratic/models/recurrent/timenet/model.py` |
| **TimeVAE** | Variational autoencoder with STL decomposition layers | `src/chronocratic/models/generative/timevae/model.py` |
| **SupervisedModule** | Generic backbone+head wrapper for classification/regression | `src/chronocratic/models/supervised/supervised.py` |
| **Augmentation Protocol** | Structural protocol for primitive transforms | `src/chronocratic/models/augmentation/base.py` |
| **AugmentationProducer[V]** | Covariant protocol for typed view-set production | `src/chronocratic/models/augmentation/base.py` |
| **SingleView / ViewPair / AlignedPair** | Frozen view-set dataclasses (Layer 2) | `src/chronocratic/models/augmentation/base.py` |
| **TrainableAugmentationProducer** | Nominal ABC for learnable augmentation (nn.Module) | `src/chronocratic/models/augmentation/base.py` |
| **Augmentation primitives** | Model-agnostic transforms (Jitter, Scaling, Permutation) | `src/chronocratic/models/augmentation/primitives.py` |
| **Augmentation producers** | Generic combinators (SingleViewProducer, RolePair, etc.) | `src/chronocratic/models/augmentation/producers.py` |
| **BasicEncodingMixin** | Uniform `encode()` API for fixed-length sequence models | `src/chronocratic/models/_mixin/encoding.py` |
| **BaseEncodingMixin** | Sliding-window + pooling `encode()` for dilated models | `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` |
| **PoolingEncodingMixin** | Pooling strategies (full_series, multiscale, integer) | `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` |
| **DecompositionEncodingMixin** | Trend+seasonality concatenation evaluation | `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` |
| **MaskMode** | Enum + factory for encoder masking strategies | `src/chronocratic/models/convolutional/dilated/encoders/masking.py` |
| **RepresentationBackbone** | Protocol for `representation_dim` property | `src/chronocratic/models/supervised/supervised.py` |
| **BatchAdapter** | Protocol for decoding model-specific batch tuples | `src/chronocratic/models/supervised/supervised.py` |
| **BackboneUnfreeze** | Gradual-unfreeze Lightning callback (discriminative LR) | `src/chronocratic/models/supervised/_callbacks.py` |
| **\*ModelParameters** | Per-model config dataclasses (kw_only) | Various `*/config.py` files |

## Pattern Overview

**Overall:** Plugin architecture using Strategy, Protocol, Mixin, and Factory patterns over a common `LightningModule` base.

**Key Characteristics:**
- Every model is a `lightning.pytorch.LightningModule` for consistent training/validation loops and checkpointing
- Augmentations use structural protocols (`AugmentationProducer[V]`) — no inheritance required. Models inject producers at construction time
- Encoding/inference behavior is composed via Mixins (`BasicEncodingMixin` for simple models, `BaseEncodingMixin` + `PoolingEncodingMixin`/`DecompositionEncodingMixin` for dilated models)
- Supervised downstream tasks use a single `SupervisedModule` with injected collaborators (Factory pattern)
- Configuration is type-safe via `@dataclass(kw_only=True)` per model
- Protocols (`BatchAdapter`, `RepresentationBackbone`) define duck-typed contracts without nominal inheritance
- Covariant type parameters enable Liskov substitution: `AugmentationProducer[AlignedPair]` satisfies `AugmentationProducer[ViewPair]`

## Layers

### Model Family Layer (Pretraining)

- Purpose: Self-supervised representation learning for time series
- Location: `src/chronocratic/models/{convolutional,transformer,recurrent,generative}/`
- Contains: `LightningModule` subclasses implementing `training_step`, `validation_step`, `configure_optimizers`
- Depends on: Encoding mixins, augmentation protocols, shared layers, loss functions, pooling utilities
- Used by: Supervised layer, external training scripts

**Standard convolutional models** (`convolutional/standard/`): TSTCC, Series2Vec, MCL/FCN — use `BasicEncodingMixin`

**Dilated convolutional models** (`convolutional/dilated/`): TS2Vec, AutoTCL, CoST — use `BaseEncodingMixin` hierarchy with sliding-window inference

**Transformer** (`transformer/tst/`): TST — uses `BasicEncodingMixin` with masked reconstruction

**Recurrent** (`recurrent/timenet/`): TimeNet — uses `BasicEncodingMixin`

**Generative** (`generative/timevae/`): TimeVAE — inherits `BaseVariationalAutoencoder` + `BasicEncodingMixin`

### Supervised Layer (Downstream)

- Purpose: Wrap pretrained backbones for classification/regression fine-tuning
- Location: `src/chronocratic/models/supervised/`
- Contains: `SupervisedModule`, factory constructors, batch adapters, representation functions, `BackboneUnfreeze` callback
- Depends on: Model layer backbones exposing `representation_dim`, `_utils` for loss functions
- Used by: External training scripts

Four modes via **configuration, not subclasses**:
1. Linear probe (pretrained backbone, `freeze_backbone=True`)
2. Full fine-tune (pretrained backbone, `freeze_backbone=False`)
3. Gradual unfreeze (`BackboneUnfreeze` callback owns freeze state)
4. Supervised from scratch (fresh backbone, `freeze_backbone=False`)

### Augmentation Layer (3-level architecture)

- Purpose: Decouple data augmentation from model logic; enable strategy swapping at runtime
- Location: `src/chronocratic/models/augmentation/` (framework), `*/augmentation/` (per-model concrete)
- **Level 1 — Primitives:** `Augmentation` Protocol (stateless tensor transforms)
- **Level 2 — ViewSets:** Frozen dataclasses (`SingleView`, `ViewPair`, `AlignedPair`)
- **Level 3 — Producers:** `AugmentationProducer[V]` Protocol (assembles primitives into typed view sets)
- **Capability — Trainable:** `TrainableAugmentationProducer` (nominal ABC + `nn.Module`)
- Generic combinators in `producers.py` are model-agnostic; concrete implementations live per-model

### Encoding Mixin Layer

- Purpose: Provide uniform `encode(data, batch_size, ...)` inference API across all models
- Location: `src/chronocratic/models/_mixin/` (basic), `src/chronocratic/models/convolutional/dilated/_mixin/` (dilated)
- Template Method pattern (`BasicEncodingMixin`): `_get_encoder`, `_prepare_inputs`, `_postprocess` are hooks
- Strategy pattern (`BaseEncodingMixin`): `_get_eval_method` returns pooling or decomposition evaluation
- Depends on: Shared utilities in `utils.py`, PyTorch DataLoader/TensorDataset
- Used by: Model layer via multiple inheritance with `LightningModule`

### Shared Infrastructure Layer

- Purpose: Reusable building blocks not tied to a single model family
- Location: `src/chronocratic/models/{layers,losses,distances,utils}`
- Contains: `BandedFourierLayer`, `TrendLayer`, `SeasonalLayer`, `LevelModel`, `ResidualConnection`, `SoftDTW`, contrastive losses, pooling functions
- Depends on: PyTorch, einops, numpy, scipy
- Used by: Model layer, augmentation layer, encoding mixins

## Data Flow

### Pretraining — Contrastive Models (TS2Vec, AutoTCL, CoST, TSTCC)

1. **Data input** — Raw batch from DataLoader
2. **Feature extraction** — `extract_features_from_batch(batch)` (`src/chronocratic/models/utils.py:19`)
3. **Augmentation** — `self._augmentation.produce(x)` returns typed `ViewSet` (`SingleView`, `ViewPair`, or `AlignedPair`)
4. **Encoding** — Views passed through `self._encoder(x)` producing embeddings
5. **Loss computation** — Model-specific contrastive loss between embeddings
6. **Backward pass** — `manual_backward(loss)` + `optimizer.step()` (manual optimization for these models)
7. **EMA update** — `self._averaged_encoder.update_parameters(self._encoder)` (dilated models)

**TS2Vec example:**
```
Batch -> extract_features -> CropShiftProducer.produce -> AlignedPair -> encoder(pair.first), encoder(pair.second) -> slice by overlap_length -> hierarchical_contrastive_loss
```

**TSTCC example:**
```
Batch -> extract_features -> RolePair.produce -> ViewPair (weak/strong) -> encoder(aug1), encoder(aug2) -> NTXentLoss + TemporalContrast
```

### Pretraining — Masked Reconstruction (TST)

1. **Data input** — Batch: `(X, targets, target_masks, padding_masks, IDs)`
2. **Reconstruction** — `self.reconstruct(x, padding_masks)` runs full transformer
3. **Loss** — `MaskedMSELoss` on masked positions (`transformer/tst/loss.py`)
4. **Backward pass** — Automatic optimization (Lightning default)

### Pretraining — VAE (TimeVAE)

1. **Data input** — Raw tensor batch
2. **Encode** — `self.encoder(x)` -> `(z_mean, z_log_var, z)` (reparameterization via `Sampling`)
3. **Decode** — `self.decoder(z)` reconstructs input
4. **Loss** — ELBO via `BaseVariationalAutoencoder` (`generative/timevae/vae_base.py`)

### Downstream Supervised Training

1. **Factory** — `make_tst_supervised(backbone, num_outputs=N)` (`supervised/factory.py:42`)
2. **Batch decode** — `batch_adapter(batch)` -> `((encoder_inputs, ...), targets)` via `BatchAdapter` Protocol
3. **Representation** — `representation_fn(backbone, *encoder_inputs)` -> `(B, rep_dim)`
4. **Head** — `FlattenLinearHead(reps)` -> `(B, num_outputs)` logits (`supervised/supervised.py:74`)
5. **Loss** — `classification_loss` or `regression_loss` (`supervised/_utils.py`)
6. **Optimizer** — Adam over trainable params; frozen backbone params excluded via generator expression

### Inference (Encoding)

**Fixed-length models (TST, Series2Vec, TimeNet, TimeVAE, TSTCC):**
1. `model.encode(data, batch_size=64)` via `BasicEncodingMixin`
2. DataLoader iteration under `torch.inference_mode()`
3. `_prepare_inputs(batch)` -> `_get_encoder()(*args)` -> `_postprocess(output)`
4. Concatenate results on dim 0, return to original device

**Dilated models (TS2Vec, AutoTCL, CoST):**
1. `model.encode(data, batch_size=64, encoding_window='multiscale', mask='binomial')` via `BaseEncodingMixin`
2. Optional sliding-window inference for long sequences
3. Per-window encoding with pooling or decomposition strategy
4. Concatenate window representations across time dimension

**State Management:**
- Dilated models maintain an `AveragedModel` (EMA) alongside primary encoder; training uses `_encoder`, validation/inference uses `_averaged_encoder`
- `SupervisedModule` freezes backbone when `freeze_backbone=True`; gradual unfreeze via `BackboneUnfreeze` callback
- Mixins preserve encoder's prior training/eval state across `encode()` calls

## Key Abstractions

### AugmentationProducer[V] (Structural Protocol)

- Location: `src/chronocratic/models/augmentation/base.py`
- Covariant type parameter `V` (returns `SingleView`, `ViewPair`, or `AlignedPair`)
- Structural protocol — no inheritance required; any class with `produce(x: Tensor) -> V` satisfies it
- Generic producers in `augmentation/producers.py` are model-agnostic combinators
- Model-specific producers handle domain logic (e.g. `CropShiftProducer` for TS2Vec)
- Replaces the previous `AugmentationMethod` ABC-based design with Protocol-based structural typing

### ViewSet Hierarchy (Frozen Dataclasses)

- `SingleView` — one augmented view
- `ViewPair` — two views (first, second)
- `AlignedPair(ViewPair)` — two views with known `overlap_length` for temporal alignment
- Liskov substitution: `AlignedPair` satisfies any `ViewPair` slot via covariance
- Covariant `V` in `AugmentationProducer[V]` enables this substitution at the type level

### TrainableAugmentationProducer (Nominal ABC)

- Location: `src/chronocratic/models/augmentation/base.py`
- Combines `nn.Module` lifecycle (parameters, state_dict) with `AugmentationTrainingStrategy`
- Named (nominal) ABC — must be runtime-checkable via `isinstance()` to gate the trainable path
- Structurally satisfies `AugmentationProducer[SingleView]`

### BasicEncodingMixin (Template Method ABC)

- Location: `src/chronocratic/models/_mixin/encoding.py`
- Abstract hooks: `_get_encoder()` (required), `_get_encoder_module()`, `_prepare_inputs()`, `_postprocess()`
- Manages DataLoader iteration, eval/inference mode, device placement, and result concatenation
- Designed to be mixed into `LightningModule` via multiple inheritance

### BaseEncodingMixin Hierarchy (Strategy + Template Method ABCs)

- Location: `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py`
- Base provides `encode()` with sliding-window support
- `PoolingEncodingMixin` — strategy returns `_evaluate_with_pooling` (multi-scale/integer/full-series pooling)
- `DecompositionEncodingMixin` — strategy returns `_evaluate_with_feature_concatenation` (trend+seasonality)

### SupervisedModule (Composition)

- Location: `src/chronocratic/models/supervised/supervised.py`
- Injects: backbone, head, representation_fn, batch_adapter, loss_fn
- Four modes via configuration, not subclasses: linear probe, full fine-tune, gradual unfreeze, supervised from scratch
- `freeze_backbone` flag controls parameter freezing at construction

### RepresentationBackbone (Runtime-Checkable Protocol)

- Location: `src/chronocratic/models/supervised/supervised.py`
- Protocol with `representation_dim` property
- Enables `FlattenLinearHead` to auto-size its input dimension from any backbone

### BatchAdapter (Structural Protocol)

- Location: `src/chronocratic/models/supervised/supervised.py`
- Callable protocol: `__call__(batch) -> ((encoder_inputs,), targets)`
- Normalizes model-specific batch formats so `SupervisedModule` stays model-agnostic

## Entry Points

### Library Public API

- Location: `src/chronocratic/models/__init__.py`
- Re-exports all model classes (`TST`, `TS2Vec`, `CoST`, `AutoTCL`, `FCN`, `Series2Vec`, `TSTCC`, `TimeNet`, `TimeVAE`) and their config dataclasses
- Single import point: `from chronocratic.models import TST, TSTModelParameters, ...`

### Supervised Training API

- Location: `src/chronocratic/models/supervised/__init__.py`
- Re-exports `SupervisedModule`, `FlattenLinearHead`, `BackboneUnfreeze`, factory functions, and adapter protocols

### Augmentation Framework API

- Location: `src/chronocratic/models/augmentation/__init__.py`
- Re-exports protocols, view-sets, generic producers, primitives, decorators, and trainable support

## Architectural Constraints

- **Threading:** Single-threaded PyTorch execution. DataLoader workers controlled per model.
- **Global state:** Module-level `np.random.Generator` in `convolutional/dilated/encoders/masking.py` for mask generation. No other module-level mutable state.
- **Device management:** Models use `self.device` (from `LightningModule`) for tensor placement. Encoding mixins preserve and restore original device of input data.
- **Manual optimization:** TS2Vec, AutoTCL, TSTCC, and CoST use `automatic_optimization = False` due to multi-optimizer or custom training steps. TST, Series2Vec, TimeNet, and TimeVAE use Lightning automatic optimization.
- **Circular imports:** Avoided via lazy imports — default augmentation classes are imported inside `__init__` methods only when `augmentation is None`. Type hints use `if TYPE_CHECKING:` guards.
- **Python version:** Requires Python 3.12+ (PEP 695 type parameter syntax used for covariant `AugmentationProducer[V]`).

## Anti-Patterns

### Encoding mixin duplication

**What happens:** Two separate encoding mixin hierarchies exist (`_mixin/` for standard models and `convolutional/dilated/_mixin/` for dilated models) with overlapping intent but different APIs.
**Why it's wrong:** New contributors face confusion about which mixin to use. The `encode()` signatures differ (basic takes `data, batch_size, num_workers`; dilated adds `mask`, `encoding_window`, `causal`, `sliding_length`, `sliding_padding`).
**Do this instead:** When adding new models, check if `BasicEncodingMixin` suffices. Only use dilated mixins when sliding-window or multi-scale pooling is required.

### Model-specific augmentation in model files

**What happens:** Models like TSTCC and TS2Vec create default augmentation producers inline when no augmentation is passed to `__init__`.
**Why it's wrong:** Couples model construction to specific augmentation logic. Makes it harder to swap augmentations for experiments.
**Do this instead:** Prefer always injecting an `AugmentationProducer` explicitly. The lazy import pattern (`if augmentation is None`) is acceptable for convenience but should not be the primary path in test/research code.

### Previous ABC-based augmentation design (resolved)

**What happened:** The old design used nominal `AugmentationMethod` and `DualAugmentation` ABCs with `.augment()` returning `TrainingViews`.
**Why it was problematic:** Tight coupling via inheritance; difficult to compose or swap strategies; model-specific augmentations lived scattered across the codebase.
**Current approach:** Protocol-based `AugmentationProducer[V]` with typed view-sets (`SingleView`, `ViewPair`, `AlignedPair`) and generic combinators. Concrete implementations still live per-model but are structurally typed.

## Error Handling

**Strategy:** Fail-fast with descriptive exceptions; no retry or fallback logic.

**Patterns:**
- Constructor validation raises `ValueError` for invalid parameter combinations (e.g., `encoding_window` not supported by decomposition models in `convolutional/dilated/_mixin/encoding.py:368`)
- `@runtime_checkable` Protocols for duck typing (`BatchAdapter`, `RepresentationBackbone`, `Augmentation`) in `supervised/supervised.py` and `augmentation/base.py`
- Type assertions via `cast()` where static analysis cannot narrow types
- Gradient clipping in TST via `configure_gradient_clipping` with `max_norm=4.0` default
- `BasicEncodingMixin._get_encoder_module()` raises `NotImplementedError` when `_get_encoder()` returns a non-Module callable without override

## Cross-Cutting Concerns

**Logging:** Per-model `self.log()` calls with configurable `sync_dist` for distributed training
**Validation:** Dataclass field types provide static validation; runtime checks in constructors
**Checkpointing:** `self.save_hyperparameters()` on every LightningModule; augmentation objects excluded via `ignore=['augmentation']`

---

*Architecture analysis: 2026-06-17*
