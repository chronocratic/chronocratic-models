# Design Discussion: Downstream Heads Architecture

## Context

PR #10 added 6 new models. Two of them (TST, Series2Vec) implement downstream classification/regression heads as **separate `pl.LightningModule` classes**. The dilated models (TS2Vec, CoST, AutoTCL) use **inline `nn.Sequential` projection heads**. TS-TCC uses a **hybrid** — training mode enum inside one module.

Three competing patterns in one codebase. Need to converge.

## Files to Read (8 files, ~600 lines total)

**Current implementations:**
1. `src/tscollection/models/transformer/tst/heads.py` — `_TSTHead(pl.LightningModule)` base + `TSTClassificationHead`/`TSTRegressionHead`. Reads backbone via `backbone.hparams['d_model']`.
2. `src/tscollection/models/convolutional/standard/series2vec/heads.py` — `Series2VecClassificationHead(pl.LightningModule)`. No base class. Duplicates training_step/val_step/optimizer from TST heads.
3. `src/tscollection/models/convolutional/dilated/cost/model.py:88-109` — dilated pattern: `query_projection_head = nn.Sequential(Linear, ReLU, Linear)`. Inline, no separate module.
4. `src/tscollection/models/convolutional/standard/tstcc/model.py:107-109` — TS-TCC pattern: freeze/unfreeze via `training_mode` enum, logits head inside encoder.

**Reference — encoding contract:**
5. `src/tscollection/models/_mixin/encoding.py` — `BasicEncodingMixin.encode()` produces representations the heads consume.

**Reference — model backbones:**
6. `src/tscollection/models/transformer/tst/model.py:102-122` — `TST.forward()` and `get_representations()`
7. `src/tscollection/models/convolutional/standard/series2vec/model.py:84-101` — `Series2Vec.forward()` and `_get_encoder()`
8. `src/tscollection/models/convolutional/standard/mcl/model.py:36-42` — `FCN` inline projection head

## Discussion Questions

### 1. What is the right abstraction level?

- **Option A (current TST):** Each head is a `pl.LightningModule`. Pro: plug into `Trainer.fit()` directly. Con: ~100 lines per head, duplicated training loop, nested LM fragility.
- **Option B (recommended):** Generic `FineTuningModule(pl.LightningModule)` wraps any backbone + plain `nn.Module` head + `encoding_fn`. One implementation serves all models. Heads become 10-line `nn.Module`s.
- **Option C (dilated pattern):** Inline `nn.Sequential` head inside the main model. Pro: zero boilerplate. Con: no linear-probe/fine-tune workflow, pretraining and downstream coupled.

### 2. Representation contract

`BasicEncodingMixin.encode()` returns tensors of varying shapes:
- Series2Vec: `(N, 1, 2*representation_dims)`
- TST: `(N, seq_len, d_model)` — heads flatten to `(N, seq_len*d_model)`
- TimeNet: `(N, seq_len, hidden_dims)` — no `_postprocess`, inconsistent

Heads need a stable input shape. Options:
- Define a `RepresentationProtocol` with shape contract.
- Let `encoding_fn` in the wrapper handle reshaping.
- Standardize `_postprocess` to always return `(N, D)`.

### 3. Freeze/unfreeze lifecycle

Current heads freeze at `__init__` time. No path to unfreeze later for gradual fine-tuning. Should the wrapper support:
- `freeze_backbone: bool` (current)
- `freeze_schedule: dict[int, bool]` (per-epoch)
- `unfreeze_layer(index)` method (manual)

### 4. Batch format coupling

TST heads expect `(X, targets, padding_masks, IDs)`. Series2Vec heads expect `(X, targets)`. A generic wrapper needs a batch parser strategy — or delegates to a `LightningDataModule` with a known output contract.

### 5. Cross-model consistency

Dilated models have no downstream heads yet. When they get them, should they use the same pattern? Currently CoST's `query_projection_head` is pretraining-only (contrastive). A classification head would be additive.

## Deliverable

Produce a recommendation with:
1. Chosen pattern (A/B/C or variant).
2. Sketch of the shared `FineTuningModule` API if Option B.
3. Migration path for existing TST and Series2Vec heads.
4. What to defer (e.g., gradual unfreeze, dilated model heads).
