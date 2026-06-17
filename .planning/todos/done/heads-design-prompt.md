# Prompt: Heads Architecture Design Discussion

Copy everything below this line into the agent.

---

We have 3 competing patterns for downstream heads (classification/regression) across our time-series models. Read these 8 files (~600 lines), then analyze and recommend a unified approach.

**Files:**
1. `src/tscollection/models/transformer/tst/heads.py` — heads as separate `pl.LightningModule`, abstract base `_TSTHead`
2. `src/tscollection/models/convolutional/standard/series2vec/heads.py` — same pattern but no base class; duplicates TST boilerplate
3. `src/tscollection/models/convolutional/dilated/cost/model.py:88-109` — dilated pattern: inline `nn.Sequential` projection head, no separate module
4. `src/tscollection/models/convolutional/standard/tstcc/model.py:107-109` — TS-TCC: training mode enum, freezes backbone at init, logits inside encoder
5. `src/tscollection/models/_mixin/encoding.py` — `BasicEncodingMixin` — the shared `encode()` contract these heads consume
6. `src/tscollection/models/transformer/tst/model.py:102-122` — `TST.get_representations()` — backbone output shape
7. `src/tscollection/models/convolutional/standard/series2vec/model.py:84-101` — `Series2Vec` backbone + encode hooks
8. `src/tscollection/models/convolutional/standard/mcl/model.py:36-42` — `FCN` inline projection head

**Current problems:**
- TST heads and Series2Vec heads duplicate ~60 lines of training_step/val_step/configure_optimizers.
- Heads access backbone internals via `backbone.hparams['d_model']` — fragile dict lookups.
- Freeze is a one-way decision at `__init__`. No gradual unfreeze path.
- Series2Vec has no regression head, no shared base class.
- Dilated models (CoST, TS2Vec, AutoTCL) use inline heads — no downstream pattern yet.
- Batch formats differ: TST expects `(X, targets, masks, IDs)`, Series2Vec expects `(X, targets)`.

**Analyze these 3 options:**

**A. Keep heads as LightningModules** — status quo. Defend or refute.

**B. Generic `FineTuningModule(pl.LightningModule)`** — one wrapper takes `(backbone, head: nn.Module, encoding_fn, loss_fn, lr)`. Heads become 10-line `nn.Module`s. Sketch the API.

**C. Inline heads** — fold downstream logic back into the main model via strategy pattern or enum (TS-TCC approach).

**D. Other** — propose a different architecture pattern that addresses the problems above. Base your design on software engineering principles (e.g., separation of concerns, DRY, single source of truth) and practical considerations (e.g., ease of adding new heads, training dynamics) in addition to best practices in the PyTorch/Lightning ecosystem. Consider how your design would handle the current issues and what trade-offs it might involve.

**Produce:**
1. Recommendation with rationale.
2. If B, sketch `FineTuningModule.__init__` and `training_step` signatures showing how it handles varying batch formats and representation shapes.
3. Migration steps for TST and Series2Vec heads.
4. What to defer to a later phase.
