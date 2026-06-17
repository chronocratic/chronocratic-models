## Context

I'm evaluating the downstream heads refactoring completed in commits d307506–4fbcd69 on branch `feature/baseline`. The task is to determine whether the TSTCC supervised training mode should remain inside the TSTCC model class or move to the shared `FineTuningModule` wrapper.

## Files to Read (in this order)

### Design Spec
- `.planning/todos/heads-design_structure.md` — Sections 4.3, 6.3, and the governing rule (Section 2). The spec says "Labeled downstream (classification/regression) lives in `FineTuningModule`."

### Current TSTCC Model (post-refactor)
- `src/tscollection/models/convolutional/standard/tstcc/model.py` — Single-purpose now. Read `__init__`, `_compute_loss`, `training_step`, `configure_optimizers`. Note: uses `automatic_optimization = False` with two manual optimizers (encoder + temporal contrast).

### TSTCC Before Refactor (git history)
- `git show HEAD~10:src/tscollection/models/convolutional/standard/tstcc/model.py` — This is the pre-refactor version with `TSTCCTrainingMode` enum. Look at the `SUPERVISED` branch in `_compute_loss` and `training_step` to see what it did: it was just `nn.CrossEntropyLoss(predictions, labels.long())` with the same manual-optimizer setup.

### FineTuningModule (new)
- `src/tscollection/models/_finetuning/finetuning.py` — The generic wrapper. Note: uses automatic optimization, single Adam optimizer.
- `src/tscollection/models/_finetuning/factory.py` — `make_tstcc_finetuner` function. Note: calls `FineTuningModule` with `freeze_backbone=False` for full supervised training.

### Comparison Models (for context on how other models handle supervised)
- `src/tscollection/models/transformer/tst/heads.py` — DELETED. Look at `git show HEAD~10:src/tscollection/models/transformer/tst/heads.py`. TST supervised was in the head classes.
- `src/tscollection/models/convolutional/standard/series2vec/model.py` — Series2Vec model only does pretraining; supervised was in `heads.py` (now deleted).

## What TSTCC's Old `SUPERVISED` Mode Did

Reading `git show HEAD~10:src/tscollection/models/convolutional/standard/tstcc/model.py` lines 121–142:
- `_compute_loss` had a branch: if `SUPERVISED` or `FINE_TUNING`, run `self._encoder(data)` → `nn.CrossEntropyLoss(predictions, labels.long())`
- `training_step` used the SAME manual-optimizer setup (2 optimizers, `manual_backward`)
- `FINE_TUNING` was identical to `SUPERVISED` except backbone weights frozen at `__init__`

So supervised was: encoder forward → cross-entropy → manual optimizer step. Same two optimizers as self-supervised, but only encoder optimizer does useful work (temporal contrast params don't participate).

## Questions to Discuss

1. **Is the `FineTuningModule` equivalent to old `SUPERVISED`?**
   - `FineTuningModule` does: `representation_fn(backbone, x) → head(reps) → loss_fn`
   - Old `SUPERVISED` does: `backbone(x) → predictions` (encoder's own logits layer) → `CrossEntropyLoss`
   - Key difference: `FineTuningModule` uses a fresh `FlattenLinearHead` on raw features, while old `SUPERVISED` used the encoder's built-in logits layer. Are these functionally equivalent for TSTCC? The encoder's logits layer is `nn.Linear(features_len * final_out_channels, num_classes)` — same input size as the fresh head. But the fresh head has different weight initialization.

2. **Manual optimization concern:**
   - TSTCC's self-supervised mode uses `automatic_optimization = False` with two optimizers.
   - `FineTuningModule` uses automatic optimization with one optimizer.
   - If supervised training stays in `FineTuningModule`, there's no conflict (different module, different trainer run).
   - If supervised training goes back into TSTCC model, it shares the same manual-optimizer infrastructure. This means the temporal contrast optimizer gets stepped even though it does no work in supervised mode. Is this a problem?

3. **API consistency:**
   - TST and Series2Vec: supervised is exclusively via `FineTuningModule`. Model class = pretrain only.
   - TSTCC post-refactor: same pattern. Model class = pretrain only. Supervised via `FineTuningModule`.
   - Does keeping TSTCC as single-purpose make the collection consistent, or does TSTCC have a unique reason to deviate?

4. **Original paper/codebase intent:**
   - The TS-TCC source repo (https://github.com/emadeldeen24/TS-TCC) has supervised training inline in the model.
   - Is deviating from the source pattern justified for collection consistency, or does it risk port fidelity?

## Expected Output

Structured comparison table:
| Dimension | FineTuningModule (current) | Inline supervised (revert) | Verdict |
|---|---|---|---|
| Functional equivalence | ... | ... | ... |
| API consistency across collection | ... | ... | ... |
| Port fidelity to source | ... | ... | ... |
| Implementation complexity | ... | ... | ... |

Then a recommendation with rationale.
