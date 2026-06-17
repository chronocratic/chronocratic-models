# Unified Downstream Heads — Implementation Plan

**Branch:** `feature/baseline` (worktree)
**Mode:** TDD (tests first, then implementation)
**Source:** Design spec at `.planning/todos/heads-design_structure.md`
**Decisions:** CONTEXT.md (D-01 through D-04)

---

## Goal

Collapse 3 competing downstream head patterns (TST `heads.py`, Series2Vec `heads.py`, TS-TCC enum-mixed modes) into one generic `FineTuningModule` wrapper with injected collaborators (batch adapters, representation functions, loss). Add `representation_dim` property to each backbone. Delete per-model head files. Full pytest coverage.

---

## Decision Coverage

| Decision | ID | Plan(s) |
|----------|----|---------|
| Fresh FlattenLinearHead for TS-TCC | D-01 | P01, P02, P05 |
| Add `self._representation_dims` to Series2VecNetwork | D-02 | P02 |
| Remove `TSTCCTrainingMode` enum entirely | D-03 | P05 |
| BackboneUnfreeze optional (included, not required) | D-04 | P01 |

---

## Multi-Source Coverage Audit

| Item | Type | Source | Covered By |
|------|------|--------|------------|
| FineTuningModule wrapper | RESEARCH | heads-design §3.1 finetuning.py | P01 |
| FlattenLinearHead | RESEARCH | heads-design §3.1 finetuning.py | P01 |
| BatchAdapter protocol | RESEARCH | heads-design §3.1 finetuning.py | P01 |
| RepresentationBackbone protocol | RESEARCH | heads-design §3.1 finetuning.py | P01 |
| tst_batch_adapter | RESEARCH | heads-design §3.2 adapters.py | P01 |
| supervised_batch_adapter | RESEARCH | heads-design §3.2 adapters.py | P01 |
| tst_representations | RESEARCH | heads-design §3.2 adapters.py | P01 |
| series2vec_representations | RESEARCH | heads-design §3.2 adapters.py | P01 |
| tstcc_representations | RESEARCH | heads-design §3.2 adapters.py | P01 |
| classification_loss | RESEARCH | heads-design §3.2 adapters.py | P01 |
| BackboneUnfreeze callback | RESEARCH | heads-design §5 | P01 |
| Factory constructors | RESEARCH | heads-design §3.3 | P01 |
| TST.representation_dim | RESEARCH | heads-design §4.1, attr: `max_len` | P02 |
| Series2Vec.representation_dim | RESEARCH | heads-design §4.2, D-02 | P02 |
| TSTCC.representation_dim | RESEARCH | heads-design §4.3, D-01 | P02 |
| Migrate TST (delete heads.py) | CONTEXT | D-01, D-03 implication | P03 |
| Migrate Series2Vec (delete heads.py) | CONTEXT | D-02 | P04 |
| Remove TS-TCC enum + downstream modes | CONTEXT | D-03 | P05 |
| Series2VecNetwork._representation_dims attr | CONTEXT | D-02 | P02 |
| Full test coverage | CONTEXT | TDD Requirements | P01, P02, P03-P05, P06 |

**Result:** All items covered. No gaps.

---

## Plan Structure (6 plans, 3 waves)

| Plan | Name | Wave | Depends On | Tasks |
|------|------|------|------------|-------|
| P01 | _finetuning package (TDD) | 1 | — | 2 |
| P02 | Backbone representation_dim | 2 | P01 | 1 |
| P03 | Migrate TST | 3 | P01, P02 | 2 |
| P04 | Migrate Series2Vec | 3 | P01, P02 | 2 |
| P05 | Remove TS-TCC enum + migrate | 4 | P01, P02 | 2 |
| P06 | Integration verification | 5 | P03, P04, P05 | 1 |

Wave 3 (P03, P04) runs in parallel — zero file overlap.

**Note:** All three factory functions (`make_tst_finetuner`, `make_series2vec_finetuner`, `make_tstcc_finetuner`) are created in P01. P03-P05 consume the factories; they do NOT modify `factory.py`. This ensures P03/P04 can run in parallel without file conflicts.

---

## P01: _finetuning Package (TDD)

**Wave:** 1 | **Tasks:** 2 | **Files created:** 6

### Files

| File | Action |
|------|--------|
| `tests/unit/test_finetuning_package.py` | Create (tests) |
| `src/tscollection/models/_finetuning/__init__.py` | Create |
| `src/tscollection/models/_finetuning/finetuning.py` | Create |
| `src/tscollection/models/_finetuning/adapters.py` | Create |
| `src/tscollection/models/_finetuning/callbacks.py` | Create |
| `src/tscollection/models/_finetuning/factory.py` | Create |

### Task 1: Write failing tests (RED)

Create `tests/unit/test_finetuning_package.py` with import statements for all `_finetuning` symbols. Tests must fail because the package does not yet exist.

**Test behaviors (write assertions before any implementation):**

1. `FineTuningModule.forward` returns shape `(batch, num_outputs)` given a dummy backbone + head + adapter.
2. `FineTuningModule.training_step` returns a scalar loss and logs `train_loss`.
3. `FineTuningModule.validation_step` returns a scalar loss and logs `val_loss`.
4. `FineTuningModule(freeze_backbone=True)` sets `requires_grad=False` on all backbone params; optimizer sees only head params.
5. `FineTuningModule(freeze_backbone=False)` backbone params receive gradients after backward.
6. `FlattenLinearHead` with `(B, seq, dim)` input → `(B, num_outputs)` output (flattens start_dim=1).
7. `FlattenLinearHead` with `(B, dim)` input → `(B, num_outputs)` output (flatten is no-op).
8. `tst_batch_adapter` unpacks `(X, targets, padding_masks, IDs)` → `((X, padding_masks), targets)`.
9. `supervised_batch_adapter` unpacks `(X, targets)` → `((X,), targets)`.
10. `tst_representations` calls `backbone.get_representations(x, padding_masks)`, zeroes padding, returns `(B, seq, d_model)`.
11. `series2vec_representations` calls `backbone.network.encode(x)`, returns `(B, 2*rep_dims)`.
12. `tstcc_representations` calls `backbone(x.float())`, extracts features from `(logits, features)`.
13. `classification_loss` calls `nn.functional.cross_entropy` with squeezed int64 targets.
14. `BackboneUnfreeze.freeze_before_training` freezes backbone modules.
15. `BackboneUnfreeze.finetune_function` at target epoch calls `unfreeze_and_add_param_group`.
16. `RepresentationBackbone` Protocol is runtime-checkable.

Use `pytest` with small synthetic modules (no real backbone needed — use `nn.Module` stubs with the right attributes).

```
uv run pytest tests/unit/test_finetuning_package.py -x  # MUST fail (imports missing)
```

### Task 2: Implement _finetuning package (GREEN)

Implement all 5 source files per the design spec (heads-design_structure.md §3). Follow these rules:

- **finetuning.py:** Copy the `FineTuningModule`, `FlattenLinearHead`, `BatchAdapter`, `RepresentationBackbone` from the design spec verbatim. Use `self.save_hyperparameters(ignore=[...])`. Freeze logic uses `self._backbone.requires_grad_(requires_grad=False)` (not per-param loop).
- **adapters.py:** Implement `tst_batch_adapter`, `supervised_batch_adapter`, `tst_representations`, `series2vec_representations`, `tstcc_representations`, `classification_loss` from the design spec §3.2. Note: `tstcc_representations` must cast input to `.float()` (per design spec).
- **callbacks.py:** Implement `BackboneUnfreeze` from the design spec §5. Use `BaseFinetuning` as the parent class. Set `freeze_backbone=False` doc-note: caller must set this when using the callback.
- **factory.py:** Implement `make_tst_finetuner`, `make_series2vec_finetuner`, `make_tstcc_finetuner`. Each factory:
  - Accepts `backbone`, `num_outputs` (or `num_classes` for TS-TCC), `task` ('classification' | 'regression'), plus hparams kwargs.
  - Creates `head = FlattenLinearHead(backbone.representation_dim, num_outputs)` — per D-01, TS-TCC uses fresh FlattenLinearHead (not encoder logits reuse).
  - Wires correct `batch_adapter`, `representation_fn`, `loss_fn`.
  - Returns `FineTuningModule(backbone=backbone, head=head, ...)`.
- **__init__.py:** Barrel with explicit `__all__`. Export `FineTuningModule`, `FlattenLinearHead`, `BatchAdapter`, `RepresentationBackbone`, all adapters, `BackboneUnfreeze`, and factory functions.

```
uv run pytest tests/unit/test_finetuning_package.py -x  # MUST pass
uv run ruff check src/tscollection/models/_finetuning/
uv run ruff format src/tscollection/models/_finetuning/ --check
```

### Commit

```
feat(01-finetuning): add _finetuning package with FineTuningModule, adapters, callbacks, factories
```

---

## P02: Backbone representation_dim Properties

**Wave:** 2 | **Tasks:** 1 | **Files modified:** 4, **Files created:** 1

### Files

| File | Action |
|------|--------|
| `src/tscollection/models/transformer/tst/model.py` | Modify |
| `src/tscollection/models/convolutional/standard/series2vec/network.py` | Modify |
| `src/tscollection/models/convolutional/standard/series2vec/model.py` | Modify |
| `src/tscollection/models/convolutional/standard/tstcc/model.py` | Modify |
| `tests/unit/test_backbone_representation_dim.py` | Create |

### Task 1: Add representation_dim property + test (TDD)

**Step 1 — Write tests (RED):**

Create `tests/unit/test_backbone_representation_dim.py`:

1. `test_tst_representation_dim_matches_forward` — Build a tiny TST (`feat_dim=2, max_seq_len=10, d_model=8`). Run `get_representations(x, mask)` → flatten → verify flattened size equals `model.representation_dim`.
2. `test_series2vec_representation_dim_matches_forward` — Build a tiny Series2Vec (`input_dims=2, ..., representation_dims=8`). Run `model.network.encode(x)` → verify output dim equals `model.representation_dim`.
3. `test_tstcc_representation_dim_matches_forward` — Build a tiny TSTCC. Run `model(x)` → extract features → flatten → verify flattened size equals `model.representation_dim`.
4. `test_representation_backbone_protocol_tst` — Verify `isinstance(tst, RepresentationBackbone)` is True after property added.

**Step 2 — Implement properties (GREEN):**

Add `@property` to each backbone class. Use verified attribute names (NOT design spec defaults):

- **TST** (`model.py`): `return self._encoder.d_model * self._encoder.max_len`. Note: encoder attr is `max_len` (not `max_seq_len`), verified at ts_transformer.py:216.
- **Series2VecNetwork** (`network.py`): Per D-02, add `self._representation_dims = representation_dims` in `__init__`. Then add `@property` returning `self._representation_dims`.
- **Series2Vec** (`model.py`): Add `@property` returning `2 * self.network._representation_dims` (per D-02, the network now stores the attribute).
- **TSTCC** (`model.py`): `return self._encoder.logits.in_features`. Per CONTEXT.md research, `final_out_channels`/`features_len` are NOT stored as attributes; the logits Linear layer's `in_features` IS the flattened size. Verified against encoder.py:52.

All properties must have Google-style docstrings and `int` return type hints.

**Step 3 — Verify factories work with real backbones:**

Run a quick smoke test that `make_tst_finetuner`, `make_series2vec_finetuner`, and `make_tstcc_finetuner` can be called with the real backbone instances (not stubs). Verify output shapes match `backbone.representation_dim`.

```
uv run pytest tests/unit/test_backbone_representation_dim.py -x  # MUST pass
```

### Commit

```
feat(02-backbones): add representation_dim property to TST, Series2Vec, TSTCC
```

---

## P03: Migrate TST

**Wave:** 3 | **Tasks:** 2 | **Files created/modified/deleted:** 4

### Files

| File | Action |
|------|--------|
| `src/tscollection/models/transformer/tst/heads.py` | DELETE |
| `src/tscollection/models/transformer/tst/__init__.py` | Modify |
| `src/tscollection/models/transformer/tst/model.py` | Modify (docstring only) |
| `tests/unit/test_tst_migration.py` | Create |

### Task 1: Write migration tests (RED)

Create `tests/unit/test_tst_migration.py`:

1. `test_tst_finetuner_classification_shape` — Build TST → `make_tst_finetuner(backbone, num_outputs=5, task='classification')` → run forward with synthetic `(X, padding_masks)` → verify output shape `(B, 5)`.
2. `test_tst_finetuner_regression_shape` — Same with `task='regression'` → verify shape `(B, num_outputs)`.
3. `test_tst_training_step_logs` — Run `training_step` with `(X, targets, padding_masks, IDs)` batch → verify scalar loss returned and `train_loss` logged.
4. `test_old_heads_import_removed` — Verify `from tscollection.models.transformer.tst import TSTClassificationHead` raises `ImportError` (or `ModuleNotFoundError`).
5. `test_tst_freeze_backbone` — Build finetuner with `freeze_backbone=True` → run one backward → verify backbone grads are None.

### Task 2: Delete heads.py, update barrel (GREEN)

1. Delete `src/tscollection/models/transformer/tst/heads.py`.
2. Update `src/tscollection/models/transformer/tst/__init__.py`:
   - Remove `TSTClassificationHead`, `TSTRegressionHead` from `__all__` and imports.
   - Keep `TST`, `TSTModelParameters`.
3. Update `model.py` docstring: replace "For downstream classification / regression, use the dedicated heads in ..." with "For downstream classification / regression, use `FineTuningModule` from `tscollection.models._finetuning`."
4. Verify no other file imports `TSTClassificationHead` or `TSTRegressionHead` (grep confirmed clean in CONTEXT.md).

```
uv run pytest tests/unit/test_tst_migration.py -x  # MUST pass
```

### Commit

```
refactor(03-tst): migrate TST downstream heads to FineTuningModule, delete heads.py
```

---

## P04: Migrate Series2Vec

**Wave:** 3 | **Tasks:** 2 | **Files created/modified/deleted:** 4

### Files

| File | Action |
|------|--------|
| `src/tscollection/models/convolutional/standard/series2vec/heads.py` | DELETE |
| `src/tscollection/models/convolutional/standard/series2vec/__init__.py` | Modify |
| `src/tscollection/models/convolutional/standard/series2vec/network.py` | Modify (docstring only) |
| `tests/unit/test_series2vec_migration.py` | Create |

### Task 1: Write migration tests (RED)

Create `tests/unit/test_series2vec_migration.py`:

1. `test_series2vec_finetuner_classification_shape` — Build Series2Vec → `make_series2vec_finetuner(backbone, num_outputs=5, task='classification')` → run forward with synthetic `X` → verify shape `(B, 5)`.
2. `test_series2vec_finetuner_regression_shape` — Same with `task='regression'` → verify shape.
3. `test_series2vec_training_step_logs` — Run `training_step` with `(X, targets)` batch → verify scalar loss.
4. `test_old_head_import_removed` — Verify `Series2VecClassificationHead` no longer importable.
5. `test_series2vec_network_has_representation_dims_attr` — Verify `model.network._representation_dims` exists (per D-02, added in P02).

### Task 2: Delete heads.py, update barrel (GREEN)

1. Delete `src/tscollection/models/convolutional/standard/series2vec/heads.py`.
2. Update `src/tscollection/models/convolutional/standard/series2vec/__init__.py`:
   - Remove `Series2VecClassificationHead` from `__all__` and imports.
   - Keep `Series2Vec`, `Series2VecModelParameters`.
3. Update `network.py` docstring: replace "downstream classification is implemented as a separate head (see series2vec/heads.py)" with "downstream classification/regression uses `FineTuningModule` from `tscollection.models._finetuning`."

```
uv run pytest tests/unit/test_series2vec_migration.py -x  # MUST pass
```

### Commit

```
refactor(04-series2vec): migrate Series2Vec downstream head to FineTuningModule, delete heads.py
```

---

## P05: Remove TS-TCC Enum + Downstream Modes

**Wave:** 4 | **Tasks:** 2 | **Files created/modified/deleted:** 6

### Files

| File | Action |
|------|--------|
| `src/tscollection/models/convolutional/standard/tstcc/enums.py` | DELETE |
| `src/tscollection/models/convolutional/standard/tstcc/model.py` | Modify |
| `src/tscollection/models/convolutional/standard/tstcc/config.py` | Modify |
| `src/tscollection/models/convolutional/standard/tstcc/__init__.py` | Modify |
| `tests/unit/test_tstcc_migration.py` | Create |
| `src/tscollection/models/convolutional/standard/__init__.py` | Verify (no changes expected) |

### Task 1: Write migration tests (RED)

Create `tests/unit/test_tstcc_migration.py`:

1. `test_enum_import_removed` — Verify `TSTCCTrainingMode` no longer importable from `tstcc.enums` or `tstcc.__init__`.
2. `test_tstcc_model_no_training_mode_param` — Verify `TSTCC.__init__` no longer accepts `training_mode`.
3. `test_tstcc_only_self_supervised` — Verify `TSTCC._compute_loss` produces contrastive loss (no supervised branch).
4. `test_tstcc_finetuner_classification_shape` — Build TSTCC → `make_tstcc_finetuner(backbone, num_classes=5)` → run forward → verify shape `(B, 5)`. Tests factory from P01 works with the now-cleaned TSTCC backbone per D-01.
5. `test_tstcc_finetuner_training_step` — Run `training_step` with `(X, targets)` → verify scalar loss (uses FineTuningModule, not TSTCC directly).
6. `test_config_no_training_mode_field` — Verify `TSTCCModelParameters` dataclass no longer has `training_mode`.

### Task 2: Remove enum, clean model, update config + barrel (GREEN)

Per D-03 (remove TSTCCTrainingMode entirely):

1. **Delete** `src/tscollection/models/convolutional/standard/tstcc/enums.py`.
2. **Modify model.py:**
   - Remove `from tscollection.models.convolutional.standard.tstcc.enums import TSTCCTrainingMode`.
   - Remove `training_mode` parameter from `__init__` signature.
   - Remove `self._training_mode = training_mode`.
   - Remove `if training_mode == TSTCCTrainingMode.FINE_TUNING` freeze branch (freeze is now FineTuningModule's job).
   - Simplify `_compute_loss`: remove the `SUPERVISED`/`FINE_TUNING` branch. The method now only handles `SELF_SUPERVISED` (contrastive) logic.
   - Update class docstring: remove references to supervised/fine_tuning modes. Keep self_supervised description.
   - Remove `self._criterion = nn.CrossEntropyLoss()` (no longer needed, was for supervised/fine-tuning).
3. **Modify config.py:**
   - Remove `from tscollection.models.convolutional.standard.tstcc.enums import TSTCCTrainingMode`.
   - Remove `training_mode` field from `TSTCCModelParameters` dataclass.
   - Update docstring to remove `training_mode` description.
4. **Modify __init__.py:**
   - Remove `TSTCCTrainingMode` from `__all__` and imports.
   - Keep `TSTCC`, `TSTCCModelParameters`.
5. **Verify** `src/tscollection/models/convolutional/standard/__init__.py` — confirm no `TSTCCTrainingMode` import leak (grep showed it was not exported there).

```
uv run pytest tests/unit/test_tstcc_migration.py -x  # MUST pass
```

### Commit

```
refactor(05-tstcc): remove TSTCCTrainingMode enum, make model single-purpose (pretrain only), migrate downstream to FineTuningModule
```

---

## P06: Integration Verification

**Wave:** 5 | **Tasks:** 1 | **Files created:** 2

### Files

| File | Action |
|------|--------|
| `tests/integration/test_finetuning_integration.py` | Create |
| `tests/integration/__init__.py` | Create |

### Task 1: Cross-model integration tests (TDD)

Create `tests/integration/test_finetuning_integration.py`:

1. `test_all_backbones_satisfy_protocol` — Verify `isinstance(backbone, RepresentationBackbone)` for TST, Series2Vec, TSTCC.
2. `test_all_factories_produce_finetuningmodule` — Call each factory → verify return type is `FineTuningModule`.
3. `test_finetuningmodule_trains_end_to_end_tst` — Build TST → factory → run 3 training steps with Trainer → verify finite loss, logged metrics.
4. `test_finetuningmodule_trains_end_to_end_series2vec` — Same for Series2Vec.
5. `test_finetuningmodule_trains_end_to_end_tstcc` — Same for TSTCC downstream.
6. `test_tstcc_pretraining_still_works` — Build TSTCC (now only self_supervised) → run 3 training steps → verify contrastive loss is finite.
7. `test_barrel_exports_clean` — Verify `tscollection.models._finetuning` exports match `__all__`. Verify no head class leaked from any barrel.
8. `test_regression_task_works` — Build finetuner with `task='regression'` → verify `nn.MSELoss` is used and training produces finite loss.

```
uv run pytest tests/integration/ -x  # MUST pass
uv run pytest tests/ -x  # Full suite MUST pass
uv run ruff check src/ tests/
uv run ruff format src/ tests/ --check
```

### Commit

```
test(integration): cross-model fine-tuning integration tests and full verification
```

---

## Verification Checklist

Run after ALL plans complete:

- [ ] `uv run pytest tests/` passes with zero failures
- [ ] `uv run ruff check src/` passes with zero errors
- [ ] `uv run ruff format src/ --check` passes
- [ ] `ty check src/` passes (if available)
- [ ] `TSTClassificationHead`, `TSTRegressionHead`, `Series2VecClassificationHead`, `TSTCCTrainingMode` are all gone from source and barrels
- [ ] `tst/heads.py`, `series2vec/heads.py`, `tstcc/enums.py` are all deleted
- [ ] Each backbone (TST, Series2Vec, TSTCC) has `representation_dim` property
- [ ] Each factory (`make_tst_finetuner`, `make_series2vec_finetuner`, `make_tstcc_finetuner`) works
- [ ] TS-TCC model no longer accepts `training_mode` parameter
- [ ] TS-TCC `TSTCCModelParameters` no longer has `training_mode` field
- [ ] `FineTuningModule` freeze/unfreeze works correctly
- [ ] Series2Vec now supports regression (was missing before)
- [ ] All keyword-args convention followed in new code

---

## Execution Order

```
P01 (TDD: tests → _finetuning package)
  ↓
P02 (TDD: tests → backbone representation_dim)
  ↓
┌─────────────────────────┐
│ P03 (TST migration)     │ ← Wave 3 parallel
│ P04 (Series2Vec migrate)│
└─────────────────────────┘
  ↓
P05 (TS-TCC enum removal + migrate)
  ↓
P06 (Integration verification)
```

Commit each plan as a single atomic commit. Do NOT squash.
