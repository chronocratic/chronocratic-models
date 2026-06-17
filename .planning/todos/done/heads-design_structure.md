# Downstream Heads — Unified Design (Execution Spec)

Status: **approved design, not yet implemented**. Companion notes:
`heads-design-prompt.md` (original problem statement), `heads-design-discussion.md` (Q&A).
This file is the authoritative build spec. Read it fully before touching code.

---

## 1. Problem being solved

We have **3 competing patterns** for downstream classification/regression heads across
the time-series models. They duplicate code, couple to backbone internals, and freeze
in a one-way fashion. The offenders:

- `src/tscollection/models/transformer/tst/heads.py` — `_TSTHead` base + two heads, each a
  `pl.LightningModule`. ~187 lines, owns its own `training_step` / `validation_step` /
  `configure_optimizers`.
- `src/tscollection/models/convolutional/standard/series2vec/heads.py` —
  `Series2VecClassificationHead`, same boilerplate, **no base class**, **no regression head**.
  ~103 lines duplicating the TST loop.
- `src/tscollection/models/convolutional/standard/tstcc/model.py` — TS-TCC folds *three*
  modes (`SELF_SUPERVISED`, `SUPERVISED`, `FINE_TUNING`) into the model via a
  `TSTCCTrainingMode` enum, freezes the backbone at `__init__` (one-way), logits head lives
  inside the encoder.

**Concrete pains:**
- TST & Series2Vec heads duplicate ~60 lines of `training_step` / `val_step` /
  `configure_optimizers`.
- Heads reach into backbone internals via fragile dict lookups:
  `backbone.hparams['d_model']`, `backbone.hparams['max_seq_len']`,
  `backbone.hparams['representation_dims']`.
- Freeze is decided once at `__init__`. No gradual-unfreeze path.
- Series2Vec has no regression head and no shared base.
- Batch formats differ: TST expects `(X, targets, padding_masks, IDs)`,
  Series2Vec/TS-TCC expect `(X, targets)`.
- Representation shapes differ: TST `(B, seq_len, d_model)` (needs flatten),
  Series2Vec `(B, 2*representation_dims)` (already flat).

---

## 2. Chosen design (and why)

**One generic Lightning wrapper + three injected collaborators.** Rejected alternatives and
rationale are in `heads-design-discussion.md`; summary:

- **A (status quo, head = LightningModule each):** rejected — N-duplication, no shared base.
- **B (generic wrapper, single `encoding_fn`):** right spirit, underspecified.
- **C (inline heads / enum, TS-TCC style):** rejected as the general pattern — couples
  pretraining & fine-tuning lifecycles, god-object backbone, one-way freeze.
- **D = chosen:** Option B **plus** a `BatchAdapter` Strategy + a differentiable
  `representation_fn` + explicit-dim heads via a `representation_dim` property.

### Governing rule (apply uniformly across the collection)

> **Pretraining loss (no labels) lives in the model. Labeled downstream lives in
> `SupervisedModule`.**

This is what makes the collection *consistent*. Pretraining stays inline because it is
model-specific and often needs custom optimization (TS-TCC manual opt, two optimizers).
Labeled downstream (classification/regression) is the same shape for every model:
`encode → head → loss`, so it lives in **one** wrapper.

"Labeled downstream" here includes **supervised-from-scratch** — training a *fresh*,
un-pretrained backbone end-to-end (`freeze_backbone=False`), not only fine-tuning a
pretrained one. Both are the same `SupervisedModule` call; the only difference is whether
the injected backbone carries pretrained weights. This is what subsumes the old TS-TCC
`SUPERVISED` mode.

| Model      | Pretraining (no labels) → stays in model class      | Downstream (labels) → `SupervisedModule` |
|------------|-----------------------------------------------------|------------------------------------------|
| TST        | masked reconstruction (`_compute_loss`)             | yes                                      |
| Series2Vec | soft-DTW pretraining (`_calculate_loss`)            | yes                                      |
| TS-TCC     | contrastive `SELF_SUPERVISED` (manual opt, 2 opts)  | yes (`SUPERVISED` + `FINE_TUNING`)       |

### Four orthogonal concerns, four owners

| Concern                                   | Today                                       | New owner                                 |
|-------------------------------------------|---------------------------------------------|-------------------------------------------|
| Decode batch tuple → `(inputs, targets)`  | hard-coded per head                         | `BatchAdapter` Strategy (callable)        |
| Backbone → differentiable representation  | `get_representations` / `network.encode`    | `representation_fn(backbone, *inputs)`    |
| Representation → logits (flatten + linear)| `nn.Linear(d_model*max_seq_len, …)`         | `head: nn.Module` (explicit `in_features`)|
| train/val/optim/log/freeze                | duplicated ×N                               | `SupervisedModule` (one LightningModule)  |

### Key decisions (locked)

1. **No LazyLinear.** Deferred materialization breaks optimizer construction before first
   forward, checkpoint load before forward, and `save_hyperparameters`. Instead, each
   backbone exposes a typed **`representation_dim` property** = the flattened feature size it
   hands the head. Single source of truth, eager, refactor-safe. This replaces all
   `hparams[...]` dict lookups.
2. **No mode enum for freeze.** The three strategies (linear-probe / full-FT / gradual) are
   **configuration**, not a type the module carries. Static cases = a `freeze_backbone: bool`
   on the module; gradual = an optional `BaseFinetuning` callback at the `Trainer`. **Single
   freeze owner**: never let the module bool AND a callback both flip `requires_grad`.
3. **`encode()` (the `BasicEncodingMixin`, `@torch.inference_mode`, batched, CPU) is for
   offline feature extraction — NOT fine-tuning.** Keep the split. `representation_fn` is the
   differentiable training path and must NOT route through `encode()`.
4. **Heads keep pooling/flatten responsibility** (`reps.flatten(start_dim=1)`), so
   `representation_fn` returns raw reps and one head type serves multiple backbones.

---

## 3. Files to create

### 3.1 `src/tscollection/models/supervised/` (new package)

New shared package. Mirror the style of `src/tscollection/models/_mixin/` (ABC/Protocol +
`__all__`, Google docstrings, `from __future__ import annotations` only if needed).

#### `supervised.py` — the wrapper + Protocols + reusable head

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import lightning.pytorch as pl
import torch
from torch import nn

__all__ = [
    'BatchAdapter',
    'SupervisedModule',
    'FlattenLinearHead',
    'RepresentationBackbone',
]


@runtime_checkable
class RepresentationBackbone(Protocol):
    """A backbone that can report the flattened feature size of its representation."""

    @property
    def representation_dim(self) -> int:
        """Flattened feature size handed to a downstream head."""


class BatchAdapter(Protocol):
    """Strategy: decode a model-specific batch tuple into encoder inputs + targets."""

    def __call__(self, batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return ``((encoder_inputs, ...), targets)``."""


class FlattenLinearHead(nn.Module):
    """Flatten a representation across all non-batch dims, then a single linear layer.

    Reused by every model whose representation is a tensor of shape ``(B, ...)``.
    Series2Vec reps are already ``(B, 2*rep)`` so the flatten is a no-op there.

    Args:
        in_features: Flattened representation size (``backbone.representation_dim``).
        num_outputs: Number of classes (classification) or targets (regression).
    """

    def __init__(self, in_features: int, num_outputs: int) -> None:
        super().__init__()
        self._fc = nn.Linear(in_features, num_outputs)

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        return self._fc(reps.flatten(start_dim=1))


class SupervisedModule(pl.LightningModule):
    """Generic downstream fine-tuning / linear-probe wrapper.

    Owns the train/val loop, optimizer, logging, and (static) freeze. Everything
    model-specific is injected:

    Args:
        backbone: A (possibly pretrained) model exposing the representation fn used below.
        head: Maps a representation tensor to ``(B, num_outputs)`` (e.g. FlattenLinearHead).
        representation_fn: ``(backbone, *encoder_inputs) -> Tensor``. Differentiable. MUST
            NOT route through ``encode()`` (that path is inference-mode / offline only).
        batch_adapter: Decodes the batch tuple into ``((encoder_inputs, ...), targets)``.
        loss_fn: ``(predictions, targets) -> scalar``.
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        freeze_backbone: Freeze backbone params (linear probe). Set ``False`` when a
            gradual-unfreeze callback owns freezing (see section 5). Never have both.
        sync_dist: Sync logged metrics across processes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        representation_fn: Callable[..., torch.Tensor],
        batch_adapter: BatchAdapter,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        freeze_backbone: bool = True,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=['backbone', 'head', 'representation_fn', 'batch_adapter', 'loss_fn']
        )
        self._backbone = backbone
        self._head = head
        self._representation_fn = representation_fn
        self._batch_adapter = batch_adapter
        self._loss_fn = loss_fn
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sync_dist = sync_dist
        if freeze_backbone:
            self._backbone.requires_grad_(requires_grad=False)

    def forward(self, *encoder_inputs: torch.Tensor) -> torch.Tensor:
        reps = self._representation_fn(self._backbone, *encoder_inputs)
        return self._head(reps)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        encoder_inputs, targets = self._batch_adapter(batch)
        predictions = self(*encoder_inputs)
        loss = self._loss_fn(predictions, targets)
        self.log(
            f'{stage}_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self._sync_dist,
        )
        return loss

    def training_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage='val')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        trainable = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.Adam(
            trainable, lr=self._learning_rate, weight_decay=self._weight_decay
        )
```

Design notes for the implementer:
- `forward(*encoder_inputs)` is variadic so it covers both `(x,)` and `(x, padding_masks)`
  without the module knowing which model it is.
- The optimizer generator `p ... if p.requires_grad` is intentional — it makes
  `freeze_backbone=True` produce a head-only optimizer, and it is compatible with the
  gradual-unfreeze callback (which *adds* a param group later via
  `unfreeze_and_add_param_group`).
- All call sites must use **keyword arguments** (project rule).

#### `_adapters.py` — batch adapters + representation fns + loss helpers

```python
from __future__ import annotations

import torch
from torch import nn

__all__ = [
    'classification_loss',
    'series2vec_representations',
    'supervised_batch_adapter',
    'tst_batch_adapter',
    'tst_representations',
    'tstcc_representations',
]


# ---- batch adapters ----------------------------------------------------------

def tst_batch_adapter(batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """TST batch ``(X, targets, padding_masks, IDs)`` -> ``((X, padding_masks), targets)``."""
    x, targets, padding_masks, _ids = batch
    return (x, padding_masks), targets


def supervised_batch_adapter(batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Standard ``(X, targets)`` -> ``((X,), targets)``. Used by Series2Vec and TS-TCC."""
    x, targets = batch
    return (x,), targets


# ---- representation fns (differentiable; mirror each backbone's encode hook) --

def tst_representations(
    backbone: nn.Module, x: torch.Tensor, padding_masks: torch.Tensor
) -> torch.Tensor:
    """Run the TST trunk and zero padded positions; head flattens (B, seq, d_model)."""
    reps = backbone.get_representations(x, padding_masks)
    return reps * padding_masks.unsqueeze(-1)


def series2vec_representations(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Concatenated temporal+frequency reps; already (B, 2*representation_dims)."""
    return backbone.network.encode(x)


def tstcc_representations(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """TS-TCC encoder returns (logits, features); take the pre-logits features."""
    _logits, features = backbone(x.float())
    return features


# ---- loss helpers ------------------------------------------------------------

def classification_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropy with the target-squeeze the old heads applied."""
    return nn.functional.cross_entropy(predictions, targets.long().squeeze())
```

Regression just uses `loss_fn=nn.MSELoss()` directly — no helper needed.

#### `__init__.py` — barrel

Export `SupervisedModule`, `FlattenLinearHead`, `BatchAdapter`, `RepresentationBackbone`,
and the adapter/representation/loss callables. Follow the project barrel style (explicit
`__all__`, no lazy imports unless a circular import forces it).

#### Optional `factory.py` — convenience constructors (recommended)

So callers don't hand-assemble four collaborators. Sketch:

```python
def make_tst_supervised(backbone, *, num_outputs, task, **kw) -> SupervisedModule: ...
def make_series2vec_supervised(backbone, *, num_outputs, task, **kw) -> SupervisedModule: ...
def make_tstcc_supervised(backbone, *, num_classes, **kw) -> SupervisedModule: ...
```

`task` in `{'classification', 'regression'}` selects `classification_loss` vs `nn.MSELoss()`.
Each factory wires `head = FlattenLinearHead(backbone.representation_dim, num_outputs)` and
the right adapter/representation_fn.

---

## 4. Backbone changes — add `representation_dim`

Add an eager, typed property to each backbone. **Verify the exact attribute source against
the current code before writing** (names below are from the reviewed snapshot; confirm).

### 4.1 TST — `src/tscollection/models/transformer/tst/model.py`

```python
@property
def representation_dim(self) -> int:
    """Flattened representation size handed to a downstream head."""
    return self._encoder.d_model * self._encoder.max_seq_len
```
- `get_representations` returns `(batch, seq_len, d_model)`; head flattens to
  `d_model * max_seq_len`. The old head computed this via
  `hparams['d_model'] * hparams['max_seq_len']`.
- **Verify:** `self._encoder.d_model` exists (used at model.py:109). Confirm the encoder also
  exposes `max_seq_len` (the head read it from `backbone.hparams['max_seq_len']`); if the
  encoder lacks the attribute, read `self.hparams['max_seq_len']` instead.

### 4.2 Series2Vec — `src/tscollection/models/convolutional/standard/series2vec/model.py`

```python
@property
def representation_dim(self) -> int:
    """Temporal + frequency reps are concatenated -> 2 * representation_dims."""
    return 2 * self.hparams['representation_dims']
```
- **Verify:** prefer a real attribute over the hparams dict if `Series2VecNetwork` stores
  `representation_dims` (e.g. `self.network.representation_dims`). Use whichever is the true
  single source; do not reintroduce a fragile lookup if an attribute exists.

### 4.3 TS-TCC — `src/tscollection/models/convolutional/standard/tstcc/model.py`

```python
@property
def representation_dim(self) -> int:
    """Pre-logits feature size of the TCC encoder."""
    return self._encoder.final_out_channels * self._encoder.features_len
```
- **Verify:** the `features` tensor returned by `TCCEncoder.forward` (second element of
  `(logits, features)`) — confirm its flattened size. Inspect
  `src/tscollection/models/convolutional/standard/tstcc/encoder.py`. It is likely
  `final_out_channels * features_len`, but CONFIRM by reading the encoder before committing.

---

## 5. Gradual unfreeze — optional `BaseFinetuning` callback (NOT a module mode)

Add to the new package (e.g. `supervised/_callbacks.py`). This is the **only** place the
three strategies differ — and they differ by *configuration*, never by an enum on the module.

```python
from __future__ import annotations

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning

__all__ = ['BackboneUnfreeze']


class BackboneUnfreeze(BaseFinetuning):
    """Freeze the backbone, then unfreeze at a target epoch with a reduced LR.

    Attach at the Trainer for the gradual strategy. When this callback is used, construct
    the SupervisedModule with ``freeze_backbone=False`` so this callback is the SOLE owner
    of freeze state.

    Args:
        unfreeze_at_epoch: Epoch at which the backbone is unfrozen.
        initial_denom_lr: Backbone enters at ``optimizer_lr / initial_denom_lr``
            (discriminative LR; 10.0 is the ULMFiT default).
    """

    def __init__(self, unfreeze_at_epoch: int = 10, initial_denom_lr: float = 10.0) -> None:
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._initial_denom_lr = initial_denom_lr

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze(pl_module._backbone)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        current_epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module._backbone,
                optimizer=optimizer,
                initial_denom_lr=self._initial_denom_lr,
            )
```

### Strategy = config table

| Strategy        | `SupervisedModule(freeze_backbone=...)` | Trainer callback                       |
|-----------------|-----------------------------------------|----------------------------------------|
| Linear probe       | `True`                               | none                                   |
| Full fine-tune     | `False`                              | none                                   |
| Gradual unfreeze   | `False` (callback owns freezing)     | `BackboneUnfreeze(unfreeze_at_epoch=k)`|
| Supervised (scratch)| `False` (fresh, un-pretrained backbone) | none                                |

Why this works with `configure_optimizers`: `BaseFinetuning.freeze_before_training` runs
before optimizer construction, so the initial Adam holds only head params;
`unfreeze_and_add_param_group` later *adds* the backbone as a new group mid-training — no
re-init, no LR clobber.

`BackboneUnfreeze` is **optional / documented**, not required for the first migration.

---

## 6. Migration steps

### 6.1 TST
1. Add `representation_dim` property to `TST` (section 4.1).
2. Add `tst_batch_adapter`, `tst_representations` to `supervised/_adapters.py` (the padding
   mask zeroing moves into `tst_representations`).
3. Replace usages of `TSTClassificationHead` / `TSTRegressionHead` with `SupervisedModule`
   (via factory): head `FlattenLinearHead(backbone.representation_dim, num_outputs)`,
   `representation_fn=tst_representations`, `batch_adapter=tst_batch_adapter`,
   loss `classification_loss` or `nn.MSELoss()`.
4. Delete `src/tscollection/models/transformer/tst/heads.py` (incl. `_TSTHead`). Update
   `__all__` / barrel exports / any imports.

### 6.2 Series2Vec
1. Add `representation_dim` property (section 4.2).
2. Add `series2vec_representations`; reuse `supervised_batch_adapter`.
3. Build the classification head via factory. **Regression head now exists for free** —
   `task='regression'` / `loss_fn=nn.MSELoss()`, no new class.
4. Delete `src/tscollection/models/convolutional/standard/series2vec/heads.py`. Update barrel.

### 6.3 TS-TCC
1. Add `representation_dim` property (section 4.3) — VERIFY feature size first.
2. Add `tstcc_representations`; reuse `supervised_batch_adapter`.
3. Move `SUPERVISED` + `FINE_TUNING` downstream out of the model into `SupervisedModule`.
   The **head can be the encoder's existing `logits` layer passed in as `head`** (preserves
   published weights / port fidelity) OR a fresh `FlattenLinearHead` — decide during impl;
   passing the existing logits submodule is preferred for fidelity. `FINE_TUNING` =
   `freeze_backbone=True`. Note: if reusing the encoder `logits` layer, the head should NOT
   flatten if the encoder already produces logits from features — match the encoder's actual
   forward; otherwise use `FlattenLinearHead` on the `features`.
4. Keep `SELF_SUPERVISED` (contrastive, `automatic_optimization=False`, two optimizers,
   augmentation) **inline in the model** — it is pretraining.
5. Collapse / remove the `TSTCCTrainingMode` enum branches that pertain to supervised/
   fine-tuning once the wrapper owns them. The model keeps one job: contrastive pretrain.
   **Caution:** this is the riskiest migration (port fidelity, manual optimization). Do it
   last and lean on tests / the source repo (https://github.com/emadeldeen24/TS-TCC).

### 6.4 Net effect
Two head files (~290 lines) collapse to one shared `supervised/` package (~90 lines of
wrapper + small adapter snippets). No more `hparams[...]` reach-ins. Series2Vec gains a
regression path for free. TS-TCC downstream becomes consistent with the rest.

---

## 7. Testing (required — current heads have zero coverage)

Add `tests/` under the matching path (`test_` prefix, pytest). Cover:
- `SupervisedModule.training_step` / `validation_step` return a scalar and log
  `train_loss` / `val_loss`.
- `freeze_backbone=True` ⇒ optimizer param count == head param count; backbone grads stay
  `None` after a backward.
- `freeze_backbone=False` ⇒ backbone params receive grads.
- Each `representation_fn` + `batch_adapter` pair produces the expected shapes for a tiny
  synthetic batch (TST with padding masks; Series2Vec/TS-TCC `(X, targets)`).
- `representation_dim` matches the actual flattened representation size from a forward pass
  (guards the section-4 "verify" items).
- Regression path: `SupervisedModule` with `nn.MSELoss()` trains on synthetic targets.
- `BackboneUnfreeze`: backbone frozen before `unfreeze_at_epoch`, trainable after; optimizer
  gains a param group. (Can defer with the callback itself.)

Run: `uv run pytest`. Lint/format: `uv run ruff check` + `uv run ruff format`. Type-check
with `ty`. Project targets Python 3.12; keyword-args everywhere; Google docstrings;
return-type hints mandatory.

---

## 8. Defer to a later phase

- **Dilated trio downstream heads (TS2Vec / CoST / AutoTCL).** They use the heavier
  sliding-window mixin under `convolutional/dilated/_mixin/` and produce multi-scale/pooled
  reps; their `representation_fn` is non-trivial. The wrapper already accepts them later
  (just another `representation_fn` + `batch_adapter` + `representation_dim`). Build only when
  a real downstream task exists. NOTE: CoST's `query_projection_head` / `key_projection_head`
  are **SSL projection heads, not downstream** — out of scope.
- **Discriminative-LR scheduler beyond `initial_denom_lr`** and multi-stage unfreeze
  schedules — extend `BackboneUnfreeze` when an experiment needs it.
- **`RepresentationBackbone` Protocol enforcement** (runtime checks / mypy-level) — add if
  drift becomes a problem; the property + tests cover it for now.

---

## 9. Verification checklist for the executing agent (do not skip)

- [ ] Read `tstcc/encoder.py` and confirm the `features` flattened size before writing the
      TS-TCC `representation_dim`.
- [ ] Confirm TST encoder exposes `d_model` and `max_seq_len` (else fall back to
      `self.hparams[...]`).
- [ ] Confirm Series2Vec `representation_dims` source (network attribute vs hparams).
- [ ] Confirm the TS-TCC encoder `logits` layer signature before deciding head reuse vs
      fresh `FlattenLinearHead`.
- [ ] Grep for every import / call site of the deleted head classes before deleting them.
- [ ] `uv run pytest`, `uv run ruff check`, `uv run ruff format`, `ty` all clean.
- [ ] Commit granularity preserved — one logical change per commit (never squash).
