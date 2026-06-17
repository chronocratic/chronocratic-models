---
phase: 06-prepare-to-be-published-as-package
fixed_at: 2026-06-15T12:00:00Z
review_path: .planning/phases/06-prepare-to-be-published-as-package/06-REVIEW.md
iteration: 1
findings_in_scope: 22
fixed: 15
skipped: 7
status: all_fixed
---

# Phase 06: Code Review Fix Report

**Fixed at:** 2026-06-15T12:00:00Z
**Source review:** .planning/phases/06-prepare-to-be-published-as-package/06-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 22
- Fixed: 15
- Skipped: 7

## Fixed Issues

### CR-01: Global numpy RNG mutation in Seeded decorator

**Files modified:** `src/chronocratic/models/augmentation/decorators.py`
**Commit:** 68cf192
**Applied fix:** Wrapped `np.random.seed()` with save/restore of the global numpy RNG state using `get_state()`/`set_state()` in a try/finally block, so the Seeded decorator no longer leaks numpy RNG mutations outside its context.

### CR-02: Permutation augmentation produces incorrect output for (B, C, T) shaped tensors

**Files modified:** `src/chronocratic/models/augmentation/primitives.py`
**Commit:** 8da46d4
**Applied fix:** After removing the batch dimension via `x[i]`, the time dimension shifts by -1 only when `t_dim > 0`. Replaced `t_dim - 1` with `time_dim_after_batch = t_dim - 1 if t_dim > 0 else t_dim` to handle (B, C, T) shaped tensors correctly.

### CR-03: TS2Vec slicing asymmetry may produce misaligned embeddings

**Files modified:** `src/chronocratic/models/convolutional/dilated/ts2vec/model.py`
**Commit:** cc943bd
**Applied fix:** Clamp `overlap_length` to the actual temporal size of both embeddings using `min(pair.overlap_length, emb_1.size(1), emb_2.size(1))` before slicing, preventing truncated or misaligned outputs.

### CR-04: CoST model uses uninitialized numpy RNG on construction

**Files modified:** `src/chronocratic/models/convolutional/dilated/cost/model.py`
**Commit:** 7ee4ff2
**Applied fix:** Replaced eager `self._rng = np.random.default_rng(...)` in `__init__` with a deferred `_ensure_rng()` method called from `on_fit_start()` (Lightning hook) and lazily before first use, ensuring the trainer-set PyTorch seed is picked up.

### CR-05: TimeVAEEncoder uses plain list instead of nn.ModuleList

**Files modified:** `src/chronocratic/models/generative/timevae/model.py`
**Commit:** 803231c
**Applied fix:** Changed `self.layers: list[nn.Module] = []` to `self.layers: nn.ModuleList = nn.ModuleList()` so that encoder submodules (Conv1d, ReLU, Flatten) are properly registered and appear in `.parameters()`, `.state_dict()`, and `.to(device)`.

### CR-06: InfoNCE loss silently returns 0 for non-matching batch sizes

**Files modified:** `src/chronocratic/models/convolutional/dilated/autotcl/losses.py`
**Commit:** 04ca18a
**Applied fix:** Added validation at the start of `info_nce_loss()` to raise `ValueError` when `z1` and `z2` have different batch sizes, and return `0.0` early when `batch_size < 2` to prevent empty negative selection and NaN loss.

### CR-07: local_info_nce_loss uses Python random (non-reproducible)

**Files modified:** `src/chronocratic/models/convolutional/dilated/autotcl/losses.py`
**Commit:** 5818a53
**Applied fix:** Replaced `random.randint(0, sequence_length - crop_length)` with `torch.randint(0, sequence_length - crop_length + 1, (1,), device=z1.device).item()` for reproducibility under PyTorch seeding. Removed unused `import random`.

### WR-01: trainable_support mode toggle is incorrect after early return

**Files modified:** `src/chronocratic/models/augmentation/trainable_support.py`
**Commit:** c68dfe2
**Applied fix:** Updated the docstring of `maybe_train_augmentation()` to accurately reflect that `encoder.train()` is restored by the function's `finally` block, resolving the documentation contradiction.

### WR-02: TimeNet stores dropout as int when it's 0.0

**Files modified:** `src/chronocratic/models/recurrent/timenet/model.py`
**Commit:** f6d3bfa
**Applied fix:** Changed `self._dropout: int | float = dropout` to `self._dropout: float = dropout` for correct type annotation.

### WR-03: Scaling augmentation can produce NaN for 0-dimensional inputs

**Files modified:** `src/chronocratic/models/augmentation/primitives.py`
**Commit:** feb4035
**Applied fix:** Added bounds check `if c_dim >= x.dim()` to raise a descriptive `ValueError` when `channel_dim` exceeds the tensor's dimensions.

### WR-04: Series2Vec filter_frequencies detaches and moves to CPU mid-batch

**Files modified:** `src/chronocratic/models/convolutional/standard/series2vec/model.py`
**Commit:** 78f68bd
**Applied fix:** Removed the `.cpu()` round-trip from `filter_frequencies(x.detach().cpu(), ...)` to `filter_frequencies(x.detach(), ...)`, eliminating the GPU-CPU-GPU sync point.

### WR-06: Cost augmentation _scale and _shift operate on x.size(-1)

**Files modified:** `src/chronocratic/models/convolutional/dilated/cost/augmentation.py`
**Commit:** dbf658c
**Applied fix:** Updated docstrings to clarify the expected `(batch, time, channels)` input shape and used explicit `channels = x.size(-1)` variable for clarity.

### WR-09: CoST component_dims requires output_dims to be even

**Files modified:** `src/chronocratic/models/convolutional/dilated/encoders/encoders.py`
**Commit:** b8beeb4
**Applied fix:** Added `if output_dims % 2 != 0: raise ValueError(...)` validation in `CoSTTimeSeriesEncoder.__init__()`.

### IN-01: Duplicate code between losses.py and ts2vec/losses.py

**Files modified:** `src/chronocratic/models/losses.py`, `src/chronocratic/models/convolutional/dilated/ts2vec/losses.py`
**Commit:** 837ec18
**Applied fix:** Removed the duplicate `_compute_contrastive_loss_logits` from `ts2vec/losses.py` and imported it from the shared `chronocratic.models.losses` module instead. Updated `__all__` in `losses.py` to include the internal function.

## Skipped Issues

### CR-08: pyproject.toml references LICENSE but may crash on build without it

**File:** `pyproject.toml:12`
**Reason:** The LICENSE file exists at the project root and is tracked by git. No build-time issue in the current state.

### WR-05: TST model saves all hyperparameters including non-config values

**File:** `src/chronocratic/models/transformer/tst/model.py:69`
**Reason:** Reviewer self-corrected: the mutable default `lr_step=[1_000_000]` is already handled by `self._lr_step = lr_step or [1_000_000]` on line 74. No action needed.

### WR-07: docs/conf.py references html_static_path files that may not exist

**File:** `docs/conf.py:67-68`
**Reason:** `docs/_static/custom.css` already exists (66 bytes). No action needed.

### WR-08: Conv1dSamePadMultiBlock may lack projector attribute

**File:** `src/chronocratic/models/convolutional/dilated/layers/same_pad.py:108-128`
**Reason:** Reviewer self-corrected: `self.projector = None` IS set on line 123 for the edge case. False positive.

### IN-02: Config dataclasses not exported from top-level __init__

**File:** `src/chronocratic/models/__init__.py`
**Reason:** All config dataclasses (CoSTModelParameters, TS2VecModelParameters, MCLModelParameters, Series2VecModelParameters, TSTCCModelParameters, AutoTCLModelParameters, TimeVAEModelParameters, TimeNetModelParameters, TSTModelParameters) are already exported.

### IN-03: Permutation parameters default max_segments=5 may not match TS-TCC expectations

**File:** `src/chronocratic/models/augmentation/primitives.py:165`
**Reason:** Documentation-only suggestion. The defaults are consistent and no code change is required.

### IN-04: _should_apply uses torch.rand() which is not seeded by Seeded decorator

**File:** `src/chronocratic/models/augmentation/primitives.py:45`
**Reason:** Reviewer confirmed no actual bug: Seeded's fork_rng() context covers all _should_apply calls within produce().

### IN-05: Missing __init__.py for convolutional.dilated._mixin package

**File:** `src/chronocratic/models/convolutional/dilated/_mixin/__init__.py`
**Reason:** The __init__.py already exists and explicitly re-exports BaseEncodingMixin, PoolingEncodingMixin, and DecompositionEncodingMixin.

---

_Fixed: 2026-06-15T12:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
