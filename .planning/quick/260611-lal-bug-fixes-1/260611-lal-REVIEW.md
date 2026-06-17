---
phase: 260611-lal-bug-fixes-1
reviewed: 2026-06-11T21:00:00Z
depth: deep
files_reviewed: 19
files_reviewed_list:
  - src/tscollection/models/convolutional/standard/mcl/config.py
  - src/tscollection/models/convolutional/standard/mcl/losses.py
  - src/tscollection/models/convolutional/standard/mcl/model.py
  - src/tscollection/models/convolutional/standard/series2vec/config.py
  - src/tscollection/models/convolutional/standard/series2vec/filters.py
  - src/tscollection/models/convolutional/standard/series2vec/losses.py
  - src/tscollection/models/convolutional/standard/series2vec/model.py
  - src/tscollection/models/convolutional/standard/series2vec/network.py
  - src/tscollection/models/convolutional/standard/tstcc/encoder.py
  - src/tscollection/models/convolutional/standard/tstcc/losses.py
  - src/tscollection/models/convolutional/standard/tstcc/model.py
  - src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py
  - src/tscollection/models/generative/timevae/model.py
  - src/tscollection/models/generative/timevae/vae_base.py
  - src/tscollection/models/recurrent/timenet/config.py
  - src/tscollection/models/recurrent/timenet/model.py
  - src/tscollection/models/transformer/tst/loss.py
  - src/tscollection/models/transformer/tst/model.py
  - src/tscollection/models/transformer/tst/ts_transformer.py
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---

# Phase: Code Review Report — 260611-lal Bug Fixes

**Reviewed:** 2026-06-11T21:00:00Z
**Depth:** deep
**Files Reviewed:** 19
**Status:** issues_found

## Summary

This review covers 17 bug fixes (C1–C4, I1–I8, R1–R3, S1–S4) applied across six models: MCL, Series2Vec, TS-TCC, TimeVAE, TimeNet, and TST. Most fixes were applied correctly and improve runtime safety, determinism, and consistency.

However, the C3 fix in `series2vec/network.py` introduces a **critical regression**: the replacement of `.squeeze(0)` with `[:, 0, :]` uses inverted dimension indexing, silently dropping all but the first batch element and producing zero loss for any batch_size > 1. The original `.squeeze(0)` was correct.

Additionally, several warnings concern cross-cutting issues: stochastic validation in VAE and MCL models, code duplication in TST, and an unguarded division-by-zero path in the TST loss.

## Critical Issues

### CR-01: C3 fix inverts dimension indexing — silently drops all but first batch element

**File:** `src/tscollection/models/convolutional/standard/series2vec/network.py:109-110`

**Issue:** The fix for C3 replaced `.squeeze(0)` with `[:, 0, :]` on both `temporal_representation` and `frequency_representation`. After `permute(2, 0, 1)`, the tensor shape is `(1, B, repr_dim)` — dimension 0 is the GAP-produced sequence length (always 1), dimension 1 is batch. The indexing `[:, 0, :]` selects "all of dim 0, index 0 of dim 1, all of dim 2", which extracts `(1, repr_dim)` — only the first batch element.

The original `.squeeze(0)` correctly removed the size-1 dimension 0, producing `(B, repr_dim)`. The fix inverted the dimension assumption.

Impact: For batch_size > 1, `torch.cdist` computes a `(1, 1)` self-distance matrix. The lower-triangular mask (diagonal=-1) excludes the diagonal, leaving an empty tensor. `pretraining_loss` hits the `numel() == 0` early exit and returns `0.0`. The model trains silently with zero loss — no gradients, no learning.

This is the most insidious bug in the entire fix set: the code "works" (no crash) but produces zero loss, making it appear that training converges immediately.

```
Trace of shapes:
  x_src = self.gap(x_src)              # (B, repr_dim, 1)
  x_src = x_src.permute(2, 0, 1)       # (1, B, repr_dim)
  ... attention + FFN ...               # (1, B, repr_dim)
  out[:, 0, :]                          # (1, repr_dim) ← WRONG: only batch[0]
  out.squeeze(0)                        # (B, repr_dim) ← correct
  out[0, :, :]                          # (B, repr_dim) ← also correct
```

**Fix:** Replace `out[:, 0, :]` with `out[0, :, :]` and `x_f[:, 0, :]` with `x_f[0, :, :]`.

```python
# Lines 109-110 — change from:
temporal_representation = out[:, 0, :]
frequency_representation = x_f[:, 0, :]

# To:
temporal_representation = out[0, :, :]
frequency_representation = x_f[0, :, :]
```

## Warnings

### WR-01: TimeVAE validation step uses stochastic latent sampling

**File:** `src/tscollection/models/generative/timevae/vae_base.py:65-71`

**Issue:** `validation_step` calls `_step`, which invokes `self.encoder(x)` and receives `(z_mean, z_log_var, z)`. The `z` value is sampled via the reparameterization trick (`Sampling.forward` calls `torch.randn`). This means validation loss varies between runs even with the same data and checkpoint.

The `forward()` method (line 43) correctly uses `z_mean` (deterministic), but `_step()` uses `z` (stochastic). Validation should use `z_mean` instead of sampled `z` for reproducible metrics.

```python
# Current (line 54):
reconstruction = self.decoder(z)  # z is randomly sampled

# Should use deterministic mean during validation:
reconstruction = self.decoder(z if self.training else z_mean)
```

**Fix:** Modify `_step` to use `z_mean` when not in training mode.

```python
def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    z_mean, z_log_var, z = self.encoder(x)
    latent = z if self.training else z_mean
    reconstruction = self.decoder(latent)
    loss, recon_loss, kl_loss = self.loss_function(x, reconstruction, z_mean, z_log_var)
    return loss, recon_loss, kl_loss
```

### WR-02: TST `encode_representations()` duplicates 7 lines from `forward()`

**File:** `src/tscollection/models/transformer/tst/ts_transformer.py:254-282` vs `284-305`

**Issue:** The new `encode_representations()` method (added for I4) shares 7 identical lines with `forward()`: permute, project_inp, pos_enc, transformer_encoder, act, permute, dropout1. The only difference is that `forward()` applies `self.output_layer()` before returning.

This duplication is a maintenance risk: any future change to the shared prefix (e.g., adding a normalization step) must be applied to both methods, and forgetting one would silently produce different behavior between `forward()` and `encode_representations()`.

**Fix:** Extract the shared prefix into a private method.

```python
def _transform(self, x: Tensor, padding_masks: Tensor) -> Tensor:
    """Run transformer trunk, returning (batch, seq_len, d_model)."""
    inp = x.permute(1, 0, 2)
    inp = self.project_inp(inp) * math.sqrt(self.d_model)
    inp = self.pos_enc(inp)
    output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
    output = self.act(output)
    output = output.permute(1, 0, 2)
    return self.dropout1(output)

def forward(self, x: Tensor, padding_masks: Tensor) -> Tensor:
    return self.output_layer(self._transform(x, padding_masks))

def encode_representations(self, x: Tensor, padding_masks: Tensor) -> Tensor:
    return self._transform(x, padding_masks)
```

### WR-03: MCL validation step computes gradients without `torch.no_grad()`

**File:** `src/tscollection/models/convolutional/standard/mcl/model.py:77-83`

**Issue:** `validation_step` calls `_step`, which computes the full MixUp contrastive loss (three forward passes + MixUpLoss). Gradients are accumulated and then discarded by Lightning's validation loop. This wastes compute and memory.

Additionally, `_step` uses `torch.randperm(len(x))` and `Beta.sample()`, making validation loss non-deterministic. While this is inherent to MixUp contrastive learning, wrapping in `torch.no_grad()` at least saves the gradient computation. Note that TSTCC was fixed to add `torch.no_grad()` (I2), but MCL was not — creating inconsistency across models.

**Fix:** Wrap `_step` call in `torch.no_grad()` for validation.

```python
def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
    with torch.no_grad():
        loss = self._step(batch)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss
```

### WR-04: TST `MaskedMSELoss` produces NaN when mask has no active elements

**File:** `src/tscollection/models/transformer/tst/loss.py:37-40` and `src/tscollection/models/transformer/tst/model.py:125-127`

**Issue:** When `combined_mask` (line 124: `target_masks * padding_masks.unsqueeze(-1)`) has all False elements, `torch.masked_select` returns empty tensors. `MaskedMSELoss` with `reduction='none'` returns an empty tensor. Then in `_compute_loss`:

```python
mean_loss = torch.sum(per_element_loss) / len(per_element_loss)
# torch.sum([]) = 0.0, len([]) = 0 → 0.0 / 0 = NaN
```

The NaN propagates through `self.log()` and the backward pass, potentially corrupting model weights. While this requires a pathological input (entire batch masked), it could occur with the last batch of a short epoch or misconfigured dataloaders.

**Fix:** Add a guard for zero active elements.

```python
# In model.py _compute_loss:
per_element_loss = self._loss_fn(predictions, targets, combined_mask)
if len(per_element_loss) == 0:
    mean_loss = torch.tensor(0.0, device=predictions.device)
else:
    mean_loss = torch.sum(per_element_loss) / len(per_element_loss)
```

### WR-05: TimeNet `_dropout` uses unnecessary `int | float` union type

**File:** `src/tscollection/models/recurrent/timenet/model.py:51`

**Issue:** The `dropout` parameter is declared as `dropout: float = 0.1` in `__init__` (line 43), but stored as `self._dropout: int | float = dropout` (line 51). The `int` branch of the union is unreachable — `dropout` is always `float`. The union type adds no value and misleads readers into thinking `int` is acceptable.

**Fix:** Use `float` type annotation consistently.

```python
self._dropout: float = dropout
```

## Info

### IN-01: MCL `encoder.py` retains unnecessary `from __future__ import annotations`

**File:** `src/tscollection/models/convolutional/standard/mcl/encoder.py:1`

**Issue:** The S3 fix removed `from __future__ import annotations` from all TSTCC files, but left it in MCL's `encoder.py`. This file uses only `int`, `torch.Tensor`, and `None` in type annotations — all resolved at import time. The future import is unnecessary.

**Fix:** Remove line 1 (`from __future__ import annotations`).

### IN-02: TimeVAE `trend_poly is not None` check is redundant

**File:** `src/tscollection/models/generative/timevae/model.py:85` (TimeVAEDecoder)

**Issue:** `trend_poly` is declared as `int = 0` (never `None`). The condition `self.trend_poly is not None` on line 85 is always True. The effective check is only `self.trend_poly > 0`.

**Fix:** Simplify the condition.

```python
if self.trend_poly > 0:
```

### IN-03: TimeVAE decoder mixes `self.custom_seas` and `custom_seas` in adjacent conditions

**File:** `src/tscollection/models/generative/timevae/model.py:85-90`

**Issue:** Line 85 uses `self.trend_poly` (instance attribute), while line 89 uses `custom_seas` (local parameter). Both are valid since `self.custom_seas = custom_seas` was set on line 84, but the inconsistent pattern is confusing. Use `self.custom_seas` for both conditions to match the instance-attribute convention.

**Fix:** Use `self.custom_seas` consistently.

```python
if self.custom_seas is not None and len(self.custom_seas) > 0:
```

### IN-04: TSTCC `__all__` placed after imports instead of at module top

**File:** `src/tscollection/models/convolutional/standard/tstcc/model.py:15`

**Issue:** The `__all__` declaration appears on line 15, after all imports. Other reviewed files (MCL, S2V, TimeNet, TST, TimeVAE) place `__all__` as the first or second statement in the module. While functionally equivalent, inconsistent placement makes the codebase harder to scan.

**Fix:** Move `__all__` to the top of the file, before imports.

---

_Reviewed: 2026-06-11T21:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
