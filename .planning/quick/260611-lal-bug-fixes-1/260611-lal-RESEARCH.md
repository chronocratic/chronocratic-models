# Phase Quick-Task: Fix Code Review Findings — Research

**Researched:** 2026-06-11
**Domain:** PyTorch model bug fixes across 6 models (MCL, S2V, TimeNet, TSTCC, TST, TimeVAE)
**Confidence:** HIGH — all claims verified against actual source code

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **TimeVAE Encoder Padding:** Not a bug — Keras `padding="same"` = `ceil(inp/stride)`, matches PyTorch `padding=1` for kernel=3, stride=2
- **TimeVAE Decoder Output Padding:** Not a bug — `output_padding=1` in `ResidualConnection` correctly compensates for PyTorch transposed conv formula
- **TST Loss Normalization:** Not a bug — `MaskedMSELoss(reduction='none')` returns only active elements; `len(per_element_loss)` counts active
- **Fix Approach:** Derive runtime values (device, batch_size) from input tensors, not init-time params. Match original source behavior exactly where correct. Use existing shared infrastructure (BasicEncodingMixin) where applicable.

### Excluded from Scope
- TimeNet decoder dropout asymmetry (model.py:63-67)
- Missing trailing encoder dropout (model.py:53-56)

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

## Summary

This quick-task fixes 17 verified bugs across 6 models, found via deep source review of PR #10 comparing local code against original reference repos. Three are critical runtime crashes (C1-C4), eight are correctness/non-determinism issues (I1-I8), three are risk-level (R1-R3), and four are style/consistency (S1-S4).

**Primary recommendation:** Group fixes into 3 logical tasks — Critical Crashes, Correctness, and Style/Cleanup — each independently mergeable.

**Files touched:** 13 source files across `mcl/`, `series2vec/`, `timenet/`, `tstcc/`, `tst/`, `timevae/`. No existing tests reference any of these models (tests only cover TS2Vec, CoST, AutoTCL), so no test breakage risk.

## Task Grouping

### Task A — Critical Crash Fixes (C1-C4)

| Bug | File(s) | Fix |
|-----|---------|-----|
| C1: MCL `MixUpLoss` | `losses.py`, `model.py`, `config.py` | Derive `batch_size` from `z_aug.shape[0]`, `device` from `z_aug.device` in `forward()`. Remove params from loss, model, and config. |
| C2: S2V `warmup` kwarg | `model.py`, `config.py` | Remove `warmup` from `configure_optimizers()` kwargs dict. Remove `warmup` from model `__init__` and config. |
| C3: S2V `.squeeze(0)` | `network.py` | Replace `.squeeze(0)` with `.squeeze(-1)` on lines 109, 110. GAP outputs `(B, D, 1)`, so `dim=-1` is the intended dimension. |
| C4: TimeNet `input_size=1` | `model.py`, `config.py` | Add `feat_dim: int` param to `TimeNet.__init__()`. Use `self._feat_dim` in GRUWrapper, final Linear, and `_postprocess`. |

**Dependencies:** None. C1 touches only mcl/. C2 touches only series2vec/. C3 touches only network.py. C4 touches only timenet/.

**C4 detail:** Current code: `GRUWrapper(1, ...)` hardcodes input_size=1. The model accepts `(batch, time, features)` input. Fix adds `feat_dim` to both config and model. The final `nn.Linear` in decoder also needs updating from `nn.Linear(self.hidden_dims, 1)` to `nn.Linear(self.hidden_dims, self._feat_dim)`. Add `_postprocess` that selects final timestep: `output[:, -1, :]`.

**C1 detail:** Replace init params with runtime derivation:
```python
def forward(self, z_aug, z_1, z_2, lam):
    batch_size = z_aug.shape[0]
    device = z_aug.device
    # ... rest uses batch_size, device
```
Remove `device` and `batch_size` from `MCLModelParameters` config. Old configs using `**vars(params)` will crash on unknown params — this is acceptable as these are bugs, not features.

### Task B — Correctness Fixes (I1-I8, R1, R3)

| Bug | File(s) | Fix |
|-----|---------|-----|
| I1: S2V optimizer default | `model.py`, `config.py` | Change default from `'Adam'` to `'RAdam'`. |
| I2: TSTCC val no_grad | `model.py` | Wrap `_compute_loss(batch)` in `torch.no_grad()` in `validation_step`. |
| I3: TSTCC seq_len guard | `temporal_contrast.py` | Add `if seq_len <= self.timestep: raise ValueError(...)` guard before `torch.randint`. |
| I4: TST internal access | `ts_transformer.py`, `model.py` | Add `encode_representations()` public method to `TSTransformerEncoder`. Update `TST.get_representations()` to use it. |
| I5: S2V filter randomness | `filters.py`, `model.py` | Add `training: bool = True` param to `filter_frequencies()`. Pass `training=self.training` from `_calculate_loss()`. Skip random branch in non-training mode. |
| I6: TimeVAE loss /n | `vae_base.py` | Remove `/ n` division on line 54. Return raw `loss, recon_loss, kl_loss`. |
| I7: TimeVAE predict() | `vae_base.py` | Replace `self.eval()` with `was_training = self.training; self.eval()`. Restore in `finally: self.train(was_training)`. |
| I8: TST gradient clip | `model.py` | Remove hardcoded `max_norm=4.0`. Use `gradient_clip_val` param or delegate to Lightning Trainer. |
| R1: TimeVAE stale layers | `model.py` | Add `del self.layers` after `nn.Sequential(*self.layers)` in `TimeVAEEncoder.__init__`. |
| R3: TimeNet postprocess | `model.py` | Add `_postprocess` method that selects final timestep: `output[:, -1, :]`. |

**Dependencies:**
- I1 touches same model.py as C2 — can be combined or done separately.
- I2 also covers R2 (augmentations.py `_should_apply()` fires in eval) — both resolved by the no_grad wrapper.
- R3 must be done with C4 (same file, same model).

**I2 detail:** Current `validation_step` calls `self._compute_loss(batch)` which in SELF_SUPERVISED mode fires `self._augmentation.augment(data)`. Fix:
```python
def validation_step(self, batch, _batch_idx):
    with torch.no_grad():
        loss = self._compute_loss(batch)
    self.log('val_loss', loss, ...)
    return loss
```

**I4 detail:** Current code reaches into `self._encoder.project_inp`, `self._encoder.pos_enc`, `self._encoder.transformer_encoder`, `self._encoder.act`, `self._encoder.dropout1`. Add to `TSTransformerEncoder`:
```python
def encode_representations(self, x: Tensor, padding_masks: Tensor) -> Tensor:
    """Return transformer representations before output_layer."""
    inp = x.permute(1, 0, 2)  # (seq, batch, feat)
    inp = self.project_inp(inp) * math.sqrt(self.d_model)
    inp = self.pos_enc(inp)
    output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
    output = self.act(output)
    output = output.permute(1, 0, 2)  # (batch, seq, d_model)
    return self.dropout1(output)
```
Then `TST.get_representations()` simply delegates: `return self._encoder.encode_representations(x, padding_masks)`.

**I5 detail:** `filter_frequencies()` calls `torch.rand(()) < LOWPASS_PROBABILITY` unconditionally. In `_calculate_loss`, add `training=self.training` to the call. Inside `filter_frequencies`, use deterministic lowpass when `training=False`:
```python
def filter_frequencies(data, lowpass_cutoff=40.0, highpass_cutoff=0.5, training=True):
    fft_results = torch.stack([apply_fft(s) for s in data])
    if training and torch.rand(()) < LOWPASS_PROBABILITY:
        return lowpass(...)
    return highpass(...)  # deterministic in val
```

**I6 detail:** Current `_step` divides by `n = x.size(0)`. The `loss_function()` already computes sums. The `/ n` makes the loss ~1/B smaller than the original source. Simply remove the division and return raw values.

**I8 detail:** Current `configure_gradient_clipping` ignores all Lightning parameters:
```python
def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    del optimizer, gradient_clip_val, gradient_clip_algorithm
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=4.0)
```
Fix: either respect `gradient_clip_val` or remove this method entirely and set `gradient_clip_val=4.0` on the Trainer. The cleanest fix is to use the param:
```python
def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    if gradient_clip_val is None:
        gradient_clip_val = 4.0  # default from original source
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_val)
```

**R1 detail:** In `TimeVAEEncoder.__init__`, `self.layers: list[nn.Module] = []` is a plain Python list. After `self.encoder = nn.Sequential(*self.layers)`, the list is no longer needed and holds stale refs. Add `del self.layers` after line 41. Also in `_get_last_dense_dim`, the loop iterates `for conv in self.layers` — but this is called BEFORE the Sequential, so it's fine. After deletion, `_get_last_dense_dim` no longer needs the raw list reference.

**R3 detail:** `TimeNet._get_encoder()` returns `self.encoder`. The GRU encoder outputs `(B, T, D)`. The BasicEncodingMixin needs a flat representation. Add `_postprocess` to select the final timestep, matching how the decoder reverses the sequence:
```python
def _postprocess(self, output):
    return output[:, -1, :]  # Select last timestep
```

### Task C — Style / Consistency (S1-S4)

| Bug | File(s) | Fix |
|-----|---------|-----|
| S1: Public attrs | `mcl/model.py`, `series2vec/model.py`, `timenet/model.py` | `self.alpha` → `self._alpha`, `self.learning_rate` → `self._learning_rate`, etc. |
| S2: Duplicate Seasonality | `timevae/model.py` | Remove `Seasonality = tuple[int, int]` line 13. Import from `layers.general` (already imported). |
| S3: Future annotations | `tstcc/encoder.py`, `tstcc/losses.py`, `tstcc/model.py`, `tstcc/temporal_contrast.py` | Remove `from __future__ import annotations`. |
| S4: Missing `__all__` | `tstcc/encoder.py`, `series2vec/filters.py`, `series2vec/losses.py`, `timevae/model.py`, `timevae/vae_base.py`, `tst/loss.py`, `tst/ts_transformer.py` | Add `__all__` declarations. |

**Dependencies:** None. S2 note — `Seasonality` IS already imported from `layers.general` on line 7, so the local alias on line 13 is redundant.

**Gotcha on S1:** Changing `self.learning_rate` to `self._learning_rate` in `FCN.__init__` also requires updating `configure_optimizers()` which references `self.learning_rate`. Same pattern for other models.

## File Change Summary

| File | Bugs | Complexity |
|------|------|------------|
| `mcl/losses.py` | C1 | Low — derive batch/device from tensor |
| `mcl/model.py` | C1, S1 | Low — remove params, underscore attrs |
| `mcl/config.py` | C1 | Low — remove device, batch_size fields |
| `series2vec/model.py` | C2, I1, I5, S1 | Medium — 4 fixes, some overlapping |
| `series2vec/network.py` | C3 | Low — squeeze dim |
| `series2vec/config.py` | C2, I1 | Low — remove warmup, change default |
| `series2vec/filters.py` | I5, S4 | Low — add training param, __all__ |
| `timenet/model.py` | C4, R3, S1 | Medium — add feat_dim, _postprocess |
| `timenet/config.py` | C4 | Low — add feat_dim field |
| `tstcc/model.py` | I2, S3 | Low — no_grad, remove future import |
| `tstcc/temporal_contrast.py` | I3, S3 | Low — add guard, remove future import |
| `tstcc/encoder.py` | S3, S4 | Low — future import, __all__ |
| `tstcc/losses.py` | S4 | Low — __all__ |
| `tst/ts_transformer.py` | I4, S4 | Medium — new public method, __all__ |
| `tst/model.py` | I8, S4 | Low — gradient clip fix |
| `tst/loss.py` | S4 | Low — __all__ |
| `timevae/model.py` | R1, S2 | Low — del layers, remove alias |
| `timevae/vae_base.py` | I6, I7, S4 | Low — remove /n, was_training flag |

## Verification Notes

### Existing Tests
No existing tests reference MCL, S2V, TSTCC, TimeVAE, TimeNet, or TST. The only test coverage is for TS2Vec, CoST, and AutoTCL (see `test_from_config.py`, `test_smoke.py`). **No test breakage risk.**

### Config Backward Compatibility
- **C1:** Removing `device` and `batch_size` from `MCLModelParameters` will crash `FCN(**vars(old_config))` because FCN no longer accepts those params. **This is acceptable** — these params were buggy (fixed at init, crash on device move).
- **C2:** Removing `warmup` from `Series2VecModelParameters` similarly breaks `**vars(old_config)`. **Acceptable** — `warmup` was never consumed by AdamW.
- **C4:** Adding `feat_dim` to `TimeNetModelParameters` means old configs missing it will need a default. Use `feat_dim: int = 1` to maintain backward compat with univariate data.

### Runtime Impact
- C1: MixUpLoss no longer allocates eye matrices at init. Memory savings on large batch sizes.
- I6: VAE loss magnitudes will be ~Bx larger (B = batch_size). If users have hardcoded loss thresholds, they may need adjustment.
- I8: If users do NOT set `gradient_clip_val` on Trainer, TST defaults to `max_norm=4.0` (same as before). Explicit Trainer values now respected.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `RAdam` is the correct default for S2V per original source | I1 | LOW — RAdam was the source default; changing back aligns behavior |
| A2 | High-pass filter is the correct deterministic choice for validation in S2V | I5 | LOW — either high or low pass is fine; just needs to be deterministic |
| A3 | `max_norm=4.0` as default in TST gradient clipping matches original | I8 | LOW — original code hardcoded 4.0 |
| A4 | Selecting `output[:, -1, :]` is the correct postprocess for TimeNet encoding | R3 | MEDIUM — depends on whether TimeNet was designed for last-timestep pooling; decoder uses `torch.flip` which suggests final-timestep focus |

## Open Questions

1. **S4 `__all__` contents:** The exact exported names for each file should be reviewed against the module's public API. Encoder files export class names; loss files export loss classes. Filters.py exports `filter_frequencies` and the constants.

2. **TSTCC `from __future__ import annotations`:** Removing this from encoder.py and temporal_contrast.py — verify that all type hints are forward-reference-safe without it (no unresolved class names in annotations). Current code uses `torch.Tensor` and `torch.BoolTensor` which are resolved at import time, so this should be safe.
