---
name: Fix Code Review Findings from PR #10
description: Fixed 17 bugs across 6 models (MCL, Series2Vec, TS-TCC, TimeVAE, TimeNet, TST) based on deep source review against original repos. Excluded TimeNet decoder dropout bugs per user decision.
status: complete
date: 2026-06-11
commits:
  - sha: 406a35d
    message: "fix(260611-lal-1): critical crash fixes (C1-C4)"
    files: 8
  - sha: 4d2c739
    message: "fix(260611-lal-2): correctness, risk, and style fixes (I1-I8, R1-R3, S1-S4)"
    files: 11
---

# Quick Task 260611-lal: Fix Code Review Findings

**Status:** Complete
**Date:** 2026-06-11
**Commits:** 406a35d, 4d2c739

## Tasks Completed

### Task 1: Critical Crash Fixes (C1-C4)

| Bug | Model | Fix |
|-----|-------|-----|
| C1 | MCL | `MixUpLoss` derives batch_size/device from input tensor at runtime, not init-time params |
| C2 | S2V | Removed `warmup` kwarg from AdamW (crashed at optimizer creation) + removed warmup from model+config |
| C3 | S2V | Replaced `.squeeze(0)` with explicit `out[:, 0, :]` indexing (handles batch_size=1 safely) |
| C4 | TimeNet | Added `feat_dim` param to TimeNet; encoder/decoder use `self._feat_dim` instead of hardcoded 1 |

### Task 2: Correctness and Risk Fixes (I1-I8, R1-R3)

| Bug | Model | Fix |
|-----|-------|-----|
| I1 | S2V | Changed default optimizer from `'Adam'` to `'RAdam'` (matches source) |
| I2 | TSTCC | `validation_step` wrapped in `torch.no_grad()`; augmentation skipped in val |
| I3 | TSTCC | Added `ValueError` guard in `TemporalContrast` when `seq_len <= timestep` |
| I4 | TST | Added public `encode_representations()` method to `TSTransformerEncoder`; `get_representations()` delegates to it |
| I5 | S2V | `filter_frequencies()` accepts `training` param; deterministic highpass in validation |
| I6 | TimeVAE | Removed `/ n` division in `_step()` loss (source is unnormalized) |
| I7 | TimeVAE | `predict()` restores training mode via `was_training` flag |
| I8 | TST | `configure_gradient_clipping()` respects `gradient_clip_val` with fallback to 4.0 |
| R1 | TimeVAE | Added `del self.layers` after `nn.Sequential` creation |
| R3 | TimeNet | Added `_postprocess()` selecting final timestep `output[:, -1, :]` |

### Task 3: Style and Consistency Fixes (S1-S4)

| Bug | Fix |
|-----|-----|
| S1 | Public attrs use underscore prefix (`_alpha`, `_learning_rate`, etc.) in MCL, S2V, TimeNet |
| S2 | Removed duplicate `Seasonality` alias from `timevae/model.py` (imported from `layers.general`) |
| S3 | Removed unnecessary `from __future__ import annotations` from 4 TSTCC files |
| S4 | Added `__all__` declarations to 7 modules missing them |

## Verification

- **ruff check:** PASSED (0 issues)
- **ty check:** PASSED (0 issues)
- **pytest:** 181 passed, 0 failed

## Files Modified (19 total)

| File | Bugs |
|------|------|
| `mcl/config.py` | C1 |
| `mcl/losses.py` | C1 |
| `mcl/model.py` | C1, S1 |
| `series2vec/config.py` | C2, I1 |
| `series2vec/filters.py` | I5 |
| `series2vec/losses.py` | S4 |
| `series2vec/model.py` | C2, I1, I5, S1 |
| `series2vec/network.py` | C3 |
| `timenet/config.py` | C4 |
| `timenet/model.py` | C4, R3, S1 |
| `tstcc/encoder.py` | S3, S4 |
| `tstcc/losses.py` | S3, S4 |
| `tstcc/model.py` | I2, S3 |
| `tstcc/temporal_contrast.py` | I3, S3 |
| `timevae/model.py` | R1, S2 |
| `timevae/vae_base.py` | I6, I7, S4 |
| `tst/model.py` | I4, I8 |
| `tst/ts_transformer.py` | I4, S4 |
| `tst/loss.py` | S4 |
