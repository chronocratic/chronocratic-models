# Quick Task 260611-lal: Fix Code Review Findings - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Task Boundary

Fix all bugs from PR #10 deep source review (combined findings from two review passes). All fixes based on comparison with original source repos.

**Excluded:** TimeNet decoder dropout asymmetry (model.py:63-67) and missing trailing encoder dropout (model.py:53-56).
</domain>

<decisions>
## Implementation Decisions

### TimeVAE Encoder Padding
- **Not a bug** — Keras `padding="same"` = `ceil(inp/stride)`, matches PyTorch `padding=1` for kernel=3, stride=2

### TimeVAE Decoder Output Padding
- **Not a bug** — `output_padding=1` in `ResidualConnection` correctly compensates for PyTorch transposed conv formula

### TST Loss Normalization
- **Not a bug** — `MaskedMSELoss(reduction='none')` returns only active elements; `len(per_element_loss)` counts active

### Fix Approach
- Derive runtime values (device, batch_size) from input tensors, not init-time params
- Match original source behavior exactly where it was correct
- Use existing shared infrastructure (BasicEncodingMixin) where applicable

</decisions>

<specifics>
## Complete Bug List

### 🔴 Critical — Will Crash at Runtime

| # | Model | File:Line | Bug |
|---|-------|-----------|-----|
| C1 | MCL | `losses.py:10-15` | `self.device` and `self.batch_size` fixed at init — crash on device move or last-batch size mismatch. Fix: derive from input tensor in `forward()` |
| C2 | S2V | `model.py:154-156` | `warmup` kwarg passed to `torch.optim.AdamW` — TypeError. Fix: remove warmup from optimizer, delete warmup param from model+config |
| C3 | S2V | `network.py:109-110` | `.squeeze(0)` relies on GAP always outputting 1 — fragile. Fix: `.squeeze()` without dim arg |
| C4 | TimeNet | `model.py:52` | `GRUWrapper(1, ...)` hardcodes `input_size=1` — crashes on multivariate. Fix: add `feat_dim` param, use `self._feat_dim` in encoder+decoder, final Linear projects to `feat_dim` |

### 🟡 Important — Correctness / Non-Determinism

| # | Model | File:Line | Bug |
|---|-------|-----------|-----|
| I1 | S2V | `model.py:60` | Default optimizer `Adam` vs source `RAdam`. Fix: change to `'RAdam'` |
| I2 | TSTCC | `model.py:173-181` | `validation_step` has no `torch.no_grad()` — gradients computed, random augmentations fire. Fix: add no_grad + skip augmentation in val |
| I3 | TSTCC | `temporal_contrast.py:168` | `torch.randint(seq_len - self.timestep)` crashes when seq_len <= timestep. Fix: add ValueError guard |
| I4 | TST | `model.py:106-114` | `get_representations()` reaches into `_encoder` internals (`project_inp`, `pos_enc`, `transformer_encoder`). Fix: add public `encode_representations()` method to TSTransformerEncoder |
| I5 | S2V | `filters.py:13` | `torch.rand(())` fires in validation — non-reproducible loss. Fix: pass `mode` parameter, use deterministic in val |
| I6 | TimeVAE | `vae_base.py:54` | Loss divided by batch size `n`. Source unnormalized. Fix: remove `/ n` |
| I7 | TimeVAE | `vae_base.py:76-83` | `predict()` calls `self.eval()` without restoring training mode. Fix: use `was_training` flag |
| I8 | TST | `model.py:180` | Hardcoded `max_norm=4.0` ignores Lightning's `gradient_clip_val`. Fix: respect param or delegate to Trainer |

### 🟡 Risk

| # | Model | File:Line | Bug |
|---|-------|-----------|-----|
| R1 | TimeVAE | `model.py:25` | `self.layers` (raw list) persists after `nn.Sequential` — stale device refs. Fix: `del self.layers` after Sequential creation |
| R2 | TSTCC | `augmentations.py:44` | `_should_apply()` fires `torch.rand()` in eval mode. Fix: skip augmentation in val (covered by I2) |
| R3 | TimeNet | `model.py:76-78` | `encode()` returns `(B,T,D)` — missing `_postprocess` to pool time dim. Fix: add `_postprocess` selecting final timestep |

### ℹ️ Style / Consistency

| # | Model | File | Fix |
|---|-------|------|-----|
| S1 | All | `mcl/model.py:30-31`, `series2vec/model.py:67-72`, `timenet/model.py:43-48` | Public attrs → underscore prefix (`_alpha`, `_learning_rate`, etc.) |
| S2 | TimeVAE | `model.py:13` | Duplicate `Seasonality` alias — import from `layers.general` instead |
| S3 | TSTCC | `encoder.py`, `losses.py`, `model.py`, `temporal_contrast.py` | Remove unnecessary `from __future__ import annotations` |
| S4 | Various | Multiple files | Add `__all__` to encoder.py, filters.py, losses.py, timevae/model.py, timevae/vae_base.py, tst/loss.py, tst/ts_transformer.py |

</specifics>

<canonical_refs>
## Source Repos
- MCL: https://github.com/Wickstrom/MixupContrastiveLearning
- TS-TCC: https://github.com/emadeldeen24/TS-TCC
- Series2Vec: https://github.com/Navidfoumani/Series2Vec
- TimeVAE: https://github.com/abudesai/timeVAE
- TimeNet: https://github.com/paudan/TimeNet
- TST: https://github.com/gzerveas/mvts_transformer

## Combined Review Sources
- Deep source review: 6 agents compared local vs original repos (May 2026)
- Independent agent review: bug-fixes.2.md cross-verified same findings
</canonical_refs>
