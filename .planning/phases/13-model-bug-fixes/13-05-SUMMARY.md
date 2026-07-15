---
phase: 13-model-bug-fixes
plan: 5
subsystem: recurrent-nan-defense
tags: [nan-defense, zero-fill-padding, masked-loss, RecurrentAE, TimeNet, reconstruction-models]
dependency_graph:
  requires:
    - 13-01 (zero_fill_padding, masked_reconstruction_loss in utils.py)
  provides:
    - RecurrentAE NaN-safe training/validation via _compute_masked_loss
    - TimeNet NaN-safe training/validation via _compute_masked_loss
  affects: []
tech_stack:
  added: []
  patterns:
    - zero_fill_padding before forward pass (prevents NaN propagation)
    - masked_reconstruction_loss with per-element reduction=none
    - isinstance dispatch for MSE vs L1 loss selection (RecurrentAE)
key_files:
  created:
    - tests/unit/test_nan_padded_recurrent.py
  modified:
    - src/chronocratic/models/recurrent/recurrentae/model.py
    - src/chronocratic/models/recurrent/timenet/model.py
decisions:
  - "Use shared _compute_masked_loss helper method in both models to avoid duplication"
  - "RecurrentAE dispatches F.mse_loss vs F.l1_loss via isinstance(self.loss_fn, nn.MSELoss)"
  - "TimeNet uses inline (output - x) ** 2 for per-element MSE (consistent with existing nn.MSELoss)"
metrics:
  duration: 8min
  completed: "2026-07-15"
status: complete
---

# Phase 13 Plan 5: RecurrentAE and TimeNet NaN Defense Summary

Wired NaN defense (zero_fill_padding + masked_reconstruction_loss) into RecurrentAE and TimeNet reconstruction models, preventing forward-pass crashes and poisoned gradients on variable-length UEA series padded with trailing NaN timesteps.

## What Was Built

Two reconstruction models updated to handle NaN-padded training data:

1. **RecurrentAE** (`_compute_masked_loss`) — zero-fills NaN timesteps before the forward pass, dispatches per-element loss computation between `F.mse_loss` and `F.l1_loss` based on the configured loss type, and applies `masked_reconstruction_loss` to exclude padded timesteps from the scalar loss.

2. **TimeNet** (`_compute_masked_loss`) — same pattern: zero-fill, per-element MSE (`(output - x) ** 2`), masked mean. Both `training_step` and `validation_step` use the shared helper.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Failing tests for NaN-padded RecurrentAE and TimeNet | a0a617b | tests/unit/test_nan_padded_recurrent.py |
| 2 | NaN defense for RecurrentAE and TimeNet | 143cadb | recurrentae/model.py, timenet/model.py |

## Tests

10 tests created, all passing:
- 4 × RecurrentAE (parametrized over mse/mae: training, validation + all-NaN + clean batch)
- 4 × TimeNet (training, validation + all-NaN + clean batch)
- Regression: 8 existing batch1-gradient tests (TimeNet + RecurrentAE) pass

## Deviations from Plan

None — plan executed exactly as written.

## Threat Surface

| Threat | Status |
|--------|--------|
| T-13-05-01 (NaN→forward crash, RecurrentAE/TimeNet) | Mitigated — zero_fill_padding + masked_reconstruction_loss |
