---
phase: 09-fixes-and-updates
plan: 04
subsystem: models/encoding-mixin
tags:
  - refactor
  - 2-hook-contract
  - TST
  - Series2Vec
  - encoding
  - tdd
status: complete
dependency_graph:
  requires: ["09-02"]
  provides:
    - "TST 2-hook contract (_get_encoder -> nn.Module, _encode_batch)"
    - "Series2Vec 2-hook contract (_get_encoder -> nn.Module, _encode_batch)"
    - "Deleted hooks regression test for all 7 BasicEncodingMixin models"
  affects:
    - "tests/unit/test_complex_models_2hook.py"
    - "tests/unit/test_no_deleted_hooks.py"
    - "src/chronocratic/models/utils.py" (docstring)
key_decisions:
  - "TST._get_encoder returns self._encoder (TSTransformerEncoder), not self.get_representations (bound method)"
  - "Series2Vec._get_encoder returns self.network (Series2VecNetwork), not self.network.encode (bound method)"
  - "TST._encode_batch builds all-true padding mask on batch_x.device, calls encoder.encode_representations"
  - "Series2Vec._encode_batch calls encoder.encode(batch_x).unsqueeze(1) — folds old _postprocess"
  - "utils.py pool_feature_map docstring updated from _postprocess to _encode_batch"
metrics:
  duration_minutes: 30
  tasks_completed: 2
  tests_added: 20
  tests_passing: 634
  files_created: 2
  files_modified: 3
completed_date: "2026-06-25"
tech_stack:
  added: []
  patterns:
    - "TDD RED/GREEN for both tasks"
    - "AST-based regression test for deleted hooks"
key_files:
  created:
    - "tests/unit/test_complex_models_2hook.py"
    - "tests/unit/test_no_deleted_hooks.py"
  modified:
    - "src/chronocratic/models/transformer/tst/model.py"
    - "src/chronocratic/models/convolutional/standard/series2vec/model.py"
    - "src/chronocratic/models/utils.py"
---

# Phase 9 Plan 04: Complex Models 2-Hook Refactor Summary

Refactor TST and Series2Vec (the 2 complex models that returned bound methods from `_get_encoder()`) to the clean 2-hook encoding contract, migrate deleted-hook tests, and verify the full test suite.

## What Was Built

### TST Refactor

`TST._get_encoder()` now returns `self._encoder` (the `TSTransformerEncoder` nn.Module) instead of `self.get_representations` (a bound method). The old `_get_encoder_module` and `_prepare_inputs` hooks were deleted. New `_encode_batch` builds an all-true padding mask on `batch_x.device` and calls `encoder.encode_representations(batch_x, padding_masks)`.

### Series2Vec Refactor

`Series2Vec._get_encoder()` now returns `self.network` (the `Series2VecNetwork` nn.Module) instead of `self.network.encode` (a bound method). The old `_get_encoder_module` and `_postprocess` hooks were deleted. New `_encode_batch` calls `encoder.encode(batch_x).unsqueeze(1)`, folding the previous `_postprocess` logic.

### Regression Tests

- `test_complex_models_2hook.py`: 13 tests verify TST and Series2Vec follow the 2-hook contract (encoder is nn.Module, output shapes unchanged, old hooks removed).
- `test_no_deleted_hooks.py`: 7 tests (one per BasicEncodingMixin model) use AST parsing to verify no model class defines `_get_encoder_module`, `_prepare_inputs`, or `_postprocess`.

### Docstring Update

`pool_feature_map` in `utils.py` updated docstring reference from `_postprocess` to `_encode_batch`.

## Deviations from Plan

None - plan executed exactly as written. Both tasks followed TDD RED/GREEN. Test migration for `test_recurrent_autoencoder.py` was already completed in plan 09-02 (RecurrentAutoEncoder is a simple model).

## Verification Results

- 634 tests pass, 2 skipped
- `grep` confirms no deleted hooks (`_get_encoder_module`, `_prepare_inputs`, `_postprocess`) remain in source code (excluding comments)
- `ruff check` and `ruff format` clean on all modified files
- TST `encode()` produces `(B, seq_len, hidden_dims)` output (unchanged)
- Series2Vec `encode()` produces `(B, 1, 2*representation_dims)` output (unchanged)

## Threat Surface

Threats T-09-04-01 (TST), T-09-04-02 (Series2Vec), T-09-04-03 (test migration) from the plan's threat model are all mitigated:
- TST._get_encoder returns nn.Module, preventing confusion about eval/train state toggle target
- Series2Vec._get_encoder returns nn.Module, same guarantee
- test_encode_batch_returns_last_timestep (already migrated in 09-02) validates _encode_batch contract

## Known Stubs

None.
