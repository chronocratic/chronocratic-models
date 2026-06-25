---
phase: 09-fixes-and-updates
plan: 02
subsystem: models/encoding-mixin
tags:
  - refactor
  - mixin
  - gradient-enabled
  - encoding
  - tdd
status: complete
dependency_graph:
  requires: ["09-01"]
  provides:
    - "2-hook BasicEncodingMixin contract (_get_encoder, _encode_batch)"
    - "gradient_enabled support on both mixin families"
  affects:
    - "TimeVAE, TimeNet, RecurrentAutoEncoder, FCN, TSTCC"
    - "supervised/_adapters.py"
key_decisions:
  - "Collapse 4-hook mixin to 2-hook: _get_encoder -> nn.Module, _encode_batch(encoder, batch_x) -> Tensor"
  - "Delete _get_encoder_module, _prepare_inputs, _postprocess from BasicEncodingMixin"
  - "Add gradient_enabled: bool = False keyword-only to encode() on both mixin families"
  - "Use nullcontext() (not enable_grad) when gradient_enabled=True"
  - "Encoder stays in eval() regardless of gradient_enabled"
  - "pin_memory=not gradient_enabled in DataLoader"
metrics:
  duration_minutes: 45
  tasks_completed: 3
  tests_added: 42
  tests_passing: 126
  files_created: 3
  files_modified: 8
completed_date: "2026-06-25"
tech_stack:
  added: []
  patterns:
    - "TDD RED/GREEN for each task"
    - "nullcontext for gradient toggle"
key_files:
  created:
    - "tests/unit/test_encoding_mixin.py"
    - "tests/unit/test_dilated_encoding_mixin.py"
    - "tests/unit/test_model_encode_batch.py"
  modified:
    - "src/chronocratic/models/_mixin/encoding.py"
    - "src/chronocratic/models/convolutional/dilated/_mixin/encoding.py"
    - "src/chronocratic/models/generative/timevae/model.py"
    - "src/chronocratic/models/recurrent/timenet/model.py"
    - "src/chronocratic/models/recurrent/recurrentae/model.py"
    - "src/chronocratic/models/convolutional/standard/mcl/model.py"
    - "src/chronocratic/models/convolutional/standard/tstcc/model.py"
    - "src/chronocratic/models/supervised/_adapters.py"
---

# Phase 9 Plan 02: Encoding Mixin Refactor Summary

Collapse `BasicEncodingMixin` from a 4-hook contract to a 2-hook contract (`_get_encoder` -> nn.Module, `_encode_batch`) and add `gradient_enabled: bool = False` to `encode()` on both mixin families, then refactor 5 simple models.

## What Was Built

### BasicEncodingMixin Rewrite

The mixin was reduced from 4 hooks (`_get_encoder`, `_get_encoder_module`, `_prepare_inputs`, `_postprocess`) to 2 hooks (`_get_encoder`, `_encode_batch`). The `encode()` method gained a `gradient_enabled` keyword-only parameter that toggles between `torch.inference_mode()` (default, severs graph) and `nullcontext()` (preserves autograd for adversarial attacks).

### Dilated Mixin Gradient Support

`BaseEncodingMixin.encode()` gained the same `gradient_enabled` parameter, keeping the hook surface (`_get_encoder`, `_get_eval_method`, `_get_slice`) unchanged.

### 5 Simple Model Refactors

- **TimeVAE**: `_postprocess(output) -> output[0]` became `_encode_batch(encoder, batch_x) -> encoder(batch_x)[0]`
- **TimeNet**: `_postprocess(output) -> output[:, -1, :]` became `_encode_batch(encoder, batch_x) -> encoder(batch_x)[:, -1, :]`
- **RecurrentAutoEncoder**: Same pattern as TimeNet
- **FCN**: `_postprocess(output) -> output.unsqueeze(1)` became `_encode_batch(encoder, batch_x) -> encoder(batch_x).unsqueeze(1)`
- **TSTCC**: Deleted `_prepare_inputs` (float cast) and `_postprocess` (pooling); folded both into `_encode_batch(encoder, batch_x) -> pool_feature_map(encoder(batch_x.float()))`

### Test Coverage

42 new tests across 3 test files verify:
- Mixin has `_encode_batch`, lacks deleted hooks
- `gradient_enabled=False` severs graph (default)
- `gradient_enabled=True` preserves autograd for backprop
- Encoder train/eval state restoration via try/finally
- Output shape correctness for all 5 models

## Deviations from Plan

None - plan executed exactly as written. The TDD cycle (RED/GREEN) was followed for each of the 3 tasks, with failing tests committed before implementation.

## Verification Results

- 126 tests pass (84 existing + 42 new)
- Structural grep confirms `_encode_batch` present in mixin, deleted hooks absent
- `gradient_enabled` present in both mixin families
- All 5 models use `_encode_batch`; no references to `_postprocess`, `_prepare_inputs`, or `_get_encoder_module` in model code

## Threat Surface

No new threat surface introduced. `gradient_enabled=False` is the default, preserving existing behavior. The `nullcontext()` choice (over `torch.enable_grad()`) respects ambient autograd state rather than forcing grad on.

## Known Stubs

None.
