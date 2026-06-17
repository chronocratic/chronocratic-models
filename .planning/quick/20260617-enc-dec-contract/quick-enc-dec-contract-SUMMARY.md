---
phase: quick
plan: enc-dec-contract
subsystem: models
tags:
  - encoder-decoder
  - protocol
  - contract
  - refactoring
dependency_graph:
  requires: []
  provides:
    - HasEncoder protocol
    - HasDecoder protocol
    - Uniform .encoder property on all 9 models
    - Uniform .decoder property on TimeVAE, TimeNet
  affects:
    - src/chronocratic/models/generative/timevae/vae_base.py
    - src/chronocratic/models/generative/timevae/model.py
    - src/chronocratic/models/recurrent/timenet/model.py
    - src/chronocratic/models/convolutional/standard/mcl/model.py
    - src/chronocratic/models/transformer/tst/model.py
    - src/chronocratic/models/convolutional/standard/tstcc/model.py
    - src/chronocratic/models/convolutional/dilated/autotcl/model.py
    - src/chronocratic/models/convolutional/dilated/cost/model.py
    - src/chronocratic/models/convolutional/standard/series2vec/model.py
tech_stack:
  added: []
  patterns:
    - @runtime_checkable Protocol
    - @property for public access
key_files:
  created:
    - src/chronocratic/models/protocols.py
    - tests/test_encoder_decoder_contract.py
  modified:
    - src/chronocratic/models/__init__.py
    - src/chronocratic/models/generative/timevae/vae_base.py
    - src/chronocratic/models/generative/timevae/model.py
    - src/chronocratic/models/recurrent/timenet/model.py
    - src/chronocratic/models/convolutional/standard/mcl/model.py
    - src/chronocratic/models/transformer/tst/model.py
    - src/chronocratic/models/convolutional/standard/tstcc/model.py
    - src/chronocratic/models/convolutional/dilated/autotcl/model.py
    - src/chronocratic/models/convolutional/dilated/cost/model.py
    - src/chronocratic/models/convolutional/standard/series2vec/model.py
decisions:
  - Protocol over ABC for runtime_checkable isinstance support
  - Uniform _encoder + @property pattern across all models
  - Breaking change on checkpoint keys accepted (no migration shim)
  - CoST .encoder returns query_encoder (inference-time encoder)
  - Series2Vec .encoder returns network (multi-purpose net exception)
  - TimeVAEEncoder inner class self.encoder preserved (scoped, not model-level)
metrics:
  duration_seconds: 3600
  completed_date: "2026-06-17"
---

# Phase Quick Plan enc-dec-contract Summary

Uniform, programmatically-checkable contract for encoder/decoder extraction across all 9 models via `@runtime_checkable` Protocols. Every model now exposes a read-only `.encoder` property returning an `nn.Module`. TimeVAE and TimeNet additionally expose `.decoder`.

## Commits

| Commit | Message |
|--------|---------|
| 8833f57 | feat: add HasEncoder/HasDecoder protocols and exports |
| bc5b924 | feat: rename encoder/decoder to \_encoder/\_decoder with properties (TimeVAE, TimeNet, FCN) |
| 01620fb | feat: add encoder property to TST, TSTCC, AutoTCL |
| e1900f6 | feat: add encoder property to CoST and Series2Vec |
| bb80c07 | feat: add encoder/decoder conformance tests |

## Deviations from Plan

None — plan executed exactly as written.

## Threat Flags

None — no new network endpoints, auth paths, or file access patterns introduced.

## Known Stubs

None.

## Verification

- 29 conformance tests: all passed
- 26 smoke + from_config tests: all passed
- 445 full test suite: 445 passed, 2 skipped
- Ruff check + format: clean on all modified files
- Graphify: updated
