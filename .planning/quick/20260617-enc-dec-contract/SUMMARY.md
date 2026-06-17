---
name: enc-dec-contract-summary
description: Encoder/decoder extraction contract implementation summary
metadata:
  type: quick-task-summary
  status: complete
  date: 2026-06-17
---

# Summary: Encoder/Decoder Extraction Contract

## What was done
- Created `protocols.py` with `HasEncoder`/`HasDecoder` Protocols
- Unified encoder/decoder storage across 9 models to `_encoder`/`_decoder` + `@property`
- Added `@property encoder` to TST, TSTCC, AutoTCL, CoST, Series2Vec
- TimeVAE, TimeNet, FCN: renamed internal attrs + added properties
- Created conformance test (29 assertions across all 9 models)
- Updated knowledge graph

## Commits
1. `12b95d5` feat: add HasEncoder/HasDecoder Protocol contracts
2. `384f9c6` refactor: unify encoder/decoder storage to _encoder/_decoder + property
3. `7a73e80` refactor: add @property encoder to TST, TSTCC, AutoTCL
4. `8b6e60b` refactor: add @property encoder to CoST and Series2Vec
5. `8ed24da` test: add conformance test for encoder/decoder contract
6. `4b5a4e3` fix: restore Sampling import lost during encoder rename

## Verification
- Conformance test: 29/29 passed
- Regression (smoke + from_config): 26/26 passed
- Full suite: 445 passed, 2 skipped (was 416/418, +29 new tests)
- Ruff: no new errors (1 pre-existing TC002 in vae_base.py)

## Known issues
- Pre-existing TC002 (numpy import) in vae_base.py — not introduced by this change
- Checkpoint keys changed (`encoder` → `_encoder`) — breaking change accepted by user
