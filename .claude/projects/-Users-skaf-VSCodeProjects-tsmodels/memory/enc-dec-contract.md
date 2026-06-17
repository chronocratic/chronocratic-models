---
name: enc-dec-contract
description: Encoder/decoder extraction contract via HasEncoder/HasDecoder Protocols
metadata:
  type: project
  date: 2026-06-17
---

Encoder/decoder extraction contract implemented. `HasEncoder`/`HasDecoder` Protocols in `protocols.py`. 9 models standardized: `_encoder`/`_decoder` storage + `@property` for public access. Conformance test: 29 assertions. Checkpoint keys changed (breaking). Related: [[augmentation-contract-redesign]].
