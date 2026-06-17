---
name: enc-dec-contract-discussion
description: Discussion findings for encoder/decoder extraction contract
metadata:
  type: discussion
  date: 2026-06-17
---

# Discussion: Encoder/Decoder Extraction Contract

## Decisions Confirmed

1. **Protocol over ABC** — Two `@runtime_checkable` Protocols (`HasEncoder`, `HasDecoder`). Matches existing repo style (`Augmentation`, `RepresentationBackbone`, `BatchAdapter`). No MRO churn.
2. **Uniform `_encoder` + `@property`** — Every model stores inner module under leading-underscore, exposes via `@property`. Consistency over leaner "property-only-where-wrapping".
3. **Checkpoint breaking change accepted** — Renaming `self.encoder` → `self._encoder` changes state_dict keys. No migration shim. Document in changelog.
4. **CoST `.encoder` → `query_encoder`** — Inference-time encoder. `key_encoder` stays accessible by name.
5. **Series2Vec `self.network` exception** — Multi-purpose net (`encode`, `pretrain_forward`, `branch_representation_dim`). Renaming to `_encoder` is invasive. Public `.encoder` property returns `self.network`.
6. **`_get_encoder()` unchanged** — Different concern from `.encoder` property. Returns callable for `encode()` loop, sometimes bound method.

## Gray Areas Resolved

- **G1: BaseVariationalAutoencoder property location** — Properties go on `BaseVariationalAutoencoder` (shared by future VAE subclasses). TimeVAE overrides for correct return types.
- **G2: FCN `self.proj_head`** — Not a decoder. FCN conforms to `HasEncoder` only.
- **G3: TimeNet type hints** — Rename `self.encoder: Sequential` → `self._encoder: Sequential`. Property returns `nn.Sequential`.
- **G4: `nn.Module.__getattr__` breaks isinstance checks** — Property is mandatory on every model (not just optional). Plain attrs fail `isinstance(m, HasEncoder)`.

## Risks

- **R1: TimeVAE initialization order** — `_build_decoder()` reads `self.encoder.encoder_last_dense_dim`. After rename: `self._encoder = ...` first, property makes `self.encoder` accessible, then `_build_decoder()` uses `self._encoder.encoder_last_dense_dim` directly.
- **R2: External `model.encoder` assignments** — Read-only property means `model.encoder = x` crashes. Acceptable (encapsulation).
- **R3: TimeVAEEncoder inner class** — Has own `self.encoder = nn.Sequential(...)` on line 42. DO NOT rename — scoped to inner class, not model-level.
