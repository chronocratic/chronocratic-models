# Encoder/Decoder Extraction Contract

## Context

Package users need to grab a trained model's encoder (and, for autoencoders, decoder)
as a standalone `nn.Module` — e.g. `model.encoder` to reuse the learned backbone, or
`model.decoder` to generate. Today this is impossible to rely on: the 9 models use **three
inconsistent storage patterns** — public `self.encoder` (TimeVAE/TimeNet/FCN), private
`self._encoder` with no public accessor (TST/TSTCC/AutoTCL), `@property encoder` (TS2Vec
only), and domain-named (`self.network` for Series2Vec, `query_encoder`/`key_encoder` for
CoST). There is no contract a user can `isinstance`-check, and no test guaranteeing new
models stay conformant.

Goal: a uniform, programmatically-checkable contract for encoder/decoder extraction.

## Decisions (locked with user)

- **Mechanism:** two `@runtime_checkable` `Protocol`s — `HasEncoder`, `HasDecoder`. Decoupled
  and opt-in (decoder only on the 2 autoencoders). Matches existing repo style
  (`Augmentation`, `RepresentationBackbone`, `BatchAdapter`). No ABC, no MRO churn. Rejected
  the hybrid (abstract property on `BasicEncodingMixin`) — it couples the extraction concern
  into the `encode()` inference mixin and only covers 6 of 9 models.
- **Uniform `_encoder`/`_decoder` + `@property`:** every model stores the inner module under a
  leading-underscore name and exposes it via a public `@property`. (User chose consistency over
  the leaner "property-only-where-wrapping" option.)
- **CoST `.encoder` → `query_encoder`** (the inference-time encoder); `key_encoder` stays
  accessible by name.
- **Enforcement = a parametrized conformance test.** Structural Protocols give no
  definition-time wall; the test is the gate (fails the moment a new model omits `.encoder`).

## The contract

New file `src/chronocratic/models/protocols.py`:

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
from torch import nn

@runtime_checkable
class HasEncoder(Protocol):
    @property
    def encoder(self) -> nn.Module: ...

@runtime_checkable
class HasDecoder(Protocol):
    @property
    def decoder(self) -> nn.Module: ...
```

Export `HasEncoder`, `HasDecoder` from `src/chronocratic/models/__init__.py`.

> Caveat to record in the docstring: `runtime_checkable` + `@property` only verifies the
> attribute *exists* at `isinstance` time — not its return type. Return-type guarantee comes
> from the conformance test, not the Protocol.

## Per-model changes

Pattern: store inner module as `self._encoder` / `self._decoder`, add
`@property encoder` / `@property decoder` returning it. Route internal references through
the property (or `self._encoder`) consistently.

| Model | File | Change | Conforms to |
|-------|------|--------|-------------|
| BaseVAE/TimeVAE | `generative/timevae/vae_base.py`, `.../model.py` | class hints + assignments `encoder/decoder` → `_encoder/_decoder`; add both properties in `vae_base.py`; update `forward`/`_step`/`predict`/`get_prior_samples*` refs | HasEncoder + HasDecoder |
| TimeNet | `recurrent/timenet/model.py` | `self.encoder/decoder` → `_encoder/_decoder` + 2 properties; update `forward` | HasEncoder + HasDecoder |
| FCN | `convolutional/standard/mcl/model.py` | `self.encoder` → `_encoder` + property; update refs | HasEncoder |
| TST | `transformer/tst/model.py` | already `_encoder`; add `@property encoder` | HasEncoder |
| TSTCC | `convolutional/standard/tstcc/model.py` | already `_encoder`; add `@property encoder` | HasEncoder |
| TS2Vec | `convolutional/dilated/ts2vec/model.py` | **none** — already `_encoder` + property | HasEncoder ✓ |
| AutoTCL | `convolutional/dilated/autotcl/model.py` | already `_encoder` (+`_averaged_encoder`); add `@property encoder` mirroring TS2Vec (`cast`) | HasEncoder |
| CoST | `convolutional/dilated/cost/model.py` | keep `query_encoder`/`key_encoder`; add `@property encoder` returning `self.query_encoder` | HasEncoder |
| Series2Vec | `convolutional/standard/series2vec/model.py` | keep `self.network`; add `@property encoder -> nn.Module` returning `self.network` | HasEncoder |

**Justified exceptions to the rename** (public `.encoder` added, inner name kept): CoST
(`query_encoder`/`key_encoder` are domain roles, not a single encoder) and Series2Vec
(`self.network` is a multi-purpose net: `.encode`, `.pretrain_forward`, `.branch_representation_dim`
— renaming to `_encoder` is invasive and misleading). The public `.encoder` surface is uniform;
only the private inner name differs where a literal `_encoder` is wrong.

> Note on `_get_encoder()` (the `BasicEncodingMixin` hook): leave it as-is. It returns the
> callable for the `encode()` inference loop (sometimes a bound method, e.g.
> `self.network.encode`). It is a *different* concern from `.encoder` (raw submodule). They
> overlap for most models but must stay separate for Series2Vec.

## Enforcement test

New `tests/test_encoder_decoder_contract.py`. Parametrize over all 9 exported models, reusing
the construction params already proven in `tests/test_from_config.py` and `tests/test_smoke.py`
(e.g. `TS2Vec(input_dims=1)`, `CoST(input_dims=1, sequence_length=100)`, the TSTCC kwargs, etc.).
For TST / TimeVAE / TimeNet / FCN / Series2Vec, pull default construction kwargs from each
model's config dataclass during execution.

Assertions:
- every model: `isinstance(m, HasEncoder)` and `isinstance(m.encoder, nn.Module)`
- TimeVAE, TimeNet: `isinstance(m, HasDecoder)` and `isinstance(m.decoder, nn.Module)`
- (optional negative) an encoder-only model is `not isinstance(m, HasDecoder)`

## Verification

```bash
uv run pytest tests/test_encoder_decoder_contract.py -v   # new conformance gate
uv run pytest tests/test_smoke.py tests/test_from_config.py -q  # model.encoder usage unbroken
uv run pytest -q                                          # full suite (was 416/418)
uv run ruff check . && uv run ruff format --check .
```
Manual smoke: construct one of each family, confirm `m.encoder` returns the module and (for
TimeVAE) `m.decoder` works.

After edits: `graphify update .` (per CLAUDE.md).

## Out of scope

- No change to `encode()` / `_get_encoder()` inference behavior.
- No new base class or ABC.
- Ties loosely to the paused augmentation-contract Protocol work — same structural-Protocol
  style, but independent.
