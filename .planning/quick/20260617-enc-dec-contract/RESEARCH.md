# Research: Encoder/Decoder Storage Patterns

## Three Existing Patterns

1. **Public `self.encoder`** (TimeVAE, TimeNet, FCN) — assignment-time attrs, no property
2. **Private `self._encoder`** (TST, TSTCC, TS2Vec, AutoTCL) — already underscore-prefixed
3. **Domain-named** (CoST: `query_encoder`/`key_encoder`; Series2Vec: `network`) — custom names

## Per-Model Reference Table

| Model | File | Current Storage | Internal Refs | Property Needed | Change Type |
|-------|------|----------------|---------------|-----------------|-------------|
| TimeVAE | `.../timevae/model.py` | `self.encoder`, `self.decoder` | 4 encoder + 0 decoder (base class handles rest) | `encoder`, `decoder` | Rename + property |
| BaseVAE | `.../timevae/vae_base.py` | Class hints `encoder: nn.Module`, `decoder: nn.Module` | 5 `self.encoder` + 5 `self.decoder` reads | `encoder`, `decoder` | Rename hints + property |
| TimeNet | `.../timenet/model.py` | `self.encoder: Sequential`, `self.decoder: Sequential` | 2 encoder + 1 decoder | `encoder`, `decoder` | Rename + property |
| FCN | `.../mcl/model.py` | `self.encoder` | 2 encoder | `encoder` | Rename + property |
| TST | `.../tst/model.py` | `self._encoder` | 6 `_encoder` refs | `encoder` | Property only |
| TSTCC | `.../standard/tstcc/model.py` | `self._encoder` | 6 `_encoder` refs | `encoder` | Property only |
| TS2Vec | `.../ts2vec/model.py` | `self._encoder` + existing `@property encoder` | 5 `_encoder` refs | None | Already conformant |
| AutoTCL | `.../autotcl/model.py` | `self._encoder` + `self._averaged_encoder` | 8 `_encoder` refs | `encoder` | Property only |
| CoST | `.../cost/model.py` | `self.query_encoder` + `self.key_encoder` | Many query/key refs | `encoder` → `query_encoder` | Property only |
| Series2Vec | `.../series2vec/model.py` | `self.network` | 5 network refs | `encoder` → `network` | Property only |

## Test Construction Params

| Model | Construction | Source |
|-------|-------------|--------|
| FCN | `FCN(n_in=1)` | Simplest |
| TST | `TST(feat_dim=1, max_seq_len=100)` | Need to verify |
| TSTCC | `TSTCC(input_channels=1, kernel_size=5, stride=1, final_out_channels=16, features_len=12, num_classes=10)` | test_from_config.py |
| TS2Vec | `TS2Vec(input_dims=1)` | test_smoke.py |
| AutoTCL | `AutoTCL(input_dims=1, augmentation=...)` | test_from_config.py |
| CoST | `CoST(input_dims=1, sequence_length=100)` | test_from_config.py |
| Series2Vec | `Series2Vec(...)` | Need to verify constructor |
| TimeVAE | `TimeVAE(seq_len=100, feat_dim=1, latent_dim=10)` | Need to verify |
| TimeNet | `TimeNet(...)` | Need to verify constructor |

## Key Findings

1. **Protocol pattern exists** — `augmentation/base.py` has `Augmentation`, `RepresentationBackbone` as `@runtime_checkable` Protocols. `protocols.py` fits.
2. **`runtime_checkable` + `@property` limitation** — Only verifies attribute exists, not return type. Return-type guarantee comes from conformance test.
3. **`test_smoke.py` line 124** — Already uses `model.encoder` on TS2Vec. Works via existing property.
4. **Supervised adapters** — `series2vec_representations()` uses `backbone.network.encode(x)`. Public `network` name stays.
5. **TS2Vec is reference model** — Already has `@property encoder` returning `self._encoder` with `cast`. Use as pattern.
