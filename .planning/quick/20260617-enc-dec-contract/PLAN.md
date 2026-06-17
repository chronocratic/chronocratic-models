# Plan: Encoder/Decoder Extraction Contract

## Goal
Uniform, programmatically-checkable contract for encoder/decoder extraction across all 9 models via `@runtime_checkable` Protocols.

## Waves

### Wave 1: Contracts + exports
- Create `src/chronocratic/models/protocols.py` with `HasEncoder` / `HasDecoder` Protocols
- Export from `src/chronocratic/models/__init__.py`

### Wave 2: Models that need rename (`self.encoder` → `self._encoder` + property)
- **TimeVAE** (`vae_base.py` + `model.py`): rename `self.encoder` → `self._encoder`, `self.decoder` → `self._decoder`. Add properties. Update all internal refs (~10 encoder + ~5 decoder refs). Class hints → underscore names.
- **TimeNet** (`model.py`): rename `self.encoder` → `self._encoder`, `self.decoder` → `self._decoder`. Add properties. Update forward() refs (~3 refs).
- **FCN** (`model.py`): rename `self.encoder` → `self._encoder`. Add property. Update forward/`_get_encoder` refs (~2 refs). No decoder.

### Wave 3: Models that need property only (already `_encoder`)
- **TST** (`model.py`): add `@property encoder` returning `self._encoder`
- **TSTCC** (`model.py`): add `@property encoder` returning `self._encoder`
- **AutoTCL** (`model.py`): add `@property encoder` returning `self._encoder` (cast like TS2Vec)

### Wave 4: Exception models — property only, keep internal names
- **CoST** (`model.py`): add `@property encoder` returning `self.query_encoder`
- **Series2Vec** (`model.py`): add `@property encoder` returning `self.network`
- **TS2Vec** (`model.py`): no change — already conformant

### Wave 5: Conformance test
- Create `tests/test_encoder_decoder_contract.py`
- Parametrize over all 9 models
- Assert `isinstance(m, HasEncoder)`, `isinstance(m.encoder, nn.Module)`
- For TimeVAE/TimeNet: assert `isinstance(m, HasDecoder)`, `isinstance(m.decoder, nn.Module)`
- Negative: encoder-only models should NOT match `HasDecoder`

### Wave 6: Verification
- `uv run pytest tests/test_encoder_decoder_contract.py -v`
- `uv run pytest tests/test_smoke.py tests/test_from_config.py -q`
- `uv run ruff check . && uv run ruff format --check .`
- `uv run pytest -q` (full suite)
- `graphify update .`

## Dependencies
Wave 1 → Wave 2/3/4 (independent) → Wave 5 → Wave 6

## Assumptions
- Breaking change on checkpoint keys accepted (no migration shim)
- `_get_encoder()` hook unchanged — different concern from `.encoder` property
- Series2Vec `self.network` rename justified (multi-purpose net)
