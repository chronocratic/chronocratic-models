---
name: unified-heads-context
description: "Research findings + discussion decisions for unified downstream heads"
---

# Unified Downstream Heads — Context

## Research Findings

### Attribute Mismatches (verified)
- **TST**: `self._encoder.max_len` (not `max_seq_len`) — ts_transformer.py:216
- **Series2Vec**: `representation_dims` NOT stored on network — needs `self._representation_dims = representation_dims` added
- **TS-TCC**: `final_out_channels`/`features_len` NOT stored — use `self._encoder.logits.in_features`

### Import Sites (verified clean)
- No external callers for `TSTClassificationHead`, `TSTRegressionHead`, `Series2VecClassificationHead`
- Zero test coverage for any head class
- `TSTCCTrainingMode` only in: `tstcc/enums.py`, `tstcc/model.py`, `tstcc/__init__.py`, `tstcc/config.py`

## Decisions (from discussion)
1. **TS-TCC head**: Fresh FlattenLinearHead, not encoder logits reuse
2. **Series2Vec**: Add `self._representation_dims` attr to Series2VecNetwork
3. **TS-TCC enum**: Remove TSTCCTrainingMode entirely; model single-purpose now (pretrain only)
4. **BackboneUnfreeze**: Optional, include in package but not required for migration

## TDD Requirements
- Test-first approach per project config (`workflow.tdd_mode: true`)
- Write tests before implementation for each commit
- Coverage: FineTuningModule, adapters, factories, representation_dim properties
