---
name: unified-heads
status: complete
completed: 2026-06-11T15:00:00Z
---

# Unified Downstream Heads — Summary

## What was done
Collapsed 3 competing head patterns into one `FineTuningModule` wrapper with injected collaborators.

## Results
- **_finetuning package**: FineTuningModule, FlattenLinearHead, Protocols, adapters, callbacks, factories
- **Backbone changes**: Added `representation_dim` property to TST, Series2Vec, TSTCC
- **TST**: Deleted heads.py, migrated to FineTuningModule via factory
- **Series2Vec**: Deleted heads.py, migrated to FineTuningModule via factory
- **TS-TCC**: Removed TSTCCTrainingMode enum, model now single-purpose (pretrain only)
- **Tests**: 55 new tests (unit + integration), 247 total pass, zero regressions

## Commits (11 atomic, TDD)
- d307506: test(01-finetuning): add failing tests for _finetuning package
- d2b7df2: feat(01-finetuning): add _finetuning package with FineTuningModule, adapters, callbacks, factories
- 5d97672: test(02-backbones): add failing tests for representation_dim on all backbones
- 9e8aebf: feat(02-backbones): add representation_dim property to TST, Series2Vec, TSTCC
- 98a4837: test(03-tst): add tests for TST migration to FineTuningModule
- 68cc9e4: refactor(03-tst): migrate TST downstream heads to FineTuningModule, delete heads.py
- 1c399db: test(04-series2vec): add tests for Series2Vec migration to FineTuningModule
- 92a45c4: refactor(04-series2vec): migrate downstream head to FineTuningModule, delete heads.py
- 9e4431a: test(05-tstcc): add tests for TSTCC enum removal and downstream migration
- 8146c09: refactor(05-tstcc): remove TSTCCTrainingMode enum, make model single-purpose (pretrain only)
- 4fbcd69: test(integration): cross-model fine-tuning integration tests and full verification

## Files changed
- **Created**: 5 (_finetuning package) + 6 (tests) = 11 files
- **Deleted**: 3 (heads.py × 2, enums.py × 1)
- **Modified**: 6 (representation_dim properties, docstrings, barrel updates)

## Decisions confirmed
- D-01: Fresh FlattenLinearHead for TS-TCC (not encoder logits reuse)
- D-02: Explicit `self._representation_dims` attr added to Series2VecNetwork
- D-03: TSTCCTrainingMode enum removed entirely
- D-04: BackboneUnfreeze callback optional, included in package
