# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-21
**Phase:** 1-Foundation
**Areas discussed:** Config dataclass scope, Config integration with models, Mixin adaptation details, Module organization

---

## Config Dataclass Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Strip runner-only fields | Remove model_name, set_input_dims, set_sequence_length. Configs = type-safe containers only. | |
| Keep structure, note usage | Keep model_name, set_input_dims as no-op for Phase 2 compatibility. | |
| Model-only now, aug later | Model config dataclasses in Phase 1. Aug config dataclasses move to Phase 3. | ✓ |

**User's choice:** Model-only now, aug later. CFG-02 moves to Phase 3.
**Notes:** Runner-specific fields (`model_name`, `set_input_dims`) are artifacts of rbspaper's dispatch system. No runner in library-only scope. Also agreed to strip these fields from the base config hierarchy.

## Config Integration with Models

| Option | Description | Selected |
|--------|-------------|----------|
| from_config() classmethod | Keep __init__ flat params for backward compat. Add @classmethod for type-safe path. | ✓ |
| Config-only constructor | Replace __init__ to take single config object. Breaking change. | |
| Both, kwargs override | __init__ accepts config= and **kwargs. Maximum flexibility, most complexity. | |

**User's choice:** from_config() classmethod.
**Notes:** Backward compat critical — existing callers use flat params. Config is the recommended new path but not mandatory.

## Mixin Adaptation

| Option | Description | Selected |
|--------|-------------|----------|
| Selective adapt | Take polymorphism + sliding-window fixes. Drop runner-specific guards. | ✓ |
| Faithful port | Port rbspaper mixin as-is. Add encoder validation. Align for future sync. | |
| Minimal split | Split into 3 classes. Keep encode() body unchanged. Lowest risk. | |

**User's choice:** Selective adapt.
**Notes:** User clarified that `encoder is None` validation in rbspaper is a runner artifact — our library always sets encoders in __init__. Asked about what "validators" are before deciding.

## Module Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror mixin hierarchy | ModelParameters → DilatedCNNModelParameters → per-model configs | ✓ |
| Flat with shared base | ModelParameters → per-model configs directly | |
| Composition over inheritance | No hierarchy. Common params in shared dataclass, models hold reference. | |

**User's choice:** Mirror mixin hierarchy with DilatedCNNModelParameters intermediate layer.
**Notes:** User proposed config hierarchy should reflect architecture type — ModelParameters (polymorphic base) → DilatedCNNModelParameters (shared encoder params) → per-model. Leaves room for transformer-based models later.

## Directory Restructure

**User's choice:** Defer to Phase 2, shift augmentation to Phase 3, cleanup to Phase 4.
**Notes:** User wanted colocated structure: model-specific augmentations inside each model folder, shared layers (`BandedFourierLayer`) outside at `models/layers/general.py`, dilated-specific layers (`Conv1dDilatedEncoder`, `same_pad`) inside `cnn/dilated/layers/`. Also augmentations: ABC stays shared, concrete per-model goes to `ts2vec/augmentation.py`, `cost/augmentation.py`, `autotcl/augmentation.py`.

---

## Deferred Ideas
- Full directory restructure → Phase 2
- Augmentation refactor → Phase 3
- Cleanup and verification → Phase 4
- Aug config dataclasses → Phase 3
- Model-specific augmentation files → Phase 2
