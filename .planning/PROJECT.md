# tsmodels — Time-Series Representation Learning Library

## What This Is

A Python library for self-supervised time-series representation learning. Provides pre-trained encoders (TS2Vec, CoST, AutoTCL) with a unified encoding API for downstream ML/DL research. Users import models, instantiate them, and train with Lightning — no runners, no experiment framework.

## Core Value

Users can add new augmentation methods and training strategies by subclassing — zero library modification required. Models accept any augmentation through a polymorphic interface, eliminating enum-based branching.

## Requirements

### Validated

- ✓ TS2Vec model with multi-scale hierarchical learning — existing
- ✓ CoST model with decomposition-based contrastive learning — existing
- ✓ AutoTCL model with neural network augmentation — existing
- ✓ TCN backbone encoder shared across models — existing
- ✓ Encoding API for inference (full-series, multiscale, sliding window) — existing
- ✓ PyTorch Lightning integration for training — existing

### Active

- [ ] Adopt split mixin hierarchy from rbspaper (BaseEncodingMixin → Pooling, Decomposition subclasses)
- [ ] Refactor augmentations to 3-layer ABC hierarchy (AugmentationMethod, TrainableAugmentation, AugmentationTrainingStrategy)
- [ ] Remove enum-based augmentation dispatch and factory patterns
- [ ] Move training logic from models into augmentation strategies
- [ ] Add config dataclasses for type-safe model parameters (without runner-specific dispatch)
- [ ] Make augmentation module the single home for all augmentations (model-agnostic and model-specific)
- [ ] Remove rbspaper source after merging useful changes
- [ ] Provide unified model API — all models accept `augmentation: AugmentationMethod` in constructor

### Out of Scope

- Experiment runners / CLI training tools — library only
- Encoding dispatch functions (`encoding.py`) — runner-oriented, not needed
- Data modules and evaluation pipelines — user responsibility

## Context

Brownfield project. Existing code has 3 models (TS2Vec, CoST, AutoTCL) working but with:
- Single mixin class using `hasattr` branching to distinguish model types
- Augmentation logic scattered across models (RIP/adversarial training in AutoTCL model code)
- Enum-based augmentation selection with factory patterns
- `_sources/rbspaper/` contains experimental improvements from another project (split mixins, config layer)

Current branch: `refactor/mixin` — work is already in progress.

Augmentation refactor design is documented in `.planning/todos/augmentation-refactor.md` (Option 3b — 3-layer ABC hierarchy).

## Constraints

- **Tech stack**: Python 3.12, PyTorch >=2.4, Lightning, uv package manager
- **Library-only**: No runners, no CLI. Users compose models with Lightning themselves
- **Backward compatibility**: Existing model behavior must be preserved. Only internal structure changes
- **Extensibility**: New models must be addable without modifying existing model code or shared abstractions

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 3-layer augmentation ABC | Eliminates enum branching, enables user-extensibility | — Pending |
| Split mixin hierarchy | Open-Closed — new model types get subclasses, no hasattr checks | — Pending |
| Config dataclasses, no runner dispatch | Type-safe params without coupling to experiment framework | — Pending |
| Library-only scope | Users control training loop, Lightning handles orchestration | — Pending |
| Augmentation as single module | One import path; model-specific augmentations named by convention | — Pending |

---
*Last updated: 2026-05-21 after initialization*

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (`/gsd:ship`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (`/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state
