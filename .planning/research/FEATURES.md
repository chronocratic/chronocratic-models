# Features Research: What Library Users Expect

**Date:** 2026-05-21
**Domain:** ML/DL research libraries for representation learning

## Table Stakes (must have)

### Clear Import Paths
- Users expect `from tscollection.models import TS2Vec` to work
- Augmentations at `from tscollection.augmentation import CropShiftAugmentation`
- One import per concern — no nested `._augmentation.enums.AugmentationMode`

### Extensible Augmentations
- Subclass `AugmentationMethod` → get a working augmentation
- No registry to update, no enum to extend
- Model code unchanged when new augmentations appear

### Unified Model API
- All models share the same constructor signature: `augmentation: AugmentationMethod, **params`
- All models expose `.encode()` with identical parameters
- Training via Lightning `trainer.fit()` — no custom runner needed

### Type Hints
- Full annotations on public API (constructor, encode, forward)
- Users get IDE autocompletion

## Differentiators (competitive advantage)

### Strategy-based Training
- Training strategies (RIP, Adversarial) are first-class objects, not hidden in model code
- Users compose: `TrainableAugmentation(strategy=RIPTrainingStrategy(...))`
- New training algorithms = new strategy class, zero model changes

### Config-Driven Reproducibility
- Model params as dataclasses → easy to serialize, share, reproduce
- `ModelConfig(hidden_dims=64, depth=10)` instead of 12 positional args

### Pooling vs Decomposition Abstraction
- Mixin hierarchy matches the two encoding families
- New models pick one: `PoolingEncodingMixin` or `DecompositionEncodingMixin`

## Anti-Features (do NOT build)

- **No experiment runners** — users have their own training loops
- **No data modules** — domain-specific, user provides these
- **No evaluation pipelines** — outside library scope
- **No registry/factory** — direct imports are clearer
- **No CLI** — library, not tool

## Feature Verdict

Focus on the augmentation refactor as the core differentiator. Strategy-based training is the unique selling point — no other time-series library does this cleanly. Config dataclasses enable reproducibility without runner coupling.
