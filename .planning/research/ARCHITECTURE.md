# Architecture Research: Library Structure

**Date:** 2026-05-21

## Target Package Layout

```
tscollection/
  __init__.py                    # Top-level exports
  augmentation/                  # ALL augmentations in one place
    __init__.py                  # Public exports
    strategies.py                # AugmentationMethod, TrainableAugmentation, AugmentationTrainingStrategy
    config.py                    # Augmentation parameter dataclasses
  models/
    __init__.py                  # Model exports
    abstract/
      __init__.py
      encoding_functionality_mixin.py  # BaseEncodingMixin, PoolingEncodingMixin, DecompositionEncodingMixin
    config.py                    # Model parameter dataclasses (TS2VecModelParameters, etc.)
    ts2vec/
      __init__.py
      model.py                   # TS2Vec class
      config.py                  # TS2Vec-specific config (if needed)
    cost/
      __init__.py
      model.py                   # CoST class
    autotcl/
      __init__.py
      model.py                   # AutoTCL class
    encoders/
      __init__.py
      encoders.py                # TCN encoders
      masking.py                 # MaskMode enum, mask utilities
    layers/
      __init__.py
      convolutions/
        dilated.py
        same_pad.py
      general.py                 # Shared layer utilities
    losses.py                    # Contrastive losses
    utils.py                     # Model utilities
```

## Build Order Implications

### Phase 1: Foundation (mixin + config)
- Split mixin hierarchy first — it's used by all models
- Config dataclasses second — they parameterize models
- These enable Phase 2

### Phase 2: Augmentation refactor
- Depends on config layer (augmentation config dataclasses)
- Must touch all 3 models simultaneously
- Strategies replace enums/factories

### Phase 3: Cleanup
- Remove rbspaper source
- Remove dead code (encoding.py runner dispatch)
- Polish public API

## Architecture Verdict

Structure follows "models consume augmentations" direction. Augmentations are the product; models are the clients. This is clean separation — augmentation module has zero imports from models directory.
