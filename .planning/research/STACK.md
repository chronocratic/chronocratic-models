# Stack Research: ML Library Patterns

**Date:** 2026-05-21
**Domain:** Self-supervised representation learning libraries

## Reference Libraries

### timm (pytorch-image-models)
- **Structure:** Flat registry pattern. All models in `timm/models/`, each in its own module file
- **Config:** Uses `PretrainedConfig` dataclasses per model family
- **Key pattern:** `build_model_with_cfg()` factory that accepts a model entrypoint + config dict. User-facing API is `timm.create_model('resnet50', pretrained=True)`
- **Augmentations:** Separate `timm.data.transforms` module. Training augmentations composed as callable pipelines, not mixed into model class
- **Lesson:** Registry + config separates model definition from instantiation

### sentence-transformers
- **Structure:** Component-based architecture. `SentenceTransformer` takes a list of components (`WordEmbeddings`, `Pooling`, `Dense`)
- **Key pattern:** `SentenceTransformer(['bert-base-nli-mean-token', 'Pooling', 'Dense'])` — components compose at runtime, not at class definition
- **Training:** `training_args` dataclass + `trainer` follows HuggingFace Trainer pattern
- **Lesson:** Composition over inheritance for model pipelines

### HuggingFace Transformers
- **Structure:** Config-driven. `AutoConfig`, `AutoModel` maps task → model class
- **Key pattern:** `model.config` is a dataclass serialized to JSON. Model reads all params from config, not constructor args
- **Mixin usage:** `PreTrainedModel` mixin provides `from_pretrained()`, `push_to_hub()`, `config` property
- **Lesson:** Config-first design makes models serializable and reproducible

## Recommendations for tsmodels

### What to adopt
1. **Config dataclasses per model** — like timm/transformers. Already planned
2. **Separate augmentation module** — like timm's `data.transforms`. Already planned
3. **Mixin for shared behavior** — like transformers' `PreTrainedModel`. Split mixin hierarchy is right approach

### What to avoid
1. **Registry pattern** — too runner-oriented. Library users prefer direct imports
2. **Component-based composition** — sentence-transformers approach is overkill for encoder-only models
3. **Factory functions** — `create_model()` adds indirection. Users should instantiate directly

## Stack Verdict

Current direction aligns with timm + transformers patterns: config dataclasses, separate augmentation module, mixin for shared inference behavior. No registry, no factories. Users import and instantiate directly.
