# Pitfalls Research: Common Mistakes

**Date:** 2026-05-21

## Pitfall 1: Breaking existing model behavior during refactor

**Warning sign:** Smoke tests fail after renaming `augmentation_mode` to `augmentation` parameter
**Prevention:** Write behavioral tests BEFORE refactoring. Capture current training_step output, compare after changes
**Phase:** Phase 2 (augmentation refactor)

## Pitfall 2: Mixin diamond inheritance

**Warning sign:** `LightningModule` + `PoolingEncodingMixin` creates MRO conflicts
**Prevention:** Mixins must not define `__init__`. Use composition where possible. Test `inspect.getm(ModelClass)` after each change
**Phase:** Phase 1 (mixin split)

## Pitfall 3: Augmentation strategy coupling to model internals

**Warning sign:** `RIPTrainingStrategy` imports from `tscollection.models.autotcl.model`
**Prevention:** Strategy interface is purely functional — takes tensors in, returns loss out. No model-aware imports
**Phase:** Phase 2 (augmentation refactor)

## Pitfall 4: Leaving dead code after refactor

**Warning sign:** `enums.py` deleted but `from .enums import AugmentationMode` still in `__init__.py`
**Prevention:** Final grep sweep for old names. Run `ty check` (type checker) on full source tree
**Phase:** Phase 3 (cleanup)

## Pitfall 5: Config dataclasses become too verbose

**Warning sign:** `TS2VecModelParameters` has 15+ fields, users prefer kwargs
**Prevention:** Config is for type hints + IDE completion, not mandatory. Models accept `augmentation: AugmentationMethod, **kwargs` — config objects are convenience, not requirement
**Phase:** Phase 1 (config layer)

## Pitfall 6: Removing rbspaper too early

**Warning sign:** Need to cherry-pick a fix but `_sources/` already deleted
**Prevention:** Keep rbspaper source until all merges are verified. Delete in final cleanup phase
**Phase:** Phase 3 (cleanup)

## Verdict

Biggest risk is Pitfall 1 (behavior regression) and Pitfall 2 (mixin MRO). Both in Phase 1-2 transition. Run smoke tests after each model refactor, not at the end.
