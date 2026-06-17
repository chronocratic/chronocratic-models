---
plan: 01-09
phase: 01-augmentation-producer-contract
status: complete
wave: 5
---

## What Was Built

Updated barrel exports in `augmentation/__init__.py` to expose the new producer contract alongside legacy symbols. New symbols: `Augmentation`, `AugmentationProducer`, `SingleView`, `ViewPair`, `AlignedPair`, `TrainableAugmentationProducer`, `SingleViewProducer`, `IndependentPair`, `RolePair`, `FullOverlapPair`, `Seeded`, `maybe_train_augmentation`, `maybe_configure_augmentation_optimizer`, and all primitive re-exports. Legacy symbols (`AugmentationMethod`, `TrainingViews`, `TrainableAugmentation`, `DualAugmentation`) retained for backward compat (D-05).

`autotcl/augmentation/__init__.py` — no change needed; already exports `AutoTCLNeuralNetworkAugmentation` and training strategies correctly.

## Files Modified

- `src/tscollection/models/augmentation/__init__.py` — added new contract imports, reorganized `__all__`

## Self-Check

- [x] New symbol imports pass: `AugmentationProducer`, `SingleView`, `ViewPair`, `AlignedPair`, `Seeded`, `IndependentPair`, `RolePair`, `FullOverlapPair`, `SingleViewProducer`, `maybe_*`
- [x] Legacy imports pass: `AugmentationMethod`, `TrainingViews`, `TrainableAugmentation`, `DualAugmentation`
- [x] Per-model re-exports pass: `AutoTCLNeuralNetworkAugmentation`, `CropShiftAugmentation`
