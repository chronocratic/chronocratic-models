# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes are managed using [towncrier](https://towncrier.readthedocs.io/) and stored in the
[`changelog.d/`](changelog.d/) directory. See [`changelog.d/README.md`](changelog.d/README.md)
for instructions on adding changelog fragments.

<!-- towncrier release notes start -->

## v0.1.0a1 (2026-06-15) — First Alpha Release

The first pre-release of chronocratic-models.

This alpha introduces the complete set of self-supervised time series models, the polymorphic augmentation framework, supervised fine-tuning infrastructure, and the Sphinx-based documentation.

Expect breaking changes before the 1.0 release.

### Added

- **Convolutional (Dilated) models:** TS2Vec, CoST, AutoTCL — multi-scale representation learning via dilated convolutions, with `PoolingEncodingMixin` for sliding-window encoding.
- **Convolutional (Standard) models:** Series2Vec, TSTCC, FCN — contrastive and clustering-based pretraining with `BasicEncodingMixin`.
- **Transformer model:** TST — masked-reconstruction pretraining with configurable encoder depth and positional encoding.
- **Recurrent model:** TimeNet — GRU-based encoder-decoder with autoencoder pretraining.
- **Generative model:** TimeVAE — variational autoencoder with KL divergence + reconstruction loss.
- **Polymorphic augmentation producer contract:** Models accept any augmentation through `{AugmentationProducer}` protocols, eliminating enum-based branching.
- **Augmentation primitives:** Jitter, Scaling, Permutation, ComposeAugmentation with configurable `*Parameters` dataclasses.
- **Augmentation producers:** `SingleViewProducer`, `IndependentPair`, `RolePair`, `FullOverlapPair`.
- **Trainable augmentation support:** `TrainableAugmentationProducer` ABC and `maybe_train_augmentation` / `maybe_configure_augmentation_optimizer` utilities.
- **Supervised fine-tuning:** `SupervisedModule` wrapper with four modes — linear probe, full fine-tune, gradual unfreeze, supervised-from-scratch.
- **Factory functions:** `make_tst_supervised`, `make_series2vec_supervised`, `make_tstcc_supervised` for quick backbone + head setup.
- **`BackboneUnfreeze` callback:** Lightning callback for gradual unfreezing of pretrained encoders.
- **`ModelParameters` dataclasses:** One per model, with `kw_only=True` and Google-style `Args:` docstrings.
- **Shared layers:** `BandedFourierLayer`, `LevelModel`, `ResidualConnection`, `SeasonalLayer`, `TrendLayer`.
- **Distance metrics:** `SoftDTW` (differentiable dynamic time warping).
- **Encoding mixins:** `BasicEncodingMixin` and `PoolingEncodingMixin`.
- **Sphinx documentation** with autodoc-generated API reference per model family.
- **BSD 3-Clause license**.

### Notes

- Namespace is `chronocratic.models` (PyPI name is `chronocratic-models`).
- Requires Python 3.12+.
- Uses PyTorch and PyTorch Lightning as the primary framework.
