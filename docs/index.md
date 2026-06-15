# chronocratic-models

Self-supervised time-series representation learning models built with PyTorch and Lightning.

> **Note:** The PyPI package name uses a hyphen (`chronocratic-models`), but the import uses the `chronocratic.models` namespace.

## Installation

```bash
pip install chronocratic-models
```

## Quick Links

- [](quickstart) — Get started with a TS2Vec encoding example
- [](api/index) — Full API reference with autodoc
- [](changelog) — Release history and changelog
- [](contributing) — How to contribute to the project

## Models

The library provides ten pre-trained encoders organized by architecture family:

**Convolutional (Dilated):** TS2Vec, CoST, AutoTCL

**Convolutional (Standard):** Series2Vec, TSTCC, FCN

**Transformer:** TST

**Recurrent:** TimeNet

**Generative:** TimeVAE

## Features

- **Polymorphic augmentation producer contract** — models accept any augmentation through a unified interface, eliminating enum-based branching.
- **Lightning integration** — all models are built on PyTorch Lightning for clean training loops and extensibility.
- **Self-supervised representation learning** — pre-trained encoders ready for downstream tasks without labeled data.
- **Pre-configured model parameters** — each model ships with tested default configuration dataclasses.
- **NumPy and PyTorch tensor support** — flexible input handling for both frameworks.
