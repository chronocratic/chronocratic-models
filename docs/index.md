# chronocratic-models

Ready-to-use time series models implemented in PyTorch and Lightning.

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

The library provides time series models organized by architecture family.

**Convolutional (Dilated):** [](api/conv_dilated) — TS2Vec, CoST, AutoTCL

**Convolutional (Standard):** [](api/conv_standard) — Series2Vec, TSTCC, MCL

**Transformer:** [](api/transformer) — TST

**Recurrent:** [](api/recurrent) — TimeNet, RecurrentAutoEncoder

**Generative:** [](api/generative) — TimeVAE

**Supervised:** [](api/supervised) — SupervisedModule with factory functions

## Features

- **Polymorphic augmentation producer contract** — models accept any augmentation through a unified interface, eliminating enum-based branching.
- **Lightning integration** — all models are built on PyTorch Lightning for clean training loops and extensibility.
- **Self-supervised representation learning** — train encoders for downstream tasks without labeled data.
- **Pre-configured model parameters** — each model ships with tested default configuration dataclasses.
- **NumPy and PyTorch tensor support** — flexible input handling for both frameworks.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
changelog
contributing
api/index
```
