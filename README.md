# chronocratic-models

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/chronocratic-models.svg)](https://pypi.org/project/chronocratic-models/)
[![Python versions](https://img.shields.io/pypi/pyversions/chronocratic-models.svg)](https://pypi.org/project/chronocratic-models/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/chronocratic-models?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/chronocratic-models)
[![Build Status](https://github.com/chronocratic/chronocratic-models/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/chronocratic/chronocratic-models/actions)
[![Documentation Status](https://readthedocs.org/projects/chronocratic-models/badge/?version=latest)](https://chronocratic-models.readthedocs.io/en/latest/?badge=latest)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub stars](https://img.shields.io/github/stars/chronocratic/chronocratic-models)](https://github.com/chronocratic/chronocratic-models/stargazers)

Ready-to-use time series models implemented in PyTorch and Lightning.

> **Note:** The PyPI package name uses a hyphen (`chronocratic-models`), but the import uses the `chronocratic.models` namespace.

## Installation

```bash
pip install chronocratic-models
```

## Quick Start

```python
import torch
from lightning.pytorch import Trainer
from chronocratic.models import TS2Vec, TS2VecModelParameters

# Create model using parameters dataclass
params = TS2VecModelParameters(input_dims=1)
model = TS2Vec(**vars(params))

# Prepare synthetic time series (n_instance, n_timestamps, n_features)
synthetic_data = torch.randn(2, 100, 1)

# Train the model first (models do not ship with pre-trained weights)
trainer = Trainer(max_epochs=1, accelerator="cpu", enable_checkpointing=False)
trainer.fit(model, train_dataloaders=synthetic_data)

# Get multi-scale representations
representations = model.encode(
    synthetic_data,
    batch_size=2,
    num_workers=0,
    encoding_window="multiscale",
)
print(representations.shape)
```

## Models

The package ships with self-supervised time-series models across these architectures:

| Category | Import |
|----------|--------|
| Convolutional (Dilated) | `TS2Vec`, `CoST`, `AutoTCL` |
| Convolutional (Standard) | `Series2Vec`, `TSTCC`, `MCL` |
| Transformer | `TST` |
| Recurrent | `TimeNet`, `RecurrentAutoEncoder` |
| Generative | `TimeVAE` |

For details (original papers, encoder architecture, default hyperparameters), see the [API reference](https://chronocratic-models.readthedocs.io/en/latest/) and the `ModelParameters` dataclass for each model. The list above is maintained by the exports in `chronocratic.models`; adding a model is just extending `__init__.py`.

> **Important:** No pre-trained weights are included — train on your own data before inference.

## Features

- **Polymorphic augmentation producer contract** — models accept any augmentation through a unified interface, eliminating enum-based branching.
- **Lightning integration** — all models are built on PyTorch Lightning for clean training loops and extensibility.
- **Self-supervised representation learning** — train encoders for downstream tasks without labeled data.
- **Pre-configured model parameters** — each model ships with tested default configuration dataclasses.
- **NumPy and PyTorch tensor support** — flexible input handling for both frameworks.

## Documentation

For full API reference, guides, and examples, visit [chronocratic-models.readthedocs.io](https://chronocratic-models.readthedocs.io/).

## License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](LICENSE) file for details.
