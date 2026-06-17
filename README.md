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
from chronocratic.models import TS2Vec, TS2VecModelParameters

# Create model using parameters dataclass
params = TS2VecModelParameters(input_dims=1)
model = TS2Vec(**vars(params))

# Prepare synthetic time series (n_instance, n_timestamps, n_features)
synthetic_data = torch.randn(2, 100, 1)

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

### Convolutional (Dilated)

| Model | Description |
|-------|-------------|
| **TS2Vec** | Multi-scale hierarchical representation learning via dilated convolutions with hierarchical clustering. Code source: [zhihanyue/ts2vec](https://github.com/zhihanyue/ts2vec) |
| **CoST** | Decomposition-based contrastive self-supervised learning with trend-seasonal decomposition and contrastive objectives. Code source: [salesforce/CoST](https://github.com/salesforce/CoST) |
| **AutoTCL** | Automatic temporal contrastive learning with a trainable augmentation module for self-supervised time-series encoding. Code source: [AslanDing/AutoTCL](https://github.com/AslanDing/AutoTCL) |

### Convolutional (Standard)

| Model | Description |
|-------|-------------|
| **Series2Vec** | Self-supervised pretraining via contrastive learning on augmented time-series segments. |
| **TSTCC** | Temporal and contextual contrastive pretraining for time-series representation learning. |
| **FCN** | Fully convolutional encoder designed for Mixup Contrastive Learning (MCL) objectives. |

### Transformer

| Model | Description |
|-------|-------------|
| **TST** | Time Series Transformer with masked-reconstruction-based self-supervised pretraining. |

### Recurrent

| Model | Description |
|-------|-------------|
| **TimeNet** | Recurrent encoder-decoder architecture for time-series representation learning. |

### Generative

| Model | Description |
|-------|-------------|
| **TimeVAE** | Variational autoencoder for time-series data with latent representation encoding and generation. |

## Features

- **Polymorphic augmentation producer contract** — models accept any augmentation through a unified interface, eliminating enum-based branching.
- **Lightning integration** — all models are built on PyTorch Lightning for clean training loops and extensibility.
- **Self-supervised representation learning** — pre-trained encoders ready for downstream tasks without labeled data.
- **Pre-configured model parameters** — each model ships with tested default configuration dataclasses.
- **NumPy and PyTorch tensor support** — flexible input handling for both frameworks.

## Documentation

For full API reference, guides, and examples, visit [chronocratic-models.readthedocs.io](https://chronocratic-models.readthedocs.io/).

## License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](LICENSE) file for details.
