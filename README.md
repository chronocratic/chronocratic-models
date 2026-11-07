<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/header/surface.svg?title=chronocratic-models&subtitle=Ready-to-use+time+series+models+implemented+in+PyTorch+and+Lightning.&align=center&font=jetbrains-mono&width=1200&height=600&bg=0E2A44&titleColor=F3F1EC&subtitleColor=A8D5D1&mode=dark&overlay=0&logo=https://raw.githubusercontent.com/chronocratic/chronocratic-models/main/.github/assets/icon-dark.png&image=https://raw.githubusercontent.com/chronocratic/chronocratic-models/main/.github/assets/header-bg-dark.png" />
    <img alt="chronocratic-models" src="https://shieldcn.dev/header/surface.svg?title=chronocratic-models&subtitle=Ready-to-use+time+series+models+implemented+in+PyTorch+and+Lightning.&align=center&font=jetbrains-mono&width=1200&height=600&bg=F3F1EC&titleColor=0E2A44&subtitleColor=2A6A7C&mode=light&overlay=0&logo=https://raw.githubusercontent.com/chronocratic/chronocratic-models/main/.github/assets/icon-light.png&image=https://raw.githubusercontent.com/chronocratic/chronocratic-models/main/.github/assets/header-bg-light.png" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/chronocratic-models/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/pypi/chronocratic-models.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="PyPI version" src="https://shieldcn.dev/pypi/chronocratic-models.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://pepy.tech/projects/chronocratic-models"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fpepy.tech%2Fapi%2Fv2%2Fprojects%2Fchronocratic-models&query=%24.total_downloads&label=downloads&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="Total downloads" src="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fpepy.tech%2Fapi%2Fv2%2Fprojects%2Fchronocratic-models&query=%24.total_downloads&label=downloads&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/chronocratic-models/actions/workflows/build-and-test.yml"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/ci/chronocratic/chronocratic-models.svg?workflow=build-and-test.yml&branch=main&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="CI status" src="https://shieldcn.dev/github/ci/chronocratic/chronocratic-models.svg?workflow=build-and-test.yml&branch=main&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/chronocratic-models/blob/main/LICENSE"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/license/chronocratic/chronocratic-models.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="License" src="https://shieldcn.dev/github/license/chronocratic/chronocratic-models.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/chronocratic-models/stargazers"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/stars/chronocratic/chronocratic-models.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="GitHub stars" src="https://shieldcn.dev/github/stars/chronocratic/chronocratic-models.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/astral-sh/ruff"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/code_style-ruff-261230.svg?logo=ruff&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="Code style: ruff" src="https://shieldcn.dev/badge/code_style-ruff-261230.svg?logo=ruff&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
</p>

<div align="center">

| Python | Docs |
| :---: | :---: |
| [![Python versions](https://img.shields.io/pypi/pyversions/chronocratic-models.svg)](https://pypi.org/project/chronocratic-models/) | [![Documentation status](https://readthedocs.org/projects/chronocratic-models/badge/?version=latest)](https://chronocratic-models.readthedocs.io/en/latest/) |

</div>

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
| Convolutional (Standard) | `Series2Vec`, `TSTCC`, `SimCLR`, `MHCCL`, `MCL` |
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

## Contributing

For development setup, linting, testing, and coding standards, see [docs/contributing.md](docs/contributing.md).

## License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](LICENSE) file for details.
