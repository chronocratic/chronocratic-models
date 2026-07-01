# Quick Start

This guide shows how to install and use `chronocratic-models` for encoding time series data with TS2Vec.

## Installation

Install the latest release from PyPI:

```bash
pip install chronocratic-models
```

## Encoding a Time Series

The following example demonstrates how to create a TS2Vec model and encode a synthetic time series.

```python
import torch
from chronocratic.models import TS2Vec

# Create model with default parameters
model = TS2Vec(input_dims=1)
model.eval()

# Encode a synthetic time series (batch, channels, seq_len)
synthetic_data = torch.randn(1, 1, 100)

# Get multi-scale representations
with torch.no_grad():
    representations = model.encode(synthetic_data)
    print(representations.shape)  # (1, channels, hidden_dim)
```

## Model Catalog

Ten models are available across five architecture families. Models take keyword arguments directly. Each family ships with a `*ModelParameters` dataclass you can configure and unpack with `vars()`.

```python
from chronocratic.models import (
    TS2Vec, TS2VecModelParameters,
    TST, TSTModelParameters,
)

# Direct keyword arguments
model = TS2Vec(input_dims=1, depth=5)

# Or configure via dataclass, then unpack
params = TS2VecModelParameters(input_dims=1, depth=5)
model = TS2Vec(**vars(params))

# Transformer models use different parameter names
tst_params = TSTModelParameters(feat_dim=1, max_seq_len=100)
model = TST(**vars(tst_params))
```

See the [](api/index) for full API documentation per model family.
