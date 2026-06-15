# Quick Start

This guide shows how to install and use `chronocratic-models` for encoding time-series data with TS2Vec.

## Installation

Install the latest release from PyPI:

```bash
pip install chronocratic-models
```

## Encoding a Time Series

The following example demonstrates how to create a TS2Vec model and encode a synthetic time series.

```python
import torch
from chronocratic.models import TS2Vec, TS2VecModelParameters

# Create model with default parameters
model = TS2Vec(TS2VecModelParameters(input_dim=1))
model.eval()

# Encode a synthetic time series (batch, channels, seq_len)
synthetic_data = torch.randn(1, 1, 100)

# Get multi-scale representations
with torch.no_grad():
    representations = model.encode(synthetic_data)
    print(representations.shape)  # (1, channels, hidden_dim)
```

## Model Catalog

Ten models are available across five architecture families. All models follow the same import pattern:

```python
from chronocratic.models import (
    TS2Vec, TS2VecModelParameters,
    CoST, CoSTModelParameters,
    AutoTCL, AutoTCLModelParameters,
    Series2Vec, Series2VecModelParameters,
    TSTCC, TSTCCModelParameters,
    FCN, MCLModelParameters,
    TST, TSTModelParameters,
    TimeNet, TimeNetModelParameters,
    TimeVAE, TimeVAEModelParameters,
)
```

Each model accepts a configuration dataclass that defines architecture hyperparameters. See the [](api/index) for full API documentation per model family.
