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

Ten models are available across five architecture families. All models take keyword arguments matching their `*ModelParameters` dataclass:

```python
from chronocratic.models import TS2Vec, TST, TimeVAE, Series2Vec, TSTCC, FCN, TimeNet, CoST, AutoTCL

# Most models accept keyword-only args
model = TS2Vec(input_dims=1)
model = Series2Vec(input_dims=1, sequence_length=100)
model = TimeNet(input_dims=1, seq_len=100)

# Some use different param names — see each model's config dataclass
model = TST(feat_dim=1, max_seq_len=100)
model = TimeVAE(input_dims=1, seq_len=100)
```

See the [](api/index) for full API documentation per model family.
