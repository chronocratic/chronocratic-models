# Model Protocols

Runtime-checkable Protocols for encoder/decoder extraction from models.

## Overview

The `chronocratic.models.protocols` module defines Protocols that allow users to check whether a model exposes its encoder (and optionally decoder) as a standalone `nn.Module`.

## Available Protocols

```{eval-rst}
.. automodule:: chronocratic.models.protocols
   :members:
   :show-inheritance:
```

## Usage

```python
from chronocratic.models.protocols import HasEncoder, HasDecoder
from chronocratic.models import TimeVAE

model = TimeVAE(sequence_length=100, input_dims=1, latent_dim=10)

# Check encoder availability
if isinstance(model, HasEncoder):
    encoder = model.encoder  # nn.Module

# Check decoder availability
if isinstance(model, HasDecoder):
    decoder = model.decoder  # nn.Module
```

## Implementation Details

- `HasEncoder` requires a `@property encoder` returning `nn.Module`
- `HasDecoder` requires a `@property decoder` returning `nn.Module`
- Both are `@runtime_checkable` — use `isinstance()` for checks
- The conformance test (`tests/test_encoder_decoder_contract.py`) verifies all 9 models satisfy these contracts
