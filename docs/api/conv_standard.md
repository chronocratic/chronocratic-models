# Standard Convolutional Models

Models that use standard (non-dilated) 1D convolutions with `BasicEncodingMixin` for simpler inference.

## Series2Vec

Temporal encoding via SoftDTW-based contrastive loss.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.standard.series2vec.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.series2vec.config
   :members:
   :show-inheritance:
```

## TSTCC

Temporal contrastive clustering for representation learning.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.standard.tstcc.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.tstcc.config
   :members:
   :show-inheritance:
```

## FCN (MCL)

Multi-scale contrastive learning with a minimal FCN architecture.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.standard.mcl.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.mcl.config
   :members:
   :show-inheritance:
```
