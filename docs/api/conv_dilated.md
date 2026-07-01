# Dilated Convolutional Models

Models that use dilated 1D convolutions for multi-scale time series representations.
All three share `PoolingEncodingMixin` for sliding-window, multi-resolution encoding.

## TS2Vec

Hierarchical contrastive learning via progressively dilated convolutions.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.dilated.ts2vec.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.dilated.ts2vec.config
   :members:
   :show-inheritance:
```

## CoST

Contrastive Seasonality-Trend decomposition for time series pretraining.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.dilated.cost.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.dilated.cost.config
   :members:
   :show-inheritance:
```

## AutoTCL

Adversarial unsupervised contrastive learning with trainable augmentation.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.dilated.autotcl.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.dilated.autotcl.config
   :members:
   :show-inheritance:
```
