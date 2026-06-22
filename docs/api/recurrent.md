# Recurrent Models

RNN-based architectures for sequential time-series modeling.

## RecurrentAutoEncoder

Multi-layer recurrent autoencoder supporting LSTM, GRU, and vanilla RNN cells.
Uses time-reversed encoder output for sequence reconstruction.

```{eval-rst}
.. automodule:: chronocratic.models.recurrent.recurrentae.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.recurrent.recurrentae.config
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.recurrent.enums
   :members:
   :show-inheritance:
```

## TimeNet

GRU-based encoder-decoder with autoencoder pretraining. Uses `BasicEncodingMixin` for inference.

```{eval-rst}
.. automodule:: chronocratic.models.recurrent.timenet.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.recurrent.timenet.config
   :members:
   :show-inheritance:
```
