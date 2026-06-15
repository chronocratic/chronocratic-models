# Encoding Mixins

Mixins that provide `encode()` interfaces for inference.

## BasicEncodingMixin

Simple encoding for models that produce single-resolution representations.

```{eval-rst}
.. automodule:: chronocratic.models._mixin.encoding
   :members:
   :show-inheritance:
```

## PoolingEncodingMixin

Multi-resolution encoding with sliding window and pooling. Used by dilated convolutional models.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.dilated._mixin.encoding
   :members:
   :show-inheritance:
```
