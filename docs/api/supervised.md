# Supervised Fine-tuning

Wrapper for downstream classification and regression tasks using pretrained encoders as backbones.

## SupervisedModule

Core module combining a pretrained backbone with a classification/regression head. Supports linear probing, full fine-tuning, gradual unfreezing, and supervised-from-scratch training.

```{eval-rst}
.. automodule:: chronocratic.models.supervised.supervised
   :members:
   :show-inheritance:
```

## Factory Functions

Convenience constructors for common backbone-head combinations.

```{eval-rst}
.. automodule:: chronocratic.models.supervised.factory
   :members:
   :show-inheritance:
```
