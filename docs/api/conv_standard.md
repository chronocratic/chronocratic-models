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

## SimCLR

Instance-level contrastive learning: a ResNet-1D backbone, a projection head, and NT-Xent over two augmented views.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.standard.simclr.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.simclr.config
   :members:
   :show-inheritance:
```

### Parameter names vs. the reference

SimCLR's parameters are named per this library's canonical vocabulary rather
than the reference's. The mapping below is for anyone comparing this
implementation against [ULTS](https://github.com/mqwfrog/ULTS); these are
renames only and none of them changes behaviour.

| this library | ULTS (`models/SimCLR/models.py`) | why renamed |
|---|---|---|
| `input_dim` | `in_channels` | canonical name for input feature count |
| `stem_conv_channels` | (inline `64` in `self.conv1`) | was a literal; named for the layer it configures |
| `encoder_stage_channels` | (inline `64,128,256,512` in `layer_block`) | were literals; `stage` distinguishes these from TS-TCC's per-block `encoder_channels` |
| `encoder_stage_depths` | `layers` | `layers` reads as modules rather than counts; `depth` is the canonical term for a layer count |
| `encoder_stage_strides` | (inline `1,2,2,2` in `layer_block`) | were literals |
| `residual_block_type` | `block` | `block` suggests an instance; this selects a *type* |
| `projection_dim` | `num_features` | `num_features` is ambiguous with `input_dim`; this is the projection width |
| `projection_hidden_dim` | (inline `512` in `learning_head`) | was a literal |
| `conv_kernel_size` | `kernel_size` | canonical name (`kernel_size` is explicitly discouraged) |
| `normalization_layer_type` | (hardcoded `nn.BatchNorm2d`) | made configurable; see the divergence entry above |
| `temperature` | `tau` | spelled out |
| `use_lr_scheduler`, `warmup_epochs` | (no equivalent) | new; the reference's scheduler suppresses training |

The reference also exposes `reparam` and a `linear` head that this port omits:
both belong to a variational variant that its SimCLR path never uses.

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
