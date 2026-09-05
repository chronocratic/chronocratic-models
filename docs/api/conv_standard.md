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

## MHCCL

Cluster-wise contrastive learning: a momentum encoder pair, a FINCH clustering
hierarchy, and contrast against both same-cluster instances and per-partition
prototypes, with upward and downward masking shaping the pairs.

```{eval-rst}
.. automodule:: chronocratic.models.convolutional.standard.mhccl.model
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.mhccl.config
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.mhccl.clustering
   :members:
   :show-inheritance:

.. automodule:: chronocratic.models.convolutional.standard.mhccl.losses
   :members:
   :show-inheritance:
```

### Parameter names vs. the reference

MHCCL's parameters are named per this library's canonical vocabulary rather than
the reference's. The mapping below is for anyone comparing this implementation
against [MHCCL](https://github.com/mqwfrog/MHCCL); these are renames only and
none of them changes behaviour.

| this library | MHCCL (`main.py` argparse, `framework.py`) | why renamed |
|---|---|---|
| `input_dim` | `in_channels`, selected by a `dataset_name` chain | canonical name; the chain has no `else` branch |
| `stem_conv_kernel_size` | inline `8` in the `net.conv1` replacement | was a literal, chosen by dataset name |
| `stem_conv_channels` | inline `64` in `net.conv1` | was a literal |
| `encoder_stage_channels`, `encoder_stage_strides` | torchvision `ResNet` internals | were not configurable |
| `encoder_stage_depths` | `[2, 2, 2, 2]` positional list | `depth` is the canonical term for a layer count |
| `residual_block_type` | `resnet.BasicBlock` | selects a *type*, not an instance |
| `conv_kernel_size` | torchvision `conv3x3` | canonical name |
| `projection_dim` | `low_dim` / `num_classes` | `num_classes` is a misnomer — nothing is classified |
| `projection_hidden_dim` | `dim_mlp` | was derived from a weight shape |
| `use_projection_mlp` | `--mlp` | spelled out |
| `key_momentum` | `--moco_m` | disambiguated from the SGD momentum |
| `optimizer_momentum` | `--momentum` | idem |
| `positive_instance_count`, `negative_instance_count` | `--posi`, `--negi` | spelled out |
| `positive_prototype_count`, `negative_prototype_count` | `--posp`, `--negp` | spelled out |
| `hierarchy_levels` | `--layers` | counts clustering partitions, not network layers |
| `use_instance_loss` | `--protoNCE_only`, inverted | inverted so the default is the "on" state |
| `instance_temperature`, `prototype_temperature` | `--tempi`, `--tempp`, gated by `--usetemp` | three fields collapsed to two; `None` is the off state |
| `mask_outliers_at_base_level`, `mask_outliers_at_upper_levels` | `--mask_layer0`, `--mask_others` | say what is masked and where |
| `outlier_mask_mode` | `--mask_mode` | `StrEnum`; the `mask_` prefix is dropped from the values |
| `outlier_distance_threshold`, `outlier_mask_proportion` | `--dist_threshold`, `--proportion` | scoped to the masking they parameterize |
| `replace_centroids_with_nearest_member` | `--replace_centroids` | says what replaces them |
| `use_lr_scheduler`, `lr_step_milestones` | `--cos` / `--schedule` | one knob: `None` is cosine, a tuple is step decay |
| `feature_bank_size` | (no equivalent) | new; see the divergences above |

The reference also exposes `--warmup_epoch` and `--req_clust`, which this port
omits: the first leaves the cluster assignments unset while the forward pass
subscripts them, so any value above its default raises before the first step,
and the second only feeds a CSV dump.

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
