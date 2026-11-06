# Model Notes

Living per-model reference. Update this page when you add or change a model; [Contributing](contributing.md) stays stable.

## Model Support Matrix

| Model | VECTOR | SEQUENCE | Notes |
|---|---|---|---|
| TS2Vec | Yes | Yes | Both via pooling |
| CoST | Yes | Yes | Both via feature concatenation |
| MCL | Yes | Yes | SEQUENCE via fallback (unsqueeze to `B,1,D`) |
| TimeNet | Yes | Yes | Both supported |
| TST | Yes | Yes | Both supported |
| TimeVAE | Yes | No | VECTOR only |
| AutoTCL | Yes | Yes | Both via pooling |
| TSTCC | Yes | Yes | SEQUENCE via `_warn_sequence_fallback` + `unsqueeze(1)` |
| SimCLR | Yes | Yes | VECTOR via mean over time, SEQUENCE via transpose |
| MHCCL | Yes | Yes | VECTOR via mean over time, SEQUENCE via transpose |
| Series2Vec | Yes | Yes | Both supported |
| RecurrentAutoEncoder | Yes | Yes | Both supported |

> Models that list `SEQUENCE=Yes` but are not natively sequence-producing (MCL, TSTCC)
> issue a `_warn_sequence_fallback` and return `(B, 1, D)` by unsqueezing the vector.
> `supported_outputs` is the authoritative declaration; check that attribute on the model class.

## Current divergences and known limits


- **TimeVAE** — `nn.AdaptiveAvgPool1d` before the `Flatten`, so the latent projection width is constant. No counterpart in the reference; required because one pretrained model encodes several forecast window lengths. Exact identity at the nominal `sequence_length`.
- **TST** — positional encoding computed in `forward()` rather than cached. Numerically identical to the reference at any length the reference supported.
- **SimCLR** — ported as a 1-D ResNet. The reference stacks `Conv2d` over an input unsqueezed to `(B, C, T, 1)`, where the outer kernel columns only ever see zero padding; the `Conv1d` stack is numerically identical (asserted at stride 1 and 2 in `tests/unit/test_simclr.py`), as is `BatchNorm1d` against the reference's `BatchNorm2d` on that axis. Six further departures, each commented at its line: each encoder stage honours its own configured depth, where the reference repeats the third stage's index in place of the fourth; the learning rate follows the *original* SimCLR's linear warmup into cosine annealing ([google-research/simclr](https://github.com/google-research/simclr) `WarmUpAndCosineDecay`) rather than the reference's `ExponentialLR(gamma=1e-6)`, which reaches `1e-15` within two epochs and would suppress training; normalization defaults to `GroupNorm(1, C)` rather than hardcoded `BatchNorm`, so the encoder remains well-defined at `batch_size=1`; the default augmentation draws a per-sample scale factor, the reference's single shared factor being meaningful only for its offline whole-corpus augmentation; that factor uses `sigma=0.1` rather than the `sigma=1.1` the reference's `DataTransform` passes, since `N(1.0, 1.1)` places a substantial fraction of its mass below zero and those draws invert the sign of the series, leaving the two views describing different signals; and the encoder returns the unpooled feature map `(B, C, T')`, matching the reference's flatten-of-`C×T` contrastive input. An earlier version applied `AdaptiveAvgPool1d(1)`, which destroyed the angular diversity NT-Xent needs; pooling is now done in `_encode_batch` (mean for VECTOR, transpose for SEQUENCE), keeping the loss path intact while leaving downstream probes unaffected. `0.1` is the default of the reference's own `scaling()`, overridden only by `DataTransform`, and that module documents the intended range as `[0.1, 2.0]`.
- **SimCLR, open limit** — the model is sensitive to the ratio between its capacity and the number of optimizer steps a run affords. The default configuration is a ResNet-50-scale backbone; where that ratio is unfavourable it can remain far from convergence, in which case a randomly-initialised backbone may match or exceed a trained one under a linear probe and repeated runs of one configuration disagree. The learning-rate schedule reduces this sensitivity without removing it, and the contributing causes — nondeterministic kernel selection versus undertraining — have not been separated. Callers can pin the former through `Trainer(deterministic=True)`.
- **MHCCL** — ported as a 1-D ResNet-18 pair sharing `Conv1dResNetEncoder`, for the reasons given under SimCLR, with one caveat that does not apply there: the width-1 equivalence holds through the `3x3` residual convolutions but not through this model's stem, whose `kernel_size=8, padding=4` widens the vestigial axis to 2 (to 80 on the `epilepsy` setting, padded by 43) before the first stage mixes those columns across a ReLU and the second stage's stride collapses them. The columns are the same signal under different slices of the stem kernel — an artifact of reusing a vision backbone — so for MHCCL the 1-D port is a simplification of the reference rather than a rewriting of it. Two departures are structural. First, FINCH clustering is reimplemented in torch on the input's device: the reference uses scikit-learn, SciPy and pynndescent, none of which this library depends on, and a host round-trip per level would dominate a per-batch clustering step; component labelling is asserted equal to `scipy.sparse.csgraph` in `tests/unit/test_mhccl.py`. Second, the reference clusters the whole training set once per epoch and addresses the assignments through a dataset-global sample index that this library's batch contract does not carry, so clustering runs per batch over the raw and augmented embeddings unioned with a momentum feature bank; that bank is capped at one epoch of enqueued rows, because copies of one sample taken from different encoder states cluster together and add a partition encoding duplication rather than structure. Sixteen departures in all, ten of them marked `DIVERGENCE` at the line where they happen; the 1-D port above and the per-batch clustering below are two of the six structural ones. Candidate scores are inner products via `einsum("bd,bmd->bm", ...)`, where the reference reshapes its `(B, M, D)` stack to `(B, D, M)` — reinterpreting the buffer instead of permuting axes, so its einsum contracts mismatched elements; candidate slots a batch cannot fill are masked out of the mean rather than raising; the cluster-level term averages over the partitions actually realized, where the reference divides by the configured count and ties the loss scale to the batch's geometry; centroids neither mutate the caller's feature matrix nor mis-count, where the reference zeroes rows in place and subtracts a constant one from every cluster; upward masking reads each level's own cluster membership, where the reference builds its member lists once from the level-0 partition and reuses them at every coarser level, measuring each coarse centroid against the members of whichever level-0 cluster shares its id; the level-to-level cluster map comes from the merge step rather than a scan of the label matrix per anchor per candidate; shuffle-BN and the `torch.distributed` dependency are dropped, being the identity at world size 1; normalization defaults to `GroupNorm(1, C)` rather than the reference's `BatchNorm`, so the encoder stays well-defined at `batch_size=1`; views come from the injected producer per batch, where the reference freezes both at `Dataset` construction and never seeds `np.random`; the default view pair keeps every channel's own values under a shared segment order, where the reference's `permutation` broadcasts channel 0's shuffled values over every channel (inherited from TS-TCC, whose data is univariate); its scale factor varies across channels rather than along time, where the reference's varies along time and is shared across channels, making it multiplicative noise rather than a rescaling; self-distances in the neighbour search are infinite rather than the reference's `1e12`, which is large enough only for unit-norm features; the reference's `min_sim` edge pruning is dropped, being a no-op there (it zeroes stored entries of a `csr_matrix`, and `csgraph` reads only the sparsity structure); and defaults follow the CLI except for `use_projection_mlp` and `use_lr_scheduler`, which take the README command's `--mlp` and `--cos` because the CLI defaults would ship no projection head and no decay at all — the command's remaining overrides (`--protoNCE_only`, the masking flags, `--mask_mode`, `--posi`, `--negi`) select an ablation rather than a baseline, so the CLI default stands there.
- **MHCCL, open limit** — the number of partitions the clustering realizes depends on the pool size and on the data's geometry, so it can fall below `hierarchy_levels`. The cluster-level term then averages over the partitions actually used, which is reported as `mhccl/hierarchy_levels_used` and warns once; whether a per-batch pool plus a feature bank is a good proxy for the reference's per-epoch corpus prototypes is empirical and unmeasured.
- **TST, open limit** — encoding at full-partition lengths is length-*correct* but O(T²) in memory and will exhaust it before raising. Chunked windowing is an open follow-up.
- **Pretrain/encode length mismatch, open question** — a model pretrained on crops of one length and used to encode windows of another is mechanically valid after patching, but whether representation quality holds across lengths is empirical. Worth measuring before publishing numbers.

## Encoder transpose locations

- `TimeVAEEncoder.forward()` — `transpose(1, 2)` at entry
- `Series2VecNetwork._to_channels_first()` — layout conversion helper
- Dilated encoders — `transpose(1, 2)` in `_common_forward()`
- `FCNEncoder.forward()` — `transpose(1, 2)` at entry
- `TCCEncoder.forward()` — `transpose(1, 2)` at entry
- `Conv1dResNetEncoder.forward()` — `transpose(1, 2)` at entry

## Mixin assignment

- `BasicEncodingMixin` (`_mixin/encoding.py`): TST, TimeVAE, TimeNet, RecurrentAutoEncoder, MCL, TSTCC, SimCLR, MHCCL, Series2Vec.
- `BaseEncodingMixin` (`convolutional/dilated/_mixin/encoding.py`): TS2Vec, AutoTCL, CoST. `PoolingEncodingMixin` serves TS2Vec and AutoTCL; `DecompositionEncodingMixin` serves CoST.

## NaN guard locations

- `BasicEncodingMixin` models: guard lives in each model's `_encode_batch()`.
- `PoolingEncodingMixin` (TS2Vec, AutoTCL): `_evaluate_with_pooling`.
- `DecompositionEncodingMixin` (CoST): `_evaluate_with_feature_concatenation`.
- `AutoTCLTimeSeriesEncoder._process_not_nan_mask`: `torch.nan_to_num` + multiplication pattern.

## Model-specific hyperparameter names

Names that are unique to one model and intentionally differ from the canonical table in [Contributing](contributing.md):

- `latent_dim` — TimeVAE
- `embedding_dim` — Series2Vec

## Length-robustness worked examples

Step 3 (`max_train_length` via `process_sample_length` in `training_step`) is followed by TS2Vec, CoST, AutoTCL, TST, and TimeVAE.

- **Pattern A (compute analytically in `forward()`), TST.** `FixedPositionalEncoding` (`transformer/tst/ts_transformer.py`) originally built a `(sequence_length, 1, hidden_dim)` buffer in `__init__`. The sinusoidal formula is closed-form, so the buffer was only a cache, and one that silently imposed a ceiling. It now builds `position` and `div_term` inside `forward()` from `x.size(0)`, on `x.device` with `x.dtype`. Cost: recomputation per forward, negligible next to attention. Benefit: no ceiling, and the silent truncation path for under-long inputs disappears. More faithful to the paper than the cache was.
- **Pattern B (resample the temporal axis), TimeVAE.** `TimeVAEEncoder` (`generative/timevae/model.py`) flattens its conv stack into `nn.Linear(encoder_last_dense_dim, latent_dim)`, so the weight shape scaled with input length. `nn.AdaptiveAvgPool1d` before the flatten resamples to a constant number of ordered slots. Its target is the conv-stack output at the nominal `sequence_length`, making it an exact identity there (16 elements into 16 bins is one element per bin): training is bit-identical and existing checkpoints keep their shapes.
- **Pattern C (raise when the limit is inherent), TST.** `LearnablePositionalEncoding` stores one learned vector per position; there is no principled value for position 5,000 when training saw 0–127. It raises a `ValueError` naming both lengths and the remedy.
- **Length-agnostic, no patch needed:** dilated and standard `Conv1d` models (TS2Vec, CoST, AutoTCL) slide over any length.
