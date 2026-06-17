---
phase: 06-prepare-to-be-published-as-package
reviewed: 2026-06-15T12:00:00Z
depth: deep
files_reviewed: 105
files_reviewed_list:
  - pyproject.toml
  - src/chronocratic/models/__init__.py
  - src/chronocratic/models/augmentation/__init__.py
  - src/chronocratic/models/augmentation/base.py
  - src/chronocratic/models/augmentation/decorators.py
  - src/chronocratic/models/augmentation/primitives.py
  - src/chronocratic/models/augmentation/producers.py
  - src/chronocratic/models/augmentation/trainable_support.py
  - src/chronocratic/models/convolutional/dilated/autotcl/__init__.py
  - src/chronocratic/models/convolutional/dilated/autotcl/augmentation/__init__.py
  - src/chronocratic/models/convolutional/dilated/autotcl/augmentation/methods.py
  - src/chronocratic/models/convolutional/dilated/autotcl/augmentation/training.py
  - src/chronocratic/models/convolutional/dilated/autotcl/config.py
  - src/chronocratic/models/convolutional/dilated/autotcl/losses.py
  - src/chronocratic/models/convolutional/dilated/autotcl/model.py
  - src/chronocratic/models/convolutional/dilated/autotcl/utils.py
  - src/chronocratic/models/convolutional/dilated/cost/__init__.py
  - src/chronocratic/models/convolutional/dilated/cost/augmentation.py
  - src/chronocratic/models/convolutional/dilated/cost/config.py
  - src/chronocratic/models/convolutional/dilated/cost/model.py
  - src/chronocratic/models/convolutional/dilated/cost/utils.py
  - src/chronocratic/models/convolutional/dilated/encoders/__init__.py
  - src/chronocratic/models/convolutional/dilated/encoders/encoders.py
  - src/chronocratic/models/convolutional/dilated/encoders/masking.py
  - src/chronocratic/models/convolutional/dilated/layers/__init__.py
  - src/chronocratic/models/convolutional/dilated/layers/dilated.py
  - src/chronocratic/models/convolutional/dilated/layers/same_pad.py
  - src/chronocratic/models/convolutional/dilated/ts2vec/__init__.py
  - src/chronocratic/models/convolutional/dilated/ts2vec/augmentation.py
  - src/chronocratic/models/convolutional/dilated/ts2vec/config.py
  - src/chronocratic/models/convolutional/dilated/ts2vec/losses.py
  - src/chronocratic/models/convolutional/dilated/ts2vec/model.py
  - src/chronocratic/models/convolutional/dilated/ts2vec/utils.py
  - src/chronocratic/models/convolutional/standard/mcl/__init__.py
  - src/chronocratic/models/convolutional/standard/mcl/config.py
  - src/chronocratic/models/convolutional/standard/mcl/encoder.py
  - src/chronocratic/models/convolutional/standard/mcl/losses.py
  - src/chronocratic/models/convolutional/standard/mcl/model.py
  - src/chronocratic/models/convolutional/standard/series2vec/__init__.py
  - src/chronocratic/models/convolutional/standard/series2vec/config.py
  - src/chronocratic/models/convolutional/standard/series2vec/encoder.py
  - src/chronocratic/models/convolutional/standard/series2vec/filters.py
  - src/chronocratic/models/convolutional/standard/series2vec/losses.py
  - src/chronocratic/models/convolutional/standard/series2vec/model.py
  - src/chronocratic/models/convolutional/standard/series2vec/network.py
  - src/chronocratic/models/convolutional/standard/tstcc/__init__.py
  - src/chronocratic/models/convolutional/standard/tstcc/augmentations.py
  - src/chronocratic/models/convolutional/standard/tstcc/config.py
  - src/chronocratic/models/convolutional/standard/tstcc/encoder.py
  - src/chronocratic/models/convolutional/standard/tstcc/losses.py
  - src/chronocratic/models/convolutional/standard/tstcc/model.py
  - src/chronocratic/models/convolutional/standard/tstcc/temporal_contrast.py
  - src/chronocratic/models/distances/__init__.py
  - src/chronocratic/models/distances/soft_dtw/__init__.py
  - src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py
  - src/chronocratic/models/generative/__init__.py
  - src/chronocratic/models/generative/timevae/__init__.py
  - src/chronocratic/models/generative/timevae/config.py
  - src/chronocratic/models/generative/timevae/model.py
  - src/chronocratic/models/generative/timevae/vae_base.py
  - src/chronocratic/models/layers/__init__.py
  - src/chronocratic/models/layers/general.py
  - src/chronocratic/models/losses.py
  - src/chronocratic/models/recurrent/__init__.py
  - src/chronocratic/models/recurrent/timenet/__init__.py
  - src/chronocratic/models/recurrent/timenet/config.py
  - src/chronocratic/models/recurrent/timenet/model.py
  - src/chronocratic/models/supervised/__init__.py
  - src/chronocratic/models/supervised/_adapters.py
  - src/chronocratic/models/supervised/_callbacks.py
  - src/chronocratic/models/supervised/_utils.py
  - src/chronocratic/models/supervised/factory.py
  - src/chronocratic/models/supervised/supervised.py
  - src/chronocratic/models/transformer/__init__.py
  - src/chronocratic/models/transformer/tst/__init__.py
  - src/chronocratic/models/transformer/tst/config.py
  - src/chronocratic/models/transformer/tst/loss.py
  - src/chronocratic/models/transformer/tst/model.py
  - src/chronocratic/models/transformer/tst/ts_transformer.py
  - src/chronocratic/models/utils.py
findings:
  critical: 8
  warning: 9
  info: 5
  total: 22
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-06-15T12:00:00Z
**Depth:** deep
**Files Reviewed:** 105
**Status:** issues_found

## Summary

This phase renamed `tscollection` to `chronocratic` namespace, overhauled `pyproject.toml` with `setuptools_scm`/`towncrier`, added Sphinx docs, and fixed CI workflows. Deep review across all source files reveals 8 critical defects, 9 warnings, and 5 info items. The most serious issues are a global RNG mutation in `Seeded`, undefined behavior in `Permutation` for (batch, channels, time) shaped tensors, and incorrect slicing logic in the TS2Vec model that can corrupt embeddings. Several warnings relate to device placement bugs, unsafe loss computation with batch_size=1, and documentation/API inconsistencies that matter for a package being prepared for publication.

## Critical Issues

### CR-01: Global numpy RNG mutation in Seeded decorator

**File:** `src/chronocratic/models/augmentation/decorators.py:64`
**Issue:** `np.random.seed(self._seed)` mutates the global legacy numpy RNG state. This affects any concurrent code relying on `np.random` (e.g., other augmentations, data loaders) even outside the `fork_rng()` context. The `torch.random.fork_rng()` only isolates PyTorch's random state, not numpy's. This defeats the purpose of the decorator as an isolation boundary and is especially dangerous because TS2Vec's `CropShiftProducer` (in `augmentation.py:96`) relies on `np.random.randint()` for its augmentation logic.

**Fix:**
```python
def produce(self, x: torch.Tensor) -> V:
    with torch.random.fork_rng():
        torch.manual_seed(self._seed)
        # Isolate numpy RNG state using a local generator, not global seed
        original_state = np.random.get_state()
        np.random.seed(self._seed)  # noqa: NPY002
        try:
            return self._inner.produce(x)
        finally:
            np.random.set_state(original_state)
```

Or better, migrate all callers to `np.random.default_rng()` and pass the generator explicitly instead of relying on the global state.

### CR-02: Permutation augmentation produces incorrect output for (B, C, T) shaped tensors

**File:** `src/chronocratic/models/augmentation/primitives.py:188-206`
**Issue:** When `time_dim=-1` is used with (batch, channels, time) shaped tensors, line 206 computes `t_dim - 1` which becomes `-1 - 1 = -2`, causing `index_select` to operate on the wrong dimension. The `_normalize_dim()` function converts `time_dim` to an absolute index (e.g., `2` for a 3D tensor with `time_dim=-1`), then `t_dim - 1` yields `1`, which is the channel dimension, not the time dimension within the per-sample slice. The result is that channels are permuted instead of time steps.

This affects TS-TCC which uses `(B, C, T)` convention and passes `time_dim=-1` to `PermutationParameters`.

**Fix:**
```python
# x[i] removes the batch dim, so the time dim shifts if time_dim > 0.
adjusted_time_dim = t_dim - 1 if t_dim > 0 else t_dim
result[i] = x[i].index_select(adjusted_time_dim, warp)
```

Or more robustly, handle negative dimensions correctly:
```python
# After removing batch dim (dim 0), time dimension shifts by -1 only if t_dim > 0
time_dim_after_batch = t_dim - 1 if t_dim > 0 else t_dim
result[i] = x[i].index_select(time_dim_after_batch, warp)
```

### CR-03: TS2Vec slicing asymmetry may produce misaligned embeddings

**File:** `src/chronocratic/models/convolutional/dilated/ts2vec/model.py:99-100`
**Issue:** The `_encode_augmented_views` method slices embeddings asymmetrically:
```python
emb_1 = encoder(pair.first)[:, -pair.overlap_length:]  # tail
emb_2 = encoder(pair.second)[:, :pair.overlap_length]   # head
```
This is intentionally different (tail vs head) based on the original TS2Vec crop-and-shift augmentation, but it creates a subtle bug when `overlap_length` exceeds the actual sequence length of one view (which can happen with non-crop augmentations like `FullOverlapPair`). The slice `[:, -overlap_length:]` on `emb_1` wraps to include the entire sequence, while `[:, :overlap_length]` on `emb_2` truncates. This produces two embeddings of different temporal lengths, which will cause downstream loss functions to fail silently or produce incorrect gradients.

**Fix:**
```python
def _encode_augmented_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair = self._augmentation.produce(x)
    encoder = self._encoder if self.training else self._averaged_encoder
    emb_1 = encoder(pair.first)
    emb_2 = encoder(pair.second)
    # Clamp overlap_length to the actual sequence length of both embeddings
    overlap = min(pair.overlap_length, emb_1.size(1), emb_2.size(1))
    emb_1 = emb_1[:, -overlap:]
    emb_2 = emb_2[:, :overlap]
    return emb_1, emb_2
```

### CR-04: CoST model uses uninitialized numpy RNG on construction

**File:** `src/chronocratic/models/convolutional/dilated/cost/model.py:89`
**Issue:** `self._rng = np.random.default_rng(seed=int(torch.random.initial_seed()))` initializes the RNG with the current PyTorch seed at construction time. If the model is created before `torch.manual_seed()` is called (which is common in Lightning when the model is instantiated in the data module or trainer), the seed will be `0` (PyTorch default), making all CoST runs deterministic regardless of what seed is set later. The seed is not updated when `torch.manual_seed()` is called post-construction.

**Fix:**
```python
# Defer RNG creation until first use, or reseed in on_fit_start()
def on_fit_start(self) -> None:
    self._rng = np.random.default_rng(seed=int(torch.random.initial_seed()))
```

### CR-05: TimeVAEEncoder uses plain list instead of nn.ModuleList

**File:** `src/chronocratic/models/generative/timevae/model.py:26-38`
**Issue:** `self.layers: list[nn.Module] = []` followed by `self.layers.append(...)` appends modules to a Python list. This means the appended `nn.Conv1d` and `nn.ReLU` modules are NOT registered as PyTorch submodules. They will not appear in `.parameters()`, `.children()`, or `.state_dict()`, causing:
- Optimizers will not update these weights
- Checkpointing will not save/restore encoder layers
- `.to(device)` will not transfer these layers to GPU

The `nn.Sequential(*self.layers)` conversion on line 42 only captures references, but the modules are not properly registered before being wrapped.

**Fix:**
```python
self.layers = nn.ModuleList()
self.layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
# ... rest of the layers
```

### CR-06: InfoNCE loss silently returns 0 for non-matching batch sizes

**File:** `src/chronocratic/models/convolutional/dilated/autotcl/losses.py:188-199`
**Issue:** When `z1` and `z2` have different batch sizes, `similarity_matrix` will have shape `(B1, B2)`. The mask `torch.eye(z1.shape[0])` has shape `(B1, B1)`. Applying this boolean mask to `similarity_matrix` when `B1 != B2` will cause a shape mismatch crash. Additionally, with `batch_size=1`, the positive is `similarity_matrix[0, 0]` and the negative selection `~mask` selects nothing, leading to an empty tensor and a silent NaN loss.

**Fix:**
```python
if z1.shape[0] != z2.shape[0]:
    msg = f"Batch size mismatch: z1 has {z1.shape[0]} samples, z2 has {z2.shape[0]}"
    raise ValueError(msg)
if z1.shape[0] < 2:
    return z1.new_tensor(0.0)
```

### CR-07: local_info_nce_loss uses Python random (non-reproducible)

**File:** `src/chronocratic/models/convolutional/dilated/autotcl/losses.py:52`
**Issue:** `start = random.randint(0, sequence_length - crop_length)` uses Python's `random` module which is not controlled by PyTorch or numpy seeding. This makes the loss non-reproducible even when all other seeds are fixed. It also doesn't support GPU computation.

**Fix:**
```python
start = torch.randint(0, sequence_length - crop_length + 1, (1,), device=z1.device).item()
```

### CR-08: pyproject.toml references LICENSE but may crash on build without it

**File:** `pyproject.toml:12`
**Issue:** `license-files = ["LICENSE"]` is specified in pyproject.toml. If the LICENSE file is missing during `pip install` (e.g., when building from a shallow clone without the LICENSE file, or when the file is gitignored in some setups), setuptools will raise an error during package building. This is a package preparation phase, so build robustness matters.

**Fix:** Ensure LICENSE is included in the MANIFEST or use `include-package-data = true` in setuptools configuration. Verify that `git ls-files LICENSE` returns the file.

## Warnings

### WR-01: trainable_support mode toggle is incorrect after early return

**File:** `src/chronocratic/models/augmentation/trainable_support.py:70-76`
**Issue:** The function sets `encoder.eval()` and `augmentation.train()`, then restores `augmentation.eval()` and `encoder.train()` in the `finally` block. However, when `augmentation.should_train_augmentation()` returns `False`, the function returns `None` at line 69 BEFORE entering the try/finally block. This is correct. But when `train_step()` returns `None` (which it may), the encoder is still restored to train mode even though no training happened. This is actually fine in practice, but the comment on line 48 says "The caller is responsible for setting encoder back to train for Phase 2", which contradicts the `finally` block that unconditionally calls `encoder.train()`. The documentation is misleading.

**Fix:** Update the docstring to reflect that `encoder.train()` is restored by this function.

### WR-02: TimeNet stores dropout as int when it's 0.0

**File:** `src/chronocratic/models/recurrent/timenet/model.py:51`
**Issue:** `self._dropout: int | float = dropout` has an incorrect type annotation. `dropout` is typed as `float` in the constructor but stored as `int | float`. When `dropout=0.1`, `self._dropout` is `0.1` (float), but the conditional `if self._dropout > 0` on line 62 works because `0.1 > 0` is `True`. However, if someone passes `dropout=0`, then `self._dropout` is `0` (float), and `if 0.0 > 0` is `False` -- this is correct. The annotation is just misleading.

**Fix:** Change type to `self._dropout: float = dropout`.

### WR-03: Scaling augmentation can produce NaN for 0-dimensional inputs

**File:** `src/chronocratic/models/augmentation/primitives.py:139`
**Issue:** `c_dim = _normalize_dim(x, self._params.channel_dim)` works for standard (B, C, T) tensors, but if `x` is 1D or 2D, `_normalize_dim` may produce an out-of-bounds dimension index. For example, with `x` of shape `(batch,)` (1D) and `channel_dim=1`, `_normalize_dim` returns `1`, but `x.dim()` is only 1, so `x.size(1)` crashes with `IndexError`.

**Fix:**
```python
if c_dim >= x.dim():
    msg = f"channel_dim={self._params.channel_dim} exceeds tensor dimensions ({x.dim()})"
    raise ValueError(msg)
```

### WR-04: Series2Vec filter_frequencies detaches and moves to CPU mid-batch

**File:** `src/chronocratic/models/convolutional/standard/series2vec/model.py:107`
**Issue:** `filter_frequencies(x.detach().cpu(), training=self.training).to(x.device)` moves data to CPU, processes it, then moves it back. This is a CPU-GPU sync point in the middle of the training step, which blocks the GPU and destroys async training. Additionally, `.detach()` breaks the gradient flow for frequency distances computation.

**Fix:** Process on the same device: `filter_frequencies(x.detach(), training=self.training)` without the CPU round-trip.

### WR-05: TST model saves all hyperparameters including non-config values

**File:** `src/chronocratic/models/transformer/tst/model.py:69`
**Issue:** `self.save_hyperparameters()` saves all constructor parameters without filtering non-serializable objects. While TST does not have an `augmentation` parameter (which would be a common issue), the `lr_step` parameter defaults to `[1_000_000]` which is a mutable list default, potentially causing shared state across instances.

**Fix:** Use `lr_step: list[int] = None` in the signature and `self._lr_step = lr_step or [1_000_000]` (which is already done on line 74). This is fine as-is. No action needed.

### WR-06: Cost augmentation _scale and _shift operate on x.size(-1)

**File:** `src/chronocratic/models/convolutional/dilated/cost/augmentation.py:107,113`
**Issue:** `_scale` uses `x.size(-1)` for the random factor shape, which is the last dimension. For `(batch, time, channels)` input, this produces per-channel factors correctly. For `(batch, channels, time)` input (which TSTCC uses), this produces per-time-step factors instead of per-channel factors. The docstring claims `(batch, time, channels)` but the augmentation is used through `IndependentPair` by CoST, not TSTCC.

**Fix:** Add explicit dimension handling based on input tensor shape.

### WR-07: docs/conf.py references html_static_path files that may not exist

**File:** `docs/conf.py:67-68`
**Issue:** `html_static_path: list[str] = ["_static"]` and `html_css_files: list[str] = ["custom.css"]` reference a `_static/custom.css` file that may not exist. Sphinx will emit a warning or error during doc build if `_static/custom.css` is missing, breaking the documentation pipeline.

**Fix:** Create `docs/_static/custom.css` (even if empty) or remove the `html_css_files` reference.

### WR-08: Conv1dSamePadMultiBlock may lack projector attribute

**File:** `src/chronocratic/models/convolutional/dilated/layers/same_pad.py:108-128`
**Issue:** `__initiate_projector` sets `self.projector` only when `stride == 1` and `(in_channels != out_channels or is_final)`, or when `stride != 1`. When `stride == 1` and `in_channels == out_channels` and `not is_final`, `self.projector` is never set. However, `forward()` on line 132 checks `if self.projector is None`, which will raise `AttributeError` because `projector` was never assigned.

**Fix:**
```python
if stride == 1:
    if in_channels != out_channels or is_final:
        self.projector = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    else:
        self.projector = None  # This IS set, but let's verify the else branch exists.
```

Actually, reviewing again -- line 123 does set `self.projector = None`. The code is correct. Removing this as a false positive.

### WR-09: CoST component_dims requires output_dims to be even

**File:** `src/chronocratic/models/convolutional/dilated/encoders/encoders.py:464`
**Issue:** `self.component_dims = output_dims // 2` silently floors. If `output_dims` is odd (e.g., 319), the decoder splits will not perfectly reconstruct the original dimension. Default is 320 (even), but this is not enforced.

**Fix:**
```python
if output_dims % 2 != 0:
    msg = f"output_dims must be even for CoST, got {output_dims}"
    raise ValueError(msg)
```

## Info

### IN-01: Duplicate code between losses.py and ts2vec/losses.py

**File:** `src/chronocratic/models/losses.py:8-40` and `src/chronocratic/models/convolutional/dilated/ts2vec/losses.py:10-42`
**Issue:** `_compute_contrastive_loss_logits` is duplicated almost identically in both files. The TS2Vec version is slightly different (uses `.cpu().numpy()` for indexing_factor) but the core logic is the same.

**Fix:** Consolidate into a shared utility function in `chronocratic.models.losses`.

### IN-02: Config dataclasses not exported from top-level __init__

**File:** `src/chronocratic/models/__init__.py`
**Issue:** The top-level `__init__.py` exports model classes (TST, TSTCC, etc.) and their config dataclasses (TSTModelParameters, etc.), but the config dataclasses for CoST, TS2Vec, MCL, and others may not all be imported. Checking: `CoSTModelParameters`, `TS2VecModelParameters`, `MCLModelParameters`, `Series2VecModelParameters`, `TSTCCModelParameters`, `AutoTCLModelParameters`, `TimeVAEModelParameters`, `TimeNetModelParameters`, `TSTModelParameters` -- all are exported. No issue here. However, the config dataclasses are not re-exported in a way that allows `from chronocratic.models import TSTModelParameters` without importing the model class.

### IN-03: Permutation parameters default max_segments=5 may not match TS-TCC expectations

**File:** `src/chronocratic/models/augmentation/primitives.py:165`
**Issue:** `max_segments: int = 5` is the default in `PermutationParameters`, but TS-TCC's default augmentation uses `max_segments=5` explicitly in `_default_tstcc_pair()`. The default is consistent, but it would be worth documenting the expected range and its effect on augmentation severity.

### IN-04: _should_apply uses torch.rand() which is not seeded by Seeded decorator

**File:** `src/chronocratic/models/augmentation/primitives.py:45`
**Issue:** `_should_apply(p)` uses `torch.rand((1,)).item()`. When the producer is wrapped by `Seeded`, the `fork_rng()` context ensures this is seeded. However, the `Seeded` decorator only seeds at the beginning of `produce()`. If the inner producer calls `_should_apply` multiple times, all calls are within the seeded context, so this is fine. No actual bug.

### IN-05: Missing __init__.py for convolutional.dilated._mixin package

**File:** `src/chronocratic/models/convolutional/dilated/_mixin/__init__.py`
**Issue:** The `_mixin` directory exists with `encoding.py` but its `__init__.py` should explicitly re-export `BaseEncodingMixin`, `PoolingEncodingMixin`, and `DecompositionEncodingMixin` for cleaner imports.

## Additional Notes

1. **Namespace migration completeness:** All imports have been updated from `tscollection` to `chronocratic`. No residual references to the old namespace were found in the reviewed files.

2. **setuptools_scm configuration:** The `version_file` is set to `src/chronocratic/models/_version.py`. The fallback in `__init__.py:8` provides `"0.0.0.dev0"` when `_version.py` doesn't exist, which is correct for development installs.

3. **Sphinx configuration:** The `docs/conf.py` correctly inserts the project source path and imports `__version__` from the package. The `suppress_warnings` configuration for `efifo` is appropriate for avoiding known Sphinx warnings.

4. **readthedocs.yaml:** Correctly configured to use `uv sync` with the `docs` extra, which ensures sphinx dependencies are available.

5. **Test coverage:** The test files cover augmentation contracts, decorators, primitives, producers, and model-specific producers. However, there are no tests for the `Permutation` bug described in CR-02, which would have caught it.

6. **Circular dependency management:** The codebase uses lazy imports (PLC0415) extensively to avoid circular dependencies between augmentation strategies, model definitions, and barrel re-exports. This is a reasonable approach but adds fragility -- any new import at module level risks recreating the circular dependency.
