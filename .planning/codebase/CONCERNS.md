# Concerns and Technical Debt

**Analysis Date:** 2026-06-17

## Code Quality Issues

**GPU-to-CPU transfers in contrastive loss functions during forward pass:**
`src/chronocratic/models/losses/contrastive.py` (lines 68, 98) call `.cpu().numpy()` to build an `indexing_factor` from a GPU tensor, then pass it to `_compute_contrastive_loss_logits` where it gets concatenated back on-device. This forces a synchronous host-device transfer every batch. Replace with pure-torch equivalents (`torch.arange(..., device=...)`) since the indexing factor is simply sequential integers.

**GPU-to-CPU round-trip in vendored SoftDTW CPU autograd function:**
`src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py` (lines 274, 287-288) in `_SoftDTW` detach tensors to CPU, convert to NumPy for Numba JIT computation, then copy results back. Every forward/backward pass incurs this sync. The CUDA path avoids it but `_SoftDTW` is the fallback for non-CUDA runs and sequences > 1024. Consider replacing vendored code with the `soft-dtw` package (pure PyTorch differentiable implementation).

**Repetitive `.cpu()` calls inside DataLoader loops in encoding mixins:**
`src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` (line 269) moves intermediate representations to CPU inside the inference loop, then `torch.cat` on CPU. `src/chronocratic/models/_mixin/encoding.py` (line 144) moves the final result back to `data_device`. This is correct for memory-constrained inference but creates unnecessary host-device synchronization in high-throughput scenarios. Consider deferring `.cpu()` until after `torch.cat`.

**Large monolithic encoder file (639 lines):**
`src/chronocratic/models/convolutional/dilated/encoders/encoders.py` contains `BaseTimeSeriesEncoder` plus four distinct subclasses (`AutoTCLTimeSeriesEncoder`, `CoSTTimeSeriesEncoder`, `TS2VecTimeSeriesEncoder`, `AutoTCLAugmentationTimeSeriesEncoder`). Each encoder class is 100-200 lines. Split into per-encoder files under a sub-package for navigability and reduced import coupling.

**Inconsistent Lightning import styles:**
Nine files use `import lightning.pytorch as pl` but `src/chronocratic/models/recurrent/timenet/model.py` uses `from lightning.pytorch import LightningModule`. Standardize to one pattern project-wide.

**Bare `print()` statement outside `__main__` guard:**
`src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py` (line 409) has a `print()` call in a module-level function (`timed_run`) that is not guarded by `if __name__ == "__main__"`. It executes whenever that function is called from library code.

**Excessive `.clone()` calls in encoder masking/processing (8 call sites):**
`src/chronocratic/models/convolutional/dilated/encoders/encoders.py` (lines 151, 162, 356, 489, 497, 603, 611) perform `.clone()` before masking operations. Some clones may be unnecessary if the input tensor is not reused. Audit each call; consider `torch.where()` for masking instead of clone-then-multiply.

**`vae_base.py` uses `next(self.parameters())` for device detection:**
`src/chronocratic/models/generative/timevae/vae_base.py` (lines 84, 96, 104) repeatedly calls `next(self.parameters())` to infer device. This is fragile if the model has no parameters or parameters span multiple devices. The class inherits from `LightningModule` which provides `self.device`.

**Inconsistent `from __future__ import annotations` usage (25 of 87 files):**
CLAUDE.md says "do not always write" it, but the selective usage means some files can use forward references (e.g., `tuple[torch.Tensor, ...]`) while others crash at runtime. A uniform policy — either all files or only where circular imports demand it — would reduce confusion.

## Architecture Concerns

**Circular import chains requiring lazy imports in model files:**
Multiple modules use `# noqa: PLC0415` lazy imports to break cycles:
- `src/chronocratic/models/convolutional/dilated/autotcl/augmentation/training.py` (line 70): `augmentation/__init__` imports from strategies, which imports from `encoders.py`, which imports from `augmentation/__init__`.
- `src/chronocratic/models/convolutional/dilated/ts2vec/augmentation.py` (line 80): similar chain.
- `src/chronocratic/models/convolutional/standard/tstcc/model.py` (line 74): lazy import of augmentations.
- `src/chronocratic/models/convolutional/dilated/autotcl/model.py` (lines 68, 200): lazy imports of augmentation and utils modules.

The augmentation contract redesign (`#18`) improved this but did not eliminate the cycles. A clean ports-and-adapters boundary between model logic and augmentation producers would remove the need for lazy imports entirely.

**Two parallel encoding mixin hierarchies with similar names:**
`src/chronocratic/models/_mixin/encoding.py` defines `BasicEncodingMixin` (lightweight, for fixed-sequence models like TST, TimeVAE, TimeNet). `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` defines `BaseEncodingMixin`, `PoolingEncodingMixin`, and `DecompositionEncodingMixin` (heavy, for sliding-window models). The naming collision (`Basic` vs. `Base`) and duplicated DataLoader/encoding logic increase maintenance burden. A unified base with strategy hooks (e.g., `_get_eval_strategy`) would share the boilerplate while keeping the branching in focused override methods.

**`RepresentationBackbone` protocol is runtime-checkable but not enforced at construction:**
`src/chronocratic/models/supervised/supervised.py` defines `RepresentationBackbone` as a `runtime_checkable` Protocol requiring `representation_dim`. `SupervisedModule.__init__` accepts any `nn.Module` for the `backbone` parameter. If a backbone lacks `representation_dim`, the error manifests at head construction time, not at module creation. The factory at `src/chronocratic/models/supervised/factory.py` should validate `isinstance(backbone, RepresentationBackbone)` early.

**Four models disable `automatic_optimization` with manual optimizer stepping:**
`src/chronocratic/models/convolutional/standard/tstcc/model.py` (line 66), `src/chronocratic/models/convolutional/dilated/autotcl/model.py` (line 81), `src/chronocratic/models/convolutional/dilated/cost/model.py` (line 76), `src/chronocratic/models/convolutional/dilated/ts2vec/model.py` (line 59). Each implements its own `optimizer.step()` / `optimizer.zero_grad()` pattern manually. A shared base or callback for multi-optimizer stepping would reduce the risk of missing `zero_grad()` or incorrect closure usage.

**Numba/CUDA hard dependency in core library:**
`src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py` imports `numba.cuda` at module load time. Users without CUDA or Numba installed get import errors even when they never use SoftDTW. Consider lazy import or a separate `soft-dtw-cuda` optional dependency.

**PyTorch SWA utils (`AveragedModel`) deprecation risk:**
`src/chronocratic/models/convolutional/dilated/ts2vec/model.py` and `src/chronocratic/models/convolutional/dilated/autotcl/model.py` use `torch.optim.swa_utils.AveragedModel` for EMA tracking. The SWA utilities have been flagged for deprecation. A `type: ignore  # noqa: PGH003` at `autotcl/model.py:175` acknowledges the tension. Replace with a custom EMA `nn.Module` (shadow parameters + momentum update).

## Missing Functionality

**No encoder-decoder infrastructure (branch purpose not yet realized):**
The `encoder-decoder-infra` branch name indicates work on generic encoder-decoder patterns, but the codebase only has VAE-style encoder/decoder pairs (`src/chronocratic/models/generative/timevae/model.py`). There is no reusable encoder-decoder abstraction for sequence-to-sequence, forecasting, or conditional generation tasks. The supervised module wraps a "backbone" with a classifier/regression head but has no decoder concept.

**No `LightningDataModule` implementations:**
CLAUDE.md states "Use LightningDataModule or well-structured data-loading classes." No `LightningDataModule` exists in the codebase. All data loading is ad-hoc per-model or handled externally by callers. This is a significant gap in Lightning best practices.

**No forecasting capability:**
`pyproject.toml` lists "forecasting" as a keyword, but the library has no forecasting head, sequence-to-sequence decoder, temporal projection layer, or rolling-origin evaluation infrastructure. The existing models are self-supervised representation learners, supervised classifiers/regressors, and a VAE.

**No config serialization round-trip tests:**
Models support `save_hyperparameters()` and can be reconstructed via `from_config()`, but there are no tests verifying `model.hparams -> Model(**hparams)` produces functionally identical instances across all 10 model classes.

**No input shape validation in `BasicEncodingMixin.encode()`:**
`src/chronocratic/models/_mixin/encoding.py` accepts arbitrary tensors without dimension validation. `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` (line 216) validates 3D input, but `BasicEncodingMixin` does not. Mismatched shapes crash inside the encoder rather than with a clear error.

**No validation on augmentation `metadata` dict:**
The augmentation contract uses `metadata: dict[str, Any]` without type enforcement. If an augmentation omits a key the model expects (e.g., `overlap_length` for TS2Vec's `AlignedPair`), the error manifests deep in the training loop as a `KeyError`.

## Maintenance Burden

**Vendored third-party SoftDTW code without version pinning:**
`src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py` is a 453-line vendored copy of Mehran Maghoumi's SoftDTW CUDA implementation (MIT license). The code has been modified from the original (e.g., added bare `assert` statements). Upstream fixes require manual cherry-picking. Consider using the `soft-dtw` package with proper attribution.

**No dedicated unit tests for 60 of 87 source files:**
Test coverage is heavily concentrated on augmentation logic (22 augmentation-related test files, 418 test functions total). Core model files (`model.py`, `encoder.py`, `losses.py`, `config.py`, `network.py`, `temporal_contrast.py`), utility files (`utils.py`), and layer files (`general.py`, `dilated.py`, `same_pad.py`) all lack dedicated unit tests. The existing tests are integration-level producer tests that verify models train without crashing.

**`tqdm` progress bars embedded in library code:**
`src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` (lines 122, 242) use `tqdm.tqdm()` for encoding progress. Progress bars in library code interfere with CI logs, automated pipelines, and Jupyter notebook outputs. Use `logging` or make tqdm opt-in via a callback parameter.

**`type: ignore` suppressions indicate incomplete type coverage:**
- `src/chronocratic/models/convolutional/dilated/autotcl/model.py` (line 175): `update_parameters` call on `AveragedModel`.
- `src/chronocratic/models/convolutional/dilated/autotcl/augmentation/methods.py` (line 114): dataclass construction from kwargs.

These suggest internal APIs are not fully annotated or are out of sync with PyTorch's evolving type stubs.

**Documentation inconsistency in docstring styles:**
Most modules use Google-style docstrings (per CLAUDE.md), but `src/chronocratic/models/convolutional/dilated/encoders/encoders.py` (line 36) uses a bare parameter-list format without return documentation, and the vendored `soft_dtw_cuda.py` has minimal docstrings.

## Known Workarounds

**Defensive `.to(device)` transfers in AutoTCL model:**
`src/chronocratic/models/convolutional/dilated/autotcl/model.py` (lines 160, 229) have comments like "Defensive device transfer -- original model called `.to(x.device)`" followed by conditional `.to()` calls. Device placement is not guaranteed to be consistent across augmentation pipelines. The proper fix is to enforce device consistency at the augmentation producer level.

**`_postprocess` uses positional tuple indexing:**
`src/chronocratic/models/generative/timevae/model.py` (line 178) extracts `z_mean` via `output[0]` from a `(z_mean, z_log_var, z)` tuple. Fragile if the encoder return signature changes. Use a `dataclass` or `NamedTuple` for the encoder output.

**`_compute_fixed_representation()` returns `None` by default:**
`src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` (line 85) returns `None` in the base class with documentation stating "pooling-based models override to return a real slice." If a subclass forgets to override, downstream code receives `None` which crashes when used as a tensor dimension. Raise `NotImplementedError` instead.

## Security Considerations

**Numba CUDA JIT compilation at import time:**
`src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py` compiles CUDA kernels via `@cuda.jit` eagerly when the module loads. This has implications for air-gapped environments and containers without GPU access. Consider lazy compilation on first use.

**`numba` import without capability check:**
If the GPU does not support the compute capability required by the Numba CUDA kernel, the kernel may fail to launch silently or produce incorrect results. No runtime capability check exists.

## Recommendations

**Priority 1 — Fix GPU-CPU transfer bottlenecks:**
- Replace `.cpu().numpy()` in `src/chronocratic/models/losses/contrastive.py` with pure-torch indexing (`torch.arange`).
- Evaluate replacing vendored SoftDTW with the `soft-dtw` PyTorch package to eliminate Numba dependency.

**Priority 2 — Consolidate encoding mixin duplication:**
- Unify `BasicEncodingMixin` and `BaseEncodingMixin` into a single hierarchy with strategy-based evaluation hooks. Share DataLoader boilerplate; keep model-specific pooling/masking as override methods.

**Priority 3 — Add unit tests for core model components:**
- Write tests for loss functions, encoder classes, pooling utilities, and SoftDTW paths. Prioritize `src/chronocratic/models/losses/contrastive.py`, `src/chronocratic/models/convolutional/dilated/encoders/encoders.py`, and `src/chronocratic/models/utils.py`.

**Priority 4 — Resolve circular imports in augmentation boundary:**
- Define a clean protocol-based boundary so model files do not need lazy imports from augmentation modules. Move augmentation default factories out of the `augmentation/__init__.py` barrel.

**Priority 5 — Build encoder-decoder infrastructure:**
- Create a generic `EncoderDecoder` base class on the `encoder-decoder-infra` branch for sequence-to-sequence and forecasting tasks. Factor out the VAE encoder/decoder pattern.

**Priority 6 — Remove `tqdm` from library internals:**
- Replace `tqdm.tqdm()` in `src/chronocratic/models/convolutional/dilated/_mixin/encoding.py` with `logging.info` or an opt-in callback.

**Priority 7 — Replace `AveragedModel` with custom EMA:**
- Implement a standalone `EMA(nn.Module)` wrapper in `src/chronocratic/models/layers/general.py` to eliminate the PyTorch SWA dependency.

**Priority 8 — Split `encoders.py` into per-model files:**
- `src/chronocratic/models/convolutional/dilated/encoders/` should have `base.py`, `auto_tcl.py`, `cost.py`, `ts2vec.py`.

---

*Concerns audit: 2026-06-17*
