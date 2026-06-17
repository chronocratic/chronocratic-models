---
phase: 260611-lal-bug-fixes-1
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - src/tscollection/models/convolutional/standard/mcl/losses.py
  - src/tscollection/models/convolutional/standard/mcl/model.py
  - src/tscollection/models/convolutional/standard/mcl/config.py
  - src/tscollection/models/convolutional/standard/series2vec/model.py
  - src/tscollection/models/convolutional/standard/series2vec/network.py
  - src/tscollection/models/convolutional/standard/series2vec/config.py
  - src/tscollection/models/convolutional/standard/series2vec/filters.py
  - src/tscollection/models/convolutional/standard/series2vec/losses.py
  - src/tscollection/models/recurrent/timenet/model.py
  - src/tscollection/models/recurrent/timenet/config.py
  - src/tscollection/models/convolutional/standard/tstcc/model.py
  - src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py
  - src/tscollection/models/convolutional/standard/tstcc/encoder.py
  - src/tscollection/models/convolutional/standard/tstcc/losses.py
  - src/tscollection/models/transformer/tst/model.py
  - src/tscollection/models/transformer/tst/ts_transformer.py
  - src/tscollection/models/transformer/tst/loss.py
  - src/tscollection/models/generative/timevae/model.py
  - src/tscollection/models/generative/timevae/vae_base.py
autonomous: true
requirements: [C1, C2, C3, C4, I1, I2, I3, I4, I5, I6, I7, I8, R1, R2, R3, S1, S2, S3, S4]

must_haves:
  truths:
    - "MixUpLoss derives batch_size and device from z_aug tensor at runtime, not init params"
    - "Series2Vec configure_optimizers does not pass warmup kwarg to AdamW"
    - "Series2Network pretrain_forward uses out[:, 0, :] indexing on GAP output, not squeeze(0)"
    - "TimeNet accepts feat_dim parameter and uses it for GRUWrapper input_size and decoder output"
    - "Series2Vec default optimizer is 'RAdam' (not 'Adam')"
    - "TSTCC validation_step wraps loss computation in torch.no_grad()"
    - "TSTCC TemporalContrast raises ValueError when seq_len <= timestep"
    - "TSTransformerEncoder exposes encode_representations() public method"
    - "TST get_representations delegates to encoder.encode_representations()"
    - "filter_frequencies uses deterministic path (highpass) when training=False"
    - "TimeVAE _step does not divide loss by batch size"
    - "TimeVAE predict() restores training mode after execution via was_training flag"
    - "TST configure_gradient_clipping respects gradient_clip_val parameter (defaults to 4.0)"
    - "TimeVAEEncoder deletes self.layers after nn.Sequential creation"
    - "TimeNet has _postprocess method selecting final timestep (output[:, -1, :])"
    - "Model public attrs use underscore prefix (_alpha, _learning_rate, etc.)"
    - "TimeVAE model imports Seasonality from layers.general (no duplicate alias)"
    - "TSTCC encoder.py, losses.py, temporal_contrast.py remove from __future__ import annotations"
    - "All modules have __all__ declarations"
  artifacts:
    - path: "src/tscollection/models/convolutional/standard/mcl/losses.py"
      provides: "MixUpLoss with runtime-derived batch_size/device"
    - path: "src/tscollection/models/convolutional/standard/series2vec/model.py"
      provides: "Series2Vec without warmup kwarg bug, RAdam default, deterministic filters"
    - path: "src/tscollection/models/convolutional/standard/series2vec/network.py"
      provides: "Series2VecNetwork with explicit [:, 0, :] indexing"
    - path: "src/tscollection/models/recurrent/timenet/model.py"
      provides: "TimeNet with feat_dim param and _postprocess"
    - path: "src/tscollection/models/convolutional/standard/tstcc/model.py"
      provides: "TSTCC with no_grad validation_step"
    - path: "src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py"
      provides: "TemporalContrast with seq_len guard"
    - path: "src/tscollection/models/transformer/tst/ts_transformer.py"
      provides: "TSTransformerEncoder with encode_representations() method"
    - path: "src/tscollection/models/transformer/tst/model.py"
      provides: "TST with gradient_clip_val support"
    - path: "src/tscollection/models/generative/timevae/model.py"
      provides: "TimeVAEEncoder without stale self.layers"
    - path: "src/tscollection/models/generative/timevae/vae_base.py"
      provides: "BaseVAE without loss/n division, predict() mode restore"
  key_links:
    - from: "src/.../mcl/model.py"
      to: "src/.../mcl/losses.py"
      via: "MixUpLoss(device=, batch_size=) removed from constructor call"
      pattern: "MixUpLoss\\(\\)"
    - from: "src/.../series2vec/model.py"
      to: "src/.../series2vec/filters.py"
      via: "filter_frequencies(training=self.training)"
      pattern: "filter_frequencies.*training"
    - from: "src/.../tst/model.py"
      to: "src/.../tst/ts_transformer.py"
      via: "get_representations delegates to encoder.encode_representations"
      pattern: "encode_representations"
    - from: "src/.../timevae/vae_base.py"
      to: "predict method"
      via: "was_training flag restores mode"
      pattern: "was_training"
---

<objective>
Fix all 17 verified bugs from PR #10 deep source review (excluding TimeNet decoder dropout bugs per user decision). Grouped into 3 tasks: critical crashes, correctness/risk, style/consistency.

Purpose: Eliminate runtime crashes, non-deterministic validation, stale state, and consistency issues across 6 models (MCL, Series2Vec, TimeNet, TSTCC, TST, TimeVAE).
Output: 17 corrected source files, all fixes verified against original source repos per D-04.
</objective>

<execution_context>
@/Users/skaf/VSCodeProjects/tsmodels/.claude/gsd-core/workflows/execute-plan.md
@/Users/skaf/VSCodeProjects/tsmodels/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@/Users/skaf/VSCodeProjects/tsmodels/.planning/quick/260611-lal-bug-fixes-1/260611-lal-CONTEXT.md
@/Users/skaf/VSCodeProjects/tsmodels/.planning/quick/260611-lal-bug-fixes-1/260611-lal-RESEARCH.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Critical crash fixes (C1-C4)</name>
  <files>
    src/tscollection/models/convolutional/standard/mcl/losses.py
    src/tscollection/models/convolutional/standard/mcl/model.py
    src/tscollection/models/convolutional/standard/mcl/config.py
    src/tscollection/models/convolutional/standard/series2vec/model.py
    src/tscollection/models/convolutional/standard/series2vec/config.py
    src/tscollection/models/convolutional/standard/series2vec/network.py
    src/tscollection/models/recurrent/timenet/model.py
    src/tscollection/models/recurrent/timenet/config.py
  </files>

  <action>
  Fix 4 critical bugs that cause runtime crashes. Per D-04 (derive runtime values from input tensors, not init-time params):

  **C1 — MCL MixUpLoss device/batch_size fixed at init:**
  - In `losses.py`: Remove `device` and `batch_size` params from `MixUpLoss.__init__()`. In `forward()`, derive `batch_size = z_aug.shape[0]` and `device = z_aug.device`. Use these local variables for `torch.eye()` calls.
  - In `model.py`: Remove `device` and `batch_size` params from `FCN.__init__()`. Update `MixUpLoss()` constructor call to `MixUpLoss()` (no args).
  - In `config.py`: Remove `batch_size` and `device` fields from `MCLModelParameters`. Update docstring.

  **C2 — S2V warmup kwarg passed to AdamW:**
  - In `model.py`: Remove the `warmup` kwarg addition in `configure_optimizers()`. Delete the `if self.optimizer_name == 'AdamW': kwargs['warmup'] = self.warmup` block entirely. Remove `warmup` from `__init__()` signature and `self.warmup` assignment.
  - In `config.py`: Remove `warmup` field from `Series2VecModelParameters`. Update docstring.

  **C3 — S2V squeeze(0) fragile:**
  - In `network.py`: After `x_src = self.gap(x_src)` producing `(B, D, 1)`, then `x_src.permute(2, 0, 1)` gives `(1, B, D)`. Same for `x_f`. Replace `.squeeze(0)` on lines 109, 110 with explicit indexing: `out[:, 0, :]` and `x_f[:, 0, :]`. This safely handles any batch_size (including 1) and any representation_dim.

  **C4 — TimeNet hardcodes input_size=1:**
  - In `model.py`: Add `feat_dim: int = 1` param to `TimeNet.__init__()`. Store as `self._feat_dim = feat_dim`. In `_build_encoder()`, use `GRUWrapper(self._feat_dim, ...)` instead of `GRUWrapper(1, ...)`. In `_build_decoder()`, change `nn.Linear(self.hidden_dims, 1)` to `nn.Linear(self.hidden_dims, self._feat_dim)`.
  - In `config.py`: Add `feat_dim: int = 1` field to `TimeNetModelParameters`. Update docstring.
  </action>

  <verify>
  - grep -c "self.batch_size" src/tscollection/models/convolutional/standard/mcl/losses.py == 0 (no more init-time batch_size)
  - grep -c "self.device" src/tscollection/models/convolutional/standard/mcl/losses.py == 0 (no more init-time device)
  - grep -c "warmup" src/tscollection/models/convolutional/standard/series2vec/model.py == 0 (warmup removed)
  - grep -c "warmup" src/tscollection/models/convolutional/standard/series2vec/config.py == 0 (warmup removed)
  - grep ".squeeze(0)" src/tscollection/models/convolutional/standard/series2vec/network.py | grep -c "" == 0 (no more squeeze(0))
  - grep "feat_dim" src/tscollection/models/recurrent/timenet/model.py | grep -c "def __init__" > 0 (feat_dim in signature)
  - grep "feat_dim" src/tscollection/models/recurrent/timenet/config.py | grep -c "int" > 0 (feat_dim in config)
  </verify>

  <done>
  All 4 critical crash fixes applied. MCL MixUpLoss uses runtime-derived batch_size/device. S2V warmup removed from model+config. S2V squeeze(0) replaced with squeeze(). TimeNet accepts feat_dim and uses it for GRUWrapper and Linear layers.
  </done>
</task>

<task type="auto">
  <name>Task 2: Correctness and risk fixes (I1-I8, R1, R3)</name>
  <files>
    src/tscollection/models/convolutional/standard/series2vec/model.py
    src/tscollection/models/convolutional/standard/series2vec/config.py
    src/tscollection/models/convolutional/standard/series2vec/filters.py
    src/tscollection/models/convolutional/standard/series2vec/losses.py
    src/tscollection/models/recurrent/timenet/model.py
    src/tscollection/models/convolutional/standard/tstcc/model.py
    src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py
    src/tscollection/models/transformer/tst/model.py
    src/tscollection/models/transformer/tst/ts_transformer.py
    src/tscollection/models/generative/timevae/model.py
    src/tscollection/models/generative/timevae/vae_base.py
  </files>

  <action>
  Fix correctness, non-determinism, and risk bugs across remaining models:

  **I1 — S2V default optimizer 'Adam' vs source 'RAdam':**
  - In `model.py`: Change `optimizer_name: str = 'RAdam'` (default parameter).
  - In `config.py`: Change `optimizer_name: OptimizerName = 'RAdam'`.

  **I2 — TSTCC validation_step has no torch.no_grad() (also covers R2):**
  - In `tstcc/model.py`: Wrap the `loss = self._compute_loss(batch)` call in `validation_step()` with `with torch.no_grad():`. This also prevents augmentations from firing in eval mode (R2).

  **I3 — TSTCC temporal_contrast crashes when seq_len <= timestep:**
  - In `temporal_contrast.py`: Before line 168 (`t_samples = torch.randint(...)`), add guard: `if seq_len <= self.timestep: raise ValueError(f"seq_len ({seq_len}) must be > timestep ({self.timestep})")`.

  **I4 — TST get_representations reaches into encoder internals:**
  - In `ts_transformer.py`: Add `encode_representations(self, x: Tensor, padding_masks: Tensor) -> Tensor` method that returns representations before output_layer. Per the research: permute input to (seq, batch, feat), apply project_inp * sqrt(d_model), pos_enc, transformer_encoder, act, permute back to (batch, seq, d_model), apply dropout1. Return that tensor.
  - In `model.py`: Replace `get_representations()` body with `return self._encoder.encode_representations(x, padding_masks)`.

  **I5 — S2V filter_frequencies fires random branch in validation:**
  - In `filters.py`: Add `training: bool = True` parameter to `filter_frequencies()`. Change `if torch.rand(()) < LOWPASS_PROBABILITY:` to `if training and torch.rand(()) < LOWPASS_PROBABILITY:`. When not training, always use deterministic highpass path.
  - In `model.py`: Update `filter_frequencies(x.detach().cpu())` call in `_calculate_loss()` to `filter_frequencies(x.detach().cpu(), training=self.training)`.

  **I6 — TimeVAE loss divided by batch size (source is unnormalized):**
  - In `vae_base.py`: In `_step()`, remove `/ n` division on line 54. Change `return loss / n, recon_loss / n, kl_loss / n` to `return loss, recon_loss, kl_loss`.

  **I7 — TimeVAE predict() calls self.eval() without restoring training mode:**
  - In `vae_base.py`: In `predict()`, replace `self.eval()` with `was_training = self.training; self.eval()`. After the `with torch.no_grad():` block, add `self.train(was_training)`.

  **I8 — TST hardcoded gradient_clip_val ignores Lightning param:**
  - In `tst/model.py`: In `configure_gradient_clipping()`, remove `del optimizer, gradient_clip_val, gradient_clip_algorithm`. Instead, use `gradient_clip_val` with a fallback: `if gradient_clip_val is None: gradient_clip_val = 4.0`. Pass `max_norm=gradient_clip_val` to `clip_grad_norm_`.

  **R1 — TimeVAE encoder self.layers stale after nn.Sequential:**
  - In `timevae/model.py`: After `self.encoder = nn.Sequential(*self.layers)` (line 41), add `del self.layers`.

  **R3 — TimeNet encode returns (B,T,D) — missing _postprocess to pool time dim:**
  - In `timenet/model.py`: Add `_postprocess(self, output: torch.Tensor) -> torch.Tensor:` method that returns `output[:, -1, :]` (select final timestep).
  </action>

  <verify>
  - grep "optimizer_name.*RAdam" src/tscollection/models/convolutional/standard/series2vec/model.py | grep -c "def __init__" > 0
  - grep "optimizer_name.*RAdam" src/tscollection/models/convolutional/standard/series2vec/config.py | grep -c "=" > 0
  - grep "torch.no_grad" src/tscollection/models/convolutional/standard/tstcc/model.py | grep -c "validation_step" > 0
  - grep "seq_len <= self.timestep" src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py | grep -c "ValueError" > 0
  - grep "encode_representations" src/tscollection/models/transformer/tst/ts_transformer.py | grep -c "def " > 0
  - grep "encode_representations" src/tscollection/models/transformer/tst/model.py | grep -c "self._encoder" > 0
  - grep "training=" src/tscollection/models/convolutional/standard/series2vec/model.py | grep -c "filter_frequencies" > 0
  - grep "/ n" src/tscollection/models/generative/timevae/vae_base.py | grep -c "return" == 0
  - grep "was_training" src/tscollection/models/generative/timevae/vae_base.py | grep -c "" > 0
  - grep "gradient_clip_val" src/tscollection/models/transformer/tst/model.py | grep -c "del" == 0
  - grep "del self.layers" src/tscollection/models/generative/timevae/model.py | grep -c "" > 0
  - grep "output\[:, -1, :\]" src/tscollection/models/recurrent/timenet/model.py | grep -c "_postprocess" > 0
  </verify>

  <done>
  All correctness and risk fixes applied: S2V defaults to RAdam, TSTCC val uses no_grad, TSTCC seq_len guarded, TST uses public encode_representations, S2V filter deterministic in val, TimeVAE loss unnormalized, TimeVAE predict restores mode, TST respects gradient_clip_val, TimeVAE deletes stale layers, TimeNet has _postprocess.
  </done>
</task>

<task type="auto">
  <name>Task 3: Style and consistency fixes (S1-S4)</name>
  <files>
    src/tscollection/models/convolutional/standard/mcl/model.py
    src/tscollection/models/convolutional/standard/series2vec/model.py
    src/tscollection/models/recurrent/timenet/model.py
    src/tscollection/models/generative/timevae/model.py
    src/tscollection/models/convolutional/standard/tstcc/model.py
    src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py
    src/tscollection/models/convolutional/standard/tstcc/encoder.py
    src/tscollection/models/convolutional/standard/tstcc/losses.py
    src/tscollection/models/convolutional/standard/series2vec/filters.py
    src/tscollection/models/convolutional/standard/series2vec/losses.py
    src/tscollection/models/generative/timevae/vae_base.py
    src/tscollection/models/transformer/tst/loss.py
    src/tscollection/models/transformer/tst/ts_transformer.py
  </files>

  <action>
  Apply style and consistency fixes across all models:

  **S1 — Public attrs to underscore prefix:**
  - In `mcl/model.py`: Change `self.alpha` to `self._alpha` (line 30) and `self.learning_rate` to `self._learning_rate` (line 31). Update `configure_optimizers()` to reference `self._learning_rate`.
  - In `series2vec/model.py`: Change `self.learning_rate` to `self._learning_rate`, `self.soft_dtw_gamma` to `self._soft_dtw_gamma`, `self.sync_dist` to `self._sync_dist`, `self.optimizer_name` to `self._optimizer_name`, `self.weight_decay` to `self._weight_decay`, `self.warmup` to `self._warmup`. Wait — warmup was already removed in Task 1 (C2). Skip warmup. Update all references to these attrs within the same file (e.g., `configure_optimizers`, `_build_soft_dtw`).
  - In `timenet/model.py`: Change `self.hidden_dims` to `self._hidden_dims`, `self.num_layers` to `self._num_layers`, `self.dropout` to `self._dropout`, `self.learning_rate` to `self._learning_rate`. Update `_build_encoder()`, `_build_decoder()`, `configure_optimizers()` to use underscore-prefixed names.

  CAUTION: The S1 changes reference attrs that were modified by Tasks 1 and 2. Review each file holistically before saving to ensure consistency. For example, if `self.hidden_dims` was used in `GRUWrapper(self._feat_dim, self.hidden_dims, ...)` in Task 1's C4 fix, it must now be `self._hidden_dims`.

  **S2 — Duplicate Seasonality alias:**
  - In `timevae/model.py`: Remove line 13 (`Seasonality = tuple[int, int]`). The `Seasonality` type is already imported from `layers.general` on line 6. Verify it is used in the `custom_seas` parameter type hints.

  **S3 — Remove unnecessary `from __future__ import annotations`:**
  - In `tstcc/model.py`: Remove line 1 (`from __future__ import annotations`).
  - In `tstcc/temporal_contrast.py`: Remove line 1 (`from __future__ import annotations`).
  - In `tstcc/encoder.py`: Remove line 1 (`from __future__ import annotations`).
  - In `tstcc/losses.py`: Remove line 1 (`from __future__ import annotations`).
  Verify all type hints use `torch.Tensor` (resolved at import) not forward references that would break.

  **S4 — Add `__all__` declarations:**
  Add `__all__` to files missing it. Use module-level `__all__` at the top of each file:
  - `tstcc/encoder.py`: `__all__ = ['TCCEncoder']`
  - `series2vec/filters.py`: `__all__ = ['filter_frequencies', 'apply_fft', 'lowpass_filter', 'highpass_filter', 'LOWPASS_PROBABILITY', 'SAMPLING_RATE']`
  - `series2vec/losses.py`: `__all__ = ['pretraining_loss', 'pairwise_soft_dtw_distances', 'pairwise_euclidean_distances']`
  - `timevae/model.py`: `__all__ = ['TimeVAEEncoder', 'TimeVAEDecoder', 'TimeVAE']`
  - `timevae/vae_base.py`: `__all__ = ['Sampling', 'BaseVariationalAutoencoder']`
  - `tst/loss.py`: `__all__ = ['MaskedMSELoss']`
  - `tst/ts_transformer.py`: `__all__ = ['FixedPositionalEncoding', 'LearnablePositionalEncoding', 'TransformerBatchNormEncoderLayer', 'TSTransformerEncoder', 'get_pos_encoder', '_get_activation_fn']`
  </action>

  <verify>
  - grep "self\._alpha" src/tscollection/models/convolutional/standard/mcl/model.py | grep -c "" > 0
  - grep "self\.alpha" src/tscollection/models/convolutional/standard/mcl/model.py | grep -v "#" | grep -c "" == 0
  - grep "self\._learning_rate" src/tscollection/models/convolutional/standard/series2vec/model.py | grep -c "" > 0
  - grep "self\._hidden_dims" src/tscollection/models/recurrent/timenet/model.py | grep -c "" > 0
  - grep "Seasonality = tuple" src/tscollection/models/generative/timevae/model.py | grep -c "" == 0
  - grep "from __future__" src/tscollection/models/convolutional/standard/tstcc/model.py | grep -c "" == 0
  - grep "from __future__" src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py | grep -c "" == 0
  - grep "__all__" src/tscollection/models/convolutional/standard/series2vec/filters.py | grep -c "" > 0
  - grep "__all__" src/tscollection/models/generative/timevae/vae_base.py | grep -c "" > 0
  - grep "__all__" src/tscollection/models/transformer/tst/loss.py | grep -c "" > 0
  - grep "__all__" src/tscollection/models/transformer/tst/ts_transformer.py | grep -c "" > 0
  </verify>

  <done>
  All style/consistency fixes applied: public attrs use underscore prefix in MCL/S2V/TimeNet models, TimeVAE no longer defines duplicate Seasonality, TSTCC files have future annotations removed, all targeted modules have __all__ declarations.
  </done>
</task>

</tasks>

<verification>
After all 3 tasks complete:
1. Run ruff check on all modified files: `uv run ruff check src/tscollection/models/convolutional/standard/mcl/ src/tscollection/models/convolutional/standard/series2vec/ src/tscollection/models/recurrent/timenet/ src/tscollection/models/convolutional/standard/tstcc/ src/tscollection/models/transformer/tst/ src/tscollection/models/generative/timevae/`
2. Run ruff format --check on all modified files
3. Run ty check: `uv run ty check src/tscollection/models/`
4. Verify no remaining references to removed attrs (e.g., `self.warmup`, `self.device` in MixUpLoss)
</verification>

<success_criteria>
- All 17 bugs (C1-C4, I1-I8, R1-R3, S1-S4) are fixed across the 6 models
- No `ruff` or `ty` errors in modified files
- MCL MixUpLoss no longer takes device/batch_size at init
- S2V no longer passes warmup to AdamW
- S2V squeeze uses dimension-agnostic form
- TimeNet accepts and uses feat_dim for GRUWrapper
- S2V defaults to RAdam optimizer
- TSTCC validation uses no_grad
- TSTCC temporal contrast guards short sequences
- TST uses public encode_representations method
- S2V filter is deterministic during validation
- TimeVAE loss is not divided by batch size
- TimeVAE predict() restores training mode
- TST gradient clipping respects gradient_clip_val
- TimeVAE encoder has no stale self.layers reference
- TimeNet has _postprocess selecting final timestep
- All model public attrs use underscore prefix
- TimeVAE has no duplicate Seasonality alias
- TSTCC files have no unnecessary future annotations
- All targeted modules have __all__ declarations
</success_criteria>

<output>
Create `.planning/quick/260611-lal-bug-fixes-1/260611-lal-SUMMARY.md` when done
</output>
