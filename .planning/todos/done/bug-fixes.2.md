# Bug Fixes — PR #10 (feature/baseline)

Generated from cross-agent review of 6 new models (MCL, Series2Vec, TS-TCC, TimeVAE, TimeNet, TST) + shared components.

**Base:** origin/dev -> feature/baseline
**Total:** 6 critical (will crash), 9 important (correctness/risk), 11 minor (style/consistency)

---

## Critical — Runtime Crashes

### BUG-01: MixUpLoss allocates identity matrix to fixed batch_size

**Files:** `src/tscollection/models/convolutional/standard/mcl/losses.py`, `src/tscollection/models/convolutional/standard/mcl/model.py`, `src/tscollection/models/convolutional/standard/mcl/config.py`
**Severity:** Critical — crashes on last batch of epoch when batch_size < configured batch_size
**PR comment:** mcl/model.py:33

**Current code (losses.py:10-29):**
```python
class MixUpLoss(torch.nn.Module):
    def __init__(self, device: str | torch.device, batch_size: int) -> None:
        super().__init__()
        self.tau = 0.5
        self.device = torch.device(device)
        self.batch_size = batch_size  # <-- fixed at init
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, z_aug, z_1, z_2, lam):
        ...
        labels_lam_0 = lam * torch.eye(self.batch_size, device=self.device)  # <-- crashes if len(z_aug) != self.batch_size
        labels_lam_1 = (1 - lam) * torch.eye(self.batch_size, device=self.device)
        labels = torch.cat((labels_lam_0, labels_lam_1), 1)
        logits = torch.cat((torch.mm(z_aug, z_1.T), torch.mm(z_aug, z_2.T)), 1)
```

**Current code (model.py:29-33):**
```python
self.alpha = alpha
self.learning_rate = learning_rate
self.criterion = MixUpLoss(device=device, batch_size=batch_size)  # <-- passes config batch_size
```

**Root cause:** `MixUpLoss` bakes `batch_size` into its identity matrix size. Last epoch batch is smaller -> `torch.mm(z_aug, z_1.T)` produces `(N_actual, 2*N_actual)` but `labels` is `(N_config, 2*N_config)`. Shape mismatch crash.

**Fix:**
1. Remove `device` and `batch_size` from `MixUpLoss.__init__`. Allocate identity tensors lazily in `forward()` using input tensor device and shape.
2. Remove `device: str = 'cuda'` from `MCLModelParameters` (config.py:28). Device is determined by Trainer.
3. Remove `batch_size: int = 8` from `MCLModelParameters` (config.py:29). Not needed by loss.

**Fixed code (losses.py:7-35):**
```python
class MixUpLoss(nn.Module):
    """MixUp contrastive loss used by MCL."""

    def __init__(self, *, tau: float = 0.5) -> None:
        super().__init__()
        self.tau = tau
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(
        self, z_aug: torch.Tensor, z_1: torch.Tensor, z_2: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Compute the MixUp contrastive loss.

        Args:
            z_aug: Projected augmented features, shape ``(B, D)``.
            z_1: Projected features of first view, shape ``(B, D)``.
            z_2: Projected features of second view, shape ``(B, D)``.
            lam: MixUp interpolation coefficient, scalar.

        Returns:
            Scalar loss.
        """
        z_1 = nn.functional.normalize(z_1)
        z_2 = nn.functional.normalize(z_2)
        z_aug = nn.functional.normalize(z_aug)

        batch = z_aug.shape[0]
        labels_lam_0 = lam * torch.eye(batch, device=z_aug.device)
        labels_lam_1 = (1 - lam) * torch.eye(batch, device=z_aug.device)
        labels = torch.cat((labels_lam_0, labels_lam_1), 1)

        logits = torch.cat((torch.mm(z_aug, z_1.T), torch.mm(z_aug, z_2.T)), 1)

        return torch.mean(torch.sum(-labels * self.logsoftmax(logits / self.tau), 1))
```

**Fixed code (model.py:29-33):**
```python
self._alpha = alpha
self._learning_rate = learning_rate
self.criterion = MixUpLoss(tau=0.5)
# Remove `device` and `batch_size` params from __init__ signature.
```

**Fixed config.py:** Delete `batch_size` and `device` fields.

---

### BUG-02: Series2Vec passes `warmup` kwarg to AdamW — not a valid arg

**Files:** `src/tscollection/models/convolutional/standard/series2vec/model.py`
**Severity:** Critical — TypeError on optimizer init when `optimizer_name='AdamW'`
**PR comment:** series2vec/model.py:152-156

**Current code (model.py:150-156):**
```python
def configure_optimizers(self) -> torch.optim.Optimizer:
    optimizer_cls = _get_optimizer(self.optimizer_name)
    kwargs: dict = {'lr': self.learning_rate, 'weight_decay': self.weight_decay}
    if self.optimizer_name == 'AdamW':
        kwargs['warmup'] = self.warmup   # <-- AdamW has no 'warmup' param
    return optimizer_cls(self.parameters(), **kwargs)
```

**Root cause:** `torch.optim.AdamW` accepts `lr`, `betas`, `weights_decay`, `eps`, `amsgrad`. It does not have a `warmup` parameter. Warmup is implemented via LR schedulers, not the optimizer itself.

**Fix:**
1. Delete the `if self.optimizer_name == 'AdamW': kwargs['warmup'] = self.warmup` block (lines 154-155).
2. Remove `warmup: int = 0` from `__init__` signature (line 62) — it has no consumer.
3. Remove `self.warmup = warmup` from line 72.
4. Remove `warmup: int = 0` from `Series2VecModelParameters` in config.py:55.
5. If warmup is desired, implement via `torch.optim.lr_scheduler.LinearLR` and return a scheduler config from `configure_optimizers()`.

**Fixed code (model.py:150-153):**
```python
def configure_optimizers(self) -> torch.optim.Optimizer:
    optimizer_cls = _get_optimizer(self._optimizer_name)
    return optimizer_cls(
        self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
    )
```

---

### BUG-03: TimeNet hardcodes GRU input_size=1 — no feat_dim parameter

**Files:** `src/tscollection/models/recurrent/timenet/model.py`, `src/tscollection/models/recurrent/timenet/config.py`
**Severity:** Critical — crashes on any multivariate input (channels > 1)
**PR comment:** timenet/model.py:38-39

**Current code (model.py:38-52):**
```python
def __init__(
    self, hidden_dims: int, num_layers: int, dropout: float = 0.1, learning_rate: float = 1e-3
) -> None:
    ...
def _build_encoder(self) -> nn.Sequential:
    encoder_layers: list[nn.Module] = [GRUWrapper(1, self.hidden_dims, batch_first=True)]  # <-- input_size=1
```

**Root cause:** No `feat_dim` parameter. GRU expects `(batch, seq, input_size)`. Hardcoded `input_size=1` rejects multivariate time series.

**Fix:**
1. Add `feat_dim: int = 1` to `TimeNet.__init__` signature and `save_hyperparameters()`.
2. Replace `GRUWrapper(1, ...)` with `GRUWrapper(self._feat_dim, ...)`.
3. Add `nn.Linear(feat_dim, 1)` after decoder GRU stack for outputs when feat_dim != 1, OR add output projection `nn.Linear(self.hidden_dims, self._feat_dim)` as final decoder layer instead of `nn.Linear(self.hidden_dims, 1)`.
4. Add `feat_dim: int = 1` to `TimeNetModelParameters` in config.py.

**Fixed code (model.py:38-40):**
```python
def __init__(
    self,
    feat_dim: int,
    hidden_dims: int,
    num_layers: int,
    *,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> None:
    super().__init__()
    self.save_hyperparameters()
    self._feat_dim = feat_dim
    self._hidden_dims = hidden_dims
    self._learning_rate = learning_rate
    ...
```

**Fixed decoder (model.py:59-68):**
```python
def _build_decoder(self) -> nn.Sequential:
    decoder_layers: list[nn.Module] = [
        GRUWrapper(self._hidden_dims, self._hidden_dims, batch_first=True)
    ]
    for i in range(1, self.num_layers):
        if i > 1 and self.dropout > 0:
            decoder_layers.append(nn.Dropout(self.dropout))
        decoder_layers.append(GRUWrapper(self._hidden_dims, self._hidden_dims, batch_first=True))
    decoder_layers.append(nn.Linear(self._hidden_dims, self._feat_dim))  # <-- was 1
    return nn.Sequential(*decoder_layers)
```

**Also fix encoder (model.py:52):**
```python
GRUWrapper(self._feat_dim, self._hidden_dims, batch_first=True)
```

---

### BUG-04: Working-tree import crash — augmentation/__init__.py references deleted composition.py

**Files:** `src/tscollection/models/augmentation/__init__.py`
**Severity:** Critical — ImportError on any augmentation import in working tree
**Note:** Only affects working tree. Committed code uses `PairedAugmentation` from `.composition`; working tree renamed to `DualAugmentation` in `.dual`.

**Current code (augmentation/__init__.py:44):**
```python
from .dual import DualAugmentation  # <-- working tree has this; committed has .composition
```

**Already fixed in current working tree.** The file reads correctly: `from .dual import DualAugmentation`. Skip if `import tscollection.models.augmentation` succeeds. Verify with `python -c "from tscollection.models.augmentation import DualAugmentation; print('OK')"`.

---

### BUG-05: TSTCC validation_step computes gradients unnecessarily

**Files:** `src/tscollection/models/convolutional/standard/tstcc/model.py`
**Severity:** Important — wasted memory during validation, non-deterministic loss due to random augmentation
**PR comment:** tstcc/model.py:173-181

**Current code (model.py:173-181):**
```python
def validation_step(self, batch, _batch_idx):
    loss = self._compute_loss(batch)  # <-- no torch.no_grad(), runs augmentations + TemporalContrast with randomness
    ...
```

**Root cause:** Two problems:
1. No `torch.no_grad()` — gradients flow through encoder + TemporalContrast during validation, wasting memory.
2. `SELF_SUPERVISED` mode calls `self._augmentation.augment(data)` which uses `torch.rand()` in `_should_apply()` (augmentations.py:44) and `TemporalContrast.forward()` uses `torch.randint()` (temporal_contrast.py:168). Validation loss is non-deterministic.

**Fix:** Skip augmentation in validation; pass raw data through both encoder branches without random sampling. Use `torch.no_grad()` wrapper.

**Fixed code (model.py:123-144):**
```python
def _compute_loss(
    self,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    compute_grads: bool = False,
) -> torch.Tensor:
    data = extract_features_from_batch(batch).float()

    if self._training_mode == TSTCCTrainingMode.SELF_SUPERVISED:
        if compute_grads:
            views = self._augmentation.augment(data)
            aug1, aug2 = views.views[0], views.views[1]
        else:
            # Deterministic validation: use two identical views of raw data.
            aug1 = data
            aug2 = data
        ...
```

**Fixed validation_step (model.py:173-181):**
```python
def validation_step(self, batch, _batch_idx):
    with torch.no_grad():
        loss = self._compute_loss(batch, compute_grads=False)
    ...
```

**Also update training_step (model.py:159):**
```python
loss = self._compute_loss(batch, compute_grads=True)
```

---

### BUG-06: TemporalContrast crashes when seq_len <= timesteps

**Files:** `src/tscollection/models/convolutional/standard/tstcc/temporal_contrast.py`
**Severity:** Important — runtime crash on short sequences
**PR comment:** temporal_contrast.py:168

**Current code (temporal_contrast.py:167-168):**
```python
batch, seq_len, _ = z1.shape
t_samples = torch.randint(seq_len - self.timestep, size=(1,), device=device).long()
```

**Root cause:** `torch.randint(high, ...)` requires `high > 0`. If `seq_len <= self.timestep`, high is <= 0 and PyTorch raises `ValueError`.

**Fix:** Add guard before `torch.randint`.

**Fixed code (temporal_contrast.py:167-170):**
```python
batch, seq_len, _ = z1.shape
if seq_len <= self.timestep:
    msg = f'seq_len ({seq_len}) must be greater than timesteps ({self.timestep})'
    raise ValueError(msg)
t_samples = torch.randint(seq_len - self.timestep, size=(1,), device=device).long()
```

---

## Important — Correctness and Risk

### BUG-07: TST loss diluted by padding/unmasked positions

**Files:** `src/tscollection/models/transformer/tst/model.py`
**Severity:** Important — biased loss, underweights active elements
**PR comment:** tst/model.py:134

**Current code (model.py:128-142):**
```python
def _compute_loss(self, batch):
    x, targets, target_masks, padding_masks, _ = batch
    predictions = self.reconstruct(x, padding_masks)
    combined_mask = target_masks * padding_masks.unsqueeze(-1)
    per_element_loss = self._loss_fn(predictions, targets, combined_mask)
    mean_loss = torch.sum(per_element_loss) / len(per_element_loss)  # <-- divides by total elements, not active
```

**Root cause:** `MaskedMSELoss` with `reduction='mean'` already returns the mean over masked elements. However, the code then divides by `len(per_element_loss)` which is the number of active elements. Wait — re-reading the loss: `MaskedMSELoss` with `reduction='none'` returns per-element losses for **masked** positions only (it uses `torch.masked_select`). Then dividing by `len(per_element_loss)` gives mean over active elements. But if `reduction='mean'`, it already divides by the active count.

Actually, re-reading loss.py: `reduction='none'` means `self.mse_loss` uses `reduction='none'`, producing per-pair squared errors on the masked vectors. Then `torch.sum(per_element_loss) / len(per_element_loss)` computes the mean. This is correct for a single sample but **not** for a batch: `masked_select` flattens across all dimensions, so `len(per_element_loss)` is the total number of active elements across the batch. Dividing sum by count IS the mean. This is actually correct.

**However**, the issue is subtler: the code constructs `per_element_loss` as a 1D tensor of squared errors per active element, then divides by its length. This is mathematically equivalent to `per_element_loss.mean()`. No bug here.

**Re-evaluating:** Let me re-read the original finding. The claim was that division by `len(per_element_loss)` dilutes the loss. But `MaskedMSELoss(reduction='none')` only returns errors for **masked** positions. The denominator is the count of those active elements. This is correct.

**Verdict:** PASS — no bug. The loss computation is correct. ~~BUG-07~~ removed.

---

### BUG-08: TST get_representations reaches into encoder internals

**Files:** `src/tscollection/models/transformer/tst/model.py`
**Severity:** Important — tight coupling, fragile to encoder refactors
**PR comment:** tst/model.py:106-114

**Current code (model.py:106-114):**
```python
def get_representations(self, x, padding_masks):
    inp = x.permute(1, 0, 2)
    inp = self._encoder.project_inp(inp) * math.sqrt(self._encoder.d_model)  # internal attrs
    inp = self._encoder.pos_enc(inp)
    out = self._encoder.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
    out = self._encoder.act(out)
    out = out.permute(1, 0, 2)
    return self._encoder.dropout1(out)
```

**Root cause:** `TST` accesses `self._encoder.project_inp`, `d_model`, `pos_enc`, `transformer_encoder`, `act`, `dropout1` — all private/internal attrs of `TSTransformerEncoder`. A refactor of the encoder breaks the model.

**Fix:** Add a public `encode_representations` method to `TSTransformerEncoder` that encapsulates the trunk logic.

**New method in ts_transformer.py (after forward):**
```python
def encode_representations(
    self,
    x_permuted: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the transformer trunk and return representations (B, T, d_model).

    Skips the reconstruction output layer. Input is expected in ``(T, B, F)``
    order (the transformer's native layout).
    """
    inp = self.project_inp(x_permuted) * math.sqrt(self.d_model)
    inp = self.pos_enc(inp)
    out = self.transformer_encoder(inp, src_key_padding_mask=src_key_padding_mask)
    out = self.act(out)
    return self.dropout1(out)
```

**Updated get_representations in model.py:**
```python
def get_representations(self, x, padding_masks):
    inp = x.permute(1, 0, 2)
    out = self._encoder.encode_representations(inp, src_key_padding_mask=~padding_masks)
    return out.permute(1, 0, 2)
```

---

### BUG-09: Series2Vec filter_frequencies uses random selection in validation

**Files:** `src/tscollection/models/convolutional/standard/series2vec/filters.py`
**Severity:** Important — non-reproducible validation loss
**PR comment:** filters.py:13

**Current code (filters.py:8-25):**
```python
def filter_frequencies(data, lowpass_cutoff=40.0, highpass_cutoff=0.5):
    fft_results = torch.stack([apply_fft(sample) for sample in data])
    if torch.rand(()) < LOWPASS_PROBABILITY:  # <-- random during val
        return torch.stack([lowpass_filter(sample, lowpass_cutoff, SAMPLING_RATE) for sample in fft_results])
    return torch.stack([highpass_filter(sample, highpass_cutoff, SAMPLING_RATE) for sample in fft_results])
```

**Root cause:** `torch.rand(())` fires even under `torch.inference_mode()` and `torch.no_grad()`. Validation alternates between lowpass and highpass, making val_loss oscillate stochastically.

**Fix:** Pass a deterministic `mode` parameter instead of using random selection, or check `not torch.is_inference_mode_enabled()`.

**Fixed code (filters.py:8-28):**
```python
def filter_frequencies(
    data: torch.Tensor,
    *,
    lowpass_cutoff: float = 40.0,
    highpass_cutoff: float = 0.5,
    mode: str = 'random',  # 'lowpass' | 'highpass' | 'random'
) -> torch.Tensor:
    """Apply filtering to FFT-transformed samples."""
    fft_results = torch.stack([apply_fft(sample) for sample in data])
    if mode == 'lowpass' or (mode == 'random' and torch.rand(()) < LOWPASS_PROBABILITY):
        return torch.stack([lowpass_filter(sample, lowpass_cutoff, data) for sample in fft_results])
    return torch.stack([highpass_filter(sample, highpass_cutoff, data) for sample in fft_results])
```

Then in `series2vec/model.py:109`:
```python
# Change from:
filtered_frequency_data = filter_frequencies(x.detach().cpu())
# To:
filtered_frequency_data = filter_frequencies(
    x.detach().cpu(),
    mode='highpass' if not self.training else 'random',
)
```

---

### BUG-10: TimeVAE uses raw list[nn.Module] for encoder layers

**Files:** `src/tscollection/models/generative/timevae/model.py`
**Severity:** Important — device placement fragility
**PR comment:** timevae/model.py:25

**Current code (model.py:25-37):**
```python
self.layers: list[nn.Module] = []
self.layers.append(nn.Conv1d(...))
self.layers.append(nn.ReLU())
...
self.encoder = nn.Sequential(*self.layers)
```

**Root cause:** `self.layers` persists as an instance attribute. If code accesses `self.layers` after `self.to(device)`, those refs point to old-device modules. While `self.encoder` (the Sequential) is correctly moved, the stale list is a latent bug if reused.

**Fix:** Delete `self.layers` after constructing `self.encoder`.

**Fixed code (model.py:37):**
```python
self.encoder = nn.Sequential(*self.layers)
del self.layers  # prevent stale refs after device move
```

---

### BUG-11: Series2Vec public attributes — naming convention mismatch

**Files:** `src/tscollection/models/convolutional/standard/series2vec/model.py`
**Severity:** Minor — consistency with dilated pattern
**PR comment:** series2vec/model.py:67-72

**Current code:**
```python
self.learning_rate = learning_rate
self.soft_dtw_gamma = soft_dtw_gamma
self.sync_dist = sync_dist
self.optimizer_name = optimizer_name
self.weight_decay = weight_decay
self.warmup = warmup
```

**Fix:** Rename all to use underscore prefix: `_learning_rate`, `_soft_dtw_gamma`, `_sync_dist`, `_optimizer_name`, `_weight_decay`. Update all usages in `training_step`, `validation_step`, `configure_optimizers`, `_build_soft_dtw`. Remove `_warmup` entirely (see BUG-02).

---

### BUG-12: TimeNet public attributes and type mismatch

**Files:** `src/tscollection/models/recurrent/timenet/model.py`
**Severity:** Minor — consistency + type correctness
**PR comment:** timenet/model.py:43-48

**Current code:**
```python
self.hidden_dims: int = hidden_dims
self.num_layers: int = num_layers
self.dropout: int | float = dropout    # <-- type includes int, param is float
self.encoder: Sequential = self._build_encoder()  # <-- Sequential from TYPE_CHECKING import
self.decoder: Sequential = self._build_decoder()
self.learning_rate = learning_rate
```

**Fix:**
```python
self._hidden_dims = hidden_dims
self._num_layers = num_layers
self._dropout: float = dropout
self._encoder = self._build_encoder()
self._decoder = self._build_decoder()
self._learning_rate = learning_rate
```
Update all references to use underscore-prefixed names. Update `Sequential` annotation to `nn.Sequential`.

---

### BUG-13: TSTCC _should_apply non-deterministic during eval

**Files:** `src/tscollection/models/convolutional/standard/tstcc/augmentations.py`
**Severity:** Important — unpredictability in eval mode
**PR comment:** augmentations.py:44-45

**Current code (augmentations.py:44-45):**
```python
def _should_apply(p: float) -> bool:
    return p >= 1.0 or torch.rand((1,)).item() < p
```

**Root cause:** Fires in eval mode. Augmentations with `p < 1.0` behave differently each validation step.

**Fix:** Check model training state. Since `_should_apply` is a standalone function without model context, pass a `training` flag.

**Fixed code:**
```python
def _should_apply(p: float, *, training: bool = True) -> bool:
    if not training:
        return False
    return p >= 1.0 or torch.rand((1,)).item() < p
```

Update callers (Jitter.augment, Scaling.augment) to pass `training=True` since augmentations should only run during training. Alternatively, remove the flag and document that augmentations are training-only — the caller (`validation_step`) already skips them when `compute_grads=False`.

---

### BUG-14: MCL public attributes — naming convention

**Files:** `src/tscollection/models/convolutional/standard/mcl/model.py`
**Severity:** Minor — consistency with dilated pattern

**Current code (model.py:29-30):**
```python
self.alpha = alpha
self.learning_rate = learning_rate
```

**Fix:** Rename to `self._alpha` and `self._learning_rate`. Update `_step` and `configure_optimizers` references.

---

## Minor — Style and Consistency

### BUG-15: Missing __all__ in 8 files

**Files and expected exports:**

| File | `__all__` value |
|------|-----------------|
| `series2vec/encoder.py` | `['DisjoinEncoder']` |
| `series2vec/filters.py` | `['filter_frequencies', 'apply_fft', 'lowpass_filter', 'highpass_filter']` |
| `series2vec/losses.py` | `['pairwise_soft_dtw_distances', 'pairwise_euclidean_distances', 'pretraining_loss']` |
| `timevae/model.py` | `['TimeVAEEncoder', 'TimeVAEDecoder', 'TimeVAE']` |
| `timevae/vae_base.py` | `['Sampling', 'BaseVariationalAutoencoder']` |
| `tst/loss.py` | `['MaskedMSELoss']` |
| `tst/ts_transformer.py` | `['TSTransformerEncoder', 'FixedPositionalEncoding', 'LearnablePositionalEncoding', 'TransformerBatchNormEncoderLayer', 'get_pos_encoder']` |
| `soft_dtw/soft_dtw_cuda.py` | `['SoftDTW']` (skip — 3rd party code, don't modify) |

**Action:** Add `__all__` to each file at top, below docstring and imports.

---

### BUG-16: MCL and TSTCC __init__ use positional args — should be kw-only

**Files:** `mcl/model.py`, `tstcc/model.py`

**Current mcl/model.py:19-27:**
```python
def __init__(
    self,
    n_in: int,
    output_dims: int = 320,
    batch_size: int = 8,
    device: str = 'cuda',
    alpha: float = 1.0,
    learning_rate: float = 1e-3,
):
```

**Fix (after BUG-01 removes batch_size/device):**
```python
def __init__(
    self,
    n_in: int,
    *,
    output_dims: int = 320,
    alpha: float = 1.0,
    learning_rate: float = 1e-3,
):
```

**Current tstcc/model.py:50-62:** First 6 params positional. Add `*,` after `num_classes` or at minimum after `temperature`.

**Fix:**
```python
def __init__(
    self,
    input_channels: int,
    kernel_size: int,
    stride: int,
    final_out_channels: int,
    features_len: int,
    num_classes: int,
    *,
    dropout: float = 0.35,
    ...
):
```

---

### BUG-17: Duplicate Seasonality type alias

**Files:** `layers/general.py:16`, `generative/timevae/model.py:13`

**Both define:** `Seasonality = tuple[int, int]`

**Fix:** Keep in `layers/general.py` only. In `timevae/model.py`, import from `layers.general`.

---

### BUG-18: Sphinx-style docstring in tst/loss.py

**File:** `tst/loss.py:15-31`

**Current:**
```
Returns:
-------
if reduction == 'none':
    ...
```

**Fix:** Convert to Google style:
```
Returns:
    Per-element or mean loss depending on ``reduction`` setting.
```

---

### BUG-19: TimeVAE unused model_name class attribute

**File:** `timevae/model.py:125`

**Current:** `model_name = 'TimeVAE'`

**Fix:** Delete. Not used anywhere.

---

### BUG-20: TimeVAE _get_encoder returns (z_mean, z_log_var, z) tuple — needs _postprocess

**File:** `timevae/model.py:171-173`

**Current code:**
```python
def _get_encoder(self) -> nn.Module:
    return self.encoder
```

The encoder `forward()` returns `(z_mean, z_log_var, z)`. Without `_postprocess`, `BasicEncodingMixin.encode()` receives a tuple, not a tensor. The code at `model.py:175-179` has `_postprocess` that extracts `output[0]` (z_mean). This is correct. No fix needed.

**Verdict:** PASS — `_postprocess` is already implemented. ~~BUG-20~~ removed.

---

### BUG-21: TimeNet encode() returns 3D tensor unlike other models

**File:** `timenet/model.py:76-78`

**Current:** `_get_encoder` returns `self.encoder` (GRU stack). Output shape is `(batch, seq_len, hidden_dims)`. No `_postprocess` override.

**Other models:** Return `(N, 1, D)` or `(N, D)` — pooled/selected representation.

**Fix:** Add `_postprocess` to select final timestep.

```python
def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
    """Pool over time dimension — return final-timestep representation."""
    return output[:, -1, :].unsqueeze(1)  # (batch, 1, hidden_dims)
```

---

### BUG-22: TST config lr_step default mismatch

**File:** `tst/config.py:63`, `tst/model.py:75`

**Config says:** `lr_step: list[int] | None = None` — "None means no decay"
**Model does:** `self._lr_step = lr_step or [1_000_000]` — None becomes one far-future milestone

These are functionally equivalent (no decay happens), but the semantics differ. Document or align.

**Fix:** Add comment in model.py:
```python
self._lr_step = lr_step if lr_step is not None else [1_000_000]  # far-future sentinel = no decay
```

---

### BUG-23: SoftDTW type hints missing

**File:** `distances/soft_dtw/soft_dtw_cuda.py:299`

**Current:**
```python
def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
```

**Fix:** Add type hints.
```python
def __init__(
    self,
    use_cuda: bool,
    gamma: float = 1.0,
    normalize: bool = False,
    bandwidth: int | None = None,
    dist_func: Callable[..., torch.Tensor] | None = None,
) -> None:
```

**Note:** This is 3rd-party code (MIT). If preferring not to modify, skip and add `# noqa: ANN001` to silence ty.

---

## Execution Order

Recommended fix order (dependencies first):

1. **BUG-01** — MixUpLoss (MCL). Unblocks everything else in MCL.
2. **BUG-02** — Series2Vec warmup. Quick removal, no downstream deps.
3. **BUG-05** — TSTCC validation no_grad. Medium refactor of `_compute_loss`.
4. **BUG-06** — TemporalContrast seq_len guard. One-line addition.
5. **BUG-03** — TimeNet feat_dim. Structural change to model + config.
6. **BUG-10** — TimeVAE list cleanup. One-line deletion.
7. **BUG-15** — Missing `__all__`. Mechanical additions.
8. **BUG-16** — Kw-only args. Signature changes.
9. **BUG-11/12/14** — Private attr renaming across models. Mechanical but widespread.
10. **BUG-08** — TST encoder coupling. Requires ts_transformer.py change.
11. **BUG-09** — Series2Vec filter determinism. API change to `filter_frequencies`.
12. **BUG-13** — TSTCC _should_apply. Augmentation refactor.
13. **BUG-17/18/19/21/22** — Minor consistency fixes.

---

## Verification

After all fixes, run:
```bash
uv run pytest tests/ -v
uv run ruff check src/tscollection/models/convolutional/standard/ src/tscollection/models/generative/ src/tscollection/models/recurrent/ src/tscollection/models/transformer/
uv run ty check src/
```
