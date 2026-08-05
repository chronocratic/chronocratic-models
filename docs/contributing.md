# Contributing

Thank you for your interest in contributing to `chronocratic-models`. This guide covers the development workflow and tooling used in the project.

## Development Setup

The project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/chronocratic/chronocratic-models.git
cd chronocratic-models

# Sync the development environment
uv sync
```

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) to run linting, formatting, and type checking automatically before each commit. This eliminates the need to run ruff and ty manually — changes are fixed and verified as you commit.

**One-time setup:**

```bash
# Install pre-commit as a uv tool (once per machine)
uv tool install pre-commit

# Install git hooks into the repository (once per clone)
pre-commit install
```

After installation, the hooks run automatically on every `git commit`:

- **ruff-check** — auto-fix linting issues
- **ruff-format** — format code
- **ty** — type checking (read-only, blocks commit on errors)

The pre-commit configuration lives in [.pre-commit-config.yaml](.pre-commit-config.yaml).

**Run manually (all files):**

```bash
pre-commit run --all-files
```

## Running Tests

```bash
# Run the full test suite
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=src/chronocratic/models --cov-report=xml
```

## Linting and Formatting

The project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for linting issues
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Type Checking

The project uses [ty](https://github.com/astral-sh/ty) for static type checking.

```bash
# Run type checking
uv run ty check src/
```

## Building Documentation

```bash
# Install documentation dependencies
uv sync --extra docs

# Build the documentation
uv run sphinx-build -b html docs/ docs/_build/
```

## Protected Project Files

Certain files affect every user of the project, so changes to them are **flagged for maintainer review** and should normally be left to core maintainers:

- **`pyproject.toml`** — Build configuration, dependencies, and project metadata. Changes here affect all users and may introduce regressions.
- **`.vscode/`** — Editor settings for linting, formatting, and type-checking (e.g. `settings.json`). Shared configuration that ensures consistency across the team.

An automated CI check (`protected-files-check.yml`) posts an advisory comment on any PR that modifies these files. The comment is informational — it does not block the merge, but it makes the change explicit for reviewers. If a change is **absolutely necessary** (e.g., a dependency fix required for the PR to work), include a clear explanation in the PR description and expect maintainer review.

Non-maintainer changes to these files should have explicit sign-off from a core maintainer. When in doubt, open an issue first.

## Adding Changelog Fragments

The project uses [towncrier](https://towncrier.readthedocs.io/) for managing changelog entries. Each PR should include a changelog fragment in the `changelog.d/` directory.

```bash
# Create a fragment (e.g., for a new feature in PR #42)
echo "Added new TimeVAE model for generative time series encoding." > changelog.d/42.added.md
```

Fragment types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`.

```bash
# Verify fragments before merging
uv run towncrier check --compare-with origin/main
```

See [`changelog.d/README.md`](../changelog.d/README.md) for detailed fragment instructions.

## Code Style

- Use **snake_case** for functions and variables, **PascalCase** for classes.
- Write **Google-style docstrings** for all public functions, classes, and config dataclasses.
- Use **type hints** for all function signatures and return types.
- Prefer **functional programming patterns** and modular code organization.
- Use **keyword arguments** for all function calls.
- Model configuration uses **`@dataclass(kw_only=True)`** — see [Config-to-Model Contract](#config-to-model-contract).

## Parameters Consistency

All model classes and their config dataclasses follow shared parameter conventions. These rules ensure that `Model(**vars(ModelParameters(...)))` works across the entire library.

### Canonical Hyperparameter Names

Use these exact names. Do not invent alternatives.

| Canonical Name | Description | Do NOT Use |
|---|---|---|
| `input_dim` | Number of input features/channels | `feat_dim`, `n_in`, `input_channels`, `input_dims` |
| `hidden_dim` | Hidden representation size | `d_model`, `hidden_dims` |
| `feedforward_dim` | FFN intermediate dimension | `dim_feedforward`, `feedforward_dims` |
| `representation_dim` | Feature dim of ``encode()`` output | `output_dims`, `representation_dims`, `encoding_output_dim` |
| `depth` | Number of layers | `num_layers` |
| `dropout_rate` | Dropout probability | `dropout` |
| `num_heads` | Attention head count | `n_heads` |
| `sequence_length` | Temporal dimension | `max_seq_len`, `seq_len` |
| `conv_kernel_size` | Convolution kernel size | `kernel_size` |
| `weight_decay` | L2 regularization | `l2_reg` |
| `reconstruction_weight` | VAE reconstruction term | `reconstruction_wt` |

Model-specific names are acceptable only when genuinely unique to that model (e.g., `latent_dim` for TimeVAE, `embedding_dim` for Series2Vec).

All dimension parameters use **singular** form (``*_dim``), never plural (``*_dims``). Every dimension is a scalar ``int`` — the ecosystem convention (PyTorch: ``embedding_dim``, ``d_model``, ``dim_feedforward``) matches this pattern.

### Config-to-Model Contract

Config dataclasses and model `__init__` signatures must mirror each other exactly.

- Every config field must have a matching `__init__` parameter with the **same name** and **same default value**.
- Use `@dataclass(kw_only=True)` on all config classes.
- Defaults must be declared in **both** the config dataclass and the model `__init__`. This allows partial config instantiation.
- Verify with `Model(**vars(ModelParameters(...)))` — it must not raise.
- Use `save_hyperparameters(ignore=["augmentation"])` in Lightning modules. Non-callable config values should not be ignored.

**Example:**

```python
# Config — chronocratic/models/<model>/config.py
@dataclass(kw_only=True)
class MyModelParameters:
    """Configuration for the MyModel model.

    Args:
        input_dim: Number of input features (channels).
        hidden_dim: Hidden representation dimensionality.
        dropout_rate: Dropout probability applied after each layer.
    """

    input_dim: int
    hidden_dim: int = 64
    dropout_rate: float = 0.1
```

```python
# Model — chronocratic/models/<model>/model.py
class MyModel(pl.LightningModule, BasicEncodingMixin):
    """Short description of the model.

    Args:
        input_dim: Number of input features (channels).
        hidden_dim: Hidden representation dimensionality.
        dropout_rate: Dropout probability applied after each layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate
```

**Rules:**
- Every config dataclass gets a class-level Google-style docstring with an `Args:` section.
- Every model `__init__` inherits the docstring in the class docstring. Do not duplicate per-parameter docs in `__init__`.
- Config and model param descriptions must match — they describe the same hyperparameter from two entry points.

### Tuple over List for Sequence Defaults

All list-typed hyperparameters use `tuple[T, ...]` instead of `list[T]`.

Hyperparameter sequences are never mutated at runtime — only iterated or indexed. Tuples allow direct defaults without `field(default_factory=...)` boilerplate and enforce immutability.

**Do:**
```python
kernel_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
hidden_layer_sizes: tuple[int, ...] = (50, 100, 200)
lr_step: tuple[int, ...] | None = None  # | None only when truly optional
```

**Don't:**
```python
kernel_sizes: list[int] = [1, 2, 4]                   # mutable default
kernel_sizes: list[int] = field(default_factory=...)  # unnecessary boilerplate
```

If an internal component expects a `list`, convert at the boundary: `list(self._kernel_sizes)`. Use `Sequence[T]` in internal type annotations to match the config layer.

### Default Value Sourcing

Source **reference repository code**, not papers. Papers omit implementation details; cloned repos are ground truth.

1. Clone the original implementation's repository.
2. Check actual constructor defaults and CLI argument defaults.
3. Document any deliberate divergence where readers of the code will find it: a comment at the diverging line, the config field's docstring, and a changelog fragment.

### Hardcoded Constant Extraction

Architecture-defining constants (channel widths, kernel sizes, dilation rates, projection dimensions) must be extracted to config parameters rather than hardcoded in encoder/decoder implementations.

**Do:**
```python
# Config
encoder_channels: tuple[int, ...] = (128, 256, 128)
encoder_kernels: tuple[int, ...] = (7, 5, 3)

# Encoder
for channels, kernel in zip(self._encoder_channels, self._encoder_kernels):
    layer = nn.Conv1d(...)
```

**Don't:**
```python
# Hardcoded in encoder
nn.Conv1d(in_channels, 128, kernel_size=7),
nn.Conv1d(128, 256, kernel_size=5),
nn.Conv1d(256, 128, kernel_size=3),
```

**Out of scope for extraction:** optimizer types, gradient clipping norms, LayerNorm epsilon values, structural invariants (e.g., fixed MaxPool kernel sizes that are part of the architecture definition).

### Runtime-Derived Dimensions

A subtler variant of hardcoding, and one you will usually **inherit rather than write**: a value is not a literal, but is computed once in `__init__` from a constructor argument and baked into a weight tensor's shape. The literal is absent; the rigidity is identical.

```python
# Upstream pattern — the flattened width is frozen at construction.
self.layers.append(nn.Flatten())
self.encoder_last_dense_dim = self._get_last_dense_dim(sequence_length, input_dim)
self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
```

This compiles, trains, and passes every test that uses the nominal length. It fails the first time a caller passes anything else — which, in this codebase, is the encode path. Reference implementations are written for a single fixed length and have no reason to avoid this; expect it, and patch it during the port. [Input Length Robustness](#input-length-robustness) covers how.

Things to know when you patch:

1. **Weight shapes that depend on `input_dim`, `hidden_dim`, `latent_dim`, or `depth` are fine.** Those are fixed properties of the data or the architecture. A shape that depends on `sequence_length` is the one to remove, because temporal length varies per call.
2. **Shape probes are not a fix.** A probe like `_get_last_dense_dim` that pushes a dummy tensor through the stack still runs once, at construction, against one length. It is a convenient way to *compute* a fixed dimension — keep it — but it does not make anything length-agnostic on its own. Insert the resampling layer *before* the probe and the probe starts reporting a constant for free.
3. **Do not call a shape probe from `forward()`** to recompute dimensions at runtime. That allocates new, untrained weights per input length and the resulting representation is noise. Same reason `nn.LazyLinear` is not the answer: it sizes to whichever length arrives first and then fails on every other one, turning a deterministic crash into an order-dependent one.

### Keyword-Only Signatures

ALL Python functions with multiple parameters must use keyword-only signatures (``def func(*, param1, param2, ...)``).
This applies to model ``__init__`` methods, encoder constructors, network layers, helpers, factories,
loss functions, augmentation classes, adapters, and shared utilities.

Config dataclasses always use ``@dataclass(kw_only=True)``.

**Do:**
```python
# Multi-param: keyword-only after bare *
def __init__(
    self,
    *,
    input_dim: int,
    hidden_dim: int = 64,
    depth: int = 2,
) -> None:
    ...

# Factory: keyword-only
def make_model(*, input_dim: int, hidden_dim: int = 32) -> Model:
    ...
```

**Acceptable exception:** Single-param methods with an obvious positional input (e.g., ``forward(x)``,
``encode(x)``, ``produce(x)``) may keep that argument positional:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
```

**Don't:**
```python
# Positional multi-param — forbidden
def __init__(self, input_dim: int, hidden_dim: int = 64):
    ...
```

### `self._{name}` Attribute Storage

Store all hyperparameters as private attributes with the `self._{name}` prefix in model `__init__`.

```python
def __init__(
    self,
    *,
    input_dim: int,
    hidden_dim: int = 64,
) -> None:
    self._input_dim = input_dim
    self._hidden_dim = hidden_dim
```

Public attributes are reserved for computed values (e.g., `self.criterion`, `self.loss_fn`) and submodules (`self._encoder`, `self._decoder`).

### Literal vs. Enum Choices

| Pattern | When to Use | Example |
|---|---|---|
| **Enum (`StrEnum`)** | Closed, small set of values | `MaskMode`, `RecurrentCellType`, `NormalizationLayerType` |
| **`str` with default** | Broad options, open-ended | `pos_encoding: str = "fixed"`, `activation: str = "gelu"` |
| **Unconstrained numeric** | Values users may override freely | `dropout_rate: float = 0.01` |
| **`Literal`** | Only in config dataclasses, not model `__init__` | `OptimizerName = Literal["Adam", "RAdam", "AdamW"]` |

Never use `Literal` to restrict numeric values — users may legitimately override them. Keep model `__init__` signatures using `str` or concrete Enum types; reserve `Literal` for config-layer type narrowing.

**Shared enums live in `chronocratic.models.enums`** (e.g., `EncodingOutputShape`, `NormalizationLayerType`). Reuse existing enums across models instead of inventing new string conventions. If a new enum is needed, add it to `models/enums/` as a `StrEnum` subclass and export from `models/enums/__init__.py`. Prefer enums over string-based `if x == "foo"` guards — the type system enforces valid values, eliminating runtime validation.

### Cross-Model Consistency Checklist

Before merging a model change, verify:

- [ ] All parameter names match the canonical names table above.
- [ ] Config dataclass fields exactly match `__init__` parameter names.
- [ ] Default values are identical in config and model signatures.
- [ ] `Model(**vars(ModelParameters(...)))` instantiates without error.
- [ ] Sequence-typed HPs use `tuple[T, ...]`, not `list[T]`.
- [ ] All HPs stored as `self._{name}` private attributes.
- [ ] `save_hyperparameters(ignore=["augmentation"])` is called (if applicable).
- [ ] Default values are sourced from reference repos, not guessed.
- [ ] Architecture constants are extracted to config, not hardcoded.
- [ ] Added/updated tests cover the config splat contract.

## Tensor Shape Convention

All model entry points in this library use **`(B, T, C)`** (batch, time, channels) as the input tensor layout. This matches PyTorch's `DataLoader` output convention and the `transformers` ecosystem.

### Encoder-Owns-the-Transpose Rule

Conv1d-based encoders must transpose `(B, T, C)` to `(B, C, T)` as the **first line** of their `forward()` method. The model wrapper, training step, and loss functions should never transpose.

**The encoder owns the transpose.**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode (B, T, C) input into (B, representation_dim) representation."""
    x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
    return self.layers(x)
```

Existing examples in the codebase:

- `TimeVAEEncoder.forward()` — `transpose(1, 2)` at entry
- `Series2VecNetwork._to_channels_first()` — layout conversion helper
- Dilated encoders — `transpose(1, 2)` in `_common_forward()`
- `FCNEncoder.forward()` — `transpose(1, 2)` at entry (D-01)
- `TCCEncoder.forward()` — `transpose(1, 2)` at entry (D-01)
- `Conv1dResNetEncoder.forward()` — `transpose(1, 2)` at entry

### Augmentation Axes

Augmentation primitives (Scaling, Permutation) operate on the raw `(B, T, C)` data before encoding. Configure axis parameters accordingly:

- `ScalingParameters(channel_dim=-1)` — scales along the channel axis (dim=2 in 3-D)
- `PermutationParameters(time_dim=1)` — permutes along the time axis

### Testing

Always use **asymmetric shapes** (`T != C`) in encoder tests to catch transpose regressions. For example, `torch.randn(4, 50, 3)` for `(B, T, C)` with `T=50` and `C=3` will crash if the encoder drops its transpose, because Conv1d would see 50 channels instead of 3.

### Encoder Output Shape Consistency

All models expose a uniform `encode()` API via encoding mixins. The output shape is controlled by `EncodingOutputShape` (`VECTOR` | `SEQUENCE`), defined in `chronocratic.models.enums.encoding`.

- **`VECTOR`** (`"vector"`): Returns 2-D tensor `(N, D)` — one representation per sample.
- **`SEQUENCE`** (`"sequence"`): Returns 3-D tensor `(N, T, D)` — one representation per timestep.

The mixin `encode()` and `encode_batch()` methods accept an `output: EncodingOutputShape = EncodingOutputShape.VECTOR` keyword argument. Models that natively produce sequence outputs apply their default reduction (last-step, mean-pool) when `VECTOR` is requested. Models that natively produce flat vectors return a length-1 sequence when `SEQUENCE` is requested.

Each model class declares `supported_outputs: frozenset[EncodingOutputShape]` as a class attribute. This frozenset documents which output shapes the model supports natively without fallback warnings.

#### Model Support Matrix

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
| Series2Vec | Yes | Yes | Both supported |
| RecurrentAutoEncoder | Yes | Yes | Both supported |

> Models that list `SEQUENCE=Yes` but are not natively sequence-producing (MCL, TSTCC)
> issue a `_warn_sequence_fallback` and return `(B, 1, D)` by unsqueezing the vector.
> `supported_outputs` is the authoritative declaration; check that attribute on the model class.

#### Encoding Mixin Architecture

Two mixin families serve different encoder topologies:

1. **`BasicEncodingMixin`** (`_mixin/encoding.py`) — Fixed-length sequence models (TST, TimeVAE, TimeNet, RecurrentAutoEncoder, MCL, TSTCC, SimCLR, Series2Vec). Subclasses implement `_get_encoder()` and optionally override `_encode_batch()`. The mixin owns DataLoader iteration, eval/inference mode, device placement, and result concatenation.

2. **`BaseEncodingMixin`** (`convolutional/dilated/_mixin/encoding.py`) — Dilated conv models (TS2Vec, AutoTCL, CoST) with sliding-window inference, multi-scale pooling, and mask-mode handling. Subclasses override `_get_encoder()`, `_get_eval_method()`, and `_get_slice()`. Specialized mixins extend the base: `PoolingEncodingMixin` (TS2Vec, AutoTCL) and `DecompositionEncodingMixin` (CoST).

All encoders implement `HasEncoder` protocol (`chronocratic.models.protocols`). The `.encoder` property returns an `nn.Module` for representation extraction, checkpointing, or fine-tuning. Decoder-bearing models implement `HasDecoder` and `HasEncoderDecoder`.

## Augmentation Architecture

Augmentation follows a three-layer design: **primitives** → **view sets** → **producers**.

### Primitives

Individual augmentation operations (Scaling, Permutation, etc.) live in `augmentation/primitives.py`. Each takes a `(B, T, C)` tensor and returns an augmented tensor of the same shape.

### View Sets

- `SingleView` — one augmented copy of the input
- `ViewPair` — two independent augmented views
- `AlignedPair` — two views with shared augmentation parameters

### Producers

- `SingleViewProducer` — wraps a single augmentation
- `IndependentPairProducer` — two independent augmentation pipelines
- `RolePairProducer` — different augmentations for first/second views
- `FullOverlapProducer` — two views with identical augmentation applied twice

### Custom Augmentation

Models accept `augmentation: AugmentationProducer[ViewType] | None = None` in `__init__`. When `None`, a default producer is created. To use a custom augmentation:

```python
model = AutoTCL(
    input_dim=3,
    augmentation=SingleViewProducer(aug=ScalingParameters()),
)
```

See `augmentation/base.py` and `augmentation/producers.py` for full API.

#### Implementation Rules

- `supported_outputs` is a class-level `frozenset`. Override in model subclasses to declare capabilities.
- `_encode_batch()` signature must accept `output: EncodingOutputShape` keyword arg. Branch on value to return correct rank.
- The `encode()` mixin verifies output rank via assert: `result.ndim == expected_ndim` (2 for VECTOR, 3 for SEQUENCE). Do not silence this assert.
- When `encoding_window` is not explicitly provided, `output` drives the pooling strategy: VECTOR → `"full_series"`, SEQUENCE → `None`.
- Never hardcode shape logic outside the mixin. Use `EncodingOutputShape` enum values, not string literals.
- `encode_batch()` is gradient-preserving and DataLoader-free. Use for adversarial loops and single-batch encoding.
- **NaN protection**: Always call `zero_fill_padding()` before encoder forward pass in inference paths. See NaN Handling section for details.

#### Testing Encoder Output Shapes

- Test both `VECTOR` and `SEQUENCE` outputs for models that declare support.
- Verify tensor rank: `assert result.ndim == 2` for VECTOR, `assert result.ndim == 3` for SEQUENCE.
- Verify `supported_outputs` frozenset matches actual capabilities — calling with unsupported shape must either fall back with warning or raise appropriately.
- Test `HasEncoder` protocol conformance: `assert isinstance(model, HasEncoder)`.
- Verify `encode_batch()` preserves gradients when `batch_x.requires_grad`: `assert result.requires_grad`.

## Device Compatibility (CPU / CUDA / MPS)

Code in this library must work correctly on all PyTorch backends. Follow these five rules to ensure cross-device compatibility.

### 1. Create on input's device, don't transfer after

Build auxiliary tensors on the same device as the input, instead of creating on CPU and then calling `.to()`.

**Do:**
```python
labels = torch.eye(k - 1, dtype=torch.float32, device=z1.device)
mask = torch.zeros(batch_size, seq_len, device=x.device)
```

**Don't:**
```python
labels = torch.eye(k - 1, dtype=torch.float32).to(z1.device)  # CPU allocation then transfer
```

### 2. Loss functions inherit device from first tensor argument

Every loss function must derive its working device from its first tensor argument. Tensors created inside `forward()` or loss computation must use `device=input.device`.

**Gold standard pattern:** `NTXentLoss._correlated_mask()` in `losses/ntxent.py`:
```python
mask = ~torch.eye(n, dtype=torch.bool, device=device)
idx = torch.arange(batch_size, device=device)
```

### 3. Host-side libraries need explicit round-trip

Libraries like SciPy only accept host (numpy) arrays. On MPS tensors, calling `.numpy()` without `.cpu()` first raises `RuntimeError`. Explicitly round-trip through CPU.

```python
from scipy.signal import lfilter
import numpy as np

def _filter_on_device(b: np.ndarray, a: np.ndarray, data: torch.Tensor) -> torch.Tensor:
    filtered = lfilter(b, a, data.cpu().numpy())
    return torch.as_tensor(filtered, dtype=torch.float32, device=data.device)
```

### 4. CUDA-only kernels fall back to CPU on MPS — that is acceptable

If a kernel has no MPS equivalent (e.g., SoftDTW's CUDA kernel), falling back to CPU when `x.is_cuda` is False is the correct behavior. Document the fallback with a comment to prevent future "fixes" that duplicate the logic.

**See:** `Series2Vec._build_soft_dtw()` — MPS tensors have `is_cuda=False`, so they correctly use the CPU path.

### 5. pin_memory=True only when no gradients flow

`pin_memory=True` in DataLoader stages a CPU buffer for non-blocking H2D copies. When gradients are enabled, pinning can warn or error because tensors require grad and pinning allocates pagelocked memory.

**Do:**
```python
loader = DataLoader(dataset, pin_memory=not gradient_enabled)
```

### Lint Guard

Run `bash scripts/check_device.sh` to detect bare tensor constructors (`torch.eye`, `torch.zeros`, `torch.ones`, `torch.arange`, etc.) without `device=` in model source files. Legitimate exceptions are annotated with `# device-ok`.

## NaN Handling

Time series data may contain trailing NaN values for padding variable-length sequences within a batch. All models must handle this gracefully — training and inference paths must never produce NaN representations.

### Core Utilities

- `zero_fill_padding()` — replaces NaN with zero and returns a keep-mask for loss weighting.
- `generate_not_nan_mask()` — produces a boolean mask identifying valid (non-NaN) timesteps.
- `masked_reconstruction_loss_mean()` — computes mean loss only over non-NaN entries.

### Rules

1. **Training path** — call `zero_fill_padding()` in `training_step` before any encoder that cannot handle NaN (e.g., Conv1d, Fourier layers).
2. **Inference path** — call `zero_fill_padding()` in the encoding mixin (`encode()` / `encode_batch()`) before the encoder forward pass.
3. **Encoder `_process_not_nan_mask`** — always zero out NaN *before* masking. Multiplication (`x * 0.0`) on NaN yields NaN, not zero. Use `torch.nan_to_num(x, nan=0.0)` first, or use boolean indexing (`modified_x[~not_nan_mask] = 0`).
4. **Reconstruction models** — always apply NaN masking in loss computation to avoid training on padding artifacts.

### Implementation Status

- `BasicEncodingMixin` models: NaN guard lives in each model's `_encode_batch()` method.
- `PoolingEncodingMixin` (TS2Vec, AutoTCL): NaN guard in `_evaluate_with_pooling`.
- `DecompositionEncodingMixin` (CoST): NaN guard in `_evaluate_with_feature_concatenation`.
- `AutoTCLTimeSeriesEncoder._process_not_nan_mask`: uses `torch.nan_to_num` + multiplication pattern.

### Testing

Every model that processes variable-length batches should include tests for:
- Trailing-NaN batch (partial padding)
- All-NaN sample (entirely padding)
- Clean-input regression (no NaN present)

See `tests/unit/test_encode_nan_guard.py` and `tests/unit/test_dilated_nan_encode.py` for examples.

## Input Length Robustness

Reference implementations are written for one fixed input length. This library runs them at several. **Assume every model you port is fixed-length until you have proven otherwise, and budget for patching it** — this is expected porting work, not a defect in the upstream code.

### Why lengths vary

Three distinct length regimes exist in a single experiment run:

| Path | Typical length | Source |
| --- | --- | --- |
| Training | full partition (e.g. 36,000 timesteps) | forecasting datamodule in `RAW_SERIES` mode emits one partition per batch |
| Validation | full partition (e.g. 11,520) | same |
| Encoding | forecast windows (e.g. 96, 168, 336) | one length per forecast case, all encoded by **one** model instance |

The nominal `sequence_length` may match none of them. A model is pretrained once and then reused across every forecast case, so per-case reconstruction is not available as an escape hatch — the encoder must absorb the variation.

### Step 1 — audit the port

Before writing tests, read the upstream code for the constructs below.

Length-agnostic by construction — leave alone:

- `Conv1d` / `ConvTranspose1d`, dilated or not — kernels slide over any length.
- Attention and `nn.TransformerEncoder` — but see the cost note below.
- `nn.Linear` applied **per timestep**, i.e. mapping the feature axis (`Linear(input_dim, hidden_dim)` on a `(B, T, C)` tensor).
- Pooling with a computed kernel (`full_series_pooling`, `multiscale_pooling` in `models/utils`).

Needs patching — each of these is a latent crash:

| Construct | Patch |
| --- | --- |
| `nn.Flatten()` → `nn.Linear` — flattened width scales with `T` | Resample the temporal axis first ([Pattern B](#pattern-b--resample-the-temporal-axis-timevae)) |
| `register_buffer("pe", torch.zeros(sequence_length, ...))` and other position-indexed buffers | Compute analytically in `forward()` ([Pattern A](#pattern-a--compute-analytically-in-forward-tst)) |
| Learned position embeddings (`nn.Parameter` per position) | Cannot extrapolate — raise, don't interpolate ([Pattern C](#pattern-c--raise-when-the-limit-is-inherent-tst)) |
| `nn.Linear` whose `in_features` came from `sequence_length` | [Runtime-Derived Dimensions](#runtime-derived-dimensions) |
| Reshapes with a literal temporal extent (`x.view(B, sequence_length, -1)`) | Derive from `x.size(...)` |

Two traps worth naming explicitly:

- **Silent truncation.** `self.pe[: x.size(0)]` raises a broadcast error on an over-long input, but on an *under*-long input it quietly returns a shorter slice and the forward pass succeeds with wrong values. A model can pass its tests while being wrong in one direction only. Prefer constructions that cannot half-work.
- **Length-agnostic ≠ length-safe.** Self-attention is O(T²) in time and memory. At `T = 11,520` one attention matrix holds ~1.33 × 10⁸ entries — ~530 MB in fp32, multiplied by heads and layers. Removing a shape limit from a transformer converts a shape error into an out-of-memory error. Transformers need a training-path cap regardless.

### Step 2 — apply the patch patterns

Three patterns cover everything encountered so far. Both worked examples live in the tree; read them before inventing a fourth.

#### Pattern A — compute analytically in `forward()` (TST)

When upstream pre-allocates a table that a closed-form expression could produce, drop the table.

`FixedPositionalEncoding` (`transformer/tst/ts_transformer.py`) originally built a
`(sequence_length, 1, hidden_dim)` buffer in `__init__`. The sinusoidal formula is closed-form, so the buffer was only ever a cache — and one that silently imposed a ceiling. It now builds `position` and `div_term` inside `forward()` from `x.size(0)`, on `x.device` with `x.dtype` (see [Device Compatibility](#device-compatibility-cpu-cuda-mps)).

Cost: recomputation per forward, which is negligible next to attention. Benefit: no ceiling, and the under-long silent-truncation path disappears. This is *more* faithful to the paper than the cache was — the paper defines a formula; the table was an implementation detail that narrowed it.

#### Pattern B — resample the temporal axis (TimeVAE)

When a layer structurally requires a fixed temporal width, insert `nn.AdaptiveAvgPool1d(target)` immediately before it.

`TimeVAEEncoder` (`generative/timevae/model.py`) flattens its conv stack into `nn.Linear(encoder_last_dense_dim, latent_dim)`, so the flattened width — and therefore the weight shape — scaled with input length. The pool resamples the temporal axis to a constant number of ordered slots, holding the flattened width fixed at any input length.

Choose `target` as **the conv-stack output at the nominal `sequence_length`**. That makes the layer an exact identity at that length (16 elements into 16 bins is one element per bin), so training behavior is bit-identical and existing checkpoints keep their shapes. That identity property is what makes the patch safe to land, and §Testing turns it into an assertion.

Do not reach for global mean/max pooling instead. Collapsing the temporal axis to width 1 makes a spike at the start indistinguishable from one at the end — it compiles, the suite stays green, and the model quietly stops being the model whose name it carries.

#### Pattern C — raise when the limit is inherent (TST)

Some limits cannot be patched away. `LearnablePositionalEncoding` stores one *learned* vector per position; there is no principled value for position 5,000 when the model only ever trained on 0–127. Interpolating fabricates weights and silently changes what the model represents.

It now raises a `ValueError` naming both lengths and the remedy. Failing loudly is the correct outcome — the alternative is a number that looks fine and means nothing.

### Step 3 — cap the training path

Patching the model handles *shape*. It does not handle *cost*: a patched transformer will still exhaust memory on a 36,000-timestep partition. Add a `max_train_length` parameter and apply the shared helper in `training_step` (or the shared loss helper, if the model has one):

```python
x = process_sample_length(sample=x, max_sample_length=self._max_train_length)
```

- Apply it as the **first** operation on the batch — before mask construction, before `zero_fill_padding` — or masks desynchronize from the cropped tensor.
- **Default to `None`**, a documented no-op, so adding the parameter cannot perturb a working configuration. Let the consuming pipeline set the value.
- **Never crop inside `encode()`.** `encode()` must be deterministic (identical input → identical representation) and must return one representation per input series; `process_sample_length` draws a *random* window. Inference-path length handling is chunked windowing, which is a separate design.

TS2Vec, CoST, AutoTCL, TST, and TimeVAE all follow this.

### Testing

Every model must have tests covering:

- **Encode at several non-nominal lengths** — shorter and longer than `sequence_length`, using **one model instance across all lengths in a single test body**. A fresh model per parametrized case does not exercise the reuse pattern that actually breaks.
- **Identity/regression at the nominal length** — if a resampling or dynamic-computation layer was added, assert the output at `sequence_length` is unchanged (exact equality where the operation is arithmetically a no-op). This is what licenses the change: a layer that cannot alter any currently working path cannot regress one.
- **Long-input training** — a batch many times `sequence_length` through `training_step`, asserting a finite scalar loss with `requires_grad`.
- **Gradient flow at a non-nominal length** — `encode_batch` is differentiable and the adversarial-attack path depends on it.

See `tests/unit/test_long_input_crop.py` for examples.

### Record the divergence

Every patch here is a deliberate departure from the reference implementation, and a research codebase that loses track of those loses track of what its numbers mean. Each one needs:

- A code comment at the patched layer stating what it does and why the reference lacks it.
- A note in the commit body and the changelog fragment.
- The class docstring, when the patch changes what a constructor argument means. `sequence_length` on a patched model no longer means "the only length this accepts" — say so where the reader will look.

When a patch could alter what representations mean, write the rationale down before landing it. Passing tests are not sufficient evidence on their own: the failure mode here is a change that keeps every test green while quietly redefining the model.

Current divergences and known limits:

- **TimeVAE** — `nn.AdaptiveAvgPool1d` before the `Flatten`, so the latent projection width is constant. No counterpart in the reference; required because one pretrained model encodes several forecast window lengths. Exact identity at the nominal `sequence_length`.
- **TST** — positional encoding computed in `forward()` rather than cached. Numerically identical to the reference at any length the reference supported.
- **SimCLR** — ported as a 1-D ResNet. The reference stacks `Conv2d` over an input unsqueezed to `(B, C, T, 1)`, where the outer kernel columns only ever see zero padding; the `Conv1d` stack is numerically identical (asserted at stride 1 and 2 in `tests/unit/test_simclr.py`), as is `BatchNorm1d` against the reference's `BatchNorm2d` on that axis. Six further departures, each commented at its line: each encoder stage honours its own configured depth, where the reference repeats the third stage's index in place of the fourth; the learning rate follows the *original* SimCLR's linear warmup into cosine annealing ([google-research/simclr](https://github.com/google-research/simclr) `WarmUpAndCosineDecay`) rather than the reference's `ExponentialLR(gamma=1e-6)`, which reaches `1e-15` within two epochs and would suppress training; normalization defaults to `GroupNorm(1, C)` rather than hardcoded `BatchNorm`, so the encoder remains well-defined at `batch_size=1`; the default augmentation draws a per-sample scale factor, the reference's single shared factor being meaningful only for its offline whole-corpus augmentation; that factor uses `sigma=0.1` rather than the `sigma=1.1` the reference's `DataTransform` passes, since `N(1.0, 1.1)` places a substantial fraction of its mass below zero and those draws invert the sign of the series, leaving the two views describing different signals; and the encoder returns the unpooled feature map `(B, C, T')`, matching the reference's flatten-of-`C×T` contrastive input. An earlier version applied `AdaptiveAvgPool1d(1)`, which destroyed the angular diversity NT-Xent needs; pooling is now done in `_encode_batch` (mean for VECTOR, transpose for SEQUENCE), keeping the loss path intact while leaving downstream probes unaffected. `0.1` is the default of the reference's own `scaling()`, overridden only by `DataTransform`, and that module documents the intended range as `[0.1, 2.0]`.
- **SimCLR, open limit** — the model is sensitive to the ratio between its capacity and the number of optimizer steps a run affords. The default configuration is a ResNet-50-scale backbone; where that ratio is unfavourable it can remain far from convergence, in which case a randomly-initialised backbone may match or exceed a trained one under a linear probe and repeated runs of one configuration disagree. The learning-rate schedule reduces this sensitivity without removing it, and the contributing causes — nondeterministic kernel selection versus undertraining — have not been separated. Callers can pin the former through `Trainer(deterministic=True)`.
- **TST, open limit** — encoding at full-partition lengths is length-*correct* but O(T²) in memory and will exhaust it before raising. Chunked windowing is an open follow-up.
- **`process_sample_length`, open limit** — uses `np.random.default_rng()` per call, so it is not controlled by `lightning.seed_everything` and training crops are not reproducible across runs. Shared by TS2Vec, CoST, AutoTCL, TST, and TimeVAE.
- **Pretrain/encode length mismatch, open question** — a model pretrained on crops of one length and used to encode windows of another is mechanically valid after patching, but whether representation quality holds across lengths is empirical. Worth measuring before publishing numbers.
