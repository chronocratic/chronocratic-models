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

## Adding Changelog Fragments

The project uses [towncrier](https://towncrier.readthedocs.io/) for managing changelog entries. Each PR should include a changelog fragment in the `changelog.d/` directory.

```bash
# Create a fragment (e.g., for a new feature in PR #42)
echo "Added new TimeVAE model for generative time-series encoding." > changelog.d/42.added.md
```

Fragment types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`.

```bash
# Verify fragments before merging
uv run towncrier check --compare-with origin/dev
```

See [`changelog.d/README.md`](../changelog.d/README.md) for detailed fragment instructions.

## Code Style

- Use **snake_case** for functions and variables, **PascalCase** for classes.
- Write **Google-style docstrings** for all public functions and classes.
- Use **type hints** for all function signatures and return types.
- Prefer **functional programming patterns** and modular code organization.
- Use **keyword arguments** for all function calls.

## Parameters Consistency

All model classes and their config dataclasses follow shared parameter conventions. These rules ensure that `Model(**vars(ModelParameters(...)))` works across the entire library.

### Canonical Hyperparameter Names

Use these exact names. Do not invent alternatives.

| Canonical Name | Description | Do NOT Use |
|---|---|---|
| `input_dims` | Number of input features/channels | `feat_dim`, `n_in`, `input_channels` |
| `hidden_dims` | Hidden representation size | `d_model` |
| `depth` | Number of layers | `num_layers` |
| `dropout_rate` | Dropout probability | `dropout` |
| `num_heads` | Attention head count | `n_heads` |
| `feedforward_dims` | FFN intermediate dimension | `dim_feedforward` |
| `sequence_length` | Temporal dimension | `max_seq_len`, `seq_len` |
| `conv_kernel_size` | Convolution kernel size | `kernel_size` |
| `weight_decay` | L2 regularization | `l2_reg` |
| `output_dims` | Output/embedding dimension | `final_out_channels` |
| `reconstruction_weight` | VAE reconstruction term | `reconstruction_wt` |

Model-specific names are acceptable only when genuinely unique to that model (e.g., `latent_dim` for TimeVAE, `embedding_dims` for Series2Vec).

### Config-to-Model Contract

Config dataclasses and model `__init__` signatures must mirror each other exactly.

- Every config field must have a matching `__init__` parameter with the **same name** and **same default value**.
- Use `@dataclass(kw_only=True)` on all config classes.
- Defaults must be declared in **both** the config dataclass and the model `__init__`. This allows partial config instantiation.
- Verify with `Model(**vars(ModelParameters(...)))` — it must not raise.
- Use `save_hyperparameters(ignore=["augmentation"])` in Lightning modules. Non-callable config values should not be ignored.

**Example:**

```python
# Config
@dataclass(kw_only=True)
class MyModelParameters:
    input_dims: int
    hidden_dims: int = 64
    dropout_rate: float = 0.1

# Model
class MyModel(pl.LightningModule):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._dropout_rate = dropout_rate
        save_hyperparameters(ignore=["augmentation"])
```

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
3. Document any deliberate divergence in `.planning/audits/`.

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

### `self._{name}` Attribute Storage

Store all hyperparameters as private attributes with the `self._{name}` prefix in model `__init__`.

```python
def __init__(self, input_dims: int, hidden_dims: int = 64):
    self._input_dims = input_dims
    self._hidden_dims = hidden_dims
```

Public attributes are reserved for computed values (e.g., `self.criterion`, `self.loss_fn`) and submodules (`self._encoder`, `self._decoder`).

### Literal vs. Enum Choices

| Pattern | When to Use | Example |
|---|---|---|
| **Enum (`StrEnum`)** | Closed, small set of values | `MaskMode`, `RecurrentCellType` |
| **`str` with default** | Broad options, open-ended | `pos_encoding: str = "fixed"`, `activation: str = "gelu"` |
| **Unconstrained numeric** | Values users may override freely | `dropout_rate: float = 0.01` |
| **`Literal`** | Only in config dataclasses, not model `__init__` | `OptimizerName = Literal["Adam", "RAdam", "AdamW"]` |

Never use `Literal` to restrict numeric values — users may legitimately override them. Keep model `__init__` signatures using `str` or concrete Enum types; reserve `Literal` for config-layer type narrowing.

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
