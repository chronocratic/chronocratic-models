# Testing Patterns

**Analysis Date:** 2026-06-17

## Test Framework

**Runner:**
- `pytest` version 8.2+ (from `[dependency-groups].dev` in `pyproject.toml`)
- Config: `pyproject.toml` (`[tool.pytest.ini_options]`)

**Assertion Library:**
- `pytest` built-in assertions (standard `assert` statements)
- `torch.testing.assert_close()` for tensor equality comparisons
- `torch.allclose()` for numerical equivalence with tolerance

**Coverage Tool:**
- `pytest-cov` version 5.0+ (from dev dependencies)

**Run Commands:**
```bash
uv run pytest                        # Run all tests
uv run pytest -v                     # Verbose output
uv run pytest --cov=src/chronocratic/models --cov-report=xml   # Coverage (CI format)
uv run pytest tests/test_smoke.py    # Run specific test file
uv run pytest tests/unit/            # Run unit test directory
uv run pytest tests/integration/     # Run integration test directory
```

## Test Configuration

**From `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

**Key settings:**
- `pythonpath = ["."]`: Adds project root to `sys.path` so `from chronocratic.models ...` works
- `testpaths = ["tests"]`: Tests live in top-level `tests/` directory

**CI execution** (`build-and-test.yml`):
```yaml
- name: Run tests
  run: uv run pytest tests/ --cov=src/chronocratic/models --cov-report=xml
```

## Test File Organization

**Directory layout (28 test files across 3 tiers):**
```
tests/
├── conftest.py                                  # Shared fixtures (training helpers, data factories)
├── test_smoke.py                                # Model training smoke tests (VER-01..VER-05)
├── test_config.py                               # Config dataclass tests
├── test_config_hierarchy.py                     # Config location/inheritance tests
├── test_aug_config.py                           # Augmentation parameter config tests
├── test_augmentation.py                         # Augmentation ABC + strategies + concrete augmentations
├── test_augmentation_base.py                    # Base module import/contract tests
├── test_augmentation_per_model.py               # Per-model augmentation module tests
├── test_aug_contract.py                         # New augmentation contract types (Protocol, Producer, ViewSet)
├── test_aug_producers.py                        # Producer combinators (VER-01..VER-07)
├── test_aug_covariance.py                       # Type covariance verification tests
├── test_aug_decorators.py                       # Seeded decorator tests
├── test_aug_cross_model.py                      # Cross-model producer reuse tests
├── test_aug_trainable_support.py                # Trainable augmentation support helpers
├── test_aug_primitives.py                       # Jitter, Scaling, Permutation, ComposeAugmentation
├── test_from_config.py                          # Model instantiation from config tests
├── test_mixin.py                                # Encoding mixin hierarchy tests
├── test_ts2vec_producer.py                      # TS2Vec producer integration
├── test_cost_producer.py                        # CoST producer integration
├── test_autotcl_producer.py                     # AutoTCL producer integration
├── test_tstcc_producer.py                       # TSTCC producer integration
├── unit/
│   ├── __init__.py
│   ├── test_backbone_representation_dim.py      # Backbone rep_dim protocol compliance
│   ├── test_series2vec_supervised.py            # Series2Vec supervised finetuning
│   ├── test_supervised_package.py               # SupervisedModule + factory + callback tests
│   ├── test_tst_supervised.py                   # TST supervised finetuning
│   └── test_tstcc_supervised.py                 # TSTCC supervised migration tests
└── integration/
    ├── __init__.py
    └── test_supervised_integration.py           # E2E supervised training (all backbones)
```

**Naming:** `test_<module_name>.py` -- one test file per feature area.

**No `@pytest.mark.parametrize` used.** Tests are written individually for clarity.

**`@pytest.mark.skip`:** Used for slow training tests:
```python
@pytest.mark.skip(reason="slow: Lightning trainer overhead")
def test_tstcc_trains_20_steps(self) -> None:
    ...
```

See: `tests/test_tstcc_producer.py`

## Shared Fixtures

**Primary fixtures:** `tests/conftest.py`

**Training helper fixture:** Provides a reusable `_run_train_steps` function that wraps Lightning `Trainer`:

```python
@pytest.fixture
def train_steps() -> Callable[..., list[torch.Tensor]]:
    """Return the _run_train_steps helper so tests can pass a model."""
    return _run_train_steps
```

Usage:
```python
def test_ts2vec_trains_5_steps(self, train_steps) -> None:
    model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())
    losses = train_steps(model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)
    assert_finite_losses(losses, expected_min=5)
```

**Data factory fixture:**

```python
@pytest.fixture
def random_data() -> Callable[..., torch.Tensor]:
    """Factory for random time-series tensors."""

    def _factory(
        batch: int = 4, seq_length: int = 100, input_dims: int = 1, layout: str = "NLC"
    ) -> torch.Tensor:
        if layout == "NLC":
            return torch.randn(batch, seq_length, input_dims)
        return torch.randn(batch, input_dims, seq_length)

    return _factory
```

**Assertion helper fixture:**

```python
@pytest.fixture
def finite_losses() -> Callable[..., None]:
    """Return the assert_finite_losses helper."""
    return assert_finite_losses
```

Where `assert_finite_losses` checks:
- At least `expected_min` losses collected
- Each loss is a scalar tensor (`ndim == 0`)
- Each loss is finite (`math.isfinite(loss.item())`)

**`autouse` fixtures:** Used to load classes into test instance attributes:

```python
@pytest.fixture(autouse=True)
def _load_classes(self) -> None:
    from chronocratic.models.convolutional.dilated._mixin.encoding import (
        BaseEncodingMixin,
        DecompositionEncodingMixin,
        PoolingEncodingMixin,
    )
    self.BaseEncodingMixin = BaseEncodingMixin
    self.PoolingEncodingMixin = PoolingEncodingMixin
    self.DecompositionEncodingMixin = DecompositionEncodingMixin
```

See: `tests/test_mixin.py`

**Module-level fixtures:** Create minimal test doubles:

```python
@pytest.fixture
def pooling_model() -> nn.Module:
    """Create a minimal pooling-based model for testing."""
    from chronocratic.models.convolutional.dilated._mixin.encoding import PoolingEncodingMixin

    class _PoolingTestModel(PoolingEncodingMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._averaged_encoder = _DummyEncoder(output_dim=64)
            self.device = torch.device("cpu")

    return _PoolingTestModel()
```

See: `tests/test_mixin.py`

## Test Structure

**Class-based suites** with descriptive class names and module-level docstrings:

```python
class TestTS2VecModelParameters:
    """Test TS2VecModelParameters fields and defaults."""

    def test_is_dataclass(self) -> None:
        assert is_dataclass(TS2VecModelParameters)

    def test_requires_only_input_dims(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.input_dims == 1
```

See: `tests/test_config.py`

**Test methods** have descriptive names explaining the scenario:

- `test_ts2vec_trains_5_steps` -- Behavior being verified
- `test_default_kernel_sizes_isolation` -- Specific property under test
- `test_encode_no_state_mutation` -- Negative property (nothing should happen)
- `test_missing_input_dims_raises` -- Error behavior
- `test_loss_equivalence_to_original_auto_tcl` -- Regression safety

**Return type annotations:** All test methods return `-> None`.

## Test Categories

### Config Tests

Verify dataclass fields, defaults, mutability isolation, and field counts:

```python
def test_is_dataclass(self) -> None:
    assert is_dataclass(TS2VecModelParameters)

def test_default_kernel_sizes_isolation(self) -> None:
    p1 = CoSTModelParameters(input_dims=1, sequence_length=100)
    p2 = CoSTModelParameters(input_dims=1, sequence_length=100)
    p1.kernel_sizes.append(256)
    assert 256 not in p2.kernel_sizes

def test_field_count(self) -> None:
    assert len(fields(TS2VecModelParameters)) == 11

def test_vars_produces_correct_keys(self) -> None:
    params = TS2VecModelParameters(input_dims=1)
    result = vars(params)
    expected_keys = {"input_dims", "hidden_dims", "output_dims", ...}
    assert set(result.keys()) == expected_keys
```

See: `tests/test_config.py`, `tests/test_aug_config.py`

### Augmentation Contract Tests

Verify Protocol structural conformance, ViewSet dataclass behavior, and Producer patterns:

```python
class TestViewSetTypes:
    """ViewSet dataclass types (SingleView, ViewPair, AlignedPair)."""

    def test_single_view_is_frozen(self) -> None:
        sv = SingleView(view=torch.randn(2, 10, 4))
        with pytest.raises(Exception):
            sv.view = torch.randn(3, 5, 2)  # type: ignore

    def test_aligned_pair_extends_view_pair(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        ap = AlignedPair(first=t1, second=t2, overlap_length=8)
        assert isinstance(ap, ViewPair)
        assert ap.overlap_length == 8
```

See: `tests/test_aug_contract.py`, `tests/test_augmentation.py`

### Producer Combinator Tests

Verify `SingleViewProducer`, `IndependentPair`, `RolePair`, `FullOverlapPair`:

```python
class TestSingleViewProducer:
    def test_produces_single_view_with_tensor(self) -> None:
        aug = Jitter()
        producer = SingleViewProducer(aug=aug)
        x = torch.randn(2, 10, 3)
        result = producer.produce(x)

        assert isinstance(result, SingleView)
        assert result.view.shape == x.shape
```

See: `tests/test_aug_producers.py`

### Type Covariance Tests

Verify PEP 695 generic variance at runtime:

```python
class TestProducerCovariance:
    """Test AugmentationProducer[V] covariance at runtime."""

    def test_full_overlap_pair_fits_viewpair_slot(self) -> None:
        """FullOverlapPair produces AlignedPair, which fits ViewPair consumer."""

        def consumer(p: AugmentationProducer[ViewPair]) -> ViewPair:
            return p.produce(torch.randn(2, 50, 3))

        producer = FullOverlapPair(aug=Jitter())
        result = consumer(producer)
        assert isinstance(result, AlignedPair)
```

See: `tests/test_aug_covariance.py`

### Decorator Tests

Verify `Seeded` decorator behavior:

```python
class TestSeededDeterminism:
    def test_seeded_producer_is_deterministic(self) -> None:
        producer = SingleViewProducer(aug=Jitter())
        seeded = Seeded(inner=producer, seed=42)
        x = torch.randn(2, 10, 3)

        result1 = seeded.produce(x)
        result2 = seeded.produce(x)

        torch.testing.assert_close(result1.view, result2.view)
```

See: `tests/test_aug_decorators.py`

### Cross-Model Reuse Tests

Verify that producers and primitives work across different model types:

```python
class TestCrossModelReuse:
    def test_jitter_works_with_any_model(self) -> None:
        aug = Jitter()
        data = torch.randn(4, 100, 3)
        result = aug(data)
        assert result.shape == data.shape
```

See: `tests/test_aug_cross_model.py`

### Model Instantiation Tests

Verify models construct from config dataclasses and have correct defaults:

```python
def test_ts2vec_instantiation_returns_instance(self) -> None:
    config = TS2VecModelParameters(input_dims=1)
    model = TS2Vec(**vars(config), augmentation=None)
    assert isinstance(model, TS2Vec)

def test_ts2vec_augmentation_pass_through(self) -> None:
    model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())
    assert model._augmentation is not None
    result = model._augmentation.produce(torch.randn(4, 100, 1))
    assert isinstance(result, ViewPair)
```

See: `tests/test_from_config.py`

### Mixin Tests

Verify inheritance, polymorphic dispatch, and source-level compliance:

```python
def test_pooling_is_subclass_of_base(self) -> None:
    assert issubclass(self.PoolingEncodingMixin, self.BaseEncodingMixin)

def test_encode_no_state_mutation(self, pooling_model: nn.Module) -> None:
    pooling_model.encode(data=data, batch_size=2, num_workers=0)
    assert not hasattr(pooling_model, "_encoder") or (
        "_encoder" not in pooling_model.__dict__
    )
```

See: `tests/test_mixin.py`

### Supervised Package Tests

Use minimal stubs to test `SupervisedModule` without real backbones:

```python
class _DummyBackbone(nn.Module):
    """A tiny backbone with a known representation_dim for tests."""

    def __init__(self, rep_dim: int = 4) -> None:
        super().__init__()
        self._rep_dim = rep_dim
        self.fc = nn.Linear(2, rep_dim)

    @property
    def representation_dim(self) -> int:
        return self._rep_dim
```

See: `tests/unit/test_supervised_package.py`

### Backbone Representation Dim Tests

Verify `representation_dim` matches actual forward output:

```python
def test_representation_dim_matches_forward(self) -> None:
    model = TST(
        feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1, dim_feedforward=16
    )
    x = torch.randn(2, 10, 2)
    padding_masks = torch.ones(2, 10, dtype=torch.bool)
    reps = model.get_representations(x, padding_masks)
    reps_masked = reps * padding_masks.unsqueeze(-1)
    flat = reps_masked.reshape(reps_masked.shape[0], -1)
    assert flat.shape[1] == model.representation_dim
```

See: `tests/unit/test_backbone_representation_dim.py`

### Smoke Tests

Run actual Lightning training and verify finite loss:

```python
class TestModelTrainingSmoke:
    """End-to-end training smoke tests for each model (VER-01 through VER-05)."""

    def test_ts2vec_trains_5_steps(self) -> None:
        model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())
        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)
        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, "Loss must be a scalar tensor"
            assert math.isfinite(loss.item())
```

See: `tests/test_smoke.py`

### Checkpoint Tests

Verify save/reload preserves model weights:

```python
def test_ts2vec_checkpoint_preserves_encoder_weights(self) -> None:
    model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())
    original = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(model.state_dict(), tmp_path)
        loaded_state = torch.load(tmp_path, weights_only=True)
        model.load_state_dict(loaded_state)
        for key in original:
            assert torch.equal(original[key], model.encoder.state_dict()[key])
    finally:
        os.unlink(tmp_path)
```

See: `tests/test_smoke.py`

### Integration Tests

Full pipeline: backbone -> factory -> SupervisedModule -> training:

```python
def test_tst_trains_end_to_end(self) -> None:
    backbone = TST(feat_dim=2, max_seq_len=10, d_model=8, n_heads=2, num_layers=1)
    module = make_tst_supervised(
        backbone, num_outputs=3, task="classification", freeze_backbone=False
    )
    dataset = _DummyTSTDataset(size=20, seq_len=10, feat_dim=2, num_classes=3)
    dataloader = DataLoader(dataset, batch_size=4)
    trainer = Trainer(
        max_epochs=1, limit_train_batches=3, limit_val_batches=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=dataloader)
    assert "train_loss" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
```

See: `tests/integration/test_supervised_integration.py`

### Barrel Export Tests

Verify `__all__` exports are clean and no internal classes leak:

```python
def test_supervised_exports_match_all(self) -> None:
    exported = set(supervised.__all__)
    actual = {
        name
        for name in dir(supervised)
        if not name.startswith("_") and name in supervised.__all__
    }
    assert exported == actual

def test_no_head_class_leaked_from_tst(self) -> None:
    assert not hasattr(tst, "TSTClassificationHead")
    assert not hasattr(tst, "TSTRegressionHead")
```

See: `tests/integration/test_supervised_integration.py`

### Source-Level Compliance Tests

Read source files and verify code patterns exist (or do not exist):

```python
def test_persistent_workers_condition(self) -> None:
    mixin_file = (
        pathlib.Path(__file__).parents[1]
        / "src" / "chronocratic" / "models" / "convolutional" / "dilated" / "_mixin" / "encoding.py"
    )
    source = mixin_file.read_text()
    assert "persistent_workers=num_workers > 0" in source

def test_no_hasattr_branching(self) -> None:
    assert "hasattr" not in self.source
```

See: `tests/test_mixin.py`

### Loss Equivalence Tests

Verify refactored code produces identical results to original:

```python
def test_loss_equivalence_to_original_auto_tcl(self) -> None:
    from unittest.mock import patch

    torch.manual_seed(42)
    x_emb = torch.randn(2, 10, 32)
    aug_x_emb = torch.randn(2, 10, 32)
    aug_factor = torch.rand(2, 10, 3)

    # Compute original loss manually...
    original_loss = vx_distance + regularization_weight * regularization_loss + ...

    strategy = RIPTrainingStrategy(...)
    with patch(
        "chronocratic.models.convolutional.dilated.autotcl.utils.calculate_regular_consistency",
        return_value=fixed_consistency,
    ):
        new_loss = strategy.compute_loss(
            x_embeddings=x_emb, aug_x_embeddings=aug_x_emb, augmentation_factor=aug_factor
        )

    assert torch.allclose(original_loss, new_loss, atol=1e-6)
```

See: `tests/test_augmentation.py`

## Mocking

**Framework:** `unittest.mock.MagicMock` and `unittest.mock.patch`

**When to mock:**
- Internal randomness: `patch("...calculate_regular_consistency", return_value=fixed_consistency)`
- Backbone methods: `backbone.get_representations.return_value = reps`
- Lazy imports: `patch` at source module level

**When NOT to mock:**
- Actual model training (use real Lightning `Trainer` with minimal data)
- Tensor operations (PyTorch is fast enough for unit tests)
- Config dataclasses (test real instances)
- Augmentation primitives (test with real tensors)

See: `tests/test_augmentation.py`, `tests/unit/test_supervised_package.py`

## Test Data

**Tensor data:** Created inline with known shapes and `torch.manual_seed()` for determinism:

```python
torch.manual_seed(42)
data = torch.randn(2, 100, 3)  # (batch=2, time=100, channels=3)
x = torch.randn(4, 10, 32, requires_grad=True)
padding_masks = torch.ones(2, 10, dtype=torch.bool)
```

**Synthetic datasets:** Minimal `torch.utils.data.Dataset` subclasses for integration tests:

```python
class _DummyTSTDataset(Dataset):
    """Synthetic TST dataset: (X, targets, padding_masks, IDs)."""

    def __init__(
        self, size: int = 20, seq_len: int = 10, feat_dim: int = 2, num_classes: int = 3
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.seq_len, self.feat_dim)
        targets = torch.tensor(idx % self.num_classes, dtype=torch.long)
        padding_masks = torch.ones(self.seq_len, dtype=torch.bool)
        ids = torch.tensor(idx, dtype=torch.long)
        return x, targets, padding_masks, ids
```

See: `tests/integration/test_supervised_integration.py`

**Minimal stubs:** `_DummyBackbone`, `_DummyEncoder`, `_DummyHead`, `_DecompositionEncoder`, `_DummyStrategy`, `_DummyTrainableProducer` for unit tests that need model-like objects without real backbones.

See: `tests/test_mixin.py`, `tests/unit/test_supervised_package.py`, `tests/test_aug_decorators.py`

**Helper functions for test doubles:** Local factory functions create concrete test implementations:

```python
def make_strategy(training_ratio_step: int = 1) -> AugmentationTrainingStrategy:
    """Create a minimal AugmentationTrainingStrategy for tests."""

    class _TestStrategy(AugmentationTrainingStrategy):
        def compute_loss(
            self,
            _x_embeddings: torch.Tensor,
            _aug_x_embeddings: torch.Tensor,
            _augmentation_factor: torch.Tensor,
        ) -> torch.Tensor:
            return torch.tensor(0.0)

    return _TestStrategy(training_ratio_step=training_ratio_step)
```

See: `tests/test_aug_contract.py`

## Coverage

**Requirements:** No coverage threshold is enforced.

**View Coverage:**
```bash
uv run pytest --cov=src/chronocratic/models --cov-report=term-missing
uv run pytest --cov=src/chronocratic/models --cov-report=html
```

**CI coverage:** XML report generated for potential upload.

## Test Types Summary

| Type | Location | Scope |
|------|----------|-------|
| Config unit tests | `tests/test_config.py`, `tests/test_aug_config.py` | Dataclass fields, defaults, mutability isolation |
| Augmentation contract tests | `tests/test_aug_contract.py` | Protocol conformance, ViewSet immutability, Producer patterns |
| Augmentation ABC tests | `tests/test_augmentation.py`, `tests/test_augmentation_base.py` | TrainingStrategy, concrete augmentations |
| Per-model augmentation | `tests/test_augmentation_per_model.py` | Model-specific augmentation modules |
| Producer combinator tests | `tests/test_aug_producers.py` | SingleViewProducer, IndependentPair, RolePair, FullOverlapPair |
| Type covariance tests | `tests/test_aug_covariance.py` | PEP 695 generic variance, Liskov substitution |
| Decorator tests | `tests/test_aug_decorators.py` | Seeded decorator determinism |
| Cross-model tests | `tests/test_aug_cross_model.py` | Producer reuse across models |
| Trainable support tests | `tests/test_aug_trainable_support.py` | Maybe-train helpers, isinstance guards |
| Primitive tests | `tests/test_aug_primitives.py` | Jitter, Scaling, Permutation, ComposeAugmentation |
| Mixin tests | `tests/test_mixin.py` | Hierarchy, dispatch, source compliance |
| Model instantiation | `tests/test_from_config.py` | Config -> model round-trip |
| Per-model producer tests | `tests/test_ts2vec_producer.py`, `test_cost_producer.py`, etc. | Producer integration with specific models |
| Supervised package | `tests/unit/test_supervised_package.py` | Module, adapters, factories (with stubs) |
| Backbone rep_dim | `tests/unit/test_backbone_representation_dim.py` | representation_dim protocol compliance |
| Smoke tests | `tests/test_smoke.py` | Real training, checkpointing, extensibility |
| Integration tests | `tests/integration/test_supervised_integration.py` | Full pipeline: backbone -> factory -> training |

## Ruff Test Configuration

**Rules suppressed in test files (`tests/**/*.py`):**
- `D` (all): No docstring requirements for tests
- `PLR2004`: Magic numbers allowed (test constants are expected)
- `S101`: `assert` statements allowed (pytest assertion rewriting)

---

*Testing analysis: 2026-06-17*
