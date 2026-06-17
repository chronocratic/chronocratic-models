# TODO: Augmentation Architecture Refactor

> **Authoritative design doc:** `/Users/skaf/.claude/plans/but-look-some-staged-ripple.md`
> **Chosen design:** Option 3b — 3-layer ABC hierarchy, `TrainableAugmentation` composes `AugmentationTrainingStrategy` internally.
> **Goal:** users add new augmentation methods + training algorithms by subclassing, with zero library modification. Eliminate enum-based branching in model code.

---

## Tasks

### Phase 1 — New abstractions

- [ ] Add default no-op `train_step` and `configure_optimizer` to `AugmentationMethod` base in `src/tscollection/models/augmentation/strategies.py`
- [ ] Add `TrainableAugmentation(AugmentationMethod, nn.Module)` ABC with `training_strategy` constructor arg, abstract `forward`, abstract `configure_optimizer`, default `train_step` skeleton
- [ ] Add `AugmentationTrainingStrategy` ABC with abstract `compute_loss` and default `should_train`

### Phase 2 — Concrete strategies (extract from AutoTCL)

- [ ] Add `RIPTrainingStrategy(AugmentationTrainingStrategy)` — port logic from `autotcl/model.py:172-193`; constructor takes `consistency_weight`, `regularization_weight`, `regularization_threshold`, `every_n_epochs=1`
- [ ] Add `AdversarialTrainingStrategy(AugmentationTrainingStrategy)` — port logic from `autotcl/model.py:195-199`
- [ ] Refactor `AutoTCLNeuralNetworkAugmentation` to inherit `TrainableAugmentation`; take `training_strategy` + `meta_learning_rate` in constructor; implement `configure_optimizer` (returns AdamW over `self.parameters()`); implement `train_step` per §4.4 of plan; delegate `forward` to wrapped encoder

### Phase 3 — Remove closed-registry machinery

- [ ] Delete `src/tscollection/models/augmentation/enums.py`
- [ ] Delete `src/tscollection/models/augmentation/factories.py`
- [ ] Update `src/tscollection/models/augmentation/__init__.py` exports — expose: `AugmentationMethod`, `TrainableAugmentation`, `AugmentationTrainingStrategy`, `CropShiftAugmentation`, `CosTRandomFunctionAugmentation`, `AutoTCLNeuralNetworkAugmentation`, `RIPTrainingStrategy`, `AdversarialTrainingStrategy`

### Phase 4 — Model refactors

- [ ] **TS2Vec** (`src/tscollection/models/ts2vec/model.py`) — replace `augmentation_mode` + nested dicts with `augmentation: AugmentationMethod`; add `save_hyperparameters(ignore=['augmentation'])`; drop `_init_augmentation_method`, `_init_augmentation_mode_params`; rename `self._augmentation_method` → `self._augmentation`
- [ ] **CoST** (`src/tscollection/models/cost/model.py`) — same shape change; update both `augment()` call sites in training_step + validation_step
- [ ] **AutoTCL** (`src/tscollection/models/autotcl/model.py`) — same constructor change PLUS:
  - [ ] Delete `_exec_training_step_function`, `_training_step_function_default`, `_training_step_function_neural_network_augmentation`
  - [ ] Delete `_calculate_augmentation_loss_neural_network_augmentation`, `_augmentation_loss_network_augmentation_relevant_information_principle`, `_augmentation_loss_neural_network_augmentation_adversarial`
  - [ ] Delete `_init_augmentation_mode_params`, `_configure_optimizers_neural_network_augmentation`, `_configure_optimizers_default`, `_init_augmentation_method`
  - [ ] Replace `configure_optimizers` with simple list-builder (encoder optimizer + aug's `configure_optimizer()` if non-None)
  - [ ] Replace `training_step` with uniform polymorphic dispatch (see plan §4.4)
  - [ ] Decide fate of `_eval_mutual_information` — grep for callers; if dead, delete; if used externally, move to `AutoTCLNeuralNetworkAugmentation`

### Phase 5 — Verification

- [ ] `uv run ruff check src/`
- [ ] `uv run ruff format --check src/`
- [ ] `uv run ty check src/` (or mypy)
- [ ] Smoke test: TS2Vec + `CropShiftAugmentation()` — 1 batch, 5 steps, loss finite & decreasing
- [ ] Smoke test: CoST + `CosTRandomFunctionAugmentation(sigma=0.5)` — same
- [ ] Smoke test: AutoTCL + `AutoTCLNeuralNetworkAugmentation(training_strategy=RIPTrainingStrategy(...))` — same
- [ ] Smoke test: AutoTCL + `AutoTCLNeuralNetworkAugmentation(training_strategy=AdversarialTrainingStrategy())` — same
- [ ] Custom-aug extension test: trivial identity `AugmentationMethod` subclass injected into TS2Vec
- [ ] Custom-strategy extension test: trivial MSE `AugmentationTrainingStrategy` subclass paired with `AutoTCLNeuralNetworkAugmentation`
- [ ] Checkpoint round-trip: save → reload (with `augmentation=` re-supplied) → encoder weights match
- [ ] Final grep sweep: `augmentation_mode`, `augmentation_method_params`, `*Factory`, `*Mode` enum names — zero hits in `src/`
