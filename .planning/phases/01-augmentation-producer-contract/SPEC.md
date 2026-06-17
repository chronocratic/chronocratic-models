# Phase 01 — Augmentation Producer Contract

**Status:** Specified (not yet planned)
**Created:** 2026-06-12
**Branch:** `augmentaiton-contract-refactor`
**Related issue:** skafai/tsmodels#17

---

## 1. Summary

Replace the current augmentation contract — every augmentation returns a loose
`TrainingViews` bag of `(views: tuple[Tensor, ...], metadata: dict[str, Any])` —
with a typed, capability-segregated contract that decouples augmentation
*primitives* from model-specific *view assembly*.

The new contract lets any augmentation primitive be reused across any model
without rewriting it per model, eliminates the `metadata` dict and the
`DualAugmentation` wrapper class, and removes all runtime branching on
augmentation type inside model code.

The redesign turns an **N×M** coupling (N augmentations × M models, each pairing
needing bespoke glue) into **N+M** (N primitives + a small fixed set of
assembler/adapter producers).

---

## 2. Problem Statement

### 2.1 Current contract

```python
@dataclass
class TrainingViews:
    views: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]

class AugmentationMethod(ABC):
    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs) -> TrainingViews: ...
```

Every augmentation — whether a pure pointwise transform (jitter) or a
coordinated multi-view generator (crop-shift) — implements the same `augment`
method and returns the same untyped container.

### 2.2 Consumer reality

Four models consume the contract, each pulling a different shape out of the same
bag:

| Model    | Views wanted          | Metadata      | How it consumes today                                   |
|----------|-----------------------|---------------|---------------------------------------------------------|
| TS2Vec   | 2 *coordinated* crops | `crop_length` | one `augment()`; slices embeddings to the overlap       |
| CoST     | 2 *independent* views | none          | calls `augment()` **twice**, takes `.views[0]` each      |
| AutoTCL  | 1 view                | none          | one `augment()`, `.views[0]`; plus a trainable path     |
| TS-TCC   | 2 *different* augs     | none          | needs a whole `DualAugmentation` ABC + wrapper subclass  |

### 2.3 Root causes

1. **Untyped bag.** `views: tuple[...]` + `metadata: dict[str, Any]`. The
   contract lives in docstrings, not types. Plugging the wrong augmentation into
   the wrong model yields a silent shape error or `KeyError('crop_length')`. No
   static safety.

2. **Two responsibilities under one ABC.** Pointwise transforms
   (`Tensor → Tensor`: jitter, scaling, permutation) and view-assembly
   (coordinated crops, weak/strong pairs, independent draws) share one
   `AugmentationMethod.augment`. They are different *kinds* of thing.

3. **View-assembly logic is the real duplication.** "Call twice for query/key"
   (CoST, inline) and "use aug A for view 1, aug B for view 2" (TS-TCC, via the
   `DualAugmentation` ABC + `TSTCCDualAugmentation` class) are *generic
   combinators* wearing model-specific costumes. This is the "we wrote another
   class just for that" pain.

4. **N×M coupling.** Because no adapter sits between augmentation output and
   model need, an augmentation written for one model cannot drop into another.
   Reuse requires rewriting.

---

## 3. Goals & Non-Goals

### 3.1 Goals

- **G1.** Decouple augmentation primitives from models: a primitive is added once
  in a model-agnostic space and is reusable across all models (N+M, not N×M).
- **G2.** Type-safe contract: replace the untyped bag with typed result objects;
  mismatches are caught by the type checker at wiring time, not at runtime.
- **G3.** Zero runtime branching on augmentation type inside any model body.
- **G4.** Delete the `DualAugmentation` ABC and `TSTCCDualAugmentation` class;
  express the weak/strong pair as a generic combinator.
- **G5.** Delete the `metadata: dict[str, Any]` channel; parameters a model needs
  (e.g. TS2Vec's crop overlap) flow through typed result fields.
- **G6.** Preserve exact training behavior of all four models (no numerical
  regression; existing tests pass).

### 3.2 Non-Goals

- Not changing model architectures, losses, or hyperparameters.
- Not touching models that do **not** consume this contract (MCL, Series2Vec,
  TimeVAE, TimeNet, TST). Scope is exactly TS2Vec, CoST, AutoTCL, TS-TCC.
- Not adding a name→producer registry / config-string selection (explicitly
  deferred — see §9, YAGNI).
- Not introducing a functional/HOF producer style (deferred; keep class-based for
  the trainable case).

---

## 4. Design

### 4.1 The core reframe

`crop_length` is **not** a crop-specific field. It is the **aligned region** —
the span over which two views line up for the contrastive loss. Every two-view
producer can answer this question:

- crop-shift answers "the sampled overlap";
- any full-length augmentation (jitter, scaling) answers "the full length `T`";
- identity answers `T`.

Once `overlap_length` is a *universal question* rather than a *crop property*, the
model needs only one branchless code path. The augmentation that has no real crop
supplies `overlap_length = T`, making the model's slice a harmless no-op.

### 4.2 Layer 1 — Primitives

A primitive is the smallest reusable transform: `Tensor → Tensor`. This is the
expandable set a contributor adds, with zero model knowledge.

```python
class Augmentation(Protocol):
    """A pointwise/sequence transform producing one view of the input."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
```

Concrete primitives (migrated from current single-view `AugmentationMethod`s):
`Jitter`, `Scaling`, `Permutation`, `ComposeAugmentation`,
`CosTRandomFunctionAugmentation`.

### 4.3 Layer 2 — Typed view results (ViewSets)

```python
@dataclass(frozen=True)
class SingleView:
    view: torch.Tensor

@dataclass(frozen=True)
class ViewPair:
    first: torch.Tensor
    second: torch.Tensor

@dataclass(frozen=True)
class AlignedPair(ViewPair):       # is-a ViewPair (Liskov)
    overlap_length: int
```

`AlignedPair` subclasses `ViewPair` so a producer of `AlignedPair` satisfies any
slot expecting `ViewPair`. "Independent" (CoST), "weak/strong roles" (TS-TCC),
and "coordinated crops" (TS2Vec) are all *a pair*; the **combinator** that builds
it carries the semantics, the result type stays minimal.

`metadata: dict[str, Any]` is **deleted**. The only parameter any current model
needs (`overlap_length`) is a typed field on `AlignedPair`.

### 4.4 Layer 3 — Producers (the assemblers / adapters)

The object injected into a model. It assembles a ViewSet from primitives.

```python
V = TypeVar("V", covariant=True)

class AugmentationProducer(Protocol[V]):
    """Assembles the view set a model's loss requires from a batch."""
    def produce(self, x: torch.Tensor) -> V: ...
```

`V` is **covariant** (it appears only in return position), so
`AugmentationProducer[AlignedPair]` is a subtype of
`AugmentationProducer[ViewPair]`. This is what lets a rich crop producer drop
into a model that only needs a plain pair.

Concrete producers:

| Producer                  | Returns       | Role                                                            |
|---------------------------|---------------|----------------------------------------------------------------|
| `SingleViewProducer(aug)` | `SingleView`  | one view from one primitive (Null-Object / identity adapter)   |
| `IndependentPair(aug)`    | `ViewPair`    | applies one primitive twice → two independent draws (CoST)     |
| `RolePair(first, second)` | `ViewPair`    | applies two primitives → two role-named views (TS-TCC)         |
| `FullOverlapPair(aug)`    | `AlignedPair` | two views, `overlap_length = T` — feeds any primitive to TS2Vec |
| `CropShiftProducer(...)`  | `AlignedPair` | coordinated random crops + real sampled `overlap_length`        |

Sketches:

```python
class FullOverlapPair(AugmentationProducer[AlignedPair]):
    def __init__(self, aug: Augmentation) -> None:
        self._aug = aug
    def produce(self, x: torch.Tensor) -> AlignedPair:
        return AlignedPair(self._aug(x), self._aug(x), overlap_length=x.size(1))

class RolePair(AugmentationProducer[ViewPair]):
    def __init__(self, first: Augmentation, second: Augmentation) -> None:
        self._first, self._second = first, second
    def produce(self, x: torch.Tensor) -> ViewPair:
        return ViewPair(self._first(x), self._second(x))
```

### 4.5 Capability — Trainable producers (AutoTCL)

A trainable augmentation is an `nn.Module` producing a learned single view plus a
training lifecycle. It is a *capability extension* of a `SingleView` producer
(Interface Segregation): models that don't train an augmentation never depend on
it.

```python
class TrainableAugmentationProducer(nn.Module, ABC):
    def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
        super().__init__()
        self._training_strategy = training_strategy

    @abstractmethod
    def produce(self, x: torch.Tensor) -> SingleView: ...

    @abstractmethod
    def train_step(self, x: torch.Tensor, encoder: nn.Module, batch_idx: int) -> torch.Tensor | None: ...

    def configure_optimizer(self, lr: float) -> torch.optim.AdamW:
        return torch.optim.AdamW(self.parameters(), lr=lr)

    def should_train_augmentation(self, epoch: int, batch_idx: int) -> bool:
        return self._training_strategy.should_train(epoch, batch_idx)
```

`AugmentationTrainingStrategy` (loss strategy) is retained unchanged.
`TrainableAugmentation` (old ABC) is renamed/reshaped into
`TrainableAugmentationProducer`.

#### 4.5.1 Why the trainable capability is nominal, not a Protocol

`AugmentationProducer[V]` is a **Protocol** (structural typing) — used only for
*static* conformance of stateless producers. Nothing runtime-checks it, so it
needs no `@runtime_checkable`.

`TrainableAugmentationProducer` is a **nominal ABC** subclassing `nn.Module`. The
runtime gate (does this producer train?) is a plain nominal
`isinstance(x, TrainableAugmentationProducer)` against a real class — it matches
only explicit subclasses, checks signatures via the ABC, and cannot false-match.

This split is deliberate, not inconsistent:

- A trainable augmentation **must** be an `nn.Module` anyway (learnable weights),
  so nominal inheritance costs nothing.
- `@runtime_checkable` Protocols only check method **names**, not signatures or
  `nn.Module`-ness — a stateless producer with a stray `train_step` would
  false-match. Rejected.
- `TrainableAugmentationProducer` *structurally* satisfies
  `AugmentationProducer[SingleView]` (it has `produce(x) -> SingleView`), so it
  type-checks in any `SingleView` slot **and** gives a rock-solid runtime
  identity. Structural for the static slot; nominal for the runtime gate — each
  mechanism used where it is strong.

#### 4.5.2 Centralized trainable-support helpers

AutoTCL accepts **both** trainable and static single-view producers (static is a
valid ablation mode). The trainable path is gated, but the `isinstance` gate is
**not** duplicated per model — it lives in exactly one place. Any future model
that supports a learned augmentation reuses these helpers (Null Object: a
non-trainable producer yields `None`, so the caller's path is branchless).

```python
# augmentation/trainable_support.py
def maybe_train_augmentation(
    augmentation: AugmentationProducer[Any],
    *, x: torch.Tensor, encoder: nn.Module, epoch: int, batch_idx: int,
) -> torch.Tensor | None:
    """Run one aug-network training step iff the producer is trainable and due."""
    if not isinstance(augmentation, TrainableAugmentationProducer):
        return None
    if not augmentation.should_train_augmentation(epoch=epoch, batch_idx=batch_idx):
        return None
    return augmentation.train_step(x=x, encoder=encoder, batch_idx=batch_idx)

def maybe_configure_augmentation_optimizer(
    augmentation: AugmentationProducer[Any], *, lr: float,
) -> torch.optim.Optimizer | None:
    """Build the aug-network optimizer iff the producer is trainable, else None."""
    if not isinstance(augmentation, TrainableAugmentationProducer):
        return None
    return augmentation.configure_optimizer(lr=lr)
```

The two `isinstance` checks (training step, optimizer setup) are the **only**
augmentation-type branches permitted outside a model — and they live here, once.

### 4.6 Decorator — cross-cutting concerns (scoped)

A generic decorator adds orthogonal behavior (determinism, logging,
normalization) without touching producers or models.

```python
class Seeded(AugmentationProducer[V], Generic[V]):
    def __init__(self, inner: AugmentationProducer[V], seed: int) -> None:
        self._inner, self._seed = inner, seed
    def produce(self, x: torch.Tensor) -> V:
        with torch.random.fork_rng():
            torch.manual_seed(self._seed)
            return self._inner.produce(x)
```

**Constraint (locked decision):** decorators are `Generic[V]` and apply to
**stateless producers only**. They must **not** wrap a
`TrainableAugmentationProducer`, because a `produce`-only decorator drops the
`train_step` / `nn.Module` capability and breaks the `isinstance` gate. A
trainable augmentation that needs reproducibility **seeds itself** inside its own
`produce` / `train_step`. No capability-forwarding decorator is built in this
phase.

### 4.7 Model bindings

Each model declares the narrowest capability it needs and runs one branchless
body. The default preserves current behavior.

**TS2Vec** — needs alignment:
```python
class TS2Vec:
    def __init__(self, augmentation: AugmentationProducer[AlignedPair] | None = None) -> None:
        self._augmentation = augmentation if augmentation is not None else CropShiftProducer()

    def _encode_augmented_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self._augmentation.produce(x)
        encoder = ...
        e1 = encoder(pair.first)[:, -pair.overlap_length:]
        e2 = encoder(pair.second)[:, :pair.overlap_length]
        return e1, e2
```
To use a non-crop augmentation: `TS2Vec(augmentation=FullOverlapPair(Jitter(...)))`.
No model change, no branch (`overlap_length = T` → slice is a no-op).

**CoST** — needs an independent pair:
```python
class CoST:
    def __init__(self, augmentation: AugmentationProducer[ViewPair] | None = None) -> None:
        self._augmentation = (
            augmentation if augmentation is not None
            else IndependentPair(CosTRandomFunctionAugmentation())
        )

    def _contrastive(self, x: torch.Tensor) -> ...:
        pair = self._augmentation.produce(x)
        query, key = pair.first, pair.second
```
Replaces the current two inline `augment(x).views[0]` calls.

**AutoTCL** — needs a single view; accepts **both** trainable and static
producers. The trainable path goes through the centralized helpers (§4.5.2); the
model body holds **no** `isinstance`:
```python
class AutoTCL:
    def __init__(self, augmentation: AugmentationProducer[SingleView] | None = None) -> None:
        self._augmentation = augmentation if augmentation is not None else AutoTCLNeuralNetworkAugmentation(...)

    def configure_optimizers(self) -> ...:
        encoder_opt = ...
        aug_opt = maybe_configure_augmentation_optimizer(self._augmentation, lr=self._meta_learning_rate)
        return [encoder_opt, *( [aug_opt] if aug_opt is not None else [] )]

    def training_step(self, batch, batch_idx) -> ...:
        x = ...
        aug_loss = maybe_train_augmentation(
            self._augmentation, x=x, encoder=self._encoder,
            epoch=self.current_epoch, batch_idx=batch_idx,
        )
        if aug_loss is not None:
            ...   # optimize aug network
        view = self._augmentation.produce(x).view
        ...       # uniform encoder training (both trainable and static augs)
```
A static augmentation (e.g. `SingleViewProducer(Jitter(...))`) drops in: the
helpers return `None`, the aug-network steps are skipped, encoder training runs
identically.

**TS-TCC** — needs a role pair; `DualAugmentation` deleted:
```python
class TSTCC:
    def __init__(self, augmentation: AugmentationProducer[ViewPair] | None = None) -> None:
        self._augmentation = augmentation if augmentation is not None else _default_tstcc_pair()

    def _pretrain_step(self, data: torch.Tensor) -> ...:
        pair = self._augmentation.produce(data)
        aug1, aug2 = pair.first, pair.second

def _default_tstcc_pair() -> AugmentationProducer[ViewPair]:
    return RolePair(
        first=Scaling(ScalingParameters(sigma=1.1, mean=2.0, per_sample=True, channel_dim=1)),
        second=ComposeAugmentation([
            Permutation(PermutationParameters(max_segments=5, time_dim=-1)),
            Jitter(JitterParameters(sigma=0.8)),
        ]),
    )
```

### 4.8 The mental model

```
              picks producer (1 line at construction)
   caller ──────────────────────────────────────────┐
                                                     ▼
   model  ──depends on──►  AugmentationProducer[ItsViewSet]   ◄── one typed contract
     │                                  ▲
     │ one branchless body              │ satisfied by (Liskov, covariance)
     └─ reads .first/.second[/.overlap_length]
                                        │
        ┌────────────────────┬──────────┴───────────────┐
   CropShiftProducer   FullOverlapPair(Jitter)   RolePair(weak, strong)  ...
```

The model sees only the contract. Each producer answers it polymorphically.
"Handle multiple augmentations" logic is not *in* the model — it is distributed
across producers, written once, and consumed through one polymorphic call.

### 4.9 File layout

The N+M split must hold at the **import layer**, not just conceptually. General
primitives and generic combinators live in a shared, model-agnostic space;
model-specific producers and default wiring stay colocated with their model. No
model imports augmentation code from another model.

```
augmentation/
  base.py              # Augmentation Protocol, AugmentationProducer Protocol,
                       #   SingleView / ViewPair / AlignedPair, TrainableAugmentationProducer,
                       #   AugmentationTrainingStrategy (retained)
  primitives.py        # NEW — shared agnostic primitives:
                       #   Jitter, Scaling, Permutation, ComposeAugmentation
  producers.py         # NEW — shared agnostic combinators:
                       #   SingleViewProducer, IndependentPair, RolePair, FullOverlapPair
  decorators.py        # NEW — Seeded[V] (stateless-only)
  trainable_support.py # NEW — maybe_train_augmentation, maybe_configure_augmentation_optimizer

convolutional/dilated/ts2vec/augmentation.py   # CropShiftProducer (model-specific producer)
convolutional/dilated/cost/augmentation.py     # CosTRandomFunctionAugmentation (model-specific primitive)
convolutional/dilated/autotcl/augmentation/    # AutoTCLNeuralNetworkAugmentation (trainable producer)
convolutional/standard/tstcc/augmentations.py  # _default_tstcc_pair() (per-model wiring; imports primitives + RolePair)
```

**Rules:**

- `primitives.py` and `producers.py` are model-agnostic — they import nothing
  model-specific (enforces G1). A new primitive added here is instantly reusable
  by every model.
- A model-specific producer (CropShift) or primitive (CoST random function)
  stays in that model's aug file.
- Default wiring functions (`_default_tstcc_pair`) live in the consuming model's
  aug file and import primitives/combinators from the shared modules.
- `base.py` keeps only the contract types + retained strategy ABC; it grows no
  concrete primitives.

**Primitive extraction is in scope for this phase.** `Jitter`, `Scaling`,
`Permutation`, and `ComposeAugmentation` move out of `tstcc/augmentations.py`
(where they currently live) into `augmentation/primitives.py`, reshaped to the
`Augmentation` (`__call__: Tensor → Tensor`) contract. `tstcc/augmentations.py`
then imports them. This prevents a cross-model import (`cost`/`ts2vec` reusing
`Jitter` from `tstcc`) and is required for N+M to hold at the import layer.

---

## 5. Deletions & Renames

| Symbol                              | Action                                                            |
|-------------------------------------|------------------------------------------------------------------|
| `TrainingViews`                     | **Delete** (replaced by `SingleView` / `ViewPair` / `AlignedPair`)|
| `TrainingViews.metadata`            | **Delete** (typed fields replace the dict)                       |
| `AugmentationMethod` (ABC)          | **Replace** with `Augmentation` Protocol (primitive) + `AugmentationProducer` Protocol (assembler) |
| `DualAugmentation` (ABC)            | **Delete** (replaced by `RolePair` combinator)                  |
| `TSTCCDualAugmentation`             | **Delete** (replaced by `_default_tstcc_pair()` builder)        |
| `TrainableAugmentation` (ABC)       | **Rename/reshape** to `TrainableAugmentationProducer`           |
| `CropShiftAugmentation`             | **Reshape** to `CropShiftProducer` (`produce → AlignedPair`)    |
| `CosTRandomFunctionAugmentation`    | **Reshape** to primitive (`__call__(x) -> Tensor`)              |
| `Jitter`, `Scaling`, `Permutation`  | **Reshape** to primitives (`__call__(x) -> Tensor`)             |
| `ComposeAugmentation`               | **Reshape** to `list[Augmentation]`, `__call__` chains          |
| `AutoTCLNeuralNetworkAugmentation`  | **Reshape** to `TrainableAugmentationProducer` (`produce → SingleView`) |

New symbols: `Augmentation`, `AugmentationProducer`, `SingleView`, `ViewPair`,
`AlignedPair`, `SingleViewProducer`, `IndependentPair`, `RolePair`,
`FullOverlapPair`, `CropShiftProducer`, `TrainableAugmentationProducer`,
`Seeded`, `maybe_train_augmentation`, `maybe_configure_augmentation_optimizer`.

**File moves (see §4.9):**

| From                          | To                          | What                                              |
|-------------------------------|-----------------------------|---------------------------------------------------|
| `tstcc/augmentations.py`      | `augmentation/primitives.py`| `Jitter`, `Scaling`, `Permutation`, `ComposeAugmentation` (reshaped to `Augmentation`) |
| (new)                         | `augmentation/producers.py` | `SingleViewProducer`, `IndependentPair`, `RolePair`, `FullOverlapPair` |
| (new)                         | `augmentation/decorators.py`| `Seeded[V]`                                       |
| (new)                         | `augmentation/trainable_support.py` | the two `maybe_*` helpers                  |
| `augmentation/dual.py`        | — (delete)                  | `DualAugmentation` removed entirely               |
| `tstcc/augmentations.py`      | stays                       | `_default_tstcc_pair()` (imports from shared modules) |

---

## 6. Success Criteria

1. `TrainingViews`, `DualAugmentation`, `TSTCCDualAugmentation`, and the
   `metadata` dict no longer exist in the codebase.
2. `Augmentation`, `AugmentationProducer[V]`, the three ViewSet types, and the
   five concrete producers exist with full type hints and Google-style docstrings.
3. No model body contains `isinstance(... augmentation ...)` branching **except**
   AutoTCL's single trainable gate on `TrainableAugmentationProducer`.
4. `FullOverlapPair(Jitter(...))` can be injected into `TS2Vec` and trains without
   any TS2Vec code change (test demonstrates it).
5. `CropShiftProducer(...)` can be injected into a `ViewPair`-only model (e.g.
   CoST) and type-checks via covariance (test or `ty` demonstrates it).
6. `ty check src/` passes with zero errors.
7. All existing tests pass; numerical training behavior of all four models is
   unchanged (smoke test / seeded equivalence where feasible).
8. `Seeded` is `Generic[V]`; a test asserts it is **not** applied to a trainable
   producer (documented constraint).
9. Shared primitives live in `augmentation/primitives.py` and generic combinators
   in `augmentation/producers.py`; both import nothing model-specific. No model
   imports augmentation code from another model (e.g. no `tstcc` → `Jitter`
   import elsewhere). `_default_tstcc_pair()` lives in `tstcc/augmentations.py`.

---

## 7. Testing Strategy

- **Primitive tests:** each primitive maps `Tensor → Tensor` with expected shape.
- **Producer tests:** each producer returns the correct ViewSet type and shapes;
  `IndependentPair` yields two independent draws; `RolePair` applies both augs;
  `FullOverlapPair` sets `overlap_length == x.size(1)`; `CropShiftProducer`
  sets `0 < overlap_length <= T`.
- **Liskov / covariance test:** a function typed for `AugmentationProducer[ViewPair]`
  accepts a `CropShiftProducer`; verified by `ty` and at runtime.
- **Cross-model reuse test:** inject `FullOverlapPair(Jitter(...))` into TS2Vec;
  assert a forward/loss step runs and the slice is a no-op at full overlap.
- **Decorator test:** `Seeded` reproduces identical output for a fixed seed on a
  stateless producer; assert/lint that it is never constructed over a trainable
  producer.
- **Model smoke tests:** existing per-model training smoke tests pass unchanged.

---

## 8. Philosophy

This section explains *why* the design is shaped this way, for both the
implementing agent and the maintainer. Read it before changing the contract.

### 8.1 Separate what changes for different reasons

The original `AugmentationMethod` conflated two responsibilities that change for
different reasons: **what a transform does to a tensor** (a research/data concern
— jitter strength, permutation segments) and **how many views a model's loss
needs and how they relate** (a model/loss concern — coordinated crops, weak/strong
roles, independent draws). When two reasons-to-change live in one type, every new
augmentation risks disturbing every model and vice versa — the N×M trap. We split
them: **primitives** (`Augmentation`) own the transform; **producers**
(`AugmentationProducer`) own the assembly. A contributor adding an augmentation
touches only a primitive. A model author needing a new view arrangement reuses a
producer. Neither edits the other's space.

### 8.2 Make the contract the type, not the docstring

The old `metadata: dict[str, Any]` made the contract invisible to the compiler:
correctness depended on every consumer remembering which keys exist. The redesign
encodes the contract in types — `SingleView`, `ViewPair`, `AlignedPair` — so a
mismatch is a type error at the construction site, not a `KeyError` at runtime in
epoch 3. The type *is* the documentation, and it is checked.

### 8.3 Universalize the special case instead of branching on it

The instinct when one model needs an extra field (`crop_length`) is to add it
everywhere with a default, or to branch on "does this augmentation have a crop?"
Both spread one model's concern across the whole system. Instead we **reframed
the field as a universal question** every pair-producer can answer
(`overlap_length` = the aligned region; full length when there is no crop). The
model then has one code path; the augmentation with no crop answers `T` and the
model's slice degrades to a no-op. The variability moves out of conditionals and
into polymorphic answers.

### 8.4 Depend on capabilities, resolve at the boundary

A model declares the narrowest capability it needs as a type
(`AugmentationProducer[AlignedPair]` vs `[ViewPair]` vs `[SingleView]`). Which
concrete producer satisfies it is decided **at the construction boundary**
(dependency injection), not by inspecting types inside the loss. Subtyping
(`AlignedPair` is-a `ViewPair`) and return-position covariance make rich
producers usable in narrow slots without adapters; a one-line adapter
(`FullOverlapPair`) bridges the rare gap in the other direction. The model body
stays closed to modification and open to new producers.

### 8.5 Keep extension scoped (YAGNI)

We deliberately *did not* add a name→producer registry, a functional/HOF producer
style, a two-phase `plan()/produce()` protocol, or capability-forwarding
decorators. Each solves a problem we do not yet have. The locked scope —
core contract + a stateless-only `Generic[V]` decorator — is the smallest design
that delivers full decoupling and type safety. The deferred ideas are recorded in
§9 so the door stays open without paying for it now.

---

## 9. Design Patterns & Best Practices Used

A precise inventory, so the implementing agent applies the intended pattern and
the maintainer can recognize it.

| Pattern / principle                         | Where it lives                                              | What it buys                                                                 |
|---------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------------|
| **Strategy**                                | `AugmentationProducer` injected into each model             | swap the view-assembly algorithm without touching the model                  |
| **Adapter**                                 | `FullOverlapPair`, `IndependentPair`, `RolePair`            | reshape a primitive (or pair of primitives) into the ViewSet a model expects |
| **Null Object**                             | `FullOverlapPair` answering `overlap_length = T`; `SingleViewProducer` | the "no real crop" / "no assembly" case is a trivial object, not an `if`      |
| **Value / Parameter Object**                | `SingleView`, `ViewPair`, `AlignedPair` (frozen dataclasses)| typed, immutable results replace the untyped `metadata` dict                 |
| **Interface Segregation (ISP)**             | `AugmentationProducer` vs `TrainableAugmentationProducer`   | models depend only on the capability they use; non-trainable models stay clean |
| **Liskov Substitution + return covariance** | `AlignedPair <: ViewPair`; `Producer[V]` covariant in `V`   | a rich producer fits a narrow slot with no adapter and no branch             |
| **Decorator**                               | `Seeded[V]` (stateless producers only)                      | orthogonal concerns (determinism/logging) without modifying producers        |
| **Dependency Injection**                    | `augmentation=...` constructor param on every model         | selection happens at the boundary; defaults preserve current behavior        |
| **Factory (builder function)**              | `_default_tstcc_pair()`                                     | encapsulates the default weak/strong construction; replaces a wrapper class  |
| **Composite**                               | `ComposeAugmentation(list[Augmentation])`                   | chain primitives as a single primitive                                       |
| **Protocol-based structural typing**        | `Augmentation`, `AugmentationProducer`                      | primitives/producers conform by shape, not inheritance — easy extension      |

**Best practices:**

- Full type hints (incl. return types) and Google-style docstrings on all new
  symbols (project rule).
- Keyword-argument call sites for producers and primitives.
- Functional-first primitives (`__call__: Tensor → Tensor`, pure where possible);
  class-based only where state is required (trainable producer).
- Frozen dataclasses for results (immutability, value semantics).
- `from __future__ import annotations` only where needed for forward refs /
  circular-import avoidance, not blanket.
- No runtime type-branching in model bodies except the single documented
  trainable gate.

---

## 10. Deferred (out of scope, recorded for later)

- **Registry + self-registration** (name→producer) for config-string selection.
  Adopt only when experiment configs must pick augmentations by string.
- **Functional/HOF producer style** alongside classes. Matches the functional-first
  guideline but complicates the trainable case; revisit if the producer set grows.
- **Two-phase `plan()/produce()`** for callers that must know augmentation
  parameters *before* applying. Not needed while `AlignedPair` carries the only
  required parameter.
- **Capability-forwarding decorators** (wrap trainable producers). Build only if a
  cross-cutting concern must apply to the trainable augmentation; until then it
  seeds itself.
- **One generic `Views[Meta]` container** instead of the ViewSet hierarchy.
  Rejected: reintroduces index access (`tensors[0]`) and loses named fields.
