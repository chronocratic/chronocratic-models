# Augmentation Framework

Polymorphic augmentation producer contract — models accept any augmentation through a unified interface, eliminating enum-based branching.

## Base Types

Protocols, typed view-sets, and abstract base classes for the augmentation system.

```{eval-rst}
.. automodule:: chronocratic.models.augmentation.base
   :members:
   :show-inheritance:
```

## Producers

Concrete augmentation producers that generate single views, view pairs, and aligned pairs.

```{eval-rst}
.. automodule:: chronocratic.models.augmentation.producers
   :members:
   :show-inheritance:
```

## Primitives

Individual augmentation operations (Jitter, Scaling, Permutation) and composition.

```{eval-rst}
.. automodule:: chronocratic.models.augmentation.primitives
   :members:
   :show-inheritance:
```

## Decorators

Wrappers that add cross-cutting behavior (e.g., deterministic seeding) to augmentations.

```{eval-rst}
.. automodule:: chronocratic.models.augmentation.decorators
   :members:
   :show-inheritance:
```

## Trainable Support

Utilities for augmentations with learnable parameters.

```{eval-rst}
.. automodule:: chronocratic.models.augmentation.trainable_support
   :members:
   :show-inheritance:
```
