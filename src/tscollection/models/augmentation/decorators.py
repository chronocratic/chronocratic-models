"""Augmentation producer decorators.

Provides decorators that wrap :class:`AugmentationProducer` instances to add
cross-cutting capabilities (e.g., deterministic seeding) without modifying
the producer's own code.

Exported symbols:
    - ``Seeded``: Deterministic wrapper for stateless producers.
"""

from __future__ import annotations

__all__ = [
    'Seeded',
]

import torch

from tscollection.models.augmentation.base import (
    AugmentationProducer,
    TrainableAugmentationProducer,
)


class Seeded[V]:
    """Deterministic decorator wrapping a stateless AugmentationProducer.

    Uses ``torch.random.fork_rng()`` and ``torch.manual_seed()`` so that
    inner randomness is isolated from the outer process random state.
    Repeated calls with the same seed produce identical output tensors.

    Constraint: must NOT wrap :class:`TrainableAugmentationProducer`.
    Trainable producers have their own parameterised state; seeding at the
    producer level is not meaningful and may hide bugs.

    Args:
        inner: A stateless :class:`AugmentationProducer` to wrap.
        seed: Fixed integer seed applied before every ``produce()`` call.
    """

    def __init__(self, *, inner: AugmentationProducer[V], seed: int) -> None:
        if isinstance(inner, TrainableAugmentationProducer):
            msg = "Seeded cannot wrap TrainableAugmentationProducer. Stateless producers only."
            raise TypeError(msg)
        self._inner = inner
        self._seed = seed

    def produce(self, x: torch.Tensor) -> V:
        """Produce a view set with deterministic randomness.

        Isolates the inner producer's random state from the outer process
        context using ``torch.random.fork_rng()``.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            The typed view set produced by the inner producer,
            generated under a reproducible random seed.
        """
        with torch.random.fork_rng():
            torch.manual_seed(self._seed)
            return self._inner.produce(x)
