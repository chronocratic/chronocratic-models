"""Abstract base classes for augmentation strategies.

This module defines the shared augmentation hierarchy extracted from the
monolithic ``strategies.py``. It contains only abstract base classes and the
``TrainingViews`` dataclass (~200 lines). Concrete implementations live in
per-model augmentation files.

Exported symbols:
    - ``AugmentationMethod``: Abstract transform interface.
    - ``AugmentationTrainingStrategy``: Abstract training-loss interface.
    - ``TrainableAugmentation``: Abstract trainable augmentation (nn.Module).
    - ``TrainingViews``: Dataclass for augmentation output views + metadata.
"""

from __future__ import annotations

__all__ = [
    'AlignedPair',
    'Augmentation',
    'AugmentationMethod',
    'AugmentationProducer',
    'AugmentationTrainingStrategy',
    'SingleView',
    'TrainableAugmentation',
    'TrainableAugmentationProducer',
    'TrainingViews',
    'ViewPair',
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn
from torch.optim import AdamW

# --------------------------------------------------------------------------- #
# TrainingViews
# --------------------------------------------------------------------------- #


@dataclass
class TrainingViews:
    """Container for augmentation output views and metadata.

    The number and shape of views is defined by the model-augmentation contract,
    not enforced at runtime. Models should document their expected view structure.

    Examples:
        TS2Vec: views has 2 tensors, metadata has 'crop_length' (int).
        CoST: views has 1 tensor, metadata is empty.
        AutoTCL: views has 1 tensor, metadata may have 'augmentation_factor'.
    """

    views: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Augmentation Protocol (primitive, model-agnostic)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Augmentation(Protocol):
    """Structural protocol for model-agnostic augmentation primitives.

    Implements this protocol to create a primitive transform that accepts
    a tensor and returns a transformed tensor of the same shape. Unlike
    :class:`AugmentationMethod`, this protocol does not wrap results in
    :class:`TrainingViews` — it returns a bare tensor.

    Examples:
        Jitter, Scaling, Permutation — shared across all models.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to ``x``.

        Args:
            x: Input tensor of shape ``(batch, channels, time)``
               or ``(batch, time, channels)``.

        Returns:
            Transformed tensor with the same shape as ``x``.
        """
        ...


# --------------------------------------------------------------------------- #
# AugmentationMethod ABC
# --------------------------------------------------------------------------- #


class AugmentationMethod(ABC):
    """Abstract base class for all time-series augmentation strategies.

    Subclass this to create a new augmentation. Implement ``augment()`` to
    define the transform. The model calls this polymorphically -- no enum
    dispatch needed.

    Pure transform. No ``train_step``, no ``configure_optimizer``, no ``_setup``.
    """

    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401
        """Return augmented views of ``data``.

        Args:
            data: Input tensor of shape ``(batch, time, channels)``.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            TrainingViews containing augmented tensor(s) and metadata.
        """
        ...


# --------------------------------------------------------------------------- #
# AugmentationTrainingStrategy ABC
# --------------------------------------------------------------------------- #


class AugmentationTrainingStrategy(ABC):
    """Defines how a trainable augmentation network is optimized.

    Subclass to create a new training strategy. Implement ``compute_loss()``
    to define the loss function. Override ``should_train()`` for epoch-gated
    schedules.
    """

    def __init__(self, training_ratio_step: int = 1) -> None:
        """Initialize the training strategy.

        Args:
            training_ratio_step: Train the aug network every N epochs.
                Default ``1`` means every epoch. Matches the original
                ``augmentation_network_training_ratio_step`` from
                ``augmentation_mode_params``.
        """
        self._training_ratio_step = training_ratio_step

    @abstractmethod
    def compute_loss(
        self,
        x_embeddings: torch.Tensor,
        aug_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the augmentation network loss.

        Args:
            x_embeddings: Encodings of the original data.
            aug_x_embeddings: Encodings of the augmented data.
            augmentation_factor: Learned augmentation weights/factors.

        Returns:
            Scalar loss tensor requiring gradients.
        """
        ...

    def should_train(self, epoch: int, batch_idx: int) -> bool:  # noqa: ARG002
        """Determine if aug-network training should run this step.

        Default: train when ``epoch % training_ratio_step == 0``.

        Args:
            epoch: Current training epoch.
            batch_idx: Current batch index within the epoch.

        Returns:
            ``True`` if the augmentation network should be trained this step.
        """
        return epoch % self._training_ratio_step == 0


# --------------------------------------------------------------------------- #
# TrainableAugmentation
# --------------------------------------------------------------------------- #


class TrainableAugmentation(AugmentationMethod, nn.Module, ABC):
    """Augmentation with learnable parameters.

    Composes an ``AugmentationTrainingStrategy`` for loss computation.
    Subclasses implement ``train_step()`` to define their own training loop.
    AutoTCL-specific; not a general pattern for TS2Vec/CoST.
    """

    def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
        """Initialize a trainable augmentation.

        Args:
            training_strategy: Strategy for computing the augmentation loss.
        """
        super().__init__()
        self._training_strategy = training_strategy

    def should_train_augmentation(self, epoch: int, batch_idx: int) -> bool:
        """Check whether the aug-network should train this step.

        Delegates to the composed training strategy to avoid exposing
        the private ``_training_strategy`` attribute.

        Args:
            epoch: Current training epoch.
            batch_idx: Current batch index within the epoch.

        Returns:
            ``True`` if the augmentation network should be trained this step.
        """
        return self._training_strategy.should_train(epoch, batch_idx)

    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401
        """Return an augmented view produced by the encoder model.

        Args:
            data: Input time-series tensor.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            TrainingViews containing the augmented tensor(s) and metadata.
        """
        ...

    def configure_optimizer(self, lr: float) -> AdamW:
        """Return optimizer over this module's parameters.

        Args:
            lr: Learning rate for the augmentation network optimizer.

        Returns:
            AdamW optimizer for this module's parameters.
        """
        return AdamW(self.parameters(), lr=lr)

    @abstractmethod
    def train_step(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Run one augmentation-network training step.

        Subclasses define their own training loop. The base provides
        ``configure_optimizer()`` and ``should_train_augmentation()``;
        the composed ``_training_strategy`` provides ``compute_loss()``.

        Args:
            x: Original input data.
            encoder: The main encoder module to compute embeddings.
            batch_idx: Current batch index within the epoch.

        Returns:
            Loss tensor if training should run this step, otherwise None.
        """
        ...


# Covariant type parameter for AugmentationProducer[V].
# V appears only in return position, enabling Liskov substitution:
# AugmentationProducer[AlignedPair] is a subtype of AugmentationProducer[ViewPair].
# PEP 695: variance is inferred by type checkers based on usage (covariant here).


# --------------------------------------------------------------------------- #
# Layer 2 — Typed view results (ViewSets)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SingleView:
    """A single augmented view returned by a producer.

    Used by models that need only one transformed copy of the input
    (e.g. AutoTCL with a neural-network augmentation).

    Args:
        view: Augmented tensor of shape ``(batch, time, channels)``.
    """

    view: torch.Tensor


@dataclass(frozen=True)
class ViewPair:
    """Two augmented views returned by a producer.

    Used by models that need a pair of views for contrastive or
    consistency losses (e.g. CoST query/key, TS-TCC weak/strong).

    Args:
        first: First augmented tensor.
        second: Second augmented tensor.
    """

    first: torch.Tensor
    second: torch.Tensor


@dataclass(frozen=True)
class AlignedPair(ViewPair):
    """A pair of augmented views with a known aligned region.

    Extends :class:`ViewPair` with an ``overlap_length`` field so that
    consumers (e.g. TS2Vec) can slice embeddings to the aligned span
    without a ``metadata`` dict.

    ``AlignedPair`` is-a ``ViewPair`` (Liskov substitution), so a producer
    returning ``AlignedPair`` satisfies any slot expecting ``ViewPair``.

    Args:
        first: First augmented tensor.
        second: Second augmented tensor.
        overlap_length: Number of time steps over which the two views align.
            For non-crop augmentations this equals the full sequence length,
            making the alignment slice a no-op.
    """

    overlap_length: int


# --------------------------------------------------------------------------- #
# Layer 3 — Producers (AugmentationProducer Protocol)
# --------------------------------------------------------------------------- #


class AugmentationProducer[V](Protocol):
    """Assembles the view set a model's loss requires from a batch.

    A producer wraps one or more :class:`Augmentation` primitives and
    returns a typed view set (:class:`SingleView`, :class:`ViewPair`, or
    :class:`AlignedPair`). This is the object injected into a model at
    construction time.

    ``V`` is covariant — it appears only in return position — so a
    ``AugmentationProducer[AlignedPair]`` can be used wherever
    ``AugmentationProducer[ViewPair]`` is expected.

    This is a structural Protocol; concrete classes satisfy it by having
    the correct ``produce`` signature, not by inheriting from it.
    """

    def produce(self, x: torch.Tensor) -> V:
        """Produce the model's view set from a batch.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            A typed view set (:class:`SingleView`, :class:`ViewPair`,
            or :class:`AlignedPair`).
        """
        ...


# --------------------------------------------------------------------------- #
# Capability — Trainable producers
# --------------------------------------------------------------------------- #


class TrainableAugmentationProducer(nn.Module, ABC):
    """A trainable augmentation producer with learnable parameters.

    Combines the :class:`nn.Module` lifecycle (parameters, state_dict)
    with a training strategy for the augmentation network. This is a
    **nominal** ABC (not a Protocol) because it must be runtime-checkable
    via ``isinstance()`` to gate the trainable path.

    ``TrainableAugmentationProducer`` structurally satisfies
    ``AugmentationProducer[SingleView]`` (it has ``produce(x) -> SingleView``),
    so it type-checks in any ``SingleView`` slot.

    Args:
        training_strategy: Strategy for computing the augmentation loss
            and determining training frequency.
    """

    def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
        """Initialize a trainable augmentation producer.

        Args:
            training_strategy: Strategy for computing the augmentation loss.
        """
        super().__init__()
        self._training_strategy = training_strategy

    @abstractmethod
    def produce(self, x: torch.Tensor) -> SingleView:
        """Return an augmented view produced by the encoder model.

        Args:
            x: Input time-series tensor of shape ``(batch, time, channels)``.

        Returns:
            A single augmented view wrapped in :class:`SingleView`.
        """
        ...

    @abstractmethod
    def train_step(
        self, x: torch.Tensor, encoder: nn.Module, batch_idx: int
    ) -> torch.Tensor | None:
        """Run one augmentation-network training step.

        Subclasses define their own training loop. The base provides
        ``configure_optimizer()`` and ``should_train_augmentation()``;
        the composed ``_training_strategy`` provides ``compute_loss()``.

        Args:
            x: Original input data.
            encoder: The main encoder module to compute embeddings.
            batch_idx: Current batch index within the epoch.

        Returns:
            Loss tensor if training should run this step, otherwise None.
        """
        ...

    def configure_optimizer(self, lr: float) -> AdamW:
        """Return optimizer over this module's parameters.

        Args:
            lr: Learning rate for the augmentation network optimizer.

        Returns:
            AdamW optimizer for this module's parameters.
        """
        return AdamW(self.parameters(), lr=lr)

    def should_train_augmentation(self, epoch: int, batch_idx: int) -> bool:
        """Check whether the aug-network should train this step.

        Delegates to the composed training strategy to avoid exposing
        the private ``_training_strategy`` attribute.

        Args:
            epoch: Current training epoch.
            batch_idx: Current batch index within the epoch.

        Returns:
            ``True`` if the augmentation network should be trained this step.
        """
        return self._training_strategy.should_train(epoch, batch_idx)
