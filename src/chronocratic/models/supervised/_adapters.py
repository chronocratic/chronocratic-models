"""Batch adapters, representation functions, and loss helpers.

Each model has its own batch format and representation extraction path.
This module provides the glue callables so :class:`SupervisedModule`
stays model-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
    from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
    from chronocratic.models.transformer.tst.model import TST

__all__ = [
    'series2vec_representations',
    'supervised_batch_adapter',
    'tst_batch_adapter',
    'tst_representations',
    'tstcc_representations',
]


# ---- batch adapters ----------------------------------------------------------


def tst_batch_adapter(batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Decode TST batch ``(X, targets, padding_masks, IDs)``.

    Returns:
        ``((X, padding_masks), targets)`` — the representation function
        for TST needs both the features and the mask.
    """
    x, targets, padding_masks, _ids = batch
    return (x, padding_masks), targets


def supervised_batch_adapter(batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Decode standard ``(X, targets)`` batch.

    Used by Series2Vec and TS-TCC downstream.

    Returns:
        ``((X,), targets)``.
    """
    x, targets = batch
    return (x,), targets


# ---- representation fns (differentiable; mirror each backbone's encode hook) --


def tst_representations(
    backbone: TST, x: torch.Tensor, padding_masks: torch.Tensor
) -> torch.Tensor:
    """Run the TST trunk and zero padded positions.

    Args:
        backbone: A :class:`TST` instance.
        x: Input features of shape ``(B, seq, feat_dim)``.
        padding_masks: Boolean mask of shape ``(B, seq)``; ``1`` = valid.

    Returns:
        Representations of shape ``(B, seq, d_model)`` with padded positions zeroed.
    """
    reps = backbone.get_representations(x, padding_masks)
    return reps * padding_masks.unsqueeze(-1)


def series2vec_representations(backbone: Series2Vec, x: torch.Tensor) -> torch.Tensor:
    """Extract temporal + frequency representations.

    Args:
        backbone: A :class:`Series2Vec` instance.
        x: Input features of shape ``(B, seq, channels)``.

    Returns:
        Concatenated representations of shape ``(B, 2*representation_dims)``.
    """
    return backbone.network.encode(x)


def tstcc_representations(backbone: TSTCC, x: torch.Tensor) -> torch.Tensor:
    """Extract pre-logits features from the TCC encoder.

    Args:
        backbone: A :class:`TSTCC` instance.
        x: Input features of shape ``(B, channels, seq)``.

    Returns:
        Pre-logits feature tensor. The encoder returns ``(logits, features)``;
        we take ``features``.

    Note:
        Casts input to ``.float()`` because the TCC encoder expects float inputs.
    """
    _logits, features = backbone(x.float())
    return features
