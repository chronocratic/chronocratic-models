"""Contrastive pair selection and losses for MHCCL.

MHCCL contrasts an anchor against sets rather than single views, so both terms
share a shape: select a set of positives and a set of negatives, score each by
inner product with the anchor, and treat the result as multi-label classification
under a sigmoid — binary cross-entropy against a multi-hot target, not softmax
InfoNCE. This module builds those sets.

The instance-level term draws its positives from the anchor's cluster at the
finest partition. The cluster-level term draws them from the anchor's own
prototype plus the prototypes *downward masking* rescues: a candidate negative
whose cluster shares a parent at the next partition is a fake negative, so it is
removed from the negatives and supplemented into the positives.

Selection is vectorized over the batch and stays on the input's device. The
reference builds both sets with per-anchor Python loops over every cluster, which
dominates its training step.
"""

from __future__ import annotations

__all__ = [
    "ContrastTerms",
    "instance_contrast_terms",
    "masked_bce_with_logits",
    "prototype_contrast_terms",
]

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class ContrastTerms:
    """One contrastive term's scored candidates and their supervision.

    Attributes:
        logits: ``(B, M)`` inner products between each anchor and its selected
            candidates, positives first.
        targets: ``(B, M)`` multi-hot target, ``1.0`` on positive slots.
        valid: ``(B, M)`` bool mask. ``False`` marks a slot the batch could not
            fill — a row with fewer candidates than requested — which the loss
            excludes rather than padding with a duplicate that would then carry
            double weight.
    """

    logits: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor


def _sample_from_mask(*, candidates: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``count`` candidates per row, uniformly and without replacement.

    Random keys are drawn for every position, invalid positions are pushed below
    every valid one, and the top ``count`` keys are taken. Rows with fewer than
    ``count`` candidates come back flagged rather than padded.

    Uses torch's global RNG, so ``lightning.seed_everything`` controls it. The
    reference selects with ``random.sample``, which its own seeding covers, but
    it draws the views themselves through ``np.random``, which it never seeds.

    Args:
        candidates: ``(B, M)`` bool mask of admissible choices per row.
        count: Number of choices to draw per row. Clamped to ``M``.

    Returns:
        ``(B, k)`` indices into the mask's second axis and a ``(B, k)`` bool
        mask marking which of them were genuinely available, where ``k`` is
        ``min(count, M)``.
    """
    width = min(count, candidates.size(1))
    if width <= 0:
        shape = (candidates.size(0), 0)
        empty_index = torch.zeros(shape, dtype=torch.long, device=candidates.device)
        return empty_index, empty_index.bool()
    keys = torch.rand(candidates.shape, device=candidates.device).masked_fill(~candidates, -1.0)
    scores, indices = keys.topk(width, dim=1)
    return indices, scores >= 0.0


def _interleave_views(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Interleave two ``(B, M, D)`` view stacks into ``(B, 2M, D)``.

    Each selected instance contributes both of its views, adjacent, which is the
    order the reference builds by appending ``q[j]`` then ``k[j]``.
    """
    return torch.stack([first, second], dim=2).flatten(1, 2)


def instance_contrast_terms(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    assignment: torch.Tensor,
    positive_count: int,
    negative_count: int,
) -> ContrastTerms:
    """Score an anchor against instances sharing its finest-partition cluster.

    Both views of every selected instance are scored, so the returned width is
    ``2 * positive_count + 2 * negative_count``. The anchor's own query vector is
    excluded from its positives — pairing a vector with itself is a constant —
    so a self slot contributes its momentum view twice, as in the reference.

    Args:
        query: ``(B, D)`` unit-norm anchor projections from the query encoder.
        key: ``(B, D)`` unit-norm projections of the same batch from the
            momentum encoder.
        assignment: ``(B,)`` finest-partition cluster id per anchor.
        positive_count: Number of positive instances to draw per anchor.
        negative_count: Number of negative instances to draw per anchor.

    Returns:
        The scored candidates and their supervision.
    """
    batch_size = query.size(0)
    device = query.device
    self_index = torch.arange(batch_size, device=device)
    same_cluster = assignment[:, None] == assignment[None, :]

    # Positives: the anchor is always a candidate, so any shortfall is padded
    # with the anchor itself rather than dropped — the reference's behaviour.
    positive_index, positive_available = _sample_from_mask(
        candidates=same_cluster, count=positive_count
    )
    positive_index = torch.where(positive_available, positive_index, self_index[:, None])

    is_self = positive_index == self_index[:, None]
    positive_first = torch.where(is_self.unsqueeze(-1), key[positive_index], query[positive_index])
    positive_views = _interleave_views(positive_first, key[positive_index])

    # DIVERGENCE: negatives are every instance outside the anchor's cluster. The
    # reference excludes only the positives it happened to sample, so when a
    # cluster holds more members than ``positive_count`` the unsampled ones
    # become negatives of an anchor they share a cluster with — the fake
    # negatives the model exists to remove.
    negative_index, negative_available = _sample_from_mask(
        candidates=~same_cluster, count=negative_count
    )
    negative_views = _interleave_views(query[negative_index], key[negative_index])

    candidates = torch.cat([positive_views, negative_views], dim=1)
    logits = torch.einsum("bd,bmd->bm", query, candidates)

    targets = torch.zeros_like(logits)
    targets[:, : positive_views.size(1)] = 1.0
    valid = torch.cat(
        [
            torch.ones(positive_views.shape[:2], dtype=torch.bool, device=device),
            negative_available.repeat_interleave(2, dim=1),
        ],
        dim=1,
    )
    return ContrastTerms(logits=logits, targets=targets, valid=valid)


def prototype_contrast_terms(
    *,
    query: torch.Tensor,
    assignment: torch.Tensor,
    centroids: torch.Tensor,
    parents: torch.Tensor,
    positive_count: int,
    negative_count: int,
) -> ContrastTerms:
    """Score an anchor against prototypes at one partition, after downward masking.

    A candidate negative prototype that shares the anchor's parent cluster at the
    next partition is a fake negative: it is withheld from the negatives and
    supplemented into the positives. The anchor's own prototype is always the
    first positive.

    Args:
        query: ``(B, D)`` unit-norm anchor projections.
        assignment: ``(B,)`` cluster id of each anchor at this partition.
        centroids: ``(K, D)`` prototypes for this partition.
        parents: ``(K,)`` map from this partition's cluster ids to the next
            partition's, which supplies the sibling relation.
        positive_count: Total positive prototypes per anchor, including the
            anchor's own.
        negative_count: Number of negative prototypes to draw per anchor.

    Returns:
        The scored candidates and their supervision.
    """
    device = query.device
    cluster_ids = torch.arange(centroids.size(0), device=device)
    is_own = cluster_ids[None, :] == assignment[:, None]
    # Downward masking: same parent at the next partition means same higher-level
    # semantics, so these are not negatives.
    shares_parent = parents[None, :] == parents[assignment][:, None]

    supplemented_index, supplemented_available = _sample_from_mask(
        candidates=shares_parent & ~is_own, count=positive_count - 1
    )
    supplemented_index = torch.where(
        supplemented_available, supplemented_index, assignment[:, None]
    )
    positive_index = torch.cat([assignment[:, None], supplemented_index], dim=1)

    negative_index, negative_available = _sample_from_mask(
        candidates=~shares_parent & ~is_own, count=negative_count
    )

    candidates = torch.cat([centroids[positive_index], centroids[negative_index]], dim=1)
    logits = torch.einsum("bd,bmd->bm", query, candidates)

    targets = torch.zeros_like(logits)
    targets[:, : positive_index.size(1)] = 1.0
    valid = torch.cat(
        [torch.ones(positive_index.shape, dtype=torch.bool, device=device), negative_available],
        dim=1,
    )
    return ContrastTerms(logits=logits, targets=targets, valid=valid)


def masked_bce_with_logits(
    *, terms: ContrastTerms, temperature: float | None = None
) -> torch.Tensor:
    """Average binary cross-entropy over the slots a batch could actually fill.

    Equal to ``reduction="mean"`` whenever every slot is valid, which is the
    non-degenerate case; the mask only changes the result when a row held fewer
    candidates than requested.

    Args:
        terms: Logits, targets, and validity from a contrast-term builder.
        temperature: Optional divisor applied to the logits. ``None`` leaves
            them as raw inner products, which is what the reference's published
            configuration does — it gates temperature behind a flag its own
            documented command does not pass.

    Returns:
        Scalar loss.
    """
    logits = terms.logits if temperature is None else terms.logits / temperature
    elementwise = functional.binary_cross_entropy_with_logits(
        logits, terms.targets, reduction="none"
    )
    return (elementwise * terms.valid).sum() / terms.valid.sum().clamp_min(1.0)
