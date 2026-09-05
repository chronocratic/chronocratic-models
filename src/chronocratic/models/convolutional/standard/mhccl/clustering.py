"""On-device FINCH hierarchical clustering for MHCCL.

FINCH (Sarfraz et al., CVPR 2019) builds a partition hierarchy bottom-up from
first-nearest-neighbour relations alone: no cluster count to specify and no
distance threshold to tune. Each level links every point to its nearest
neighbour, takes the weakly connected components of that graph as clusters, and
recurses on the resulting centroids.

Everything here runs in torch on the input's device. The reference reaches for
``sklearn.metrics.pairwise_distances`` for the neighbour search,
``scipy.sparse.csgraph.connected_components`` for the components, and
``pynndescent`` above 40k points; none is a dependency of this library, and a
host round-trip per level would dominate a per-batch clustering step. The
component labelling is verified equal to SciPy's in ``tests/unit/test_mhccl.py``.

Divergences from the reference, each commented at its line: centroids are
computed without mutating the caller's feature matrix and with true per-cluster
counts; the level-to-level cluster map is returned rather than re-derived by
searching the label matrix; and the reference's ``min_sim`` edge pruning is
omitted because it is a no-op there (it zeroes stored entries of a
``csr_matrix``, which ``csgraph`` ignores, since those routines read only the
sparsity structure).
"""

from __future__ import annotations

__all__ = ["ClusterHierarchy", "finch"]

from dataclasses import dataclass

import torch

from chronocratic.models.enums.clustering import OutlierMaskMode

# Two points can only ever merge into one cluster, so a level with fewer than
# three points cannot produce a coarser non-degenerate partition.
_MIN_POINTS_TO_CLUSTER = 3
# A cluster must keep at least one member for its centroid to be defined.
_MIN_CLUSTER_SIZE_TO_MASK = 2


@dataclass(frozen=True)
class ClusterHierarchy:
    """A FINCH partition hierarchy over one feature matrix.

    Frozen so a hierarchy cannot be mutated between the instance-level and
    cluster-level terms that both consume it.

    Attributes:
        assignments: One ``(N,)`` int64 tensor per level, holding contiguous
            cluster ids in ``[0, num_clusters)``. Level 0 is the finest.
        centroids: One ``(num_clusters, D)`` tensor per level, aligned with
            ``assignments``.
        parents: One ``(num_clusters,)`` int64 tensor per level ``n``, mapping
            each level-``n`` cluster id to its level-``n+1`` cluster id. Length
            is ``len(assignments) - 1``: the coarsest level has no parent, and
            so cannot supply the sibling relation cluster-level contrast needs.
    """

    assignments: tuple[torch.Tensor, ...]
    centroids: tuple[torch.Tensor, ...]
    parents: tuple[torch.Tensor, ...]

    @property
    def num_levels(self) -> int:
        """Number of partitions in the hierarchy."""
        return len(self.assignments)


def _one_nn_neighbors(features: torch.Tensor) -> torch.Tensor:
    """Return each point's first nearest neighbour under Euclidean distance.

    Args:
        features: ``(N, D)`` feature matrix with at least two rows.

    Returns:
        ``(N,)`` int64 tensor where entry ``i`` is the index of ``i``'s nearest
        neighbour, excluding ``i`` itself.
    """
    distances = torch.cdist(features, features)
    # The reference fills the diagonal with 1e12, which is only "large enough"
    # for its normalized features; infinity is unconditional.
    distances.fill_diagonal_(torch.inf)
    return distances.argmin(dim=1)


def _weak_connected_components(neighbors: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Label the weakly connected components of a first-neighbour graph.

    Min-label propagation with pointer jumping over the symmetrised edge set.
    Components are numbered by ascending smallest member index, which is also
    SciPy's convention, so the labels match
    ``scipy.sparse.csgraph.connected_components(directed=True,
    connection="weak")`` element for element.

    Args:
        neighbors: ``(N,)`` int64 tensor of first-neighbour indices.

    Returns:
        A ``(N,)`` int64 tensor of contiguous component ids, and the component
        count.

    Raises:
        RuntimeError: If propagation does not reach a fixed point, which would
            mean the iteration bound is wrong rather than the graph unusual.
    """
    num_points = neighbors.numel()
    device = neighbors.device
    index = torch.arange(num_points, device=device)

    # Weak connectivity is connectivity of the undirected closure, so every
    # edge is followed in both directions.
    sources = torch.cat([index, neighbors])
    targets = torch.cat([neighbors, index])

    labels = index.clone()
    # Each round hooks every node onto the smallest label in its neighbourhood,
    # then compresses the resulting pointer forest to a fixed point. Hooking
    # spreads a label one edge per round, so the round count is bounded by the
    # component diameter; first-neighbour graphs of real point clouds have small
    # components and settle in a handful of rounds. The bound below is the
    # unreachable worst case — labels decrease monotonically and are bounded
    # below by zero, so the loop cannot cycle.
    max_hook_rounds = num_points + 2
    # Pointer doubling halves the remaining depth each step, so compression is
    # logarithmic in the number of nodes.
    max_compress_rounds = num_points.bit_length() + 2
    for _ in range(max_hook_rounds):
        hooked = labels.clone()
        hooked.scatter_reduce_(0, targets, labels[sources], reduce="amin")
        for _ in range(max_compress_rounds):
            jumped = hooked[hooked]
            if torch.equal(jumped, hooked):
                break
            hooked = jumped
        else:
            msg = f"pointer compression did not converge in {max_compress_rounds} rounds"
            raise RuntimeError(msg)
        if torch.equal(hooked, labels):
            break
        labels = hooked
    else:
        msg = f"connected-component labelling did not converge in {max_hook_rounds} rounds"
        raise RuntimeError(msg)

    unique_labels, assignment = torch.unique(labels, return_inverse=True)
    return assignment, int(unique_labels.numel())


def _cluster_means(
    *,
    features: torch.Tensor,
    assignment: torch.Tensor,
    num_clusters: int,
    excluded: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average each cluster's members, optionally excluding masked members.

    Args:
        features: ``(N, D)`` feature matrix.
        assignment: ``(N,)`` cluster id per row.
        num_clusters: Number of clusters, fixing the output's first dimension.
        excluded: Optional ``(N,)`` bool mask; ``True`` rows are left out of
            their cluster's mean.

    Returns:
        ``(num_clusters, D)`` tensor of centroids.
    """
    # DIVERGENCE: the reference's ``cool_mean`` implements exclusion by writing
    # zeros into the caller's feature matrix in place — corrupting every later
    # level, which reads the same matrix — and then subtracts exactly one from
    # *every* cluster's count regardless of how many members were masked, which
    # is wrong for THRESHOLD and PROPORTION and divides by zero for a singleton
    # cluster under FARTHEST. Excluding by mask leaves the input untouched and
    # divides by the true retained count.
    kept_features = features if excluded is None else features[~excluded]
    kept_assignment = assignment if excluded is None else assignment[~excluded]

    sums = torch.zeros(num_clusters, features.size(1), device=features.device, dtype=features.dtype)
    sums.index_add_(0, kept_assignment, kept_features)
    counts = torch.zeros(num_clusters, device=features.device, dtype=features.dtype)
    counts.index_add_(0, kept_assignment, torch.ones_like(kept_assignment, dtype=features.dtype))
    return sums / counts.clamp_min(1.0).unsqueeze(1)


def _distance_to_own_centroid(
    *, features: torch.Tensor, assignment: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Return each row's Euclidean distance to the centroid of its own cluster."""
    return (features - centroids[assignment]).norm(dim=1)


def _outlier_mask(
    *,
    features: torch.Tensor,
    assignment: torch.Tensor,
    centroids: torch.Tensor,
    num_clusters: int,
    mask_mode: OutlierMaskMode,
    distance_threshold: float,
    mask_proportion: float,
) -> torch.Tensor:
    """Select cluster members to exclude when recomputing centroids.

    Vectorized over all clusters at once. The reference walks every member of
    every cluster in a Python loop calling ``scipy.spatial.distance.euclidean``,
    which is the dominant cost of its training step.

    A cluster is never emptied: clusters of fewer than two members are exempt,
    and PROPORTION always retains at least one member.

    Args:
        features: ``(N, D)`` feature matrix.
        assignment: ``(N,)`` cluster id per row.
        centroids: ``(num_clusters, D)`` current centroids.
        num_clusters: Number of clusters.
        mask_mode: Which members to exclude.
        distance_threshold: Absolute cutoff for ``THRESHOLD``.
        mask_proportion: Fraction of each cluster to drop for ``PROPORTION``.

    Returns:
        ``(N,)`` bool mask; ``True`` marks a row excluded from its centroid.
    """
    device = features.device
    distances = _distance_to_own_centroid(
        features=features, assignment=assignment, centroids=centroids
    )
    counts = torch.zeros(num_clusters, device=device, dtype=torch.long)
    counts.index_add_(0, assignment, torch.ones_like(assignment))
    maskable = counts[assignment] >= _MIN_CLUSTER_SIZE_TO_MASK

    if mask_mode == OutlierMaskMode.THRESHOLD:
        return (distances > distance_threshold) & maskable

    if mask_mode == OutlierMaskMode.FARTHEST:
        cluster_max = torch.full((num_clusters,), -torch.inf, device=device, dtype=distances.dtype)
        cluster_max.scatter_reduce_(0, assignment, distances, reduce="amax")
        return (distances == cluster_max[assignment]) & maskable

    # PROPORTION: rank members within their cluster by descending distance and
    # drop the leading fraction, keeping at least one member per cluster.
    by_distance = torch.argsort(distances, descending=True)
    grouped = by_distance[torch.argsort(assignment[by_distance], stable=True)]
    grouped_assignment = assignment[grouped]
    starts = torch.cumsum(counts, dim=0) - counts
    rank_in_cluster = torch.arange(features.size(0), device=device) - starts[grouped_assignment]
    drop_count = (counts.to(distances.dtype) * mask_proportion).round().long()
    drop_count = torch.minimum(drop_count, counts - 1).clamp_min(0)

    excluded = torch.zeros(features.size(0), device=device, dtype=torch.bool)
    excluded[grouped] = rank_in_cluster < drop_count[grouped_assignment]
    return excluded


def _nearest_member_per_cluster(
    *, features: torch.Tensor, assignment: torch.Tensor, centroids: torch.Tensor, num_clusters: int
) -> torch.Tensor:
    """Return the index of the member closest to each cluster's centroid."""
    distances = _distance_to_own_centroid(
        features=features, assignment=assignment, centroids=centroids
    )
    by_distance = torch.argsort(distances)
    nearest = torch.zeros(num_clusters, device=features.device, dtype=torch.long)
    # Duplicate indices resolve to the last write, so reversing the ascending
    # order leaves the smallest distance written last, hence winning.
    nearest[assignment[by_distance].flip(0)] = by_distance.flip(0)
    return nearest


def _refine_centroids(
    *,
    features: torch.Tensor,
    assignment: torch.Tensor,
    num_clusters: int,
    apply_mask: bool,
    mask_mode: OutlierMaskMode,
    distance_threshold: float,
    mask_proportion: float,
    replace_with_nearest_member: bool,
) -> torch.Tensor:
    """Compute a level's centroids, applying upward masking when enabled."""
    centroids = _cluster_means(features=features, assignment=assignment, num_clusters=num_clusters)
    if apply_mask:
        excluded = _outlier_mask(
            features=features,
            assignment=assignment,
            centroids=centroids,
            num_clusters=num_clusters,
            mask_mode=mask_mode,
            distance_threshold=distance_threshold,
            mask_proportion=mask_proportion,
        )
        centroids = _cluster_means(
            features=features, assignment=assignment, num_clusters=num_clusters, excluded=excluded
        )
    if replace_with_nearest_member:
        nearest = _nearest_member_per_cluster(
            features=features, assignment=assignment, centroids=centroids, num_clusters=num_clusters
        )
        centroids = features[nearest]
    return centroids


@torch.no_grad()
def finch(
    features: torch.Tensor,
    *,
    max_levels: int,
    mask_base_level: bool = False,
    mask_upper_levels: bool = False,
    mask_mode: OutlierMaskMode = OutlierMaskMode.FARTHEST,
    distance_threshold: float = 0.3,
    mask_proportion: float = 0.5,
    replace_centroids_with_nearest_member: bool = False,
) -> ClusterHierarchy:
    """Build a FINCH partition hierarchy over ``features``.

    Level 0 clusters the rows themselves; each subsequent level clusters the
    previous level's centroids and composes the assignment, so every level
    partitions the original rows. Centroids at every level are means over the
    original rows, as in the reference.

    Growth stops when a level would hold a single cluster or would reduce the
    cluster count by one or fewer — the reference's rule — or when
    ``max_levels`` partitions exist, or when too few points remain to cluster.

    Args:
        features: ``(N, D)`` feature matrix. Detached and never modified.
        max_levels: Maximum number of partitions to return.
        mask_base_level: Apply upward masking when refining level-0 centroids.
        mask_upper_levels: Apply upward masking at every level above 0.
        mask_mode: Which members upward masking excludes.
        distance_threshold: Absolute cutoff for ``OutlierMaskMode.THRESHOLD``.
        mask_proportion: Fraction dropped for ``OutlierMaskMode.PROPORTION``.
        replace_centroids_with_nearest_member: Replace each centroid with the
            real member closest to it.

    Returns:
        A :class:`ClusterHierarchy` with at least one level.

    Raises:
        ValueError: If ``features`` is not 2-D, holds fewer than two rows, or
            ``max_levels`` is not positive.
    """
    expected_ndim = 2
    if features.ndim != expected_ndim:
        msg = f"features must be 2-D (N, D), got shape {tuple(features.shape)}"
        raise ValueError(msg)
    if features.size(0) < _MIN_POINTS_TO_CLUSTER - 1:
        msg = f"features must hold at least two rows, got {features.size(0)}"
        raise ValueError(msg)
    if max_levels < 1:
        msg = f"max_levels must be positive, got {max_levels}"
        raise ValueError(msg)

    points = features.detach()

    def level_centroids(
        assignment: torch.Tensor, num_clusters: int, *, is_base: bool
    ) -> torch.Tensor:
        return _refine_centroids(
            features=points,
            assignment=assignment,
            num_clusters=num_clusters,
            apply_mask=mask_base_level if is_base else mask_upper_levels,
            mask_mode=mask_mode,
            distance_threshold=distance_threshold,
            mask_proportion=mask_proportion,
            replace_with_nearest_member=replace_centroids_with_nearest_member,
        )

    assignment, num_clusters = _weak_connected_components(_one_nn_neighbors(points))
    assignments = [assignment]
    centroids = [level_centroids(assignment, num_clusters, is_base=True)]
    parents: list[torch.Tensor] = []

    while len(assignments) < max_levels and num_clusters >= _MIN_POINTS_TO_CLUSTER:
        # Cluster the current level's centroids; the resulting labelling is
        # exactly the level-to-level parent map, so no search is needed later.
        parent_map, next_num_clusters = _weak_connected_components(_one_nn_neighbors(centroids[-1]))
        # The reference's stopping rule: a level that collapses to one cluster,
        # or that merges at most one pair, carries no new structure.
        if next_num_clusters == 1 or num_clusters - next_num_clusters <= 1:
            break
        assignment = parent_map[assignment]
        assignments.append(assignment)
        centroids.append(level_centroids(assignment, next_num_clusters, is_base=False))
        parents.append(parent_map)
        num_clusters = next_num_clusters

    return ClusterHierarchy(
        assignments=tuple(assignments), centroids=tuple(centroids), parents=tuple(parents)
    )
