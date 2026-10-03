"""Outlier masking mode enum.

Selects which members a cluster drops before its centroid is recomputed, in
hierarchical-clustering models that refine prototypes bottom-up. Distinct from
:class:`~chronocratic.models.convolutional.dilated.encoders.masking.MaskMode`,
which masks input timesteps rather than cluster members.

Mapping to behaviour
    FARTHEST:   drop the single member furthest from the centroid.
    THRESHOLD:  drop every member beyond an absolute distance.
    PROPORTION: drop a fixed fraction of each cluster, furthest first.

FARTHEST removes a constant one member per cluster and so cannot empty a
cluster of two or more; the other two are governed by their parameter and can,
which is why the centroid computation excludes rather than deletes.
"""

from __future__ import annotations

from enum import StrEnum


class OutlierMaskMode(StrEnum):
    """How to select cluster members to exclude when refining a centroid.

    Attributes:
        FARTHEST: Exclude the single member furthest from the centroid.
        THRESHOLD: Exclude every member whose distance to the centroid
            exceeds an absolute threshold.
        PROPORTION: Exclude a fixed fraction of each cluster's members,
            furthest from the centroid first.
    """

    FARTHEST = "farthest"
    THRESHOLD = "threshold"
    PROPORTION = "proportion"
