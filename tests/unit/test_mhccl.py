"""Unit tests for the MHCCL model.

Covers the config-to-model contract, the FINCH hierarchy and its equivalences to
the reference's SciPy-backed implementation, both masking strategies, the
momentum encoder and feature bank, the encoding-output-shape contract,
NaN-padding defense, length robustness, and training.

All shapes are asymmetric (``T != C``) so a dropped transpose crashes rather
than silently passing.
"""

from __future__ import annotations

import dataclasses
import inspect
from itertools import pairwise
import math
from types import SimpleNamespace
import warnings

import lightning.pytorch as pl
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair
from chronocratic.models.augmentation.primitives import Jitter
from chronocratic.models.augmentation.producers import FullOverlapProducer, IndependentPairProducer
from chronocratic.models.convolutional.standard.encoders.resnet import Conv1dResNetEncoder
from chronocratic.models.convolutional.standard.mhccl.augmentations import _default_mhccl_pair
from chronocratic.models.convolutional.standard.mhccl.clustering import (
    _cluster_means,
    _one_nn_neighbors,
    _weak_connected_components,
    finch,
)
from chronocratic.models.convolutional.standard.mhccl.config import MHCCLModelParameters
from chronocratic.models.convolutional.standard.mhccl.losses import (
    _sample_from_mask,
    instance_contrast_terms,
    masked_bce_with_logits,
    prototype_contrast_terms,
)
from chronocratic.models.convolutional.standard.mhccl.model import _reset_hierarchy_warning, MHCCL
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.clustering import OutlierMaskMode
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.protocols import HasEncoder

# Asymmetric on purpose: T=50, C=3.
BATCH, TIME, CHANNELS = 16, 50, 3

# Tiny architecture so the suite stays fast. The library defaults build a
# ResNet-18 pair (~22M parameters across query and key).
SMALL = {
    "encoder_stage_channels": (8, 16),
    "encoder_stage_depths": (1, 1),
    "encoder_stage_strides": (1, 2),
    "stem_conv_channels": 8,
    "projection_hidden_dim": 16,
    "projection_dim": 8,
}


def _small_model(**overrides: object) -> MHCCL:
    """Build a small MHCCL instance, overriding any keyword."""
    kwargs = {"input_dim": CHANNELS} | SMALL | overrides
    return MHCCL(**kwargs)  # type: ignore[arg-type]


def _data(batch: int = BATCH, time: int = TIME, channels: int = CHANNELS) -> torch.Tensor:
    """Return a random ``(B, T, C)`` batch."""
    return torch.randn(batch, time, channels)


def _clustered_features(
    groups: int = 8, per_group: int = 5, dim: int = 16, *, seed: int = 0
) -> torch.Tensor:
    """Return features with genuine group structure for the clustering tests.

    White noise has no between-sample structure, so a partition over it says
    nothing about whether the clustering found anything.
    """
    generator = torch.Generator().manual_seed(seed)
    centres = 8.0 * torch.randn(groups, 1, dim, generator=generator)
    noise = torch.randn(groups, per_group, dim, generator=generator)
    return (centres + noise).reshape(groups * per_group, dim)


def _random_first_neighbour_graph(num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random functional graph: one out-edge per node, no self-loops."""
    return np.array(
        [rng.choice([j for j in range(num_points) if j != i]) for i in range(num_points)]
    )


@pytest.fixture(autouse=True)
def _reset_warnings() -> None:
    """Keep the once-per-process hierarchy warning from leaking between tests."""
    _reset_hierarchy_warning()


# --------------------------------------------------------------------------- #
# Config-to-model contract
# --------------------------------------------------------------------------- #


class TestConfigContract:
    """MHCCLModelParameters and MHCCL.__init__ must mirror each other."""

    def test_config_splat_instantiates(self) -> None:
        """Model(**vars(ModelParameters(...))) must not raise."""
        config = MHCCLModelParameters(input_dim=CHANNELS, **SMALL)  # type: ignore[arg-type]
        assert isinstance(MHCCL(**vars(config)), MHCCL)

    def test_config_splat_with_defaults_only(self) -> None:
        """Partial config instantiation works — every optional field has a default."""
        config = MHCCLModelParameters(input_dim=1)
        assert isinstance(MHCCL(**vars(config)), MHCCL)

    def test_field_names_match_init_parameters(self) -> None:
        """Every config field has an __init__ parameter of the same name."""
        init_params = set(inspect.signature(MHCCL.__init__).parameters) - {"self"}
        field_names = {f.name for f in dataclasses.fields(MHCCLModelParameters)}
        assert field_names == init_params

    def test_defaults_match_init_defaults(self) -> None:
        """Every config default equals the matching __init__ default."""
        init_params = inspect.signature(MHCCL.__init__).parameters
        for field in dataclasses.fields(MHCCLModelParameters):
            if field.default is dataclasses.MISSING:
                continue
            assert field.default == init_params[field.name].default, (
                f"default mismatch for {field.name}"
            )

    def test_init_is_keyword_only(self) -> None:
        """No positional parameters besides self."""
        params = list(inspect.signature(MHCCL.__init__).parameters.values())[1:]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)

    def test_sequence_hyperparameters_are_tuples(self) -> None:
        """Sequence-typed hyperparameters default to tuples, not lists."""
        config = MHCCLModelParameters(input_dim=1)
        assert isinstance(config.encoder_stage_channels, tuple)
        assert isinstance(config.encoder_stage_depths, tuple)
        assert isinstance(config.encoder_stage_strides, tuple)

    def test_hyperparameters_stored_privately(self) -> None:
        """Hyperparameters land on self._{name} attributes."""
        model = _small_model()
        for name in ("input_dim", "key_momentum", "hierarchy_levels", "feature_bank_size"):
            assert hasattr(model, f"_{name}")

    def test_augmentation_excluded_from_saved_hyperparameters(self) -> None:
        """save_hyperparameters(ignore=["augmentation"]) keeps the producer out."""
        model = _small_model(augmentation=IndependentPairProducer(aug=Jitter()))
        assert "augmentation" not in model.hparams

    def test_defaults_match_the_reference(self) -> None:
        """Contrast counts and momentum come from the reference's argparse."""
        config = MHCCLModelParameters(input_dim=1)
        assert config.positive_instance_count == 3
        assert config.negative_instance_count == 4
        assert config.positive_prototype_count == 3
        assert config.negative_prototype_count == 4
        assert config.hierarchy_levels == 3
        assert config.key_momentum == pytest.approx(0.999)
        assert config.learning_rate == pytest.approx(0.03)
        assert config.optimizer_momentum == pytest.approx(0.9)
        assert config.weight_decay == pytest.approx(1e-4)
        # The reference gates temperature behind a flag its published command
        # does not pass, so both terms use raw inner products by default.
        assert config.instance_temperature is None
        assert config.prototype_temperature is None
        # ResNet-18 with BasicBlock, which is what framework.py builds.
        assert config.encoder_stage_depths == (2, 2, 2, 2)
        assert config.residual_block_type is ResidualBlockType.BASIC

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_dim": 0},
            {"projection_dim": 0},
            {"hierarchy_levels": 0},
            {"key_momentum": 1.0},
            {"key_momentum": -0.1},
            {"feature_bank_size": -1},
            {"positive_instance_count": 0},
            {"negative_instance_count": -1},
            {"positive_prototype_count": 0},
            {"instance_temperature": 0.0},
            {"prototype_temperature": -1.0},
            {"outlier_mask_proportion": 0.0},
            {"outlier_mask_proportion": 1.5},
            {"outlier_distance_threshold": 0.0},
            {"singleton_split_count": 1},
            {"max_train_length": 0},
        ],
    )
    def test_config_rejects_invalid_values(self, kwargs: dict) -> None:
        """__post_init__ rejects each out-of-range field."""
        base = {"input_dim": CHANNELS}
        with pytest.raises(ValueError):  # noqa: PT011
            MHCCLModelParameters(**{**base, **kwargs})


# --------------------------------------------------------------------------- #
# Equivalences that license the port
# --------------------------------------------------------------------------- #


class TestReferenceEquivalence:
    """Numerical claims the port rests on, asserted rather than asserted-to."""

    def test_components_match_scipy(self) -> None:
        """Weak-component labels equal scipy.sparse.csgraph, element for element.

        The reference builds its partitions with
        ``connected_components(directed=True, connection="weak")``; this replaces
        it to keep the clustering on-device. Both label components by ascending
        smallest member index, so the agreement is exact, not merely a matching
        partition.
        """
        rng = np.random.default_rng(0)
        for _ in range(60):
            num_points = int(rng.integers(2, 60))
            neighbours = _random_first_neighbour_graph(num_points, rng)
            ours, count = _weak_connected_components(torch.as_tensor(neighbours))
            adjacency = sp.csr_matrix(
                (np.ones(num_points, dtype=np.float32), (np.arange(num_points), neighbours)),
                shape=(num_points, num_points),
            )
            expected_count, expected = connected_components(
                csgraph=adjacency, directed=True, connection="weak", return_labels=True
            )
            assert count == expected_count
            assert np.array_equal(ours.numpy(), expected)
            # The partition is the semantically load-bearing claim; keep it
            # asserted separately so a future SciPy relabelling degrades this
            # test rather than breaking it.
            ours_np = ours.numpy()
            assert np.array_equal(
                ours_np[:, None] == ours_np[None, :], expected[:, None] == expected[None, :]
            )

    def test_components_handle_duplicate_points(self) -> None:
        """Identical rows collapse into one component rather than confusing argmin."""
        features = torch.ones(6, 4)
        assignment, count = _weak_connected_components(_one_nn_neighbors(features))
        assert count == 1
        assert torch.equal(assignment, torch.zeros(6, dtype=torch.long))

    def test_cluster_means_match_the_reference_formula(self) -> None:
        """_cluster_means equals the reference's sparse ``cool_mean``."""
        features = _clustered_features()
        assignment = torch.randint(0, 5, (features.size(0),))
        ours = _cluster_means(features=features, assignment=assignment, num_clusters=5)

        # Transcribed from main.py::cool_mean.
        data = features.numpy()
        partition = assignment.numpy()
        _, counts = np.unique(partition, return_counts=True)
        one_hot = sp.csr_matrix(
            (np.ones(data.shape[0], dtype="float32"), (np.arange(data.shape[0]), partition)),
            shape=(data.shape[0], 5),
        )
        expected = (one_hot.T @ data) / counts[..., np.newaxis]
        assert np.allclose(ours.numpy(), expected, atol=1e-5)

    def test_parent_map_matches_the_reference_lookup(self) -> None:
        """The returned parent map equals the reference's ``argwhere`` search.

        ``framework.py`` recovers a cluster's parent by scanning the label matrix
        for the first row carrying that cluster id and reading its next-level
        label. The merge step already produced exactly that map.
        """
        hierarchy = finch(_clustered_features(groups=12, per_group=6), max_levels=4)
        label_matrix = np.stack([a.numpy() for a in hierarchy.assignments], axis=1)
        for level, parents in enumerate(hierarchy.parents):
            for cluster_id in range(parents.numel()):
                rows = np.argwhere(label_matrix[:, level] == cluster_id)
                expected = label_matrix[rows[0][0], level + 1]
                assert int(parents[cluster_id]) == int(expected)

    def test_min_sim_pruning_is_a_no_op_in_the_reference(self) -> None:
        """Zeroing a csr_matrix entry does not remove the edge from csgraph.

        This is what licenses omitting the reference's ``ensure_early_exit``
        pruning: it writes zeros into stored entries, and the csgraph routines
        read only the sparsity structure.
        """
        rng = np.random.default_rng(1)
        num_points = 40
        neighbours = _random_first_neighbour_graph(num_points, rng)
        adjacency = sp.csr_matrix(
            (np.ones(num_points, dtype=np.float32), (np.arange(num_points), neighbours)),
            shape=(num_points, num_points),
        )
        before = connected_components(csgraph=adjacency, directed=True, connection="weak")[1]
        pruned = adjacency.copy()
        pruned[pruned.nonzero()] = 0  # what get_clust does for long edges
        after = connected_components(csgraph=pruned, directed=True, connection="weak")[1]
        assert np.array_equal(before, after)

    def test_logits_are_inner_products(self) -> None:
        """Scoring equals explicit per-row dot products, which a reshape does not.

        The reference converts its ``(B, M, D)`` candidate stack with
        ``torch.reshape(t, (B, D, M))`` where the comment above the line asks for
        a transpose. Reshape reinterprets the buffer, so its einsum contracts
        mismatched elements.
        """
        batch, count, dim = 4, 7, 16
        query = torch.nn.functional.normalize(torch.randn(batch, dim), dim=1)
        candidates = torch.nn.functional.normalize(torch.randn(batch, count, dim), dim=2)

        ours = torch.einsum("bd,bmd->bm", query, candidates)
        explicit = torch.stack(
            [torch.stack([query[b] @ candidates[b, m] for m in range(count)]) for b in range(batch)]
        )
        assert torch.allclose(ours, explicit, atol=1e-6)

        reshaped = torch.reshape(candidates, (batch, dim, count))
        reference = torch.einsum("nab,nbc->nac", query.unsqueeze(1), reshaped).squeeze(1)
        assert not torch.allclose(reference, explicit, atol=1e-3)

    def test_reshape_and_transpose_differ_on_collapsed_candidates(self) -> None:
        """With identical candidates the reshape still varies across slots.

        A genuine inner product gives one value per identical candidate. The
        reference's reshape does not, because ``gcd(M, D) = 1`` makes each slot a
        dot product with a different cyclic shift of the flattened block — which
        is why the defect is not visible as a degenerate loss.
        """
        batch, count, dim = 2, 7, 16
        assert math.gcd(count, dim) == 1
        query = torch.nn.functional.normalize(torch.randn(batch, dim), dim=1)
        single = torch.nn.functional.normalize(torch.randn(batch, 1, dim), dim=2)
        candidates = single.expand(batch, count, dim).contiguous()

        ours = torch.einsum("bd,bmd->bm", query, candidates)
        assert torch.allclose(ours, ours[:, :1].expand_as(ours), atol=1e-6)

        reshaped = torch.reshape(candidates, (batch, dim, count))
        reference = torch.einsum("nab,nbc->nac", query.unsqueeze(1), reshaped).squeeze(1)
        assert (reference.max(dim=1).values - reference.min(dim=1).values > 0.1).all()

    def test_cosine_schedule_matches_the_reference_formula(self) -> None:
        """CosineAnnealingLR reproduces the reference's hand-computed --cos decay."""
        max_epochs = 12
        model = _small_model()
        model._trainer = SimpleNamespace(max_epochs=max_epochs)  # type: ignore[assignment]
        config = model.configure_optimizers()
        assert isinstance(config, dict)
        optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]

        base_lr = model._learning_rate
        for epoch in range(max_epochs):
            # Transcribed from main.py::adjust_learning_rate under --cos.
            expected = base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs))
            assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, abs=1e-9)
            scheduler.step()

    def test_encoder_reproduces_resnet18_structure(self) -> None:
        """The default encoder is a ResNet-18: four stages of two basic blocks."""
        config = MHCCLModelParameters(input_dim=CHANNELS)
        assert config.encoder_stage_depths == (2, 2, 2, 2)
        assert config.encoder_stage_channels == (64, 128, 256, 512)
        assert config.encoder_stage_strides == (1, 2, 2, 2)
        encoder = Conv1dResNetEncoder(
            input_dim=CHANNELS,
            conv_kernel_size=config.conv_kernel_size,
            stem_conv_kernel_size=config.stem_conv_kernel_size,
            stem_conv_channels=config.stem_conv_channels,
            encoder_stage_channels=config.encoder_stage_channels,
            encoder_stage_depths=config.encoder_stage_depths,
            encoder_stage_strides=config.encoder_stage_strides,
            residual_block_type=config.residual_block_type,
        )
        assert encoder.representation_dim == 512
        assert len(encoder._stages) == 4
        assert all(len(stage) == 2 for stage in encoder._stages)


# --------------------------------------------------------------------------- #
# FINCH
# --------------------------------------------------------------------------- #


class TestFinch:
    """The clustering's structural invariants."""

    def test_hierarchy_coarsens_monotonically(self) -> None:
        """Every level holds strictly fewer clusters than the one below it."""
        hierarchy = finch(_clustered_features(groups=12, per_group=6), max_levels=5)
        sizes = [int(a.max()) + 1 for a in hierarchy.assignments]
        assert all(later < earlier for earlier, later in pairwise(sizes))

    def test_labels_are_contiguous(self) -> None:
        """Cluster ids cover 0..k-1 with no gaps at every level."""
        hierarchy = finch(_clustered_features(), max_levels=4)
        for assignment in hierarchy.assignments:
            assert set(assignment.tolist()) == set(range(int(assignment.max()) + 1))

    def test_centroids_are_means_of_original_rows(self) -> None:
        """Coarser centroids average the original rows, not the level below."""
        features = _clustered_features()
        hierarchy = finch(features, max_levels=4)
        for assignment, centroids in zip(hierarchy.assignments, hierarchy.centroids, strict=True):
            for cluster_id in range(int(assignment.max()) + 1):
                expected = features[assignment == cluster_id].mean(dim=0)
                assert torch.allclose(centroids[cluster_id], expected, atol=1e-5)

    def test_parents_compose_into_the_next_assignment(self) -> None:
        """assignments[n+1] == parents[n][assignments[n]] at every level."""
        hierarchy = finch(_clustered_features(groups=12, per_group=6), max_levels=5)
        assert len(hierarchy.parents) == hierarchy.num_levels - 1
        for level, parents in enumerate(hierarchy.parents):
            assert torch.equal(
                hierarchy.assignments[level + 1], parents[hierarchy.assignments[level]]
            )
            assert int(parents.max()) <= int(hierarchy.assignments[level + 1].max())

    def test_no_degenerate_level_is_returned(self) -> None:
        """A level that collapses to one cluster is discarded, not returned."""
        hierarchy = finch(_clustered_features(), max_levels=8)
        assert all(int(a.max()) + 1 > 1 for a in hierarchy.assignments)

    def test_respects_max_levels(self) -> None:
        """max_levels caps the returned partition count."""
        features = _clustered_features(groups=16, per_group=8)
        assert finch(features, max_levels=2).num_levels <= 2

    def test_is_deterministic(self) -> None:
        """The same input gives the same hierarchy — no RNG in the clustering."""
        features = _clustered_features()
        first, second = finch(features, max_levels=4), finch(features, max_levels=4)
        assert first.num_levels == second.num_levels
        for left, right in zip(first.assignments, second.assignments, strict=True):
            assert torch.equal(left, right)

    def test_does_not_mutate_its_input(self) -> None:
        """Masking excludes rows; it never writes into the caller's matrix."""
        features = _clustered_features()
        original = features.clone()
        finch(
            features,
            max_levels=4,
            mask_base_level=True,
            mask_upper_levels=True,
            mask_mode=OutlierMaskMode.PROPORTION,
        )
        assert torch.equal(features, original)

    def test_identical_rows_give_a_single_level(self) -> None:
        """Degenerate input terminates instead of looping."""
        hierarchy = finch(torch.ones(8, 4), max_levels=4)
        assert hierarchy.num_levels == 1
        assert hierarchy.parents == ()

    @pytest.mark.parametrize("kwargs", [{"max_levels": 0}, {"max_levels": -1}])
    def test_rejects_invalid_max_levels(self, kwargs: dict) -> None:
        """max_levels must be positive."""
        with pytest.raises(ValueError):  # noqa: PT011
            finch(_clustered_features(), **kwargs)

    def test_rejects_wrong_rank(self) -> None:
        """Only 2-D feature matrices are accepted."""
        with pytest.raises(ValueError):  # noqa: PT011
            finch(torch.randn(4, 5, 6), max_levels=2)


# --------------------------------------------------------------------------- #
# Upward masking
# --------------------------------------------------------------------------- #


class TestUpwardMasking:
    """Outlier exclusion when a prototype is recomputed."""

    def test_disabled_by_default_is_bitwise_identical(self) -> None:
        """The default path must not silently apply masking."""
        features = _clustered_features()
        plain = finch(features, max_levels=3)
        explicit = finch(features, max_levels=3, mask_base_level=False, mask_upper_levels=False)
        for left, right in zip(plain.centroids, explicit.centroids, strict=True):
            assert torch.equal(left, right)

    def test_masking_moves_centroids(self) -> None:
        """Excluding the farthest member changes the centroid it was pulling."""
        features = _clustered_features()
        plain = finch(features, max_levels=1)
        masked = finch(features, max_levels=1, mask_base_level=True)
        assert not torch.allclose(plain.centroids[0], masked.centroids[0])

    def test_counts_drop_by_exactly_the_number_masked(self) -> None:
        """The retained mean divides by the true count, not by a constant.

        The reference subtracts exactly one from every cluster's count whatever
        the mode masked, which is wrong for THRESHOLD and PROPORTION.
        """
        features = _clustered_features(groups=4, per_group=6)
        assignment = torch.arange(4).repeat_interleave(6)
        excluded = torch.zeros(features.size(0), dtype=torch.bool)
        excluded[[0, 1, 6]] = True  # two from cluster 0, one from cluster 1
        masked = _cluster_means(
            features=features, assignment=assignment, num_clusters=4, excluded=excluded
        )
        for cluster_id in range(4):
            keep = (assignment == cluster_id) & ~excluded
            assert torch.allclose(masked[cluster_id], features[keep].mean(dim=0), atol=1e-5)

    @pytest.mark.parametrize(
        "mode", [OutlierMaskMode.FARTHEST, OutlierMaskMode.THRESHOLD, OutlierMaskMode.PROPORTION]
    )
    def test_every_mode_produces_finite_centroids(self, mode: OutlierMaskMode) -> None:
        """No mode may empty a cluster and divide by zero."""
        hierarchy = finch(
            _clustered_features(),
            max_levels=3,
            mask_base_level=True,
            mask_upper_levels=True,
            mask_mode=mode,
            distance_threshold=0.5,
            mask_proportion=0.9,
        )
        for centroids in hierarchy.centroids:
            assert torch.isfinite(centroids).all()

    def test_replace_centroids_uses_real_members(self) -> None:
        """Each prototype becomes a row that exists in the input."""
        features = _clustered_features()
        hierarchy = finch(features, max_levels=1, replace_centroids_with_nearest_member=True)
        for centroid in hierarchy.centroids[0]:
            assert (features == centroid).all(dim=1).any()


# --------------------------------------------------------------------------- #
# Downward masking and the contrastive terms
# --------------------------------------------------------------------------- #


class TestDownwardMasking:
    """Fake-negative filtering and positive supplementation."""

    def test_siblings_become_positives_and_leave_the_negatives(self) -> None:
        """Exactly the same-parent, different-cluster prototypes are promoted."""
        # Six clusters in three sibling pairs: {0,1} -> 0, {2,3} -> 1, {4,5} -> 2.
        parents = torch.tensor([0, 0, 1, 1, 2, 2])
        centroids = torch.eye(6, 8)
        assignment = torch.arange(6)
        query = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)

        torch.manual_seed(0)
        terms = prototype_contrast_terms(
            query=query,
            assignment=assignment,
            centroids=centroids,
            parents=parents,
            positive_count=2,
            negative_count=2,
        )
        # Positive slot 0 is the anchor's own prototype; slot 1 is its sibling.
        assert terms.targets[:, :2].eq(1.0).all()
        assert terms.targets[:, 2:].eq(0.0).all()

        # Recover which prototype each scored slot used: centroids are one-hot,
        # so the argmax of the scored candidate identifies it.
        for anchor in range(6):
            sibling = int(parents[anchor])
            expected_siblings = {i for i in range(6) if int(parents[i]) == sibling}
            own = terms.logits[anchor, 0]
            assert own == pytest.approx(float(query[anchor] @ centroids[anchor]), abs=1e-6)
            supplemented = terms.logits[anchor, 1]
            candidates = {
                i: float(query[anchor] @ centroids[i]) for i in expected_siblings - {anchor}
            }
            assert any(supplemented == pytest.approx(v, abs=1e-6) for v in candidates.values())

    def test_negatives_never_share_a_parent(self) -> None:
        """No promoted sibling can appear among the negatives."""
        parents = torch.tensor([0, 0, 1, 1, 2, 2])
        centroids = torch.eye(6, 8)
        assignment = torch.zeros(4, dtype=torch.long)
        query = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        torch.manual_seed(1)
        terms = prototype_contrast_terms(
            query=query,
            assignment=assignment,
            centroids=centroids,
            parents=parents,
            positive_count=2,
            negative_count=4,
        )
        # Anchors are all in cluster 0, whose parent is 0, so clusters 0 and 1
        # are excluded from the negatives; only 2..5 remain.
        for anchor in range(4):
            for slot in range(2, terms.logits.size(1)):
                if not bool(terms.valid[anchor, slot]):
                    continue
                score = float(terms.logits[anchor, slot])
                forbidden = [float(query[anchor] @ centroids[i]) for i in (0, 1)]
                assert all(score != pytest.approx(f, abs=1e-6) for f in forbidden)

    def test_single_parent_level_yields_no_negatives_but_a_finite_loss(self) -> None:
        """When every cluster shares one parent there is nothing to push away."""
        parents = torch.zeros(4, dtype=torch.long)
        terms = prototype_contrast_terms(
            query=torch.nn.functional.normalize(torch.randn(5, 8), dim=1),
            assignment=torch.randint(0, 4, (5,)),
            centroids=torch.randn(4, 8),
            parents=parents,
            positive_count=2,
            negative_count=3,
        )
        assert not terms.valid[:, 2:].any()
        assert torch.isfinite(masked_bce_with_logits(terms=terms))

    def test_instance_positives_share_the_anchors_cluster(self) -> None:
        """Instance-level positives come from the anchor's own cluster."""
        assignment = torch.tensor([0, 0, 0, 1, 1, 1])
        query = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        key = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        torch.manual_seed(2)
        terms = instance_contrast_terms(
            query=query, key=key, assignment=assignment, positive_count=2, negative_count=2
        )
        # Width is 2*(pos + neg) because each instance contributes both views.
        assert terms.logits.shape == (6, 8)
        assert terms.targets[:, :4].eq(1.0).all()
        assert terms.targets[:, 4:].eq(0.0).all()

    def test_instance_negatives_exclude_the_whole_cluster(self) -> None:
        """A same-cluster instance is never a negative, sampled or not.

        The reference excludes only the positives it drew, leaving unsampled
        cluster members available as negatives of an anchor they share a cluster
        with — the fake negatives the model exists to remove.
        """
        assignment = torch.zeros(5, dtype=torch.long)  # one cluster: no negatives exist
        query = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        key = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        terms = instance_contrast_terms(
            query=query, key=key, assignment=assignment, positive_count=2, negative_count=3
        )
        assert not terms.valid[:, 4:].any()

    def test_sampling_shortfall_is_masked_not_padded(self) -> None:
        """Rows with fewer candidates than requested come back flagged."""
        candidates = torch.tensor([[True, False, False], [True, True, True]])
        _, valid = _sample_from_mask(candidates=candidates, count=3)
        assert valid[0].sum() == 1
        assert valid[1].sum() == 3

    def test_masked_loss_equals_mean_when_all_valid(self) -> None:
        """The validity mask costs nothing in the non-degenerate case."""
        torch.manual_seed(0)
        logits = torch.randn(4, 6)
        targets = torch.zeros(4, 6)
        targets[:, :2] = 1.0
        terms = type(
            "T", (), {"logits": logits, "targets": targets, "valid": torch.ones_like(logits).bool()}
        )()
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        assert masked_bce_with_logits(terms=terms) == pytest.approx(float(expected), abs=1e-6)


# --------------------------------------------------------------------------- #
# Momentum encoder
# --------------------------------------------------------------------------- #


class TestMomentumEncoder:
    """The key branch is an EMA of the query branch, never an optimization target."""

    def test_key_parameters_start_equal_and_frozen(self) -> None:
        """Key parameters are copied from the query branch and require no grad."""
        model = _small_model()
        for online, momentum in model._momentum_module_pairs():
            for param_online, param_momentum in zip(
                online.parameters(), momentum.parameters(), strict=True
            ):
                assert torch.equal(param_online, param_momentum)
                assert not param_momentum.requires_grad

    def test_update_moves_key_toward_query(self) -> None:
        """One EMA step applies the coefficient and closes the gap."""
        model = _small_model(key_momentum=0.9)
        with torch.no_grad():
            for param in model._encoder.parameters():
                param.add_(torch.randn_like(param))

        before = [p.clone() for p in model._key_encoder.parameters()]
        online = [p.clone() for p in model._encoder.parameters()]
        model._momentum_update_key_encoder()

        for old, source, updated in zip(
            before, online, model._key_encoder.parameters(), strict=True
        ):
            assert torch.allclose(updated, 0.9 * old + 0.1 * source, atol=1e-6)
            assert (updated - source).norm() < (old - source).norm()

    def test_key_receives_no_gradient(self) -> None:
        """Backward through the loss leaves the key branch untouched."""
        model = _small_model()
        model._compute_loss(_data()).backward()
        assert all(p.grad is None for p in model._key_encoder.parameters())
        assert any(p.grad is not None for p in model._encoder.parameters())

    def test_optimizer_excludes_frozen_parameters(self) -> None:
        """configure_optimizers passes only trainable parameters to SGD."""
        model = _small_model(use_lr_scheduler=False)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.SGD)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert not any(id(p) in optimized for p in model._key_encoder.parameters())
        assert all(id(p) in optimized for p in model._encoder.parameters())


# --------------------------------------------------------------------------- #
# Feature bank
# --------------------------------------------------------------------------- #


class TestFeatureBank:
    """The bank enlarges the clustering pool without importing stale structure."""

    def test_starts_empty(self) -> None:
        """No rows join the pool before the first enqueue."""
        model = _small_model()
        assert int(model.feature_bank_fill) == 0
        assert model._bank_rows().shape[0] == 0

    def test_disabled_bank_clusters_the_batch_alone(self) -> None:
        """feature_bank_size=0 keeps the pool at exactly 3B rows."""
        model = _small_model(feature_bank_size=0)
        model._compute_loss(_data())
        assert model._bank_rows().shape[0] == 0
        assert int(model.feature_bank_fill) == 0

    def test_enqueue_wraps_and_saturates(self) -> None:
        """The FIFO overwrites the oldest rows once full."""
        model = _small_model(feature_bank_size=8)
        model.train()
        for _ in range(4):
            model._enqueue(torch.randn(6, model._projection_dim))
        assert int(model.feature_bank_fill) == 8
        assert int(model.feature_bank_index) == (24 % 8)

    def test_epoch_cap_bounds_retained_rows(self) -> None:
        """The bank never retains more than one epoch of rows.

        Without the cap a small dataset fills a large bank with copies of the
        same samples drawn from many encoder states.
        """
        model = _small_model(feature_bank_size=1024)
        model.train()
        for _ in range(6):
            model._enqueue(torch.randn(64, model._projection_dim))
        model.rows_last_epoch.fill_(96)
        assert int(model.feature_bank_fill) == 384
        assert model._bank_rows().shape[0] == 96

    def test_validation_does_not_touch_the_bank(self) -> None:
        """Validation must not move training state."""
        model = _small_model()
        model.eval()
        model.validation_step(_data(), 0)
        assert int(model.feature_bank_fill) == 0
        assert int(model.rows_this_epoch) == 0

    def test_epoch_end_records_the_row_count(self) -> None:
        """on_train_epoch_end rolls this epoch's count into the cap."""
        model = _small_model()
        model.train()
        model._enqueue(torch.randn(30, model._projection_dim))
        model.on_train_epoch_end()
        assert int(model.rows_last_epoch) == 30
        assert int(model.rows_this_epoch) == 0

    def test_duplicate_rows_manufacture_spurious_depth(self) -> None:
        """Pin the hazard the epoch cap exists to prevent.

        Replacing distinct rows with copies of fewer samples makes the hierarchy
        *deeper*, not shallower: each sample's copies collapse into a tight
        cluster of their own at the finest partition, adding a level that encodes
        duplication rather than structure. A bank spanning several epochs holds
        exactly such copies — the same samples at different encoder states — so
        "the hierarchy got deeper" is not on its own evidence that the bank
        helped. ``test_epoch_cap_bounds_retained_rows`` covers the guard.
        """
        distinct = _clustered_features(groups=12, per_group=6)
        duplicated = torch.cat([distinct[:24]] * 3)
        assert duplicated.size(0) == distinct.size(0)
        assert finch(duplicated, max_levels=6).num_levels > finch(distinct, max_levels=6).num_levels


# --------------------------------------------------------------------------- #
# Encoder and encoding output
# --------------------------------------------------------------------------- #


class TestEncodingOutput:
    """The encode() contract: rank, width, and gradient flow."""

    def test_is_basic_encoding_mixin(self) -> None:
        """MHCCL uses the fixed-length mixin."""
        assert isinstance(_small_model(), BasicEncodingMixin)

    def test_conforms_to_has_encoder(self) -> None:
        """The .encoder property satisfies the protocol."""
        assert isinstance(_small_model(), HasEncoder)

    def test_supported_outputs(self) -> None:
        """Both shapes are produced natively, so both are declared."""
        assert MHCCL.supported_outputs == frozenset(
            {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
        )

    def test_vector_output_is_two_dimensional(self) -> None:
        """VECTOR returns (B, representation_dim)."""
        model = _small_model()
        result = model.encode_batch(_data())
        assert result.ndim == 2
        assert result.shape == (BATCH, model.representation_dim)

    def test_sequence_output_keeps_a_real_temporal_axis(self) -> None:
        """SEQUENCE returns (B, T', D) with T' > 1 and no fallback warning."""
        model = _small_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.encode_batch(_data(), output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3
        assert result.shape[1] > 1

    def test_encode_returns_backbone_not_projection(self) -> None:
        """encode() bypasses the projection head."""
        model = _small_model()
        assert model.encode_batch(_data()).shape[1] == model.representation_dim
        assert model(_data()).shape[1] == model._projection_dim

    def test_forward_is_unit_norm(self) -> None:
        """Projections are L2-normalized, which the clustering assumes."""
        projections = _small_model()(_data())
        assert torch.allclose(projections.norm(dim=1), torch.ones(BATCH), atol=1e-5)

    def test_encode_batch_preserves_gradients(self) -> None:
        """The adversarial path depends on encode_batch being differentiable."""
        model = _small_model()
        batch = _data().requires_grad_(requires_grad=True)
        assert model.encode_batch(batch).requires_grad

    def test_unsupported_output_raises(self) -> None:
        """An unknown output shape is an error, not a silent fallback."""
        model = _small_model()
        with pytest.raises(ValueError, match="does not support output"):
            model._encode_batch(model.encoder, _data(), output="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize("norm", [NormalizationLayerType.CHANNEL, NormalizationLayerType.BATCH])
    def test_both_normalization_types_run(self, norm: NormalizationLayerType) -> None:
        """CHANNEL (default) and BATCH (the reference's choice) both work."""
        model = _small_model(normalization_layer_type=norm)
        assert torch.isfinite(model.encode_batch(_data())).all()


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #


class TestNaNHandling:
    """Padded batches must never produce NaN representations or losses."""

    def test_trailing_nan_padding(self) -> None:
        """Partially padded samples encode finitely."""
        batch = _data()
        batch[0, 30:] = float("nan")
        batch[1, 40:] = float("nan")
        assert torch.isfinite(_small_model().encode_batch(batch)).all()

    def test_all_nan_sample(self) -> None:
        """A wholly padded sample does not poison the batch."""
        batch = _data()
        batch[2] = float("nan")
        assert torch.isfinite(_small_model().encode_batch(batch)).all()

    def test_clean_input_regression(self) -> None:
        """Clean input is unaffected by the NaN guard."""
        assert torch.isfinite(_small_model().encode_batch(_data())).all()

    def test_training_step_with_padding(self) -> None:
        """A NaN row would otherwise corrupt cdist and the whole partition."""
        batch = _data()
        batch[0, 25:] = float("nan")
        loss = _small_model().training_step(batch, 0)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Input length robustness
# --------------------------------------------------------------------------- #


class TestInputLengthRobustness:
    """One instance must serve every length it is asked for."""

    def test_one_instance_encodes_several_lengths(self) -> None:
        """The reuse pattern that breaks fixed-length ports."""
        model = _small_model()
        for length in (16, TIME, 137, 400):
            result = model.encode_batch(_data(time=length))
            assert result.shape == (BATCH, model.representation_dim)

    def test_gradient_flows_at_a_non_nominal_length(self) -> None:
        """encode_batch stays differentiable away from the nominal length."""
        model = _small_model()
        batch = _data(time=137).requires_grad_(requires_grad=True)
        model.encode_batch(batch).sum().backward()
        assert batch.grad is not None

    def test_long_input_training_step(self) -> None:
        """A batch many times the nominal length trains without exhausting memory."""
        loss = _small_model().training_step(_data(batch=4, time=1000), 0)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_max_train_length_crops_training_only(self) -> None:
        """Cropping is a training-path concern; encode() must never crop."""
        model = _small_model(max_train_length=64)
        assert torch.isfinite(model.training_step(_data(time=400), 0))
        assert model.encode_batch(_data(time=400)).shape[0] == BATCH

    def test_max_train_length_defaults_to_none(self) -> None:
        """Adding the parameter cannot perturb a working configuration."""
        assert MHCCLModelParameters(input_dim=1).max_train_length is None


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


class TestAugmentation:
    """The producer contract and the reference's constants."""

    def test_default_producer_returns_a_view_pair(self) -> None:
        """The default is a weak/strong RolePairProducer."""
        pair = _default_mhccl_pair().produce(_data())
        assert isinstance(pair, ViewPair)
        assert pair.first.shape == pair.second.shape == (BATCH, TIME, CHANNELS)

    def test_default_constants_match_the_reference(self) -> None:
        """scaling(sigma=1.1, mean=2.0) and permutation(max_segments=8) + jitter(0.8)."""
        producer = _default_mhccl_pair()
        scaling = producer._first._params  # type: ignore[attr-defined]
        assert scaling.sigma == pytest.approx(1.1)
        assert scaling.mean == pytest.approx(2.0)
        assert scaling.channel_dim == -1
        permutation, jitter = producer._second._augmentations  # type: ignore[attr-defined]
        assert permutation._params.max_segments == 8
        assert permutation._params.time_dim == 1
        assert jitter._params.sigma == pytest.approx(0.8)

    def test_views_differ(self) -> None:
        """Two identical views would give the loss nothing to discriminate."""
        pair = _default_mhccl_pair().produce(_data())
        assert not torch.allclose(pair.first, pair.second)

    def test_injected_producer_is_used(self) -> None:
        """A custom producer is stored verbatim and called once per step."""
        calls = 0
        inner = IndependentPairProducer(aug=Jitter())

        class CountingProducer:
            def produce(self, x: torch.Tensor) -> ViewPair:
                nonlocal calls
                calls += 1
                return inner.produce(x)

        model = _small_model(augmentation=CountingProducer())
        model.training_step(_data(), 0)
        assert calls == 1

    def test_accepts_an_aligned_pair_producer(self) -> None:
        """AlignedPair satisfies ViewPair, so covariance must hold."""
        producer: AugmentationProducer[ViewPair] = FullOverlapProducer(aug=Jitter())
        assert torch.isfinite(_small_model(augmentation=producer).training_step(_data(), 0))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class TestTraining:
    """The Lightning path end to end."""

    @pytest.mark.parametrize("with_labels", [False, True])
    def test_trains_with_finite_losses(self, with_labels: bool) -> None:
        """Bare-tensor and (data, labels) batches both train."""
        pl.seed_everything(0, workers=True)
        model = _small_model()
        tensors = (torch.randn(48, TIME, CHANNELS),)
        if with_labels:
            tensors = (*tensors, torch.randint(0, 3, (48,)))
        loader = DataLoader(TensorDataset(*tensors), batch_size=BATCH)
        trainer = pl.Trainer(
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
        )
        trainer.fit(model, loader)
        assert torch.isfinite(torch.tensor(trainer.callback_metrics["train_loss"].item()))

    def test_loss_decreases_on_a_fixed_batch(self) -> None:
        """The objective is optimizable, not merely finite.

        Kept short (a tiny backbone, 40 steps) so it stays a unit test.
        """
        pl.seed_everything(0, workers=True)
        model = _small_model(feature_bank_size=0)
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.05, momentum=0.9
        )
        batch = _data(batch=24)
        model.train()
        losses = []
        for _ in range(40):
            optimizer.zero_grad()
            loss = model._compute_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

    def test_validation_step_runs(self) -> None:
        """Validation produces a finite loss without moving training state."""
        model = _small_model()
        assert torch.isfinite(model.validation_step(_data(), 0))

    def test_optimizer_hyperparameters_match_the_reference(self) -> None:
        """SGD with the reference's learning rate, momentum, and weight decay."""
        model = _small_model(use_lr_scheduler=False)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.SGD)
        group = optimizer.param_groups[0]
        assert group["lr"] == pytest.approx(0.03)
        assert group["momentum"] == pytest.approx(0.9)
        assert group["weight_decay"] == pytest.approx(1e-4)

    def test_step_schedule_branch(self) -> None:
        """lr_step_milestones reproduces the reference's non---cos branch."""
        model = _small_model(lr_step_milestones=(2, 4))
        config = model.configure_optimizers()
        assert isinstance(config, dict)
        optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]
        rates = []
        for _ in range(6):
            rates.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        assert rates[1] == pytest.approx(rates[0])
        assert rates[2] == pytest.approx(rates[0] * 0.1)
        assert rates[4] == pytest.approx(rates[0] * 0.01)

    @pytest.mark.parametrize("max_epochs", [None, -1, 0])
    def test_no_horizon_returns_a_bare_optimizer(self, max_epochs: int | None) -> None:
        """Cosine annealing needs a horizon; without one the rate stays constant."""
        model = _small_model()
        model._trainer = SimpleNamespace(max_epochs=max_epochs)  # type: ignore[assignment]
        assert isinstance(model.configure_optimizers(), torch.optim.SGD)

    def test_scheduler_can_be_disabled(self) -> None:
        """use_lr_scheduler=False returns the optimizer alone."""
        assert isinstance(
            _small_model(use_lr_scheduler=False).configure_optimizers(), torch.optim.SGD
        )


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


class TestDegradation:
    """What happens when the batch cannot supply what the config asks for."""

    def test_shallow_hierarchy_still_trains(self) -> None:
        """Fewer partitions than requested reduces the term, it does not break it."""
        model = _small_model(hierarchy_levels=8, feature_bank_size=0)
        assert torch.isfinite(model.training_step(_data(batch=8), 0))

    def test_shallow_hierarchy_warns_once_after_the_first_epoch(self) -> None:
        """The warning is held back while the bank is still filling."""
        model = _small_model(hierarchy_levels=8, feature_bank_size=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.training_step(_data(batch=8), 0)
        assert not [w for w in caught if "hierarchy levels" in str(w.message)]

        model.rows_last_epoch.fill_(24)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.training_step(_data(batch=8), 0)
            model.training_step(_data(batch=8), 0)
        assert len([w for w in caught if "hierarchy levels" in str(w.message)]) == 1

    def test_over_requested_negatives_are_clamped(self) -> None:
        """Asking for more negatives than the batch holds must not raise.

        The reference calls random.sample for a fixed count and raises here.
        """
        model = _small_model(negative_instance_count=500, negative_prototype_count=500)
        assert torch.isfinite(model.training_step(_data(batch=8), 0))

    def test_no_usable_level_without_instance_loss_raises(self) -> None:
        """A configuration with nothing to train on fails loudly."""
        model = _small_model(use_instance_loss=False, feature_bank_size=0)
        with pytest.raises(RuntimeError, match="no trainable term"):
            # Identical rows collapse to a single partition, leaving no parent.
            model._compute_loss(torch.ones(6, TIME, CHANNELS))

    def test_prototype_only_configuration_trains(self) -> None:
        """use_instance_loss=False reproduces the reference's --protoNCE_only."""
        model = _small_model(use_instance_loss=False)
        assert torch.isfinite(model.training_step(_data(batch=24), 0))


# --------------------------------------------------------------------------- #
# Singleton batch
# --------------------------------------------------------------------------- #


class TestSingletonBatch:
    """B == 1 has no negatives until the series is split into windows."""

    def test_singleton_batch_produces_gradients(self) -> None:
        """Splitting restores a real training signal at batch_size=1."""
        model = _small_model()
        loss = model._compute_loss(_data(batch=1, time=300))
        loss.backward()
        assert torch.isfinite(loss)
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model._encoder.parameters()
        )

    def test_split_is_a_no_op_above_one(self) -> None:
        """Multi-sample batches are untouched by the singleton guard."""
        model = _small_model()
        assert torch.isfinite(model._compute_loss(_data(batch=4)))

    def test_short_singleton_is_left_alone(self) -> None:
        """A series too short to split is passed through rather than mangled."""
        model = _small_model()
        assert torch.isfinite(model._compute_loss(_data(batch=1, time=12)))
