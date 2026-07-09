"""Phase 12 UAT: Standardize representation_dim across all models.

Verifies that the dimension naming unification (phase 12) correctly:
- Exposes representation_dim on all 10 models
- Returns the encode() output feature dim (not flattened T*D)
- Fixes TST to return hidden_dim (not hidden_dim * seq_len)
- Fixes TST supervised factory to compute in_features inline
- Preserves latent_dim (TimeVAE) and layers (RecurrentAE)
- Does NOT introduce encoding_output_dim
- Renames all *_dims -> *_dim in config dataclasses
"""

from __future__ import annotations

import torch

from chronocratic.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from chronocratic.models.convolutional.dilated.autotcl.model import AutoTCL
from chronocratic.models.convolutional.dilated.cost.config import CoSTModelParameters
from chronocratic.models.convolutional.dilated.cost.model import CoST
from chronocratic.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec
from chronocratic.models.convolutional.standard.mcl.config import MCLModelParameters
from chronocratic.models.convolutional.standard.mcl.model import MCL
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.convolutional.standard.tstcc.config import TSTCCModelParameters
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.generative.timevae.config import TimeVAEModelParameters
from chronocratic.models.generative.timevae.model import TimeVAE
from chronocratic.models.recurrent.recurrentae.config import RecurrentAutoEncoderModelParameters
from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder
from chronocratic.models.recurrent.timenet.config import TimeNetModelParameters
from chronocratic.models.recurrent.timenet.model import TimeNet
from chronocratic.models.supervised import RepresentationBackbone, make_tst_supervised
from chronocratic.models.transformer.tst.config import TSTModelParameters
from chronocratic.models.transformer.tst.model import TST


# ---------------------------------------------------------------------------
# All 10 models expose representation_dim
# ---------------------------------------------------------------------------


class TestAllModelsExposeRepresentationDim:
    """D-03: All 10 models must expose a representation_dim property."""

    def test_mcl(self) -> None:
        model = MCL(MCLModelParameters(input_dim=1))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_tstcc(self) -> None:
        model = TSTCC(
            input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16
        )
        assert hasattr(model, "representation_dim")
        assert model.representation_dim == 16

    def test_ts2vec(self) -> None:
        model = TS2Vec(TS2VecModelParameters(input_dim=1))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_autotcl(self) -> None:
        model = AutoTCL(AutoTCLModelParameters(input_dim=1))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_cost(self) -> None:
        model = CoST(CoSTModelParameters(input_dim=1, sequence_length=100))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_series2vec(self) -> None:
        model = Series2Vec(
            input_dim=2,
            embedding_dim=8,
            num_heads=2,
            feedforward_dim=16,
            representation_dim=4,
            dropout_rate=0.1,
        )
        assert hasattr(model, "representation_dim")
        assert model.representation_dim == 4

    def test_tst(self) -> None:
        model = TST(TSTModelParameters(input_dim=1, sequence_length=100))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_timenet(self) -> None:
        model = TimeNet(TimeNetModelParameters(input_dim=1))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_timevae(self) -> None:
        model = TimeVAE(TimeVAEModelParameters(input_dim=1))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_recurrent_autoencoder(self) -> None:
        model = RecurrentAutoEncoder(
            RecurrentAutoEncoderModelParameters(input_dim=1)
        )
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0


# ---------------------------------------------------------------------------
# representation_dim matches encode() output feature dim
# ---------------------------------------------------------------------------


class TestRepresentationDimMatchesEncodeOutput:
    """D-03: representation_dim == last axis of encode() output."""

    def test_mcl_encode_output_matches(self) -> None:
        model = MCL(MCLModelParameters(input_dim=1))
        x = torch.randn(2, 100, 1)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_tstcc_encode_output_matches(self) -> None:
        model = TSTCC(
            input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16
        )
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_ts2vec_encode_output_matches(self) -> None:
        model = TS2Vec(TS2VecModelParameters(input_dim=1))
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_autotcl_encode_output_matches(self) -> None:
        model = AutoTCL(AutoTCLModelParameters(input_dim=1))
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_cost_encode_output_matches(self) -> None:
        model = CoST(CoSTModelParameters(input_dim=1, sequence_length=100))
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        # CoST: encode() output width == representation_dim (full, not 2x)
        assert reps.shape[-1] == model.representation_dim

    def test_series2vec_encode_output_matches(self) -> None:
        model = Series2Vec(
            input_dim=2,
            embedding_dim=8,
            num_heads=2,
            feedforward_dim=16,
            representation_dim=4,
            dropout_rate=0.1,
        )
        x = torch.randn(2, 20, 2)
        reps = model.network.encode(x)
        assert reps.shape[1] == model.representation_dim

    def test_timenet_encode_output_matches(self) -> None:
        model = TimeNet(TimeNetModelParameters(input_dim=1))
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_timevae_encode_output_matches(self) -> None:
        model = TimeVAE(TimeVAEModelParameters(input_dim=1))
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_recurrent_autoencoder_encode_output_matches(self) -> None:
        model = RecurrentAutoEncoder(
            RecurrentAutoEncoderModelParameters(input_dim=1)
        )
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim


# ---------------------------------------------------------------------------
# TST fix: representation_dim = hidden_dim (NOT hidden_dim * seq_len)
# ---------------------------------------------------------------------------


class TestTSTRepresentationDimFix:
    """D-03, D-06: TST property returns hidden_dim, factory computes flatten inline."""

    def test_tst_representation_dim_is_hidden_dim(self) -> None:
        model = TST(
            input_dim=2,
            sequence_length=10,
            hidden_dim=8,
            num_heads=2,
            depth=1,
        )
        assert model.representation_dim == 8
        # Must NOT be hidden_dim * sequence_length
        assert model.representation_dim != 80

    def test_tst_supervised_factory_in_features(self) -> None:
        backbone = TST(
            input_dim=2,
            sequence_length=10,
            hidden_dim=8,
            num_heads=2,
            depth=1,
        )
        module = make_tst_supervised(
            backbone=backbone,
            num_outputs=3,
            task="classification",
            freeze_backbone=False,
        )
        expected = backbone.representation_dim * backbone.encoder.sequence_length
        assert module._head._fc.in_features == expected

    def test_tst_supervised_forward_shape(self) -> None:
        backbone = TST(
            input_dim=2,
            sequence_length=10,
            hidden_dim=8,
            num_heads=2,
            depth=1,
        )
        module = make_tst_supervised(
            backbone=backbone,
            num_outputs=3,
            task="classification",
            freeze_backbone=False,
        )
        x = torch.randn(2, 10, 2)
        padding_masks = torch.ones(2, 10, dtype=torch.bool)
        out = module(x, padding_masks)
        assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# CoST: encode width == representation_dim (full, not 2x)
# ---------------------------------------------------------------------------


class TestCoSTAlignment:
    """D-05: CoST representation_dim is full encode width, not 2x component_dim."""

    def test_cost_encode_width_equals_representation_dim(self) -> None:
        model = CoST(
            CoSTModelParameters(input_dim=1, sequence_length=100, representation_dim=16)
        )
        x = torch.randn(2, 1, 100)
        reps = model.encode(x)
        assert reps.shape[-1] == model.representation_dim

    def test_cost_representation_dim_is_even(self) -> None:
        model = CoST(
            CoSTModelParameters(input_dim=1, sequence_length=100, representation_dim=16)
        )
        assert model.representation_dim % 2 == 0


# ---------------------------------------------------------------------------
# Preserved names: latent_dim and layers
# ---------------------------------------------------------------------------


class TestPreservedNames:
    """D-04: latent_dim (TimeVAE) and layers (RecurrentAE) remain unchanged."""

    def test_timevae_latent_dim_preserved(self) -> None:
        model = TimeVAE(TimeVAEModelParameters(input_dim=1, latent_dim=16))
        assert model.latent_dim == 16
        assert model.representation_dim == 16

    def test_recurrent_autoencoder_layers_preserved(self) -> None:
        model = RecurrentAutoEncoder(
            RecurrentAutoEncoderModelParameters(input_dim=1, layers=(16, 8))
        )
        assert model.layers == (16, 8)
        assert model.representation_dim == 8


# ---------------------------------------------------------------------------
# encoding_output_dim NOT introduced
# ---------------------------------------------------------------------------


class TestNoEncodingOutputDim:
    """Spec section 8: encoding_output_dim must NOT exist."""

    def test_no_encoding_output_dim(self) -> None:
        models = [
            MCL(MCLModelParameters(input_dim=1)),
            TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16),
            TS2Vec(TS2VecModelParameters(input_dim=1)),
            AutoTCL(AutoTCLModelParameters(input_dim=1)),
            CoST(CoSTModelParameters(input_dim=1, sequence_length=100)),
            Series2Vec(
                input_dim=2,
                embedding_dim=8,
                num_heads=2,
                feedforward_dim=16,
                representation_dim=4,
                dropout_rate=0.1,
            ),
            TST(TSTModelParameters(input_dim=1, sequence_length=100)),
            TimeNet(TimeNetModelParameters(input_dim=1)),
            TimeVAE(TimeVAEModelParameters(input_dim=1)),
            RecurrentAutoEncoder(
                RecurrentAutoEncoderModelParameters(input_dim=1)
            ),
        ]
        for m in models:
            assert not hasattr(m, "encoding_output_dim"), (
                f"{m.__class__.__name__} must NOT have encoding_output_dim"
            )


# ---------------------------------------------------------------------------
# Config dataclasses use singular names
# ---------------------------------------------------------------------------


class TestConfigFieldNames:
    """D-01: Config dataclasses must use singular *_dim names."""

    def test_mcl_config_fields(self) -> None:
        fields = MCLModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "representation_dim" in fields
        assert "input_dims" not in fields
        assert "output_dims" not in fields

    def test_ts2vec_config_fields(self) -> None:
        fields = TS2VecModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "representation_dim" in fields
        assert "input_dims" not in fields
        assert "output_dims" not in fields

    def test_autotcl_config_fields(self) -> None:
        fields = AutoTCLModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "representation_dim" in fields
        assert "input_dims" not in fields
        assert "output_dims" not in fields

    def test_cost_config_fields(self) -> None:
        fields = CoSTModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "representation_dim" in fields
        assert "input_dims" not in fields
        assert "output_dims" not in fields

    def test_tst_config_fields(self) -> None:
        fields = TSTModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "hidden_dim" in fields
        assert "input_dims" not in fields
        assert "hidden_dims" not in fields

    def test_timenet_config_fields(self) -> None:
        fields = TimeNetModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "hidden_dim" in fields
        assert "input_dims" not in fields
        assert "hidden_dims" not in fields

    def test_timevae_config_fields(self) -> None:
        fields = TimeVAEModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "latent_dim" in fields  # preserved
        assert "input_dims" not in fields

    def test_recurrent_autoencoder_config_fields(self) -> None:
        fields = RecurrentAutoEncoderModelParameters.__dataclass_fields__
        assert "input_dim" in fields
        assert "layers" in fields  # preserved
        assert "input_dims" not in fields


# ---------------------------------------------------------------------------
# RepresentationBackbone protocol
# ---------------------------------------------------------------------------


class TestRepresentationBackboneProtocol:
    """All 10 models satisfy RepresentationBackbone."""

    def test_all_are_representation_backbones(self) -> None:
        models = [
            MCL(MCLModelParameters(input_dim=1)),
            TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16),
            TS2Vec(TS2VecModelParameters(input_dim=1)),
            AutoTCL(AutoTCLModelParameters(input_dim=1)),
            CoST(CoSTModelParameters(input_dim=1, sequence_length=100)),
            Series2Vec(
                input_dim=2,
                embedding_dim=8,
                num_heads=2,
                feedforward_dim=16,
                representation_dim=4,
                dropout_rate=0.1,
            ),
            TST(TSTModelParameters(input_dim=1, sequence_length=100)),
            TimeNet(TimeNetModelParameters(input_dim=1)),
            TimeVAE(TimeVAEModelParameters(input_dim=1)),
            RecurrentAutoEncoder(
                RecurrentAutoEncoderModelParameters(input_dim=1)
            ),
        ]
        for m in models:
            assert isinstance(m, RepresentationBackbone), (
                f"{m.__class__.__name__} must satisfy RepresentationBackbone"
            )
