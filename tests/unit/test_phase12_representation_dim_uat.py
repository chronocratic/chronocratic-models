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

import re
from pathlib import Path

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
import pytest
from chronocratic.models.supervised import RepresentationBackbone, make_tst_supervised
from chronocratic.models.transformer.tst.config import TSTModelParameters
from chronocratic.models.transformer.tst.model import TST


# ---------------------------------------------------------------------------
# All 10 models expose representation_dim
# ---------------------------------------------------------------------------


class TestAllModelsExposeRepresentationDim:
    """D-03: All 10 models must expose a representation_dim property."""

    def test_mcl(self) -> None:
        model = MCL(**vars(MCLModelParameters(input_dim=1)))
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
        model = TS2Vec(**vars(TS2VecModelParameters(input_dim=1)))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_autotcl(self) -> None:
        model = AutoTCL(**vars(AutoTCLModelParameters(input_dim=1)))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_cost(self) -> None:
        model = CoST(**vars(CoSTModelParameters(input_dim=1, sequence_length=100)))
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
        model = TST(**vars(TSTModelParameters(input_dim=1, sequence_length=100)))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_timenet(self) -> None:
        model = TimeNet(**vars(TimeNetModelParameters(input_dim=1)))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_timevae(self) -> None:
        model = TimeVAE(**vars(TimeVAEModelParameters(input_dim=1, sequence_length=32)))
        assert hasattr(model, "representation_dim")
        assert isinstance(model.representation_dim, int)
        assert model.representation_dim > 0

    def test_recurrent_autoencoder(self) -> None:
        model = RecurrentAutoEncoder(
            **vars(RecurrentAutoEncoderModelParameters(input_dim=1))
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
        model = MCL(**vars(MCLModelParameters(input_dim=1)))
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2)
        assert reps.shape[-1] == model.representation_dim

    def test_tstcc_encode_output_matches(self) -> None:
        model = TSTCC(
            input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16
        )
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2)
        assert reps.shape[-1] == model.representation_dim

    def test_ts2vec_encode_output_matches(self) -> None:
        model = TS2Vec(**vars(TS2VecModelParameters(input_dim=1)))
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2, num_workers=0)
        assert reps.shape[-1] == model.representation_dim

    def test_autotcl_encode_output_matches(self) -> None:
        model = AutoTCL(**vars(AutoTCLModelParameters(input_dim=1)))
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2, num_workers=0)
        assert reps.shape[-1] == model.representation_dim

    def test_cost_encode_output_matches(self) -> None:
        model = CoST(**vars(CoSTModelParameters(input_dim=1, sequence_length=100)))
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2, num_workers=0)
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
        model = TimeNet(**vars(TimeNetModelParameters(input_dim=1)))
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2)
        assert reps.shape[-1] == model.representation_dim

    def test_timevae_encode_output_matches(self) -> None:
        model = TimeVAE(**vars(TimeVAEModelParameters(input_dim=1, sequence_length=32)))
        x = torch.randn(2, 32, 1)
        reps = model.encode(x, batch_size=2)
        assert reps.shape[-1] == model.representation_dim

    def test_recurrent_autoencoder_encode_output_matches(self) -> None:
        model = RecurrentAutoEncoder(
            **vars(RecurrentAutoEncoderModelParameters(input_dim=1))
        )
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2)
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
            **vars(CoSTModelParameters(input_dim=1, sequence_length=100, representation_dim=16))
        )
        x = torch.randn(2, 100, 1)
        reps = model.encode(x, batch_size=2, num_workers=0)
        assert reps.shape[-1] == model.representation_dim

    def test_cost_representation_dim_is_even(self) -> None:
        model = CoST(
            **vars(CoSTModelParameters(input_dim=1, sequence_length=100, representation_dim=16))
        )
        assert model.representation_dim % 2 == 0


# ---------------------------------------------------------------------------
# Preserved names: latent_dim and layers
# ---------------------------------------------------------------------------


class TestPreservedNames:
    """D-04: latent_dim (TimeVAE) and layers (RecurrentAE) remain unchanged."""

    def test_timevae_latent_dim_preserved(self) -> None:
        model = TimeVAE(**vars(TimeVAEModelParameters(input_dim=1, sequence_length=32, latent_dim=16)))
        assert model.latent_dim == 16
        assert model.representation_dim == 16

    def test_recurrent_autoencoder_layers_preserved(self) -> None:
        model = RecurrentAutoEncoder(
            **vars(RecurrentAutoEncoderModelParameters(input_dim=1, layers=(16, 8)))
        )
        assert model._layers == (16, 8)
        assert model.representation_dim == 8


# ---------------------------------------------------------------------------
# encoding_output_dim NOT introduced
# ---------------------------------------------------------------------------


class TestNoEncodingOutputDim:
    """Spec section 8: encoding_output_dim must NOT exist."""

    def test_no_encoding_output_dim(self) -> None:
        models = [
            MCL(**vars(MCLModelParameters(input_dim=1))),
            TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16),
            TS2Vec(**vars(TS2VecModelParameters(input_dim=1))),
            AutoTCL(**vars(AutoTCLModelParameters(input_dim=1))),
            CoST(**vars(CoSTModelParameters(input_dim=1, sequence_length=100))),
            Series2Vec(
                input_dim=2,
                embedding_dim=8,
                num_heads=2,
                feedforward_dim=16,
                representation_dim=4,
                dropout_rate=0.1,
            ),
            TST(**vars(TSTModelParameters(input_dim=1, sequence_length=100))),
            TimeNet(**vars(TimeNetModelParameters(input_dim=1))),
            TimeVAE(**vars(TimeVAEModelParameters(input_dim=1, sequence_length=32))),
            RecurrentAutoEncoder(
                **vars(RecurrentAutoEncoderModelParameters(input_dim=1))
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
            MCL(**vars(MCLModelParameters(input_dim=1))),
            TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16),
            TS2Vec(**vars(TS2VecModelParameters(input_dim=1))),
            AutoTCL(**vars(AutoTCLModelParameters(input_dim=1))),
            CoST(**vars(CoSTModelParameters(input_dim=1, sequence_length=100))),
            Series2Vec(
                input_dim=2,
                embedding_dim=8,
                num_heads=2,
                feedforward_dim=16,
                representation_dim=4,
                dropout_rate=0.1,
            ),
            TST(**vars(TSTModelParameters(input_dim=1, sequence_length=100))),
            TimeNet(**vars(TimeNetModelParameters(input_dim=1))),
            TimeVAE(**vars(TimeVAEModelParameters(input_dim=1, sequence_length=32))),
            RecurrentAutoEncoder(
                **vars(RecurrentAutoEncoderModelParameters(input_dim=1))
            ),
        ]
        for m in models:
            assert isinstance(m, RepresentationBackbone), (
                f"{m.__class__.__name__} must satisfy RepresentationBackbone"
            )


# ---------------------------------------------------------------------------
# Protocol docstring: lists all 10 models
# ---------------------------------------------------------------------------


class TestRepresentationBackboneDocstring:
    """Plan 12: RepresentationBackbone docstring lists all 10 implementers."""

    @pytest.fixture
    def docstring(self) -> str:
        return RepresentationBackbone.__doc__ or ""

    def test_lists_all_ten_models(self, docstring: str) -> None:
        """Protocol docstring should reference all 10 model names."""
        expected = {"TST", "Series2Vec", "TSTCC", "MCL", "TS2Vec", "AutoTCL",
                     "CoST", "TimeNet", "TimeVAE", "RecurrentAutoEncoder"}
        found = {name for name in expected if name in docstring}
        missing = expected - found
        assert not missing, f"Missing models in RepresentationBackbone docstring: {missing}"

    def test_defines_representation_dim_as_encode_output(self, docstring: str) -> None:
        """Protocol docstring should define representation_dim as feature dim of encode() output."""
        lower = docstring.lower()
        assert "encode" in lower, "Docstring should reference encode()"
        assert "representation_dim" in docstring, "Docstring should mention representation_dim"


# ---------------------------------------------------------------------------
# supervised/__init__.py example uses singular names
# ---------------------------------------------------------------------------


class TestSupervisedInitDocstring:
    """Plan 12: supervised package docstring example uses singular param names."""

    @pytest.fixture
    def docstring(self) -> str:
        from chronocratic.models import supervised
        return supervised.__doc__ or ""

    def test_example_uses_singular_names(self, docstring: str) -> None:
        """Example code should use input_dim, hidden_dim (not plural)."""
        assert "input_dim" in docstring
        assert "hidden_dim" in docstring
        # Should NOT have the old plural forms in example code
        assert "input_dims" not in docstring, "Example should not use input_dims"
        assert "hidden_dims" not in docstring, "Example should not use hidden_dims"


# ---------------------------------------------------------------------------
# CONTRIBUTING.md canonical names table
# ---------------------------------------------------------------------------


class TestContributingMd:
    """Plan 12: CONTRIBUTING.md uses singular canonical names."""

    @pytest.fixture
    def content(self) -> str:
        return (Path(__file__).resolve().parents[2] / "docs" / "CONTRIBUTING.md").read_text()

    def test_canonical_names_include_representation_dim(self, content: str) -> None:
        """Canonical names table should include representation_dim."""
        assert "representation_dim" in content

    def test_canonical_names_are_singular(self, content: str) -> None:
        """Canonical names table should use singular forms."""
        # Check key singular names appear
        assert "input_dim" in content
        assert "hidden_dim" in content
        assert "feedforward_dim" in content

    def test_example_code_uses_singular_params(self, content: str) -> None:
        """Example config/model code should use singular param names."""
        code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
        for block in code_blocks:
            lines = block.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Param names in code should be singular
                if re.search(r"\b(input_dims|hidden_dims|feedforward_dims|output_dims)\b", stripped):
                    pytest.fail(
                        f"Example code uses plural param name: {stripped}"
                    )

    def test_has_kw_only_convention_documented(self, content: str) -> None:
        """CONTRIBUTING.md should document the keyword-only signature convention."""
        lower = content.lower()
        assert "keyword" in lower and "only" in lower, (
            "CONTRIBUTING.md should document kw-only convention"
        )


# ---------------------------------------------------------------------------
# CHANGELOG has breaking rename entry
# ---------------------------------------------------------------------------


class TestChangelog:
    """Plan 12: CHANGELOG has v0.1.0a13 breaking rename entry."""

    @pytest.fixture
    def content(self) -> str:
        return (Path(__file__).resolve().parents[2] / "CHANGELOG.md").read_text()

    def test_v01a13_entry_exists(self, content: str) -> None:
        """CHANGELOG should have v0.1.0a13 entry."""
        assert "v0.1.0a13" in content

    def test_breaking_rename_documented(self, content: str) -> None:
        """CHANGELOG a13 entry should document dimension parameter renaming."""
        # Find the a13 section
        assert "v0.1.0a13" in content
        # Should reference the rename
        lines = content.split("\n")
        in_a13 = False
        a13_section = []
        for line in lines:
            if "v0.1.0a13" in line:
                in_a13 = True
                continue
            if in_a13 and line.startswith("## v"):
                break
            if in_a13:
                a13_section.append(line)

        section_text = "\n".join(a13_section)
        assert "rename" in section_text.lower() or "singular" in section_text.lower(), (
            f"a13 changelog should mention rename: {section_text[:200]}"
        )


# ---------------------------------------------------------------------------
# Towncrier fragment
# ---------------------------------------------------------------------------


class TestTowncrierFragment:
    """Plan 12: Towncrier fragment exists for this phase."""

    def test_fragment_exists(self) -> None:
        """changelog.d/12.changed.md should exist."""
        fragment = Path(__file__).resolve().parents[2] / "changelog.d" / "12.changed.md"
        assert fragment.exists(), f"Towncrier fragment not found at {fragment}"

    def test_fragment_has_content(self) -> None:
        """Fragment should describe the dimension parameter naming changes."""
        fragment = Path(__file__).resolve().parents[2] / "changelog.d" / "12.changed.md"
        content = fragment.read_text()
        assert len(content.strip()) > 0
        assert "dimension" in content.lower() or "rename" in content.lower() or "singular" in content.lower()


# ---------------------------------------------------------------------------
# Shared test files use singular param names
# ---------------------------------------------------------------------------


class TestSharedTestFilesUseSingular:
    """Plan 12: Shared test files should use singular param names."""

    @pytest.mark.parametrize("test_file", [
        "test_from_config.py",
        "test_backbone_representation_dim.py",
        "test_encoder_decoder_contract.py",
        "test_smoke.py",
        "test_supervised_package.py",
        "test_tstcc_supervised.py",
    ])
    def test_file_uses_singular_params(self, test_file: str) -> None:
        """Test file should use singular param names for model construction."""
        filepath = Path(__file__).resolve().parent / test_file
        content = filepath.read_text()
        # Check for old plural param names that should have been renamed
        # Exclude encoder_channels, encoder_kernels, encoder_dilations
        plural_names = {
            "input_dims", "hidden_dims", "output_dims", "representation_dims",
            "feedforward_dims", "embedding_dims", "projection_dims"
        }
        violations = []
        for match in re.finditer(r"\b(" + "|".join(plural_names) + r")\b", content):
            line_num = content[:match.start()].count("\n") + 1
            line = content.split("\n")[line_num - 1]
            # Allow in comments about old names (migration breadcrumbs)
            if line.strip().startswith("#"):
                continue
            violations.append(f"  Line {line_num}: {match.group()} in: {line.strip()}")

        assert not violations, (
            f"{test_file}: found plural param names that should be singular:\n" + "\n".join(violations)
        )
