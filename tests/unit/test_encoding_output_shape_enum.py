"""Tests for EncodingOutputShape enum and its re-exports.

Verifies the enum values, member count, import-cycle-safety, and that
the symbol is importable from both the enum module and the top-level
models package.
"""

from __future__ import annotations

import ast
import enum
from pathlib import Path

import pytest


class TestEncodingOutputShapeValues:
    """Enum members have the expected string values."""

    def test_vector_value(self) -> None:
        from chronocratic.models.enums.encoding import EncodingOutputShape

        assert EncodingOutputShape.VECTOR.value == "vector"

    def test_sequence_value(self) -> None:
        from chronocratic.models.enums.encoding import EncodingOutputShape

        assert EncodingOutputShape.SEQUENCE.value == "sequence"


class TestEncodingOutputShapeMembers:
    """Enum has exactly two members."""

    def test_member_count(self) -> None:
        from chronocratic.models.enums.encoding import EncodingOutputShape

        assert len(EncodingOutputShape) == 2

    def test_member_names(self) -> None:
        from chronocratic.models.enums.encoding import EncodingOutputShape

        names = [m.name for m in EncodingOutputShape]
        assert names == ["VECTOR", "SEQUENCE"]


class TestEncodingOutputShapeIsEnum:
    """EncodingOutputShape inherits from enum.Enum."""

    def test_is_subclass_of_enum(self) -> None:
        from chronocratic.models.enums.encoding import EncodingOutputShape

        assert issubclass(EncodingOutputShape, enum.Enum)


class TestEncodingOutputShapeNoLocalImports:
    """Enum module depends only on stdlib (import-cycle-safe per D-01)."""

    def test_no_chronocratic_imports(self) -> None:
        """enums/encoding.py must not import from chronocratic."""
        module_path = (
            Path(__file__).parents[2] / "src" / "chronocratic" / "models" / "enums" / "encoding.py"
        )
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        chronocratic_imports = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("chronocratic")
            ):
                chronocratic_imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("chronocratic"):
                        chronocratic_imports.append(alias.name)

        assert chronocratic_imports == [], (
            f"encoding.py must not import from chronocratic (found: {chronocratic_imports})"
        )


class TestEnumsInitReExport:
    """enums/__init__.py exports EncodingOutputShape via __all__."""

    def test_all_contains_encoding_output_shape(self) -> None:
        import chronocratic.models.enums as enums_pkg

        assert "EncodingOutputShape" in getattr(enums_pkg, "__all__", [])

    def test_encoding_output_shape_accessible_from_barrel(self) -> None:
        from chronocratic.models.enums import EncodingOutputShape

        assert EncodingOutputShape is not None
        assert EncodingOutputShape.VECTOR.value == "vector"


class TestTopLevelModelsReExport:
    """chronocratic.models top-level package re-exports EncodingOutputShape."""

    def test_importable_from_models(self) -> None:
        from chronocratic.models import EncodingOutputShape

        assert EncodingOutputShape is not None

    def test_models_all_contains_encoding_output_shape(self) -> None:
        import chronocratic.models

        assert "EncodingOutputShape" in chronocratic.models.__all__

    def test_value_via_top_level_import(self) -> None:
        from chronocratic.models import EncodingOutputShape

        assert EncodingOutputShape.SEQUENCE.value == "sequence"
