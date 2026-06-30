"""Tests for utils.py → utils/ package split (backward compat barrel re-export)."""

from pathlib import Path

import pytest


def _utils_dir() -> Path:
    """Return the path to the models/utils/ package directory."""
    return Path(__file__).resolve().parents[2] / "src" / "chronocratic" / "models" / "utils"


class TestUtilsPackageStructure:
    """Verify the utils package structure exists and flat utils.py is gone."""

    def test_utils_is_directory(self) -> None:
        utils_path = _utils_dir()
        assert utils_path.is_dir(), f"utils/ should be a directory, got {utils_path}"

    def test_utils_init_exists(self) -> None:
        init_path = _utils_dir() / "__init__.py"
        assert init_path.is_file(), f"utils/__init__.py should exist, got {init_path}"

    def test_utils_utils_exists(self) -> None:
        utils_impl = _utils_dir() / "utils.py"
        assert utils_impl.is_file(), f"utils/utils.py should exist, got {utils_impl}"

    def test_flat_utils_py_removed(self) -> None:
        flat_utils = _utils_dir().parent / "utils.py"
        assert not flat_utils.exists(), (
            f"Flat utils.py should be removed (replaced by utils/ package), "
            f"but {flat_utils} still exists"
        )


class TestBarrelReExports:
    """Verify all 10 __all__ symbols + pad_tensor_with_nan are importable via barrel."""

    def test_import_extract_features_from_batch(self) -> None:
        from chronocratic.models.utils import extract_features_from_batch  # noqa: F401

    def test_import_concat_last_step_features(self) -> None:
        from chronocratic.models.utils import concat_last_step_features  # noqa: F401

    def test_import_full_series_pooling(self) -> None:
        from chronocratic.models.utils import full_series_pooling  # noqa: F401

    def test_import_multiscale_pooling(self) -> None:
        from chronocratic.models.utils import multiscale_pooling  # noqa: F401

    def test_import_integer_pooling(self) -> None:
        from chronocratic.models.utils import integer_pooling  # noqa: F401

    def test_import_apply_slicing(self) -> None:
        from chronocratic.models.utils import apply_slicing  # noqa: F401

    def test_import_process_sample_length(self) -> None:
        from chronocratic.models.utils import process_sample_length  # noqa: F401

    def test_import_process_sliding_window(self) -> None:
        from chronocratic.models.utils import process_sliding_window  # noqa: F401

    def test_import_pad_tensor_with_nan(self) -> None:
        """pad_tensor_with_nan is not in __all__ but imported by downstream code."""
        from chronocratic.models.utils import pad_tensor_with_nan  # noqa: F401

    def test_all_symbols_defined_in_barrel(self) -> None:
        import chronocratic.models.utils as utils_module  # noqa: F401

        expected_symbols = [
            "apply_slicing",
            "concat_last_step_features",
            "extract_features_from_batch",
            "full_series_pooling",
            "integer_pooling",
            "multiscale_pooling",
            "process_sample_length",
            "process_sliding_window",
            "pad_tensor_with_nan",
        ]
        for name in expected_symbols:
            assert hasattr(utils_module, name), f"utils barrel missing {name}"
