"""Tests for _warn_sequence_fallback helper (once-per-class warning dedup)."""

import warnings

import pytest


class FakeModelA:
    pass


class FakeModelB:
    pass


class TestWarnSequenceFallback:
    """Verify _warn_sequence_fallback emits exactly once per class name."""

    def setup_method(self) -> None:
        """Reset the module-level dedup set before each test."""
        try:
            from chronocratic.models.utils.helpers import _warned_sequence_fallback as _set

            _set.clear()
        except ImportError:
            pass  # helpers not created yet (RED phase)

    def test_first_call_emits_warning(self) -> None:
        from chronocratic.models.utils.helpers import _warn_sequence_fallback

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_sequence_fallback(FakeModelA)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "FakeModelA" in str(w[0].message)

    def test_second_call_same_class_no_warning(self) -> None:
        from chronocratic.models.utils.helpers import _warn_sequence_fallback

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_sequence_fallback(FakeModelA)
            _warn_sequence_fallback(FakeModelA)
            assert len(w) == 1, f"Expected 1 warning (dedup), got {len(w)}"

    def test_different_class_emits_new_warning(self) -> None:
        from chronocratic.models.utils.helpers import _warn_sequence_fallback

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_sequence_fallback(FakeModelA)
            _warn_sequence_fallback(FakeModelB)
            assert len(w) == 2, f"Expected 2 warnings (different classes), got {len(w)}"

    def test_warning_message_format(self) -> None:
        from chronocratic.models.utils.helpers import _warn_sequence_fallback

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_sequence_fallback(FakeModelA)
            msg = str(w[0].message)
            assert "FakeModelA" in msg
            assert "length-1 sequence" in msg
            assert "(N, 1, D)" in msg

    def test_helpers_not_exported_from_barrel(self) -> None:
        """helpers.py is internal — _warn_sequence_fallback must NOT be importable from barrel."""
        from chronocratic.models.utils import __all__ as barrel_all  # noqa: F401

        # Barrel __all__ must not leak internal helpers
        assert "_warn_sequence_fallback" not in barrel_all
        # Verify the function itself is not accessible via "from X import Y"
        with pytest.raises(ImportError):
            from chronocratic.models.utils import _warn_sequence_fallback  # noqa: F401
