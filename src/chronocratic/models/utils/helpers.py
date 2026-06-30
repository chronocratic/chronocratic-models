"""Internal helpers for the encoding contract.

Private utilities not exported from the ``utils`` barrel.
See :mod:`chronocratic.models.utils` for the public API.
"""

import warnings


# Module-level dedup tracker for once-per-class warnings
_warned_sequence_fallback: set[str] = set()


def _warn_sequence_fallback(cls: type) -> None:
    """Warn once per class that SEQUENCE output is a length-1 fallback.

    Called when a model with no native temporal axis (Tier C) is asked
    to produce ``output=SEQUENCE``.  The warning fires at most once per
    model class to avoid flooding eval/attack loops.

    Parameters
    ----------
    cls : type
        The model class requesting the SEQUENCE fallback.
    """
    cls_name = cls.__name__
    if cls_name in _warned_sequence_fallback:
        return
    _warned_sequence_fallback.add(cls_name)
    warnings.warn(
        f"{cls_name} has no temporal axis; output=SEQUENCE returns "
        f"a length-1 sequence (N, 1, D).",
        category=UserWarning,
        stacklevel=3,
    )
