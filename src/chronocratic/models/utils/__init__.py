"""Backward-compatible barrel re-export for the models.utils package.

All 10 public symbols from the original ``utils.py`` ``__all__`` are re-exported
here, plus ``pad_tensor_with_nan`` which was imported directly by downstream
callers but never listed in ``__all__``.

Existing importers (11 files) that use:
    ``from chronocratic.models.utils import X``
continue to work without modification.
"""

from chronocratic.models.utils.utils import (
    apply_slicing,
    concat_last_step_features,
    extract_features_from_batch,
    full_series_pooling,
    integer_pooling,
    multiscale_pooling,
    pad_tensor_with_nan,
    pool_feature_map,
    process_sample_length,
    process_sliding_window,
)

__all__ = [
    "apply_slicing",
    "concat_last_step_features",
    "extract_features_from_batch",
    "full_series_pooling",
    "integer_pooling",
    "multiscale_pooling",
    "pad_tensor_with_nan",
    "pool_feature_map",
    "process_sample_length",
    "process_sliding_window",
]
