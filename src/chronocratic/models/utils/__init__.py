"""Barrel re-export for the models.utils package.

All public symbols from ``utils.py`` ``__all__`` are re-exported here, plus
``pad_tensor_with_nan`` which was imported directly by downstream callers but
never listed in ``__all__``.

New exports (Phase 13, plan 1):
- ``generate_not_nan_mask``: moved from
  ``chronocratic.models.convolutional.dilated.encoders.masking``.
- ``zero_fill_padding``: replaces NaN timesteps with 0, returns keep-mask.
- ``masked_reconstruction_loss_mean``: masked mean loss for reconstruction
  models (used by TimeNet and RecurrentAE).
- ``masked_reconstruction_loss_sum``: masked sum loss, matching TF
  ``reduce_sum`` semantics (used by TimeVAE).

Existing importers that use:
    ``from chronocratic.models.utils import X``
continue to work without modification.
"""

from chronocratic.models.utils.utils import (
    apply_slicing,
    concat_last_step_features,
    extract_features_from_batch,
    full_series_pooling,
    generate_not_nan_mask,
    integer_pooling,
    masked_reconstruction_loss_mean,
    masked_reconstruction_loss_sum,
    multiscale_pooling,
    pad_tensor_with_nan,
    process_sample_length,
    process_sliding_window,
    zero_fill_padding,
)

__all__ = [
    "apply_slicing",
    "concat_last_step_features",
    "extract_features_from_batch",
    "full_series_pooling",
    "generate_not_nan_mask",
    "integer_pooling",
    "masked_reconstruction_loss_mean",
    "masked_reconstruction_loss_sum",
    "multiscale_pooling",
    "pad_tensor_with_nan",
    "process_sample_length",
    "process_sliding_window",
    "zero_fill_padding",
]
