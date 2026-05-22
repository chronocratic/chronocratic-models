"""Backward-compatible central import paths for model configs and augmentation parameters.

Re-exports config classes, augmentation parameter dataclasses, and training
strategies from their per-model locations via a single ``models.configs``
package. Consumers use ``configs.models`` or ``configs.augmentation``
submodules rather than importing from this barrel directly.
"""

__all__ = []
