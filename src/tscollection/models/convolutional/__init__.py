"""Barrel for the convolutional model family — lazy re-exports.

Names listed in ``__all__`` are resolved lazily via ``__getattr__`` to defer
loading of ``torch`` / ``lightning.pytorch`` until a specific model or its
parameter dataclass is actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'FCN',
    'TSTCC',
    'AutoTCL',
    'AutoTCLModelParameters',
    'CoST',
    'CoSTModelParameters',
    'MCLModelParameters',
    'Series2Vec',
    'Series2VecModelParameters',
    'TS2Vec',
    'TS2VecModelParameters',
    'TSTCCModelParameters',
]

_LAZY: dict[str, str] = {
    'AutoTCL': '.dilated',
    'AutoTCLModelParameters': '.dilated',
    'CoST': '.dilated',
    'CoSTModelParameters': '.dilated',
    'TS2Vec': '.dilated',
    'TS2VecModelParameters': '.dilated',
    'FCN': '.standard',
    'MCLModelParameters': '.standard',
    'Series2Vec': '.standard',
    'Series2VecModelParameters': '.standard',
    'TSTCC': '.standard',
    'TSTCCModelParameters': '.standard',
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Resolve a public name by importing its owning subpackage on first access."""
    if name in _LAZY:
        from importlib import import_module  # noqa: PLC0415

        return getattr(import_module(_LAZY[name], __name__), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
