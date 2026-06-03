"""Provide convenient imports for the main model classes and their configs.

Names listed in ``__all__`` are resolved lazily via ``__getattr__`` so that
importing ``tscollection.models`` itself does not load every model's heavy
dependencies (``torch``, ``lightning.pytorch``, etc.). Each name is materialized
the first time it is accessed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'FCN',
    'TST',
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
    'TSTModelParameters',
    'TimeNet',
    'TimeNetModelParameters',
    'TimeVAE',
    'TimeVAEModelParameters',
]

_LAZY: dict[str, str] = {
    'AutoTCL': '.convolutional.dilated',
    'AutoTCLModelParameters': '.convolutional.dilated',
    'CoST': '.convolutional.dilated',
    'CoSTModelParameters': '.convolutional.dilated',
    'TS2Vec': '.convolutional.dilated',
    'TS2VecModelParameters': '.convolutional.dilated',
    'FCN': '.convolutional.standard',
    'MCLModelParameters': '.convolutional.standard',
    'TSTCC': '.convolutional.standard',
    'TSTCCModelParameters': '.convolutional.standard',
    'Series2Vec': '.convolutional.standard',
    'Series2VecModelParameters': '.convolutional.standard',
    'TimeVAE': '.generative',
    'TimeVAEModelParameters': '.generative',
    'TimeNet': '.recurrent',
    'TimeNetModelParameters': '.recurrent',
    'TST': '.transformer',
    'TSTModelParameters': '.transformer',
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Resolve a public name by importing its owning subpackage on first access."""
    if name in _LAZY:
        from importlib import import_module  # noqa: PLC0415

        return getattr(import_module(_LAZY[name], __name__), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
