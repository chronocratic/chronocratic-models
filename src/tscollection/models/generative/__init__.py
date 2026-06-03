"""Barrel for generative models (TimeVAE) — lazy re-exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    'TimeVAE',
    'TimeVAEModelParameters',
]

_LAZY: dict[str, str] = {
    'TimeVAE': '.timevae',
    'TimeVAEModelParameters': '.timevae',
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Resolve a public name by importing its owning subpackage on first access."""
    if name in _LAZY:
        from importlib import import_module  # noqa: PLC0415

        return getattr(import_module(_LAZY[name], __name__), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
