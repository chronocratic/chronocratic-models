from __future__ import annotations

__all__ = ['AutoTCL', 'CoST', 'TS2Vec']


def __getattr__(name: str) -> object:
    """Lazily import model classes to avoid circular imports."""
    if name == 'AutoTCL':
        from tscollection.models.cnn.dilated.autotcl.model import AutoTCL  # noqa: PLC0415

        return AutoTCL
    if name == 'CoST':
        from tscollection.models.cnn.dilated.cost.model import CoST  # noqa: PLC0415

        return CoST
    if name == 'TS2Vec':
        from tscollection.models.cnn.dilated.ts2vec.model import TS2Vec  # noqa: PLC0415

        return TS2Vec
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
