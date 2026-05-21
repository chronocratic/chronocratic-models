__all__ = ['TS2Vec']


def __getattr__(name: str) -> object:
    """Lazy import for backward compatibility - avoids circular imports."""
    if name == 'TS2Vec':
        from tscollection.models.cnn.dilated.ts2vec.model import TS2Vec  # noqa: PLC0415

        return TS2Vec
    if name == 'utils':
        import tscollection.models.ts2vec.utils as _utils  # noqa: PLC0415

        return _utils
    if name == 'model':
        import tscollection.models.cnn.dilated.ts2vec.model as _model  # noqa: PLC0415

        return _model
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
