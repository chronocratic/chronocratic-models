__all__ = ['CoST']


def __getattr__(name: str) -> object:
    """Lazy import for backward compatibility - avoids circular imports."""
    if name == 'CoST':
        from tscollection.models.cnn.dilated.cost.model import CoST  # noqa: PLC0415

        return CoST
    if name == 'utils':
        import tscollection.models.cost.utils as _utils  # noqa: PLC0415

        return _utils
    if name == 'model':
        import tscollection.models.cnn.dilated.cost.model as _model  # noqa: PLC0415

        return _model
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
