__all__ = ['AutoTCL']


def __getattr__(name: str) -> object:
    """Lazy import for backward compatibility - avoids circular imports."""
    if name == 'AutoTCL':
        from tscollection.models.cnn.dilated.autotcl.model import AutoTCL  # noqa: PLC0415

        return AutoTCL
    if name == 'utils':
        import tscollection.models.autotcl.utils as _utils  # noqa: PLC0415

        return _utils
    if name == 'model':
        import tscollection.models.cnn.dilated.autotcl.model as _model  # noqa: PLC0415

        return _model
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
