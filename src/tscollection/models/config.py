"""Root model configuration — only the abstract base class.

All concrete parameter dataclasses live in per-model config modules:
    - TS2Vec: ``ts2vec/config.py``
    - CoST: ``cost/config.py``
    - AutoTCL: ``autotcl/config.py``

This file provides ``ModelParameters`` as the common root type.
"""

__all__ = ['ModelParameters']

import abc
from dataclasses import dataclass
from typing import Self


@dataclass
class ModelParameters(abc.ABC):  # noqa: B024
    """Abstract base class for all model parameter configurations.

    Provides no fields itself; serves as a common type for IDE
    autocompletion and static type checking across all model configs.

    Raises:
        TypeError: If instantiated directly. Subclasses must be used.
    """

    def __new__(cls: type, **kwargs: object) -> Self:  # noqa: ARG004
        """Prevent direct instantiation of the abstract base class.

        Args:
            cls: The class being instantiated.
            **kwargs: Dataclass fields passed by the generated __init__.
                Retained for signature compatibility but unused in __new__.

        Raises:
            TypeError: If ``cls`` is ``ModelParameters`` itself.
        """
        if cls is ModelParameters:
            msg = 'ModelParameters is abstract and cannot be instantiated directly'
            raise TypeError(msg)
        return object.__new__(cls)
