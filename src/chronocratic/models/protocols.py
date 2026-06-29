"""Runtime-checkable protocols for encoder/decoder extraction.

These Protocols provide a uniform, programmatically-verified interface for
accessing encoder and decoder modules across all model implementations.

A model satisfies ``HasEncoder`` if it exposes a read-only ``.encoder`` property
that returns an ``nn.Module``. Similarly, ``HasDecoder`` requires a
``.decoder`` property. Both use ``@runtime_checkable`` so they work with
``isinstance()`` checks — unlike plain class attributes, which fail
``isinstance`` verification due to ``nn.Module.__getattr__`` overriding.

For models with both encoder and decoder, ``HasEncoderDecoder`` combines the
two protocols into a single runtime-checkable composite.
"""

from typing import Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class HasEncoder(Protocol):
    """Protocol for models with a publicly accessible encoder module.

    The ``encoder`` property must return an ``nn.Module`` that can be used
    for representation extraction, checkpointing, or fine-tuning.
    """

    @property
    def encoder(self) -> nn.Module:
        """Return the encoder submodule."""
        ...


@runtime_checkable
class HasDecoder(Protocol):
    """Protocol for models with a publicly accessible decoder module.

    The ``decoder`` property must return an ``nn.Module`` that can be used
    for reconstruction or generation.
    """

    @property
    def decoder(self) -> nn.Module:
        """Return the decoder submodule."""
        ...


@runtime_checkable
class HasEncoderDecoder(HasEncoder, HasDecoder, Protocol):
    """Protocol for models with both encoder and decoder modules.

    Combines ``HasEncoder`` and ``HasDecoder`` into a single runtime-checkable
    protocol for models that support both representation extraction and
    reconstruction.
    """
