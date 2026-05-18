__all__ = ['compute_amplitude_and_phase']

import torch


def compute_amplitude_and_phase(
    x: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the amplitude and phase of a complex-valued tensor.

    Parameters
    ----------
    x : torch.Tensor
        A complex-valued tensor, typically the output of a Fourier transform.
    eps : float, optional
        A small epsilon value to prevent division by zero
        and stabilize computations (default is 1e-6).

    Returns:
    -------
    amplitude : torch.Tensor
        The amplitude (magnitude) of the complex tensor.
    phase : torch.Tensor
        The phase angle of the complex tensor.

    Notes:
    -----
    The amplitude is computed using the formula:
        amplitude = sqrt((real(x) + eps)^2 + (imag(x) + eps)^2)

    The phase is computed using the formula:
        phase = arctangent2(imag(x), real(x) + eps)

    The epsilon `eps` is added to both the real and imaginary parts
    to prevent numerical issues such as division by zero.

    Examples:
    --------
    >>> import torch
    >>> x = torch.tensor([1+2j, 3+4j], dtype=torch.cfloat)
    >>> amplitude, phase = compute_amplitude_and_phase(x)
    >>> amplitude
    tensor([2.2361, 5.0000])
    >>> phase
    tensor([1.1071, 0.9273])
    """
    amplitude = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    return amplitude, phase
