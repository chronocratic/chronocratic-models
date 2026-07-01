import numpy as np
from scipy.signal import butter, lfilter
import torch


def _filter_on_device(b: np.ndarray, a: np.ndarray, data: torch.Tensor) -> torch.Tensor:
    """Run a SciPy IIR filter host-side, returning on ``data``'s device.

    SciPy's ``lfilter`` only accepts host (numpy) arrays. On MPS tensors,
    calling ``.numpy()`` without ``.cpu()`` first raises RuntimeError.
    This helper explicitly round-trips through CPU/numpy then restores device.

    ponytail: dtype=torch.float32 drops any imaginary part from complex FFT
    input, which is intentional — the filter is applied to complex data but
    the original Series2Vec pipeline discards phase information downstream.
    """
    filtered = lfilter(b, a, data.cpu().numpy())
    return torch.as_tensor(filtered, dtype=torch.float32, device=data.device)


__all__ = [
    "LOWPASS_PROBABILITY",
    "SAMPLING_RATE",
    "apply_fft",
    "filter_frequencies",
    "highpass_filter",
    "lowpass_filter",
]

LOWPASS_PROBABILITY = 0.5
SAMPLING_RATE = 128


def filter_frequencies(
    data: torch.Tensor,
    lowpass_cutoff: float = 40.0,
    highpass_cutoff: float = 0.5,
    *,
    training: bool = True,
) -> torch.Tensor:
    """Randomly apply low-pass or high-pass filtering to FFT-transformed samples."""
    fft_results = torch.stack([apply_fft(sample) for sample in data])
    if training and torch.rand(()) < LOWPASS_PROBABILITY:  # device-ok: CPU scalar probability
        return torch.stack(
            [
                lowpass_filter(sample, lowpass_cutoff, sampling_rate=SAMPLING_RATE)
                for sample in fft_results
            ]
        )
    return torch.stack(
        [
            highpass_filter(sample, highpass_cutoff, sampling_rate=SAMPLING_RATE)
            for sample in fft_results
        ]
    )


def apply_fft(sample: torch.Tensor) -> torch.Tensor:
    """Return the complex FFT of one time series sample."""
    return torch.fft.fft(sample)


def lowpass_filter(data: torch.Tensor, cutoff_frequency: float, sampling_rate: int) -> torch.Tensor:
    """Apply a Butterworth low-pass filter to ``data``."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype="low", analog=False)
    return _filter_on_device(b, a, data)


def highpass_filter(
    data: torch.Tensor, cutoff_frequency: float, sampling_rate: int
) -> torch.Tensor:
    """Apply a Butterworth high-pass filter to ``data``."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype="high", analog=False)
    return _filter_on_device(b, a, data)
