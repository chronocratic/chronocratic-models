from scipy.signal import butter, lfilter
import torch

__all__ = [
    'LOWPASS_PROBABILITY',
    'SAMPLING_RATE',
    'apply_fft',
    'filter_frequencies',
    'highpass_filter',
    'lowpass_filter',
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
    if training and torch.rand(()) < LOWPASS_PROBABILITY:
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
    """Return the complex FFT of one time-series sample."""
    return torch.fft.fft(sample)


def lowpass_filter(data: torch.Tensor, cutoff_frequency: float, sampling_rate: int) -> torch.Tensor:
    """Apply a Butterworth low-pass filter to ``data``."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return torch.tensor(filtered_data, dtype=torch.float32)


def highpass_filter(
    data: torch.Tensor, cutoff_frequency: float, sampling_rate: int
) -> torch.Tensor:
    """Apply a Butterworth high-pass filter to ``data``."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data)
    return torch.tensor(filtered_data, dtype=torch.float32)
