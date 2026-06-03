import random

from scipy.signal import butter, lfilter
import torch


def filter_frequencies(
    data: torch.Tensor, lowpass_cutoff: float = 40.0, highpass_cutoff: float = 0.5
) -> torch.Tensor:
    fft_results = torch.stack([apply_fft(sample) for sample in data])
    if random.random() < 0.5:
        return torch.stack(
            [lowpass_filter(sample, lowpass_cutoff, sampling_rate=128) for sample in fft_results]
        )
    return torch.stack(
        [highpass_filter(sample, highpass_cutoff, sampling_rate=128) for sample in fft_results]
    )


def apply_fft(sample: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft(sample)


def lowpass_filter(data: torch.Tensor, cutoff_frequency: float, sampling_rate: int) -> torch.Tensor:
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return torch.tensor(filtered_data, dtype=torch.float32)


def highpass_filter(
    data: torch.Tensor, cutoff_frequency: float, sampling_rate: int
) -> torch.Tensor:
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data)
    return torch.tensor(filtered_data, dtype=torch.float32)
