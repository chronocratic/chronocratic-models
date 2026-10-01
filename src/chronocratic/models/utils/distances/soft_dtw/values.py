"""Memory-lean soft-DTW values for no-gradient use (e.g. supervision targets).

``SoftDTW`` in ``soft_dtw_cuda`` stores the full ``(T+2) x (T+2)`` dynamic-programming
table for every pair, because its backward pass needs it. When only the forward value is
needed (Series2Vec's temporal targets), two rolling columns suffice: memory drops from
O(P * T^2) to O(B * T * C + threads * T) with the same recurrence and the same result.
"""

__all__ = ["pairwise_soft_dtw_values"]

from typing import TYPE_CHECKING

from numba import njit
import numpy as np
import torch

if TYPE_CHECKING:

    def prange(*args: int) -> range:
        return range(*args)

else:
    from numba import prange

_MIN_PAIRABLE_BATCH_SIZE = 2


@njit(parallel=True)
def _soft_dtw_pair_values(
    series: np.ndarray,
    first_index: np.ndarray,
    second_index: np.ndarray,
    gamma: float,
    bandwidth: float,
) -> np.ndarray:
    """Return soft-DTW(series[first_index[p]], series[second_index[p]]) for every pair p.

    Same recurrence, loop order and pruning rule as ``soft_dtw_cuda.compute_softdtw``,
    but keeps only columns ``j - 1`` (``previous``) and ``j`` (``current``) of ``R``.
    """
    num_pairs = first_index.shape[0]
    length = series.shape[1]
    num_channels = series.shape[2]
    values = np.empty(num_pairs)
    for pair in prange(num_pairs):  # numba: positional args only
        x = series[first_index[pair]]
        y = series[second_index[pair]]
        previous = np.full(length + 1, np.inf)  # column j - 1 of R
        current = np.full(length + 1, np.inf)  # column j of R
        previous[0] = 0.0  # R[0, 0]
        for j in range(1, length + 1):
            current[0] = np.inf  # R[0, j]
            for i in range(1, length + 1):
                if 0 < bandwidth < abs(i - j):
                    current[i] = np.inf  # pruned cell; buffer is reused, so reset explicitly
                    continue
                cost = 0.0
                for channel in range(num_channels):
                    diff = x[i - 1, channel] - y[j - 1, channel]
                    cost += diff * diff
                r0 = -previous[i - 1] / gamma  # R[i-1, j-1]
                r1 = -current[i - 1] / gamma  # R[i-1, j]
                r2 = -previous[i] / gamma  # R[i, j-1]
                rmax = max(r0, r1, r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                current[i] = cost - gamma * (np.log(rsum) + rmax)
            previous, current = current, previous
        values[pair] = previous[length]  # after the final swap, previous holds column T
    return values


def pairwise_soft_dtw_values(
    time_series: torch.Tensor, *, gamma: float, bandwidth: float | None = None
) -> torch.Tensor:
    """Soft-DTW value for every lower-triangular pair of a batch, without gradients.

    Pair order matches ``torch.tril_indices(B, B, offset=-1)``: pair ``p`` compares
    ``time_series[rows[p]]`` with ``time_series[cols[p]]``.

    Args:
        time_series: Batch of shape ``(B, T, C)`` on any device.
        gamma: Soft-min smoothing, must be > 0.
        bandwidth: Sakoe-Chiba band half-width in time steps; cells with
            ``|i - j| > bandwidth`` are skipped. ``None`` disables pruning (exact).

    Returns:
        Tensor of shape ``(B * (B - 1) / 2,)`` on the input's device and dtype. Empty when
        ``B < 2``. Never requires grad.

    Raises:
        ValueError: If ``gamma <= 0`` or ``bandwidth <= 0``.
    """
    if gamma <= 0:
        msg = f"gamma must be > 0, got {gamma}"
        raise ValueError(msg)
    if bandwidth is not None and bandwidth <= 0:
        msg = f"bandwidth must be > 0 or None (no pruning), got {bandwidth}"
        raise ValueError(msg)
    batch_size = time_series.size(0)
    if batch_size < _MIN_PAIRABLE_BATCH_SIZE:
        return time_series.new_empty(0)
    pair_indices = torch.tril_indices(batch_size, batch_size, offset=-1)
    series = time_series.detach().to(device="cpu", dtype=torch.float64).numpy()
    values = _soft_dtw_pair_values(
        series,
        pair_indices[0].numpy(),
        pair_indices[1].numpy(),
        gamma,
        0.0 if bandwidth is None else float(bandwidth),
    )
    return torch.from_numpy(values).to(device=time_series.device, dtype=time_series.dtype)
