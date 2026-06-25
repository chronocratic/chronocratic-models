"""Device smoke tests parametrized over available backends (CPU, CUDA, MPS).

Verifies that loss functions and filter helpers work on all available devices
without tensor device mismatch crashes.
"""

from __future__ import annotations

import pytest
import torch


def _available_devices() -> list[str]:
    """Return list of devices available on the current machine."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = _available_devices()


@pytest.mark.parametrize("device", DEVICES)
def test_autotcl_local_info_nce_runs(device: str) -> None:
    """local_info_nce_loss must run on any device and return finite loss on input device."""
    from chronocratic.models.convolutional.dilated.autotcl.losses import local_info_nce_loss

    z1 = torch.randn(2, 64, 8, device=device)
    z2 = torch.randn(2, 64, 8, device=device)
    loss = local_info_nce_loss(z1, z2, k=16)
    assert loss.device.type == device
    assert torch.isfinite(loss)


@pytest.mark.parametrize("device", DEVICES)
def test_autotcl_l1_out_runs(device: str) -> None:
    """l1_out_loss must run on any device and return finite loss on input device."""
    from chronocratic.models.convolutional.dilated.autotcl.losses import l1_out_loss

    z1 = torch.randn(4, 32, 8, device=device)
    z2 = torch.randn(4, 32, 8, device=device)
    loss = l1_out_loss(z1, z2)
    assert loss.device.type == device
    assert torch.isfinite(loss)


@pytest.mark.parametrize("device", DEVICES)
def test_contrastive_losses_run(device: str) -> None:
    """temporal and instance contrastive losses must run on any device with finite output."""
    from chronocratic.models.losses.contrastive import (
        instance_contrastive_loss,
        temporal_contrastive_loss,
    )

    a = torch.randn(4, 16, 8, device=device)
    b = torch.randn(4, 16, 8, device=device)
    assert torch.isfinite(temporal_contrastive_loss(a, b))
    assert torch.isfinite(instance_contrastive_loss(a, b))


@pytest.mark.parametrize("device", DEVICES)
def test_series2vec_filters_preserve_device(device: str) -> None:
    """filter_frequencies must return output on the same device as input."""
    from chronocratic.models.convolutional.standard.series2vec.filters import filter_frequencies

    x = torch.randn(2, 128, device=device)
    out = filter_frequencies(x, training=True)
    assert out.device.type == device
