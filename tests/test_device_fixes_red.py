"""RED-phase tests for device-fix correctness (09-03 Task 1).

These tests verify structural properties of the device hardening fixes:
- filters.py has a _filter_on_device helper that preserves device
- autotcl losses build labels on input device (not CPU)
- contrastive indexing_factor is a torch.Tensor (not np.ndarray)

Run: uv run pytest tests/test_device_fixes_red.py -v
"""

from __future__ import annotations

import inspect
from types import ModuleType

import numpy as np
import pytest
import torch


class TestFilterOnDeviceHelper:
    """filters.py must have _filter_on_device that preserves device."""

    @pytest.fixture(scope="class")
    def filters_module(self) -> ModuleType:
        from chronocratic.models.convolutional.standard.series2vec import filters
        return filters  # noqa: R502

    def test_filter_on_device_exists(self, filters_module: ModuleType) -> None:
        """_filter_on_device helper must exist."""
        assert hasattr(filters_module, "_filter_on_device")

    def test_filter_on_device_preserves_cpu_device(self, filters_module: ModuleType) -> None:
        """_filter_on_device must return tensor on same device as input."""
        _filter_on_device = filters_module._filter_on_device
        # Create simple Butterworth filter coefficients
        b, a = np.array([1.0]), np.array([1.0])
        data = torch.randn(128)
        result = _filter_on_device(b, a, data)
        assert result.device.type == data.device.type

    def test_lowpass_uses_helper(self, filters_module: ModuleType) -> None:
        """lowpass_filter source must reference _filter_on_device."""
        source = inspect.getsource(filters_module.lowpass_filter)
        assert "_filter_on_device" in source

    def test_highpass_uses_helper(self, filters_module: ModuleType) -> None:
        """highpass_filter source must reference _filter_on_device."""
        source = inspect.getsource(filters_module.highpass_filter)
        assert "_filter_on_device" in source


class TestAutotclLossesDeviceAware:
    """autotcl losses must build labels on input device."""

    def test_local_info_nce_loss_device_in_source(self) -> None:
        """local_info_nce_loss must use device=z1.device in torch constructors."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import local_info_nce_loss
        source = inspect.getsource(local_info_nce_loss)
        assert "device=z1.device" in source

    def test_l1_out_loss_device_in_source(self) -> None:
        """l1_out_loss must use device=z1.device in torch constructors."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import l1_out_loss
        source = inspect.getsource(l1_out_loss)
        assert "device=z1.device" in source

    def test_local_info_nce_no_cpu_eye_pattern(self) -> None:
        """local_info_nce must not build eye on CPU then transfer."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import local_info_nce_loss
        source = inspect.getsource(local_info_nce_loss)
        # The old pattern was torch.eye(k-1).to(z1.device)
        assert ".to(z1.device)" not in source or "device=z1.device" in source


class TestContrastiveNoNumpy:
    """contrastive.py indexing_factor must be torch.Tensor, not np.ndarray."""

    def test_indexing_factor_type_hint(self) -> None:
        """_compute_contrastive_loss_logits must type-hint indexing_factor as torch.Tensor."""
        from chronocratic.models.losses import contrastive
        sig = inspect.signature(contrastive._compute_contrastive_loss_logits)
        param = sig.parameters["indexing_factor"]
        assert param.annotation is torch.Tensor or "torch.Tensor" in str(param.annotation)

    def test_temporal_does_not_use_numpy(self) -> None:
        """temporal_contrastive_loss must not call .cpu().numpy() on indexing_factor."""
        from chronocratic.models.losses import contrastive
        source = inspect.getsource(contrastive.temporal_contrastive_loss)
        assert ".cpu().numpy()" not in source

    def test_instance_does_not_use_numpy(self) -> None:
        """instance_contrastive_loss must not call .cpu().numpy() on indexing_factor."""
        from chronocratic.models.losses import contrastive
        source = inspect.getsource(contrastive.instance_contrastive_loss)
        assert ".cpu().numpy()" not in source
