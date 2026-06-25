"""Tests for device-aware fixes in filters, autotcl losses, and contrastive losses.

Verifies that:
- series2vec filters use _filter_on_device helper (CPU round-trip for scipy)
- autotcl loss labels are built on input device (not CPU then .to())
- contrastive losses keep indexing_factor as torch.Tensor (not numpy)
"""

from __future__ import annotations

import inspect

import torch


def _source_contains(func_or_class, pattern: str) -> bool:
    """Check if a function's source code contains a pattern."""
    return pattern in inspect.getsource(func_or_class)


def _source_lines(func_or_class) -> list[str]:
    """Return source lines of a function."""
    return inspect.getsource(func_or_class).splitlines()


class TestFilterOnDeviceHelper:
    """filters.py must have _filter_on_device helper used by lowpass and highpass."""

    def test_filter_on_device_exists(self) -> None:
        """_filter_on_device helper must exist in filters module."""
        from chronocratic.models.convolutional.standard.series2vec.filters import (
            _filter_on_device,
        )

        assert callable(_filter_on_device)

    def test_lowpass_uses_filter_on_device(self) -> None:
        """lowpass_filter must delegate to _filter_on_device, not call lfilter directly."""
        from chronocratic.models.convolutional.standard.series2vec.filters import (
            lowpass_filter,
        )

        assert _source_contains(lowpass_filter, "_filter_on_device")
        assert not _source_contains(lowpass_filter, "lfilter(")

    def test_highpass_uses_filter_on_device(self) -> None:
        """highpass_filter must delegate to _filter_on_device, not call lfilter directly."""
        from chronocratic.models.convolutional.standard.series2vec.filters import (
            highpass_filter,
        )

        assert _source_contains(highpass_filter, "_filter_on_device")
        assert not _source_contains(highpass_filter, "lfilter(")

    def test_filter_on_device_preserves_cpu_tensor_device(self) -> None:
        """_filter_on_device must return tensor on same device as input (CPU test)."""
        from chronocratic.models.convolutional.standard.series2vec.filters import (
            _filter_on_device,
        )

        data = torch.randn(128)
        result = _filter_on_device([1.0], [1.0], data)
        assert result.device.type == "cpu"

    def test_lowpass_preserves_cpu_device(self) -> None:
        """lowpass_filter output must stay on CPU when input is on CPU."""
        from chronocratic.models.convolutional.standard.series2vec.filters import (
            lowpass_filter,
        )

        data = torch.randn(128)
        result = lowpass_filter(data, cutoff_frequency=40.0, sampling_rate=128)
        assert result.device.type == "cpu"


class TestAutoTCLLossDeviceAwareness:
    """autotcl/losses.py must build labels on input device, not CPU."""

    def test_local_info_nce_uses_device_kwarg(self) -> None:
        """local_info_nce_loss must use device= in torch.eye/zeros/arange."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import (
            local_info_nce_loss,
        )

        src = _source_lines(local_info_nce_loss)
        # Must have at least one `device=z1.device` pattern
        assert any("device=z1.device" in line for line in src), (
            "local_info_nce_loss must use device=z1.device in tensor constructors"
        )

    def test_local_info_nce_output_on_cpu_device(self) -> None:
        """local_info_nce_loss must return loss on same device as input (CPU test)."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import (
            local_info_nce_loss,
        )

        z1 = torch.randn(2, 64, 8)
        z2 = torch.randn(2, 64, 8)
        loss = local_info_nce_loss(z1, z2, k=16)
        assert loss.device.type == "cpu"
        assert torch.isfinite(loss)

    def test_l1_out_uses_device_kwarg(self) -> None:
        """l1_out_loss must use device= in torch.arange/eye."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import l1_out_loss

        src = _source_lines(l1_out_loss)
        assert any("device=z1.device" in line for line in src), (
            "l1_out_loss must use device=z1.device in tensor constructors"
        )

    def test_l1_out_output_on_cpu_device(self) -> None:
        """l1_out_loss must return loss on same device as input (CPU test)."""
        from chronocratic.models.convolutional.dilated.autotcl.losses import l1_out_loss

        z1 = torch.randn(4, 32, 8)
        z2 = torch.randn(4, 32, 8)
        loss = l1_out_loss(z1, z2)
        assert loss.device.type == "cpu"
        assert torch.isfinite(loss)


class TestContrastiveLossNoNumpyRoundTrip:
    """contrastive.py must keep indexing_factor as torch.Tensor, not numpy."""

    def test_temporal_contrastive_no_numpy(self) -> None:
        """temporal_contrastive_loss must not use .cpu().numpy() for indexing_factor."""
        from chronocratic.models.losses.contrastive import temporal_contrastive_loss

        src = _source_lines(temporal_contrastive_loss)
        assert not any(".cpu().numpy()" in line for line in src), (
            "temporal_contrastive_loss must not convert to numpy"
        )

    def test_instance_contrastive_no_numpy(self) -> None:
        """instance_contrastive_loss must not use .cpu().numpy() for indexing_factor."""
        from chronocratic.models.losses.contrastive import instance_contrastive_loss

        src = _source_lines(instance_contrastive_loss)
        assert not any(".cpu().numpy()" in line for line in src), (
            "instance_contrastive_loss must not convert to numpy"
        )

    def test_compute_contrastive_logits_accepts_tensor(self) -> None:
        """_compute_contrastive_loss_logits indexing_factor param must be torch.Tensor type."""
        from chronocratic.models.losses.contrastive import (
            _compute_contrastive_loss_logits,
        )

        sig = inspect.signature(_compute_contrastive_loss_logits)
        indexing_factor_annotation = sig.parameters["indexing_factor"].annotation
        assert indexing_factor_annotation is torch.Tensor, (
            f"indexing_factor must be annotated as torch.Tensor, got {indexing_factor_annotation}"
        )

    def test_temporal_contrastive_output_on_cpu_device(self) -> None:
        """temporal_contrastive_loss must return finite loss on CPU."""
        from chronocratic.models.losses.contrastive import temporal_contrastive_loss

        a = torch.randn(4, 16, 8)
        b = torch.randn(4, 16, 8)
        loss = temporal_contrastive_loss(a, b)
        assert loss.device.type == "cpu"
        assert torch.isfinite(loss)

    def test_instance_contrastive_output_on_cpu_device(self) -> None:
        """instance_contrastive_loss must return finite loss on CPU."""
        from chronocratic.models.losses.contrastive import instance_contrastive_loss

        a = torch.randn(4, 16, 8)
        b = torch.randn(4, 16, 8)
        loss = instance_contrastive_loss(a, b)
        assert loss.device.type == "cpu"
        assert torch.isfinite(loss)
