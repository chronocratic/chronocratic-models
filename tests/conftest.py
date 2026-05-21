"""Shared pytest fixtures for config tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_input_dims() -> int:
    """Return a sample input dimension for model configs."""
    return 1
