"""Shared fixtures for tscollection model tests."""
from __future__ import annotations

import pytest

import torch


@pytest.fixture
def sample_input_dims() -> int:
    """Return a sample input dimension for model configs."""
    return 1
