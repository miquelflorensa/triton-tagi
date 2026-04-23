"""Pytest configuration and shared fixtures for triton-tagi tests."""

import sys
from pathlib import Path

import pytest
import torch

# Allow tests to import from examples/ (e.g. run_resnet18.build_resnet18)
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring a CUDA GPU (skip if unavailable)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (run explicitly with -m slow)"
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
