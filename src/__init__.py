"""
TAGI-Triton: Tractable Approximate Gaussian Inference on Triton
================================================================

A modular, GPU-accelerated library for Bayesian neural networks using TAGI.
All heavy operations are implemented as fused Triton kernels for maximum performance.

Modules
-------
- ``layers``  : Bayesian layers (Linear, ReLU, Remax, Bernoulli, Conv2D, ...)
- ``update``  : Observation and parameter update rules
- ``network`` : Network builder (Sequential API)
- ``kernels`` : Low-level Triton kernels
"""

from .network import Sequential
from .layers import Linear, ReLU, Remax, Bernoulli, Conv2D, AvgPool2D, Flatten

__version__ = "0.1.0"
__all__ = [
    "Sequential",
    "Linear", "ReLU", "Remax", "Bernoulli",
    "Conv2D", "AvgPool2D", "Flatten",
]
