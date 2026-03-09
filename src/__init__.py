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
from .layers import (Linear, ReLU, Remax, Bernoulli,
                     Conv2D, AvgPool2D, BatchNorm2D, Flatten,
                     ResBlock, Add, EvenSoftplus)
from .param_init import (he_init, xavier_init, gaussian_param_init,
                         init_weight_bias_linear, init_weight_bias_conv2d,
                         init_weight_bias_norm)
from .init import reinit_net, init_residual_aware
from .monitor import TAGIMonitor, sweep_gains, sweep_sigma_v, compare_heads
from .auto_tune import auto_tune, find_best_gain, find_best_sigma_v
from .optimizer import AdamTAGI

__version__ = "0.1.0"
__all__ = [
    "Sequential",
    "Linear", "ReLU", "Remax", "Bernoulli",
    "Conv2D", "AvgPool2D", "BatchNorm2D", "Flatten",
    "ResBlock", "Add", "EvenSoftplus",
    "he_init", "xavier_init", "gaussian_param_init",
    "init_weight_bias_linear", "init_weight_bias_conv2d",
    "init_weight_bias_norm",
    "reinit_net", "init_residual_aware",
    "TAGIMonitor", "sweep_gains", "sweep_sigma_v", "compare_heads",
    "auto_tune", "find_best_gain", "find_best_sigma_v",
    "AdamTAGI",
]
