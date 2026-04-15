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

Numerical precision
-------------------
TF32 (tensor float 32) is disabled for CUDA matrix multiplications at import
time.  cuTAGI uses scalar FMA loops (``__fmaf_rn``) which give near-fp64
accuracy; leaving TF32 enabled in PyTorch/Triton would introduce systematic
~1e-3 errors in the variance forward pass, breaking numerical parity.
"""

import torch

# Disable TF32 globally so cuBLAS matmuls match cuTAGI's scalar FMA precision.
# TF32 truncates the fp32 mantissa to 10 bits during tensor-core accumulation,
# causing ~1e-3 systematic errors in Sz that cascade through deep networks.
torch.backends.cuda.matmul.allow_tf32 = False

from .auto_tune import auto_tune, find_best_gain, find_best_sigma_v
from .base import Layer, LearnableLayer
from .inference_init import inference_init
from .init import init_residual_aware, reinit_net
from .layers import (
    FRN2D,
    TLU,
    Add,
    AvgPool2D,
    BatchNorm2D,
    Bernoulli,
    Conv2D,
    EvenSoftplus,
    Flatten,
    FRNResBlock,
    Linear,
    ReLU,
    Remax,
    ResBlock,
    SharedVarBatchNorm2D,
    SharedVarConv2D,
    SharedVarLinear,
    SharedVarResBlock,
)
from .momentum import StateSpaceMomentum
from .monitor import TAGIMonitor, compare_heads, sweep_gains, sweep_sigma_v
from .nadam_optimizer import NadamTAGI
from .network import Sequential
from .optimizer import AdamTAGI
from .param_init import (
    gaussian_param_init,
    he_init,
    init_weight_bias_conv2d,
    init_weight_bias_linear,
    init_weight_bias_norm,
    xavier_init,
)

__version__ = "0.1.0"
__all__ = [
    # ABCs
    "Layer",
    "LearnableLayer",
    # Layers
    "Sequential",
    "Linear",
    "ReLU",
    "Remax",
    "Bernoulli",
    "Conv2D",
    "AvgPool2D",
    "BatchNorm2D",
    "Flatten",
    "ResBlock",
    "Add",
    "EvenSoftplus",
    "FRN2D",
    "TLU",
    "FRNResBlock",
    "SharedVarLinear",
    "SharedVarConv2D",
    "SharedVarBatchNorm2D",
    "SharedVarResBlock",
    # Init
    "he_init",
    "xavier_init",
    "gaussian_param_init",
    "init_weight_bias_linear",
    "init_weight_bias_conv2d",
    "init_weight_bias_norm",
    "reinit_net",
    "init_residual_aware",
    "inference_init",
    # Training utilities
    "TAGIMonitor",
    "sweep_gains",
    "sweep_sigma_v",
    "compare_heads",
    "auto_tune",
    "find_best_gain",
    "find_best_sigma_v",
    "AdamTAGI",
    "NadamTAGI",
    "StateSpaceMomentum",
]
