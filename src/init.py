"""
Network initialization utilities for TAGI.

All layer-level initialization is handled by param_init.py, which each layer
calls in its own __init__().  This module provides a single convenience
function that re-initializes every learnable layer in a Sequential network
using the same param_init routines (useful after architecture changes or to
get a fresh set of parameters).
"""

from .layers.linear import Linear
from .layers.conv2d import Conv2D
from .layers.batchnorm2d import BatchNorm2D
from .layers.resblock import ResBlock
from .param_init import (init_weight_bias_linear, init_weight_bias_conv2d,
                         init_weight_bias_norm)


def _reinit_linear(layer):
    """Re-initialize a Linear layer using param_init."""
    layer.mw, layer.Sw, layer.mb, layer.Sb = init_weight_bias_linear(
        layer.in_features, layer.out_features,
        init_method="He",
        gain_w=1.0, gain_b=1.0,
        bias=layer.has_bias,
        device=layer.device,
    )


def _reinit_conv2d(layer):
    """Re-initialize a Conv2D layer using param_init."""
    layer.mw, layer.Sw, layer.mb, layer.Sb = init_weight_bias_conv2d(
        kernel_size=layer.kH,
        in_channels=layer.C_in,
        out_channels=layer.C_out,
        init_method="He",
        gain_w=1.0, gain_b=1.0,
        device=layer.device,
    )


def _reinit_batchnorm(layer):
    """Re-initialize a BatchNorm2D layer using param_init."""
    layer.mw, layer.Sw, layer.mb, layer.Sb = init_weight_bias_norm(
        layer.num_features,
        gain_w=1.0, gain_b=1.0,
        device=layer.device,
    )
    layer._is_initialized = False


def _reinit_resblock(layer):
    """Re-initialize all sub-layers of a ResBlock using param_init."""
    _reinit_conv2d(layer.conv1)
    _reinit_batchnorm(layer.bn1)
    _reinit_conv2d(layer.conv2)
    _reinit_batchnorm(layer.bn2)
    if layer.use_projection:
        _reinit_conv2d(layer.proj_conv)
        _reinit_batchnorm(layer.proj_bn)


def reinit_net(net, verbose=True):
    """
    Re-initialize all learnable layers in a Sequential network.

    Delegates entirely to param_init.py (He init for weights, Xavier-style
    for batch-norm).  This is the same initialization each layer performs
    in its own constructor.

    Parameters
    ----------
    net : Sequential
        The network whose parameters will be re-initialized.
    verbose : bool
        If True, print a summary of re-initialized layers.
    """
    if verbose:
        print("=" * 60)
        print("  TAGI Initialization (via param_init)")
        print("=" * 60)

    for i, layer in enumerate(net.layers):
        if isinstance(layer, Linear):
            _reinit_linear(layer)
            if verbose:
                print(f"  Layer {i:2d} (Linear  ): in={layer.in_features}, out={layer.out_features}")

        elif isinstance(layer, ResBlock):
            _reinit_resblock(layer)
            if verbose:
                print(f"  Layer {i:2d} (ResBlock): {layer.in_channels}→{layer.out_channels}, stride={layer.stride}")

        elif isinstance(layer, Conv2D):
            _reinit_conv2d(layer)
            if verbose:
                print(f"  Layer {i:2d} (Conv2D  ): {layer.C_in}→{layer.C_out}, k={layer.kH}")

        elif isinstance(layer, BatchNorm2D):
            _reinit_batchnorm(layer)
            if verbose:
                print(f"  Layer {i:2d} (BN2D    ): C={layer.num_features}")

    if verbose:
        print("=" * 60)