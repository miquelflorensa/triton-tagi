"""
Network initialization utilities for TAGI.

All layer-level initialization is handled by param_init.py, which each layer
calls in its own __init__().  This module provides:

  - reinit_net()             — symmetric He re-init for all layers
  - init_residual_aware()    — asymmetric ResBlock init (Conv2 attenuated,
                               skip carries most variance)
"""

import math
import torch

from .layers.linear import Linear
from .layers.conv2d import Conv2D
from .layers.batchnorm2d import BatchNorm2D
from .layers.resblock import ResBlock
from .layers.shared_var_linear import SharedVarLinear
from .layers.shared_var_conv2d import SharedVarConv2D
from .layers.shared_var_batchnorm2d import SharedVarBatchNorm2D
from .layers.shared_var_resblock import SharedVarResBlock
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

    return net


# ======================================================================
#  Residual-aware initialization
#
#  Problem: with symmetric He init, each ResBlock addition doubles the
#  variance: σ²_out = σ²_branch + σ²_skip ≈ 2.  After L blocks: σ² ≈ 2^L.
#
#  Fix: attenuate the LAST BatchNorm's γ (scale) in each ResBlock so the
#  branch contributes only η fraction of the output variance.  This is
#  the TAGI-native SkipInit — instead of γ=0 we use γ=√η to preserve
#  Bayesian uncertainty.
#
#  After the second BN with γ=√η, branch output has variance ≈ η.
#  Skip contributes ≈ 1.  Total: σ²_out ≈ 1 + η.
#  After L blocks: σ² ≈ (1+η)^L.
#
#  All conv/linear weights stay at the existing (working) initialization.
# ======================================================================

def init_residual_aware(net, eta=0.1, verbose=True):
    """
    Residual-aware initialization for TAGI networks with ResBlocks.

    Sets the second BatchNorm's γ (scale parameter) in each ResBlock to
    √η, so the residual branch contributes only η fraction of variance
    at each addition.  This prevents variance doubling (2^L → (1+η)^L).

    All other parameters (conv weights, biases, first BN) are left at
    their existing initialization.

    Parameters
    ----------
    net     : Sequential   network to initialize
    eta     : float        residual branch variance fraction (default 0.1)
    verbose : bool         print init summary
    """
    gamma = math.sqrt(eta)

    if verbose:
        print("=" * 60)
        print("  Residual-Aware Initialization (BN γ scaling)")
        print(f"  η={eta:.4f}  →  bn2.γ = √η = {gamma:.4f}")
        print("=" * 60)

    for i, layer in enumerate(net.layers):
        if isinstance(layer, (ResBlock, SharedVarResBlock)):
            # Set second BN's gamma to √η (attenuate branch output)
            layer.bn2.mw = torch.full_like(layer.bn2.mw, gamma)

            if verbose:
                proj_str = ""
                if layer.use_projection:
                    proj_str = f"  [projection]"
                print(f"  Layer {i:2d} (ResBlock): "
                      f"bn2.γ = {gamma:.4f}{proj_str}")

    if verbose:
        n_blocks = sum(1 for l in net.layers
                       if isinstance(l, (ResBlock, SharedVarResBlock)))
        expected_var = (1.0 + eta) ** n_blocks
        print(f"  ──────────────────────────────────────────")
        print(f"  Expected σ² after {n_blocks} blocks: "
              f"(1+{eta})^{n_blocks} = {expected_var:.3f}")
        print("=" * 60)

    return net