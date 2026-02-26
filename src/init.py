"""
Decoupled Bayesian Initialization for Deep TAGI
===============================================
Prevents variance explosion in deep architectures by separating the 
initialization of the weight means (signal) from the weight variances (uncertainty).
"""

import torch
import math
from .layers.linear import Linear
from .layers.conv2d import Conv2D
from .layers.relu import ReLU
from .layers.flatten import Flatten
from .layers.avgpool2d import AvgPool2D
from .layers.batchnorm2d import BatchNorm2D

# ======================================================================
#  Helper functions
# ======================================================================

def _compute_preact(mu_a, var_a, mw, Sw, mb, Sb):
    """Z = A @ W + B: propagate mean and variance."""
    mu_z = mu_a @ mw + mb
    var_z = (mu_a ** 2) @ Sw + var_a @ (mw ** 2) + var_a @ Sw + Sb
    return mu_z, var_z

def _bayesian_relu(mu, var):
    """Exact Bayesian ReLU moments (Φ/φ form)."""
    eps = 1e-9
    std = torch.sqrt(torch.clamp(var, min=eps))
    alpha = mu / std
    INV_SQRT_2PI = 0.3989422804014327
    INV_SQRT_2 = 0.7071067811865476

    phi = torch.exp(-0.5 * alpha ** 2) * INV_SQRT_2PI
    Phi = 0.5 + 0.5 * torch.erf(alpha * INV_SQRT_2)
    Phi = torch.clamp(Phi, min=1e-20)
    phi = torch.clamp(phi, min=1e-20)

    mu_a = torch.clamp(std * phi + mu * Phi, min=1e-20)
    var_a = torch.clamp(
        -mu_a ** 2 + 2 * mu_a * mu - mu * std * phi +
        (var - mu ** 2) * Phi, min=eps)
    return mu_a, var_a

def _measure_empirical_moments(mu_z, var_z):
    """
    Measures the global empirical scalar mean and variance 
    of the Gaussian pre-activations across the batch and output units.
    """
    # Global mean: Average of all predicted means
    m_Z = mu_z.mean().item()
    
    # Global variance: E[Var] + Var[E] (Law of Total Variance)
    total_variance = var_z + (mu_z - m_Z)**2
    v_Z = total_variance.mean().item()
    
    return m_Z, max(v_Z, 1e-9)



def initialize_deep_network(net, gain_mu=2.0, gain_sigma=0.1, verbose=True):
    """
    Initializes a deep TAGI network with decoupled variances.
    
    Parameters
    ----------
    net        : Sequential   the TAGI network
    gain_mu    : float        Gain for the mean distribution (default 2.0 for He)
    gain_sigma : float        Gain for the prior uncertainty (must be < gain_mu)
    """
    if verbose:
        print("=" * 56)
        print(f"  Decoupled Bayesian Init (mu_gain={gain_mu}, sigma_gain={gain_sigma})")
        print("=" * 56)

    for i, layer in enumerate(net.layers):
        if isinstance(layer, (Linear, Conv2D)):
            K = layer.mw.shape[0]  # fan-in
            
            var_mu = gain_mu / K
            var_sigma = gain_sigma / K

            # 1. Initialize Means (Signal Propagation)
            layer.mw.normal_(0, math.sqrt(var_mu))
            
            # 2. Initialize Variances (Restricted Epistemic Uncertainty)
            layer.Sw.fill_(var_sigma)
            
            # 3. Biases
            layer.mb.fill_(0.0)
            layer.Sb.fill_(1e-6)

            if verbose:
                l_type = "Conv2D" if isinstance(layer, Conv2D) else "Linear"
                print(f"  Layer {i:2d} ({l_type:6s}): μ_W~N(0, {var_mu:.5f}), σ²_W={var_sigma:.5f}")

    if verbose:
        print("=" * 56)

def autotune_decoupled_gains(net, data_batch, verbose=True):
    """
    Automatically finds the optimal gain_mu and gain_sigma for a deep TAGI network
    by simulating forward passes and evaluating the health of the final layer's variance.
    """
    if verbose:
        print("=" * 56)
        print("  Auto-Tuning Decoupled Bayesian Gains...")
        print("=" * 56)

    # Grids to search over
    mu_gains = [1.0, 1.5, 2.0, 2.5, 3.0]
    sigma_gains = [0.01, 0.05, 0.1, 0.2, 0.5]

    best_score = -float('inf')
    best_params = (2.0, 0.1) # Default fallback

    # We need to find the last parametrized layer to measure its output
    last_param_idx = -1
    for idx, l in enumerate(net.layers):
        if isinstance(l, (Linear, Conv2D)):
            last_param_idx = idx

    # Small subset of batch for faster tuning
    x_tune = data_batch[:128].clone()
    
    for g_mu in mu_gains:
        for g_sig in sigma_gains:
            
            # 1. Temporarily initialize network with test gains
            initialize_deep_network(net, gain_mu=g_mu, gain_sigma=g_sig, verbose=False)

            # 2. Run Forward Pass
            mu_a = x_tune.clone()
            var_a = torch.full_like(x_tune, 1e-6)
            
            nan_detected = False

            for i, layer in enumerate(net.layers):
                if i > last_param_idx:
                    break # Stop at the last Linear/Conv layer

                try:
                    if isinstance(layer, (ReLU, Flatten, AvgPool2D, BatchNorm2D)):
                        if isinstance(layer, ReLU):
                            flat_mu = mu_a.view(mu_a.shape[0], -1)
                            flat_var = var_a.view(var_a.shape[0], -1)
                            mu_a_flat, var_a_flat = _bayesian_relu(flat_mu, flat_var)
                            mu_a = mu_a_flat.view(mu_a.shape)
                            var_a = var_a_flat.view(var_a.shape)
                        elif isinstance(layer, Flatten):
                            mu_a = mu_a.view(mu_a.shape[0], -1)
                            var_a = var_a.view(var_a.shape[0], -1)
                        elif isinstance(layer, (AvgPool2D, BatchNorm2D)):
                            mu_a, var_a = layer.forward(mu_a, var_a)
                    
                    elif isinstance(layer, (Conv2D, Linear)):
                        if isinstance(layer, Conv2D):
                            from .layers.conv2d import _triton_im2col
                            N, C, H, W = mu_a.shape
                            patches_mu = _triton_im2col(mu_a, layer.kH, layer.kW, layer.stride, layer.padding)
                            patches_var = _triton_im2col(var_a, layer.kH, layer.kW, layer.stride, layer.padding)
                            mu_a, var_a = _compute_preact(patches_mu, patches_var, layer.mw, layer.Sw, layer.mb, layer.Sb)
                            
                            H_out = (H + 2 * layer.padding - layer.kH) // layer.stride + 1
                            W_out = (W + 2 * layer.padding - layer.kW) // layer.stride + 1
                            n_out = layer.mw.shape[1]
                            mu_a = mu_a.view(N, H_out, W_out, n_out).permute(0, 3, 1, 2).contiguous()
                            var_a = var_a.view(N, H_out, W_out, n_out).permute(0, 3, 1, 2).contiguous()
                        else:
                            mu_a, var_a = _compute_preact(mu_a, var_a, layer.mw, layer.Sw, layer.mb, layer.Sb)
                    
                    if torch.isnan(mu_a).any() or torch.isnan(var_a).any():
                        nan_detected = True
                        break
                except Exception:
                    nan_detected = True
                    break

            # 3. Evaluate Health of Final Output
            if nan_detected:
                continue
                
            m_Z, v_Z = _measure_empirical_moments(mu_a, var_a)
            
            # Scoring logic: We want variance to be healthy (e.g., between 5 and 50)
            # We heavily penalize exploding variance (> 100) or vanishing variance (< 1.0)
            if v_Z < 1.0 or v_Z > 100.0:
                score = -abs(v_Z - 20.0) # Penalize distance from an ideal "20"
            else:
                score = v_Z # Within healthy range, higher variance provides better logit spread

            if score > best_score:
                best_score = score
                best_params = (g_mu, g_sig)

    # 4. Apply the winning parameters
    best_mu, best_sig = best_params
    if verbose:
        print(f"  Best params found: mu_gain = {best_mu}, sigma_gain = {best_sig}")
        print("  Applying optimal initialization...")
    
    initialize_deep_network(net, gain_mu=best_mu, gain_sigma=best_sig, verbose=verbose)
    return best_mu, best_sig