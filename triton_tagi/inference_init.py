"""
Inference-based initialization for TAGI networks.

Reference: Contribution 3 — Probabilistic Initialization and Regularization

Target distribution for each hidden unit Z_i:
    M_i ~ N(0, σ_M²),   Z_i | M_i ~ N(M_i, σ_Z²)

Two aggregate constraints are applied iteratively via the RTS smoother:

  1. Sum  S  = Σ Z_i
         Targets (conditional on latent means):
           μ_S̃   = 0
           σ²_S̃  = A · σ_Z²

  2. Sum-of-squares  S2 = Σ Z_i²
         Targets:
           μ_S2̃  = A · (σ_M² + σ_Z²)
           σ²_S2̃ = A · (2·σ_Z⁴ + 4·σ_M²·σ_Z²)

Weight update uses *Decoupled Parameter Updates via Inverse Moment-Matching*:
  Step 1 — Weights scale for the variance target:
           γ_i = √(σ²_{Z_i|·} / σ²_{Z_i})
           μ_W_ji ← γ_i · μ_W_ji,   σ²_W_ji ← γ_i² · σ²_W_ji
           σ²_B_i ← γ_i² · σ²_B_i
  Step 2 — Bias absorbs the residual mean shift:
           μ_B_i  ←  μ_B_i + Δμ_Z_i
"""

import torch

from .kernels.common import triton_fused_var_forward
from .layers.avgpool2d import AvgPool2D
from .layers.even_softplus import EvenSoftplus
from .layers.flatten import Flatten
from .layers.leaky_relu import LeakyReLU
from .layers.linear import Linear
from .layers.relu import ReLU
from .layers.remax import Remax
from .layers.silu import SiLU

_PASS_THROUGH = (ReLU, LeakyReLU, Remax, AvgPool2D, Flatten, EvenSoftplus, SiLU)


# ======================================================================
#  RTS constraint update on hidden-unit moments
# ======================================================================


def _apply_constraints(mz, Sz, A, sigma_M_sq, sigma_Z_sq, n_iter=1, tol=1e-6):
    """
    Apply S and S2 constraints iteratively until convergence.

    Per sample (row), the RTS surrogate moment-matching projection drives:
        S  = Σ mz_i   →  μ_S̃ = 0,        σ²_S̃ = A·σ_Z²
        S2 = Σ(mz²+Sz)→  μ_S2̃ = A(σ_M²+σ_Z²),
                          σ²_S2̃ = A(2σ_Z⁴ + 4σ_M²σ_Z²)

    Parameters
    ----------
    mz, Sz       : (B, A)   forward-pass moments
    A            : int       layer width
    sigma_M_sq   : float     σ_M²
    sigma_Z_sq   : float     σ_Z²
    n_iter       : int       max iterations
    tol          : float     convergence threshold

    Returns
    -------
    mz_post, Sz_post : (B, A)
    """
    mz = mz.clone()
    Sz = Sz.clone().clamp(min=1e-8)

    Af = float(A)
    # ── Targets ──────────────────────────────────────────────────────
    # Sum constraint (conditional on latent means)
    t_var_S = Af * sigma_Z_sq

    # Sum-of-squares constraint
    t_mu_S2 = Af * (sigma_M_sq + sigma_Z_sq)
    t_var_S2 = Af * (2.0 * sigma_Z_sq**2 + 4.0 * sigma_M_sq * sigma_Z_sq)

    for _it in range(n_iter):
        mz_prev = mz.clone()

        # ── 1. Sum constraint (S) ────────────────────────────────────
        mu_S = mz.sum(dim=1, keepdim=True)
        sig2_S = Sz.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Kalman Gain for S
        K_S = Sz / sig2_S

        # RTS Updates (Using strictly K for mean, K^2 for variance)
        mz = mz + K_S * (-mu_S)
        Sz = (Sz + (K_S) * (t_var_S - sig2_S)).clamp(min=1e-8)

        # ── 2. Sum-of-squares constraint (S2) ────────────────────────
        #  GMA moments for Z_i²:
        #    E[Z²]   = μ² + σ²
        #    Var[Z²]  = 2σ⁴ + 4σ²μ²
        #    Cov(Z,Z²) = 2μσ²
        mu_Z2 = mz**2 + Sz
        sig2_Z2 = 2.0 * Sz**2 + 4.0 * Sz * mz**2
        cov_Z_Z2 = 2.0 * mz * Sz

        mu_S2 = mu_Z2.sum(dim=1, keepdim=True)
        sig2_S2 = sig2_Z2.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Kalman Gain for S2
        K_S2 = cov_Z_Z2 / sig2_S2  # (B, A)

        # RTS Updates
        mz = mz + K_S2 * (t_mu_S2 - mu_S2)
        Sz = (Sz + (K_S2**2) * (t_var_S2 - sig2_S2)).clamp(min=1e-8)

        # ── Convergence ──────────────────────────────────────────────
        if (mz - mz_prev).abs().max().item() < tol:
            break

    return mz, Sz


# ======================================================================
#  Decoupled parameter formulation via inverse moment-matching
# ======================================================================


def _update_weights(layer, mz_prior, Sz_prior, mz_post, Sz_post):
    """
    Decoupled prior instantiation mapping activation targets to parameters.

    Step 1 — Per-neuron scaling factor absorbs the variance innovation:
        γ_i  =  √( mean_batch(σ²_{Z_i|·}) / mean_batch(σ²_{Z_i}) )
        μ_{W_ji}  ←  γ_i · μ_{W_ji}
        σ²_{W_ji} ←  γ_i² · σ²_{W_ji}
        σ²_{B_i}  ←  γ_i² · σ²_{B_i}

    Step 2 — Bias absorbs the residual mean shift after scaling:
        μ_{Z_scaled} = γ_i(μ_{Z_i} - μ_{B_i}) + μ_{B_i}
        μ_{B_i}  ←  μ_{B_i}  +  mean_batch(μ_{Z_i|·} − μ_{Z_scaled})

    Parameters
    ----------
    layer    : Linear
    mz_prior : (B, A_out)   forward-pass means (before update)
    Sz_prior : (B, A_out)   forward-pass variances (before update)
    mz_post  : (B, A_out)   posterior means (from constraints)
    Sz_post  : (B, A_out)   posterior variances (from constraints)
    """
    eps = 1e-12

    # ── Step 1: Variance update via weight scaling ───────────────────
    Sz_post_avg = Sz_post.mean(dim=0, keepdim=True).clamp(min=eps)
    Sz_prior_avg = Sz_prior.mean(dim=0, keepdim=True).clamp(min=eps)

    gamma = (Sz_post_avg / Sz_prior_avg).sqrt()  # (1, A_out)
    gamma = gamma.clamp(min=0.01, max=100.0)

    # Scale weight means by γ, weight variances by γ²
    layer.mw = layer.mw * gamma  # (A_in, A_out)
    layer.Sw = (layer.Sw * gamma**2).clamp(min=1e-8)

    # Scale bias variance by γ²
    if layer.has_bias:
        layer.Sb = (layer.Sb * gamma**2).clamp(min=1e-8)

    # ── Step 2: Mean update via bias absorption ──────────────────────
    if layer.has_bias:
        # Recompute the activation mean purely under the scaled weights
        # Z_prior = W*A + B  =>  W*A = Z_prior - B
        # Z_scaled = (γW)*A + B = γ(Z_prior - B) + B
        mz_scaled = gamma * (mz_prior - layer.mb) + layer.mb

        # Calculate the required additive shift to reach the target mz_post
        delta_mu = (mz_post - mz_scaled).mean(dim=0, keepdim=True)
        layer.mb = layer.mb + delta_mu


# ======================================================================
#  Public API
# ======================================================================


def inference_init(
    net, x_probe, sigma_M=1.0, sigma_Z=1.0, n_iter=1, tol=1e-6, skip_last=True, verbose=True
):
    """
    Inference-based initialization for a TAGI Sequential network.

    Passes a representative batch through the network and, at each
    Linear layer, iterates:  forward → constrain → parameter projection
    until the hidden-unit moments converge to the target distribution:

        M_i ~ N(0, σ_M²),   Z_i | M_i ~ N(M_i, σ_Z²)

    Each outer iteration re-forwards with the updated weights so the
    constraint always sees the actual current moments.

    Parameters
    ----------
    net       : Sequential   network (modified in-place)
    x_probe   : Tensor (B, ...)   representative batch; B >= 512 recommended
    sigma_M   : float   latent-mean std (epistemic spread, default 1.0)
    sigma_Z   : float   conditional noise std (variance floor, default 1.0)
    n_iter    : int     outer iterations per layer (forward→constrain→update)
    tol       : float   convergence threshold on moments (default 1e-6)
    skip_last : bool    skip the final Linear layer (default True)
    verbose   : bool    print per-layer diagnostics (default True)

    Returns
    -------
    net : Sequential
    """
    device = net.device
    B = x_probe.shape[0]
    sigma_M_sq = sigma_M**2
    sigma_Z_sq = sigma_Z**2
    sigma_total_sq = sigma_M_sq + sigma_Z_sq

    if verbose:
        print("=" * 72)
        print("  Inference-Based Initialization")
        print(f"  σ_M={sigma_M:.3f}  σ_Z={sigma_Z:.3f}  →  σ²_total={sigma_total_sq:.4f}")
        print(f"  Probe batch: B={B}  |  Layers: {len(net.layers)}")
        print(f"  Outer iterations per layer: {n_iter}")
        print("-" * 72)

    linear_indices = [i for i, lay in enumerate(net.layers) if isinstance(lay, Linear)]
    last_linear_idx = linear_indices[-1] if linear_indices else -1

    ma = x_probe.to(device)
    Sa = torch.zeros_like(ma)

    with torch.no_grad():
        for idx, layer in enumerate(net.layers):
            if isinstance(layer, Linear):
                A = layer.out_features
                is_last = skip_last and (idx == last_linear_idx)

                if is_last:
                    # Just forward, no update
                    mz = torch.matmul(ma, layer.mw) + layer.mb
                    Sz = triton_fused_var_forward(ma, Sa, layer.mw, layer.Sw, layer.Sb)
                    if verbose:
                        print(
                            f"  [{idx:2d}] Linear "
                            f"{layer.in_features}→{layer.out_features}: "
                            f"[output — skipped]"
                        )
                else:
                    # ── Outer loop: forward → constrain → update ─────
                    for oi in range(n_iter):  # noqa: B007
                        mz = torch.matmul(ma, layer.mw) + layer.mb
                        Sz = triton_fused_var_forward(ma, Sa, layer.mw, layer.Sw, layer.Sb)

                        # Single-pass moment matching constraint
                        mz_post, Sz_post = _apply_constraints(
                            mz, Sz, A, sigma_M_sq, sigma_Z_sq, n_iter=1, tol=tol
                        )

                        # Decoupled parameter prior formulation
                        # Note: _update_weights no longer needs ma or Sa
                        _update_weights(layer, mz, Sz, mz_post, Sz_post)

                        # Check convergence: re-forward and compare
                        mz_new = torch.matmul(ma, layer.mw) + layer.mb

                        delta = (mz_new - mz).abs().max().item()
                        if delta < tol:
                            break

                    # Final forward with converged weights
                    mz = torch.matmul(ma, layer.mw) + layer.mb
                    Sz = triton_fused_var_forward(ma, Sa, layer.mw, layer.Sw, layer.Sb)

                    if verbose:
                        Af = float(A)
                        # Targets
                        t_mu_S = 0.0
                        t_var_S = Af * sigma_Z_sq
                        t_mu_S2 = Af * (sigma_M_sq + sigma_Z_sq)
                        t_var_S2 = Af * (2.0 * sigma_Z_sq**2 + 4.0 * sigma_M_sq * sigma_Z_sq)

                        # Achieved (per-sample aggregates, averaged over batch)
                        a_mu_S = mz.sum(dim=1).mean().item()
                        a_var_S = Sz.sum(dim=1).mean().item()
                        mu_Z2 = mz**2 + Sz
                        sig2_Z2 = 2.0 * Sz**2 + 4.0 * Sz * mz**2
                        a_mu_S2 = mu_Z2.sum(dim=1).mean().item()
                        a_var_S2 = sig2_Z2.sum(dim=1).mean().item()

                        print(
                            f"  [{idx:2d}] Linear "
                            f"{layer.in_features}→{layer.out_features}  "
                            f"(A={A}, {oi + 1} iters)"
                        )
                        print(f"       {'':8s}  {'target':>12s}  {'achieved':>12s}")
                        print(f"       μ_S       {t_mu_S:12.4f}  {a_mu_S:12.4f}")
                        print(f"       σ²_S      {t_var_S:12.4f}  {a_var_S:12.4f}")
                        print(f"       μ_S2      {t_mu_S2:12.4f}  {a_mu_S2:12.4f}")
                        print(f"       σ²_S2     {t_var_S2:12.4f}  {a_var_S2:12.4f}")

                # Propagate corrected moments to the next layer
                ma, Sa = mz, Sz.clamp(min=1e-8)

            elif isinstance(layer, _PASS_THROUGH):
                ma, Sa = layer.forward(ma, Sa)

    if verbose:
        print("=" * 72)

    return net
