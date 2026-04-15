"""
Adam-TAGI: Integrating Gradient-Based Optimization Dynamics into TAGI.

Maps Adam's temporal smoothing and adaptive scaling into the TAGI framework
without violating its probabilistic structure. Three mechanisms:

1. Parameter Momentum — EMA of parameter updates (m_t, v_t)
2. Adaptive Gain Modulation — dimensionless modulation A_t ∈ [0,1]
3. Process Noise Injection — variance floor to prevent plasticity loss

Usage
-----
    from src.optimizer import AdamTAGI

    net = Sequential([...])
    opt = AdamTAGI(net)

    for epoch in range(n_epochs):
        for xb, yb in loader:
            y_pred_mu, y_pred_var = opt.step(xb, yb, sigma_v)
"""

import torch
import triton
import triton.language as tl

from .layers.batchnorm2d import BatchNorm2D
from .layers.conv2d import Conv2D
from .layers.linear import Linear
from .layers.resblock import ResBlock
from .update.observation import compute_innovation
from .update.parameters import get_cap_factor

_LEARNABLE_LAYERS = (Linear, Conv2D, BatchNorm2D, ResBlock)

BLOCK = 1024


# ======================================================================
#  Triton kernel — Adam-TAGI parameter update
# ======================================================================


@triton.jit
def _adam_tagi_update_kernel(
    # Parameter tensors (in-place)
    m_ptr,
    S_ptr,
    # Raw deltas from backward pass
    dm_ptr,
    dS_ptr,
    # EMA states (in-place)
    ema_m_ptr,
    ema_v_ptr,
    # Scalars
    beta1,
    beta2,
    eps,
    eps_Q,
    bias_corr1,  # 1 - beta1^t
    bias_corr2,  # 1 - beta2^t
    cap_factor,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    S = tl.load(S_ptr + offs, mask=valid)
    dm = tl.load(dm_ptr + offs, mask=valid)
    dS = tl.load(dS_ptr + offs, mask=valid)
    ema_m = tl.load(ema_m_ptr + offs, mask=valid)
    ema_v = tl.load(ema_v_ptr + offs, mask=valid)

    # ── 1. Update EMA of parameter updates ──
    # m_t = β₁ m_{t-1} + (1 - β₁) Δμ
    # v_t = β₂ v_{t-1} + (1 - β₂) Δμ²
    ema_m_new = beta1 * ema_m + (1.0 - beta1) * dm
    ema_v_new = beta2 * ema_v + (1.0 - beta2) * dm * dm

    # Bias-corrected estimates
    m_hat = ema_m_new / bias_corr1
    v_hat = ema_v_new / bias_corr2

    # ── 2. Adaptive Gain Modulation ──
    # A_t = |m̂_t| / (√v̂_t + ε)  ∈ [0, 1]
    A = tl.abs(m_hat) / (tl.sqrt(tl.maximum(v_hat, 0.0)) + eps)
    A = tl.minimum(A, 1.0)  # clamp to [0, 1]

    # ── 3. Modulated mean update with capping ──
    # μ_new = μ + A_t ⊙ m̂_t  (smoothed, modulated momentum)
    dm_adam = A * m_hat

    # Apply cap for safety: delta_bar = √S / cap_factor
    delta_bar = tl.sqrt(tl.maximum(S, 1e-10)) / cap_factor
    dm_sign = tl.where(dm_adam > 0.0, 1.0, tl.where(dm_adam < 0.0, -1.0, 0.0))
    dm_capped = dm_sign * tl.minimum(tl.abs(dm_adam), delta_bar)
    m_new = m + dm_capped

    # ── 4. Modulated variance update ──
    # Σ_new = Σ + A² ⊙ ΔΣ  (scale variance reduction by A²)
    dS_adam = A * A * dS

    dS_sign = tl.where(dS_adam > 0.0, 1.0, tl.where(dS_adam < 0.0, -1.0, 0.0))
    dS_capped = dS_sign * tl.minimum(tl.abs(dS_adam), delta_bar)
    S_new = tl.maximum(S + dS_capped, 1e-5)

    # ── 5. Process noise injection (plasticity floor) ──
    S_new = S_new + eps_Q

    # Store results
    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)
    tl.store(ema_m_ptr + offs, ema_m_new, mask=valid)
    tl.store(ema_v_ptr + offs, ema_v_new, mask=valid)


def _adam_tagi_update(
    m, S, dm, dS, ema_m, ema_v, beta1, beta2, eps, eps_Q, bias_corr1, bias_corr2, cap_factor
):
    """Apply Adam-TAGI update to a single parameter tensor (in-place)."""
    n = m.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _adam_tagi_update_kernel[grid](
        m.view(-1),
        S.view(-1),
        dm.view(-1),
        dS.view(-1),
        ema_m.view(-1),
        ema_v.view(-1),
        beta1,
        beta2,
        eps,
        eps_Q,
        bias_corr1,
        bias_corr2,
        cap_factor,
        n,
        BLOCK=BLOCK,
    )


# ======================================================================
#  Adam-TAGI Optimizer
# ======================================================================


class AdamTAGI:
    """
    Adam-TAGI optimizer for Bayesian neural networks.

    Wraps a Sequential network and replaces the standard capped update
    with Adam-style temporal smoothing and adaptive gain modulation.

    Parameters
    ----------
    net       : Sequential   the TAGI network
    beta1     : float        EMA decay for first moment (default 0.9)
    beta2     : float        EMA decay for second moment (default 0.999)
    eps       : float        numerical stability for A_t (default 1e-8)
    eps_Q     : float        process noise magnitude (default 1e-6)
    """

    def __init__(self, net, beta1=0.9, beta2=0.999, eps=1e-8, eps_Q=1e-6):
        self.net = net
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eps_Q = eps_Q
        self.t = 0  # global step counter

        # EMA states keyed by (id(layer), param_name) to survive tensor reassignment
        self._states = {}
        # Cache layer list so iteration order is stable
        self._learnable_list = list(self._iter_learnable_layers())

    def _get_state(self, layer, attr):
        """Get or lazily create EMA state for a parameter tensor."""
        key = (id(layer), attr)
        if key not in self._states:
            param = getattr(layer, attr)
            self._states[key] = {
                "ema_m": torch.zeros_like(param),
                "ema_v": torch.zeros_like(param),
            }
        return self._states[key]

    def _iter_learnable_layers(self):
        """Yield all leaf learnable layers (expanding ResBlocks)."""
        for layer in self.net.layers:
            if isinstance(layer, ResBlock):
                yield from layer._learnable
            elif isinstance(layer, (Linear, Conv2D, BatchNorm2D)):
                yield layer

    def step(self, x_batch, y_batch, sigma_v):
        """
        Perform one Adam-TAGI training step.

        Parameters
        ----------
        x_batch : Tensor  input mini-batch
        y_batch : Tensor  target mini-batch
        sigma_v : float   observation noise std

        Returns
        -------
        y_pred_mu  : Tensor  predicted means (before update)
        y_pred_var : Tensor  predicted variances (before update)
        """
        batch_size = x_batch.shape[0]
        self.t += 1

        # ── 1. Forward ──
        y_pred_mu, y_pred_var = self.net.forward(x_batch)

        # ── 2. Output innovation ──
        delta_mu, delta_var = compute_innovation(y_batch, y_pred_mu, y_pred_var, sigma_v)

        # ── 3. Backward (compute + store deltas, NO param update) ──
        for layer in reversed(self.net.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # ── 4. Adam-TAGI update ──
        cap_factor = get_cap_factor(batch_size)
        bias_corr1 = 1.0 - self.beta1**self.t
        bias_corr2 = 1.0 - self.beta2**self.t

        for layer in self._learnable_list:
            self._update_param(
                layer, "mw", "Sw", "delta_mw", "delta_Sw", cap_factor, bias_corr1, bias_corr2
            )
            if getattr(layer, "has_bias", True):
                self._update_param(
                    layer, "mb", "Sb", "delta_mb", "delta_Sb", cap_factor, bias_corr1, bias_corr2
                )

        return y_pred_mu, y_pred_var

    def _update_param(
        self, layer, m_attr, S_attr, dm_attr, dS_attr, cap_factor, bias_corr1, bias_corr2
    ):
        """Apply Adam-TAGI update to one parameter pair (mean, variance)."""
        m = getattr(layer, m_attr)
        S = getattr(layer, S_attr)
        dm = getattr(layer, dm_attr)
        dS = getattr(layer, dS_attr)

        state = self._get_state(layer, m_attr)
        ema_m = state["ema_m"]
        ema_v = state["ema_v"]

        _adam_tagi_update(
            m,
            S,
            dm,
            dS,
            ema_m,
            ema_v,
            self.beta1,
            self.beta2,
            self.eps,
            self.eps_Q,
            bias_corr1,
            bias_corr2,
            cap_factor,
        )

    def reset(self):
        """Reset optimizer state (EMA buffers and step counter)."""
        self.t = 0
        self._states.clear()

    def __repr__(self):
        n_states = len(self._states)
        n_params = sum(s["ema_m"].numel() for s in self._states.values())
        return (
            f"AdamTAGI(β1={self.beta1}, β2={self.beta2}, "
            f"ε={self.eps}, ε_Q={self.eps_Q}, "
            f"tensors={n_states}, params={n_params}, step={self.t})"
        )
