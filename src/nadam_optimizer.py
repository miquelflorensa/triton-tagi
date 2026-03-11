"""
Nadam-TAGI: Integrating Kinematic Parameter Dynamics into TAGI.

Promotes parameter velocity to a fully probabilistic state variable with
explicit cross-covariance tracking.  Two-phase step:

1. **Predict** (before forward): Nesterov-like look-ahead via the linear
   state transition F, with adaptive process noise Q derived from the
   running innovation variance (Eq. 10).
2. **Update** (after backward): Kalman posterior correction on both
   parameters and velocity via the cross-covariance ratio r (Eqs. 13-20).

Hyperparameters
---------------
beta1       : float  momentum coefficient (default 0.9)
gamma       : float  process noise EMA smoothing (default 0.99)
alpha_theta : float  process noise scaling for parameters (default 1e-3)
alpha_m     : float  process noise scaling for velocity (default 1e-3)
sigma_m0_sq : float  initial velocity variance (default 1e-8)
eps_Q       : float  initial innovation variance seed (default 1e-6)

Per-parameter storage: mu_m, Sigma_m, C, Sigma_delta_mu (4 scalars)
vs. 2 for vanilla TAGI and 3 for Adam.

Usage
-----
    from src.nadam_optimizer import NadamTAGI

    net = Sequential([...])
    opt = NadamTAGI(net)

    for epoch in range(n_epochs):
        for xb, yb in loader:
            y_pred_mu, y_pred_var = opt.step(xb, yb, sigma_v)
"""

import torch
import triton
import triton.language as tl

from .update.observation import compute_innovation
from .update.parameters import get_cap_factor
from .layers.linear import Linear
from .layers.conv2d import Conv2D
from .layers.batchnorm2d import BatchNorm2D
from .layers.resblock import ResBlock

_LEARNABLE_LAYERS = (Linear, Conv2D, BatchNorm2D, ResBlock)

BLOCK = 1024


# ======================================================================
#  Triton kernel — Predict phase (Eqs. 5–10)
# ======================================================================

@triton.jit
def _nadam_predict_kernel(
    # Parameter tensors (in-place)
    m_ptr, S_ptr,
    # Velocity state (in-place)
    mu_m_ptr, Sigma_m_ptr, C_ptr,
    # Running innovation variance (read-only in predict)
    Sigma_dmu_ptr,
    # Scalars
    beta1, alpha_theta, alpha_m,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    # Load all state into registers
    m       = tl.load(m_ptr + offs, mask=valid)
    S       = tl.load(S_ptr + offs, mask=valid)
    mu_m    = tl.load(mu_m_ptr + offs, mask=valid)
    Sigma_m = tl.load(Sigma_m_ptr + offs, mask=valid)
    C       = tl.load(C_ptr + offs, mask=valid)
    Sdmu    = tl.load(Sigma_dmu_ptr + offs, mask=valid)

    # ── Eq. 10: Adaptive process noise from previous step's innovation EMA ──
    # Q^(i)_{θ,t} = α_θ · Σ^(i)_{Δμ,t-1}
    # Q^(i)_{m,t} = α_m · Σ^(i)_{Δμ,t-1}
    # Q^(i)_{θm,t} = 0
    Q_theta = alpha_theta * Sdmu
    Q_m     = alpha_m * Sdmu

    # ── Eqs. 5–6: Predictive prior mean (Nesterov look-ahead) ──
    # μ^(i)_{θ,t} = μ^(i)_{θ|y,t-1} + β₁ · μ^(i)_{m,t-1|y}
    # μ^(i)_{m,t} = β₁ · μ^(i)_{m,t-1|y}
    m_new    = m + beta1 * mu_m
    mu_m_new = beta1 * mu_m

    # ── Eqs. 7–9: Predictive prior covariance (F P_{t-1|y} F^T + Q) ──
    # Σ^(i)_{θ,t} = Σ^(i)_{θ|y,t-1} + 2β₁C^(i)_{t-1|y} + β₁²Σ^(i)_{m,t-1|y} + Q_θ
    # C^(i)_t = β₁(C^(i)_{t-1|y} + β₁Σ^(i)_{m,t-1|y}) + Q_{θm}   [Q_{θm} = 0]
    # Σ^(i)_{m,t} = β₁²Σ^(i)_{m,t-1|y} + Q_m
    beta1_sq = beta1 * beta1
    S_new       = S + 2.0 * beta1 * C + beta1_sq * Sigma_m + Q_theta
    C_new       = beta1 * (C + beta1 * Sigma_m)
    Sigma_m_new = beta1_sq * Sigma_m + Q_m

    # Numerical floors
    S_new       = tl.maximum(S_new, 1e-5)
    Sigma_m_new = tl.maximum(Sigma_m_new, 0.0)

    # Store predicted state
    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)
    tl.store(mu_m_ptr + offs, mu_m_new, mask=valid)
    tl.store(Sigma_m_ptr + offs, Sigma_m_new, mask=valid)
    tl.store(C_ptr + offs, C_new, mask=valid)


# ======================================================================
#  Triton kernel — Update phase (Eqs. 11–20)
# ======================================================================

@triton.jit
def _nadam_update_kernel(
    # Parameter tensors (in-place)
    m_ptr, S_ptr,
    # Raw deltas from backward pass (read-only)
    dm_ptr, dS_ptr,
    # Velocity state (in-place)
    mu_m_ptr, Sigma_m_ptr, C_ptr,
    # Running innovation variance (in-place)
    Sigma_dmu_ptr,
    # Initial parameter variance (read-only, for innovation clamp)
    S0_ptr,
    # Scalars
    gamma, cap_factor, innov_clamp_c,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m       = tl.load(m_ptr + offs, mask=valid)
    S       = tl.load(S_ptr + offs, mask=valid)
    dm      = tl.load(dm_ptr + offs, mask=valid)
    dS      = tl.load(dS_ptr + offs, mask=valid)
    mu_m    = tl.load(mu_m_ptr + offs, mask=valid)
    Sigma_m = tl.load(Sigma_m_ptr + offs, mask=valid)
    C       = tl.load(C_ptr + offs, mask=valid)
    Sdmu    = tl.load(Sigma_dmu_ptr + offs, mask=valid)
    S0      = tl.load(S0_ptr + offs, mask=valid)

    # ── Eq. 16: Cross-covariance ratio (from predicted Σ_θ, before update) ──
    # r^(i)_t = C^(i)_t / Σ^(i)_{θ,t}
    r = C / tl.maximum(S, 1e-10)

    # ── Eq. 13: Parameter mean posterior (with cuTAGI capping) ──
    # μ^(i)_{θ|y,t} = μ^(i)_{θ,t} + Δμ^(i)_{θ,t}
    delta_bar = tl.sqrt(tl.maximum(S, 1e-10)) / cap_factor
    dm_sign   = tl.where(dm > 0.0, 1.0, tl.where(dm < 0.0, -1.0, 0.0))
    dm_capped = dm_sign * tl.minimum(tl.abs(dm), delta_bar)
    m_new     = m + dm_capped

    # ── Eq. 15: Parameter variance posterior (with cuTAGI capping + floor) ──
    # Σ^(i)_{θ|y,t} = Σ^(i)_{θ,t} + (J^(i)_{θ,t})² · δ^(i)_{Σ,t}
    dS_sign   = tl.where(dS > 0.0, 1.0, tl.where(dS < 0.0, -1.0, 0.0))
    dS_capped = dS_sign * tl.minimum(tl.abs(dS), delta_bar)
    S_new     = tl.maximum(S + dS_capped, 1e-5)

    # ── Eqs. 17–19: Velocity posterior via cross-covariance ──
    #
    # The paper proves PSD when J²|δ_Σ| ≤ Σ_θ, which holds for uncapped
    # TAGI.  With cuTAGI capping, |dS_capped| can exceed S (the cap
    # sqrt(S)/cap_factor > S when S < 1/cap_factor²), violating PSD.
    # We use the actual S change (after capping + floor) which is bounded
    # by construction: |actual_dS| ≤ S - floor < S.
    actual_dS = S_new - S

    # Eq. 17: μ^(i)_{m,t|y} = μ^(i)_{m,t} + r^(i)_t · Δμ^(i)_{θ,t}
    mu_m_new = mu_m + r * dm_capped

    # Eq. 18: Σ^(i)_{m,t|y} = Σ^(i)_{m,t} + (r^(i)_t)² · (J^(i))² · δ_Σ
    Sigma_m_new = Sigma_m + r * r * actual_dS

    # Eq. 19: C^(i)_{t|y} = C^(i)_t + r^(i)_t · (J^(i))² · δ_Σ
    C_new = C + r * actual_dS

    # Numerical floor
    Sigma_m_new = tl.maximum(Sigma_m_new, 0.0)

    # ── Eq. 11: Running innovation variance EMA (with innovation clamp) ──
    # The innovation fed into the EMA is clamped at ±c·√S₀ where S₀ is the
    # initial parameter variance.  This breaks the compound feedback loop
    # where dm_capped² ∝ S (via the sqrt(S) cap) would make Σ_Δμ grow with
    # S, feeding back through Q → Σ_m → C → S.  By referencing S₀ (fixed),
    # Σ_Δμ is bounded independently of S's growth.
    innov_cap = innov_clamp_c * tl.sqrt(tl.maximum(S0, 1e-10))
    dm_for_ema = tl.minimum(tl.abs(dm_capped), innov_cap)
    Sdmu_new = gamma * Sdmu + (1.0 - gamma) * dm_for_ema * dm_for_ema

    # Store posterior state
    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)
    tl.store(mu_m_ptr + offs, mu_m_new, mask=valid)
    tl.store(Sigma_m_ptr + offs, Sigma_m_new, mask=valid)
    tl.store(C_ptr + offs, C_new, mask=valid)
    tl.store(Sigma_dmu_ptr + offs, Sdmu_new, mask=valid)


# ======================================================================
#  Python wrappers
# ======================================================================

def _nadam_predict(m, S, mu_m, Sigma_m, C, Sigma_dmu,
                   beta1, alpha_theta, alpha_m):
    """Apply Nesterov predict step to a single parameter tensor (in-place)."""
    n = m.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _nadam_predict_kernel[grid](
        m.view(-1), S.view(-1),
        mu_m.view(-1), Sigma_m.view(-1), C.view(-1),
        Sigma_dmu.view(-1),
        beta1, alpha_theta, alpha_m,
        n, BLOCK=BLOCK,
    )


def _nadam_update(m, S, dm, dS, mu_m, Sigma_m, C, Sigma_dmu, S0,
                  gamma, cap_factor, innov_clamp_c):
    """Apply Kalman posterior update to a single parameter tensor (in-place)."""
    n = m.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _nadam_update_kernel[grid](
        m.view(-1), S.view(-1),
        dm.view(-1), dS.view(-1),
        mu_m.view(-1), Sigma_m.view(-1), C.view(-1),
        Sigma_dmu.view(-1),
        S0.view(-1),
        gamma, cap_factor, innov_clamp_c,
        n, BLOCK=BLOCK,
    )


# ======================================================================
#  Nadam-TAGI Optimizer
# ======================================================================

class NadamTAGI:
    """
    Nadam-TAGI optimizer for Bayesian neural networks.

    Wraps a Sequential network and replaces the standard capped update
    with a two-phase augmented-state Kalman filter: Nesterov-like
    momentum prediction followed by posterior correction of both
    parameters and velocity.

    Parameters
    ----------
    net           : Sequential  the TAGI network
    beta1         : float       momentum coefficient (default 0.9)
    gamma         : float       process noise EMA smoothing (default 0.99)
    alpha_theta   : float       process noise scaling for Σ_θ (default 1e-3)
    alpha_m       : float       process noise scaling for Σ_m (default 1e-3)
    sigma_m0_sq   : float       initial velocity variance (default 1e-8)
    eps_Q         : float       initial Σ_{Δμ} seed (default 1e-6)
    innov_clamp_c : float       innovation clamp at ±c·√S₀ for Σ_Δμ EMA (default 3.0)
    """

    def __init__(self, net, beta1=0.9, gamma=0.99,
                 alpha_theta=1e-3, alpha_m=1e-3,
                 sigma_m0_sq=1e-8, eps_Q=1e-6,
                 innov_clamp_c=3.0):
        self.net = net
        self.beta1 = beta1
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_m = alpha_m
        self.sigma_m0_sq = sigma_m0_sq
        self.eps_Q = eps_Q
        self.innov_clamp_c = innov_clamp_c
        self.t = 0

        # Augmented state keyed by (id(layer), param_name)
        self._states = {}
        self._learnable_list = list(self._iter_learnable_layers())

    # ------------------------------------------------------------------
    #  State management
    # ------------------------------------------------------------------

    def _get_state(self, layer, m_attr, S_attr):
        """Get or lazily create augmented state for a parameter tensor.

        Per Eq. 3:  μ_m,0 = 0,  Σ_m,0 = σ²_{m,0},  C_0 = 0
        Per §2.3.4: Σ_{Δμ,0} = ε_Q

        Also stores S0 (snapshot of initial parameter variance) for the
        innovation clamp reference.
        """
        key = (id(layer), m_attr)
        if key not in self._states:
            param = getattr(layer, m_attr)
            S = getattr(layer, S_attr)
            self._states[key] = {
                'mu_m':      torch.zeros_like(param),
                'Sigma_m':   torch.full_like(param, self.sigma_m0_sq),
                'C':         torch.zeros_like(param),
                'Sigma_dmu': torch.full_like(param, self.eps_Q),
                'S0':        S.clone(),
            }
        return self._states[key]

    def _iter_learnable_layers(self):
        """Yield all leaf learnable layers (expanding ResBlocks)."""
        for layer in self.net.layers:
            if isinstance(layer, ResBlock):
                for sub in layer._learnable:
                    yield sub
            elif isinstance(layer, (Linear, Conv2D, BatchNorm2D)):
                yield layer

    # ------------------------------------------------------------------
    #  Per-parameter predict and update
    # ------------------------------------------------------------------

    def _predict_param(self, layer, m_attr, S_attr):
        """Apply predict step (Eqs. 5–10) to one parameter pair."""
        m = getattr(layer, m_attr)
        S = getattr(layer, S_attr)
        state = self._get_state(layer, m_attr, S_attr)
        _nadam_predict(
            m, S,
            state['mu_m'], state['Sigma_m'], state['C'],
            state['Sigma_dmu'],
            self.beta1, self.alpha_theta, self.alpha_m,
        )

    def _update_param(self, layer, m_attr, S_attr, dm_attr, dS_attr,
                      cap_factor):
        """Apply posterior update (Eqs. 11–20) to one parameter pair."""
        m  = getattr(layer, m_attr)
        S  = getattr(layer, S_attr)
        dm = getattr(layer, dm_attr)
        dS = getattr(layer, dS_attr)
        state = self._get_state(layer, m_attr, S_attr)
        _nadam_update(
            m, S, dm, dS,
            state['mu_m'], state['Sigma_m'], state['C'],
            state['Sigma_dmu'],
            state['S0'],
            self.gamma, cap_factor, self.innov_clamp_c,
        )

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------

    def step(self, x_batch, y_batch, sigma_v):
        """
        Perform one Nadam-TAGI training step.

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

        # ── Step 1–2: Compute process noise + Predict (Eqs. 5–10) ──
        for layer in self._learnable_list:
            self._predict_param(layer, 'mw', 'Sw')
            if getattr(layer, 'has_bias', True):
                self._predict_param(layer, 'mb', 'Sb')

        # ── Forward pass (on look-ahead parameters) ──
        y_pred_mu, y_pred_var = self.net.forward(x_batch)

        # ── Output innovation ──
        delta_mu, delta_var = compute_innovation(
            y_batch, y_pred_mu, y_pred_var, sigma_v
        )

        # ── Step 3: Backward (compute TAGI deltas from look-ahead position) ──
        for layer in reversed(self.net.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # ── Steps 4–6: Update parameters, velocity, cross-cov, EMA ──
        cap_factor = get_cap_factor(batch_size)
        for layer in self._learnable_list:
            self._update_param(layer, 'mw', 'Sw', 'delta_mw', 'delta_Sw',
                               cap_factor)
            if getattr(layer, 'has_bias', True):
                self._update_param(layer, 'mb', 'Sb', 'delta_mb', 'delta_Sb',
                                   cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def reset(self):
        """Reset optimizer state (augmented state buffers and step counter)."""
        self.t = 0
        self._states.clear()

    def __repr__(self):
        n_states = len(self._states)
        n_params = sum(s['mu_m'].numel() for s in self._states.values())
        return (f"NadamTAGI(β1={self.beta1}, γ={self.gamma}, "
                f"α_θ={self.alpha_theta}, α_m={self.alpha_m}, "
                f"σ²_m0={self.sigma_m0_sq}, ε_Q={self.eps_Q}, "
                f"c={self.innov_clamp_c}, "
                f"tensors={n_states}, params={n_params}, step={self.t})")
