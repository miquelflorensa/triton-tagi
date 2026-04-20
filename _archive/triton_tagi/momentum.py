"""
State-Space Momentum for TAGI.

Each parameter θ is augmented with a velocity v, forming a 2D state [θ, v].
Before each mini-batch, a kinematic transition is applied:

    [θ, v]_{t|t-1} = F [θ, v]_{t-1} + w,   F = [[1, dt], [0, γ]]

where dt controls how much velocity is integrated into the position.

The forward pass uses only the θ marginal (unchanged from standard TAGI).
The backward pass computes the augmented RTS gain:

    J_x = [J_θ;  (σ_θv / σ²_θ) J_θ]

and updates both θ and v simultaneously.  Because each parameter's 2×2 block
is independent, the O(N) linear complexity of standard TAGI is preserved.

Decoupled mode (default)
------------------------
In decoupled mode, the predict step applies velocity to the mean and evolves
Sc/Sv for the augmented gain, but does NOT inflate S from velocity.  This
avoids the per-mini-batch variance explosion that otherwise constrains the
momentum ratio r = Sc/S to negligible values.

Usage
-----
    from src.momentum import StateSpaceMomentum

    net = Sequential([...])
    opt = StateSpaceMomentum(net, gamma=0.9, dt=0.1)

    for epoch in ...:
        for x_batch, y_batch in loader:
            opt.step(x_batch, y_batch, sigma_v)
"""

import torch
import triton
import triton.language as tl

from .layers.batchnorm2d import BatchNorm2D
from .layers.conv2d import Conv2D
from .layers.linear import Linear
from .update.observation import compute_innovation
from .update.parameters import get_cap_factor

BLOCK = 1024

# Layer types that support per-element momentum states
_MOMENTUM_SUPPORTED = (Linear, Conv2D, BatchNorm2D)


# ======================================================================
#  Triton kernel — Kinematic prediction step
#
#  F = [[1, dt], [0, γ]]
#  μ_new  = F μ_old
#  Σ_new  = F Σ_old F^T + Q
# ======================================================================


@triton.jit
def _predict_kernel(
    m_ptr,
    S_ptr,
    mv_ptr,
    Sv_ptr,
    Sc_ptr,
    gamma,
    dt,
    q_theta,
    q_v,
    vel_scale,  # 1.0 = full Bayesian, 0.0 = decoupled (don't inflate S)
    vel_cap_frac,  # max velocity contribution as fraction of sqrt(S)
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    S = tl.load(S_ptr + offs, mask=valid)
    mv = tl.load(mv_ptr + offs, mask=valid)
    Sv = tl.load(Sv_ptr + offs, mask=valid)
    Sc = tl.load(Sc_ptr + offs, mask=valid)

    # Mean: [θ + capped(dt·v),  γ·v]
    vel_contrib = dt * mv
    vel_bar = tl.sqrt(tl.maximum(S, 1e-10)) * vel_cap_frac
    vel_sign = tl.where(vel_contrib > 0.0, 1.0, tl.where(vel_contrib < 0.0, -1.0, 0.0))
    vel_capped = vel_sign * tl.minimum(tl.abs(vel_contrib), vel_bar)
    m_new = m + vel_capped
    mv_new = gamma * mv

    # Covariance: F Σ F^T + Q
    #   σ²_θ_new  = σ²_θ + vel_scale·(2·dt·σ_θv + dt²·σ²_v) + q_θ
    #   σ_θv_new  = γ·(σ_θv + dt·σ²_v)
    #   σ²_v_new  = γ²·σ²_v + q_v
    S_new = S + vel_scale * (2.0 * dt * Sc + dt * dt * Sv) + q_theta
    Sc_new = gamma * (Sc + dt * Sv)
    Sv_new = gamma * gamma * Sv + q_v

    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)
    tl.store(mv_ptr + offs, mv_new, mask=valid)
    tl.store(Sv_ptr + offs, Sv_new, mask=valid)
    tl.store(Sc_ptr + offs, Sc_new, mask=valid)


# ======================================================================
#  Triton kernel — Augmented RTS posterior update
#
#  Updates both θ and v using the standard θ deltas and the augmented
#  gain ratio  r = σ_θv / σ²_θ.
# ======================================================================


@triton.jit
def _augmented_update_kernel(
    m_ptr,
    S_ptr,
    mv_ptr,
    Sv_ptr,
    Sc_ptr,
    dm_ptr,
    dS_ptr,
    cap_factor,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    S = tl.load(S_ptr + offs, mask=valid)
    mv = tl.load(mv_ptr + offs, mask=valid)
    Sv = tl.load(Sv_ptr + offs, mask=valid)
    Sc = tl.load(Sc_ptr + offs, mask=valid)
    dm = tl.load(dm_ptr + offs, mask=valid)
    dS = tl.load(dS_ptr + offs, mask=valid)

    # Augmented gain ratio
    r = Sc / tl.maximum(S, 1e-10)

    # ── θ update (standard capped update) ──
    delta_bar = tl.sqrt(tl.maximum(S, 1e-10)) / cap_factor

    dm_sign = tl.where(dm > 0.0, 1.0, tl.where(dm < 0.0, -1.0, 0.0))
    dm_capped = dm_sign * tl.minimum(tl.abs(dm), delta_bar)
    m_new = m + dm_capped

    dS_sign = tl.where(dS > 0.0, 1.0, tl.where(dS < 0.0, -1.0, 0.0))
    dS_capped = dS_sign * tl.minimum(tl.abs(dS), delta_bar)
    S_new = tl.maximum(S + dS_capped, 1e-5)

    # ── v update (augmented gain = r · J_θ) ──
    dm_v = r * dm
    dS_v = r * r * dS

    delta_bar_v = tl.sqrt(tl.maximum(Sv, 1e-10)) / cap_factor

    dmv_sign = tl.where(dm_v > 0.0, 1.0, tl.where(dm_v < 0.0, -1.0, 0.0))
    dmv_capped = dmv_sign * tl.minimum(tl.abs(dm_v), delta_bar_v)
    mv_new = mv + dmv_capped

    dSv_sign = tl.where(dS_v > 0.0, 1.0, tl.where(dS_v < 0.0, -1.0, 0.0))
    dSv_capped = dSv_sign * tl.minimum(tl.abs(dS_v), delta_bar_v)
    Sv_new = tl.maximum(Sv + dSv_capped, 1e-12)  # velocity needs no positive floor

    # ── Cross-covariance update + Cauchy-Schwarz bound ──
    dSc = r * dS
    Sc_new = Sc + dSc
    bound = tl.sqrt(tl.maximum(S_new * Sv_new, 0.0))
    Sc_new = tl.maximum(tl.minimum(Sc_new, bound), -bound)

    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)
    tl.store(mv_ptr + offs, mv_new, mask=valid)
    tl.store(Sv_ptr + offs, Sv_new, mask=valid)
    tl.store(Sc_ptr + offs, Sc_new, mask=valid)


# ======================================================================
#  Python wrappers
# ======================================================================


def predict_params(m, S, mv, Sv, Sc, gamma, dt, q_theta, q_v, vel_scale=1.0, vel_cap_frac=0.1):
    """In-place kinematic prediction for one parameter tensor."""
    n = m.numel()
    _predict_kernel[(triton.cdiv(n, BLOCK),)](
        m.view(-1),
        S.view(-1),
        mv.view(-1),
        Sv.view(-1),
        Sc.view(-1),
        gamma,
        dt,
        q_theta,
        q_v,
        vel_scale,
        vel_cap_frac,
        n,
        BLOCK=BLOCK,
    )


def augmented_update(m, S, mv, Sv, Sc, dm, dS, cap_factor):
    """In-place augmented posterior update for one parameter tensor."""
    n = m.numel()
    _augmented_update_kernel[(triton.cdiv(n, BLOCK),)](
        m.view(-1),
        S.view(-1),
        mv.view(-1),
        Sv.view(-1),
        Sc.view(-1),
        dm.view(-1),
        dS.view(-1),
        cap_factor,
        n,
        BLOCK=BLOCK,
    )


# ======================================================================
#  StateSpaceMomentum
# ======================================================================


class StateSpaceMomentum:
    """
    State-space momentum optimizer for TAGI networks.

    Wraps a Sequential network and provides a step() that includes
    kinematic prediction before the forward pass and an augmented RTS
    posterior update that simultaneously updates parameters and velocities.

    Parameters
    ----------
    net          : Sequential
    gamma        : float   friction / decay for velocity  (default 0.9)
    dt           : float   velocity integration step size (default 1.0)
                           Controls F = [[1, dt], [0, γ]].
    q_theta      : float   process noise on parameter variance (default None → auto)
    q_v          : float   process noise on velocity variance  (default None → auto)
    sigma_v0     : float   initial velocity std (default None → auto)
    decouple     : bool    If True (default), don't inflate S from velocity.
    vel_cap_frac : float   Max velocity contribution per step as fraction of
                           sqrt(S).  Prevents overshooting regardless of dt.
                           Default 0.1 → velocity can shift mean by at most
                           10% of parameter std per step.
    eps_Q        : float   Minimum process noise floor (default 1e-6).
                           Prevents S collapse on hard tasks (e.g. CIFAR-100).
    """

    def __init__(
        self,
        net,
        gamma=0.9,
        dt=1.0,
        q_theta=None,
        q_v=None,
        sigma_v0=None,
        decouple=True,
        vel_cap_frac=0.1,
        eps_Q=1e-6,
    ):
        self.net = net
        self.gamma = gamma
        self.dt = dt
        self.decouple = decouple
        self.vel_cap_frac = vel_cap_frac
        self.eps_Q = eps_Q
        self._vel_scale = 0.0 if decouple else 1.0

        # (layer, param_name, mv, Sv, Sc)  for every managed parameter tensor
        self.param_states = []
        # ids of layers fully managed by momentum (skip standard update)
        self._managed_layer_ids = set()

        # Compute adaptive defaults from actual parameter variances
        median_S = self._compute_median_variance()
        if sigma_v0 is None:
            # Velocity std ≈ 30% of parameter std → max r ≈ 0.3
            sigma_v0 = 0.3 * (median_S**0.5)
        if q_v is None:
            # Process noise to maintain velocity variance in steady state
            # Sv_ss ≈ q_v / (1 - γ²),  target Sv_ss ≈ sigma_v0²
            q_v = sigma_v0**2 * (1.0 - gamma**2)
        if q_theta is None:
            if decouple:
                # In decoupled mode, the mean shifts by dt·mv every step but
                # S is not inflated.  Without process noise, S collapses to
                # the floor and learning stalls.  Inject ~0.1% of median S
                # per step to keep S healthy, with a hard floor of eps_Q.
                q_theta = max(1e-3 * median_S, eps_Q)
            else:
                q_theta = eps_Q

        self.q_theta = q_theta
        self.q_v = q_v
        self._init_states(sigma_v0)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _iter_leaf_layers(self):
        """Yield every leaf learnable layer, expanding compound layers."""
        from .layers.frn_resblock import FRNResBlock
        from .layers.resblock import ResBlock
        from .layers.shared_var_resblock import SharedVarResBlock
        from .network import _LEARNABLE_LAYERS

        _COMPOUND = (ResBlock, FRNResBlock, SharedVarResBlock)
        for layer in self.net.layers:
            if isinstance(layer, _COMPOUND):
                yield from layer._learnable
            elif isinstance(layer, _LEARNABLE_LAYERS):
                yield layer

    def _compute_median_variance(self):
        """Compute median parameter variance across all supported layers."""
        all_S = []
        for layer in self._iter_leaf_layers():
            if not isinstance(layer, _MOMENTUM_SUPPORTED):
                continue
            if hasattr(layer, "Sw"):
                all_S.append(layer.Sw.detach().view(-1))
            if getattr(layer, "has_bias", False):
                all_S.append(layer.Sb.detach().view(-1))
        if all_S:
            return torch.cat(all_S).median().item()
        return 1e-4  # fallback

    def _init_states(self, sigma_v0):
        """Create velocity / cross-covariance tensors for every parameter."""
        for layer in self._iter_leaf_layers():
            if not isinstance(layer, _MOMENTUM_SUPPORTED):
                continue

            self._managed_layer_ids.add(id(layer))

            # Weights
            if hasattr(layer, "mw"):
                mv = torch.zeros_like(layer.mw)
                Sv = torch.full_like(layer.mw, sigma_v0**2)
                Sc = torch.zeros_like(layer.mw)
                self.param_states.append((layer, "w", mv, Sv, Sc))

            # Biases
            if getattr(layer, "has_bias", False):
                mv = torch.zeros_like(layer.mb)
                Sv = torch.full_like(layer.mb, sigma_v0**2)
                Sc = torch.zeros_like(layer.mb)
                self.param_states.append((layer, "b", mv, Sv, Sc))

    @staticmethod
    def _get_param(layer, pname):
        if pname == "w":
            return layer.mw, layer.Sw
        return layer.mb, layer.Sb

    @staticmethod
    def _get_deltas(layer, pname):
        if pname == "w":
            return layer.delta_mw, layer.delta_Sw
        return layer.delta_mb, layer.delta_Sb

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def predict(self):
        """Apply kinematic transition to all managed parameter states."""
        for layer, pname, mv, Sv, Sc in self.param_states:
            m, S = self._get_param(layer, pname)
            predict_params(
                m,
                S,
                mv,
                Sv,
                Sc,
                self.gamma,
                self.dt,
                self.q_theta,
                self.q_v,
                vel_scale=self._vel_scale,
                vel_cap_frac=self.vel_cap_frac,
            )

    def update(self, cap_factor):
        """Apply augmented posterior update to all managed parameters."""
        for layer, pname, mv, Sv, Sc in self.param_states:
            m, S = self._get_param(layer, pname)
            dm, dS = self._get_deltas(layer, pname)
            if dm is not None and dS is not None:
                augmented_update(m, S, mv, Sv, Sc, dm, dS, cap_factor)

    def _update_unmanaged(self, cap_factor):
        """Standard capped update for layers not managed by momentum."""
        from .layers.frn_resblock import FRNResBlock
        from .layers.resblock import ResBlock
        from .layers.shared_var_resblock import SharedVarResBlock
        from .network import _LEARNABLE_LAYERS

        _COMPOUND = (ResBlock, FRNResBlock, SharedVarResBlock)
        for layer in self.net.layers:
            if isinstance(layer, _COMPOUND):
                for sub in layer._learnable:
                    if id(sub) not in self._managed_layer_ids:
                        sub.update(cap_factor)
            elif isinstance(layer, _LEARNABLE_LAYERS):
                if id(layer) not in self._managed_layer_ids:
                    layer.update(cap_factor)

    def step(self, x_batch, y_batch, sigma_v):
        """
        Full training step with state-space momentum.

        1. Kinematic prediction  (transition)
        2. Forward pass          (standard TAGI, uses θ marginal)
        3. Output innovation
        4. Backward pass         (standard TAGI, stores deltas)
        5. Augmented RTS update  (updates both θ and v)
        6. Standard update for any non-managed layers

        Returns
        -------
        y_pred_mu, y_pred_var : Tensor  predictions (before update)
        """
        batch_size = x_batch.shape[0]

        # 1. Kinematic transition
        self.predict()

        # 2. Forward
        y_pred_mu, y_pred_var = self.net.forward(x_batch)

        # 3. Innovation
        delta_mu, delta_var = compute_innovation(y_batch, y_pred_mu, y_pred_var, sigma_v)

        # 4. Backward
        for layer in reversed(self.net.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # 5. Augmented update (managed parameters)
        cap_factor = get_cap_factor(batch_size)
        self.update(cap_factor)

        # 6. Standard update (unmanaged layers, e.g. FRN2D, TLU)
        self._update_unmanaged(cap_factor)

        return y_pred_mu, y_pred_var

    def __repr__(self):
        n_params = sum(self._get_param(lay, p)[0].numel() for lay, p, *_ in self.param_states)
        mode = "decoupled" if self.decouple else "full"
        return (
            f"StateSpaceMomentum(γ={self.gamma}, dt={self.dt}, "
            f"q_θ={self.q_theta:.2e}, q_v={self.q_v:.2e}, "
            f"vel_cap={self.vel_cap_frac}, "
            f"mode={mode}, managed_params={n_params})"
        )
