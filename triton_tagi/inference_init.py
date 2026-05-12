"""Inference-Based Initialization (IBI) for TAGI networks.

Pre-training calibration pass. For each calibratable layer with width A, the
algorithm drives the empirical batch-aggregate pre-activation moments
(mu_Zi, S_Zi) toward targets derived from layer-sum statistics S = sum_i Zi
and S2 = sum_i Zi^2.

Targets (scalar, width-A layer, user-set sigma_m, sigma_z):
    mu_S_tilde   = 0
    var_S_tilde  = A * sigma_z^2
    mu_S2_tilde  = A * (sigma_m^2 + sigma_z^2)
    var_S2_tilde = A * (2 sigma_z^4 + 4 sigma_m^2 sigma_z^2)

sigma_z modes:
    - float:  Global activation-scale prior (matches PLAN.md V1).
    - "auto": Per-layer sigma_z derived from a probe forward through the
              He-initialized network. For each Conv2D/Linear, sigma_z² is set
              to the layer's empirical batch-mean output variance (sqrt of
              S_Z mean across channels). This preserves He's per-layer scale
              so BN's data-dep gamma init sees the input distribution it
              expects, while still applying IBI's S/S2 moment-matching
              projection (channel mean equalized, channel-variance equalized).

Per-layer step on a single batch:
    1. Forward the already-calibrated prefix to obtain (ma, Sa) at this layer.
    2. Forward the layer to obtain per-sample (mz, Sz).
    3. Aggregate to scalar (A,) per output unit (PLAN.md D1: batch-mean first).
    4. S projection  (closed-form Kalman on S).
    5. S2 RTS update (linearized Kalman on S2, applied after S).
    6. Decoupled inverse: rescale mw, Sw, Sb by gamma_i (gamma_i^2) and shift mb
       by delta_mu_Zi so that the re-forward output moments land on the targets.

Layer dispatch (Phase 2 + Phase 3 — PLAN.md):
    - Linear:       A = out_features, aggregate over the batch dim.
    - Conv2D:       A = C_out, aggregate over (N * H_out * W_out) — the C_out
                    channels at one spatial position are exactly the "layer of
                    width A" the algorithm assumes (shared weights across
                    positions, per-channel bias).
    - BatchNorm2D:  Pass-through. BN's output variance is dominated by the
                    empirical batch_var (used as the normalization denominator),
                    while the TAGI moment S_z propagated through BN reflects
                    only parameter uncertainty (orders of magnitude smaller).
                    Calibrating BN's S_z blows up gamma. The IBI prior is
                    enforced at the surrounding Conv2D / Linear layers.
    - ResBlock:     Phase 3 option (a): per-sub-layer calibration. Walk the main
                    path (Conv->ReLU->BN->Conv->ReLU->BN), then the projection
                    path if present, then add. The residual addition has no
                    learnable parameters.
    - All others (ReLU, AvgPool2D, Flatten, Remax, ...): pass through.

Convergence. S hits its target exactly every batch. S2 approaches asymptotically
because the S2 observation is linearized around current (mu_Zi, S_Zi).

Failure modes. If the batch-aggregate var_S or var_S2 falls below eps, the
corresponding projection is skipped for that batch. If S_Zi < eps for a unit,
gamma_i is undefined and that unit is left untouched.

BatchNorm running stats. During IBI each BN runs in a special "_ibi_mode" that
uses batch stats for normalization but skips both running-stat updates and the
data-dependent gamma init. After the IBI loop, a clean warm-up pass populates
running stats from the calibrated forward (with preserve_var temporarily off so
the data-dep init doesn't clobber the IBI-set gamma). This keeps train-mode
calibration consistent with eval-mode running-stat normalization.

Reference: experiments/inference_init/PLAN.md.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .layers.batchnorm2d import BatchNorm2D
from .layers.conv2d import Conv2D
from .layers.linear import Linear
from .layers.resblock import ResBlock, triton_add_shortcut
from .network import Sequential

# BatchNorm is intentionally NOT calibrated. BN's output variance is
# dominated by the empirical batch_var (used as the normalization denominator),
# while the TAGI moment S_z propagated through BN reflects only parameter
# uncertainty (Sw, Sb) — orders of magnitude smaller. Calibrating S_z to the
# IBI target (sigma_z^2) blows up gamma to compensate, producing pathological
# affine scales (verified empirically: gamma ~4-8x intended on CIFAR-10 CNN).
# BN is treated as pass-through; the calibration prior is enforced at the
# Conv2D / Linear layers.
_CALIBRATABLE = (Linear, Conv2D)


def _layer_targets(
    A: int, sigma_m: float, sigma_z: float
) -> tuple[float, float, float, float]:
    """Return (mu_S_tilde, var_S_tilde, mu_S2_tilde, var_S2_tilde) for width A."""
    var_S = A * sigma_z**2
    mu_S2 = A * (sigma_m**2 + sigma_z**2)
    var_S2 = A * (2.0 * sigma_z**4 + 4.0 * sigma_m**2 * sigma_z**2)
    return 0.0, var_S, mu_S2, var_S2


def _aggregate_moments(
    mz: Tensor, Sz: Tensor, layer
) -> tuple[Tensor, Tensor, int]:
    """Batch-mean per-output-unit aggregation. Returns (mu_Z, S_Z, A)."""
    if isinstance(layer, Linear):
        A = layer.out_features
        return mz.reshape(-1, A).mean(dim=0), Sz.reshape(-1, A).mean(dim=0), A
    if isinstance(layer, Conv2D):
        A = layer.C_out
        # (N, C, H, W) -> (N*H*W, C) -> mean over rows -> (C,)
        return (
            mz.permute(0, 2, 3, 1).reshape(-1, A).mean(dim=0),
            Sz.permute(0, 2, 3, 1).reshape(-1, A).mean(dim=0),
            A,
        )
    raise TypeError(f"_aggregate_moments: unsupported layer {type(layer).__name__}")


def _s_projection(
    mu_Z: Tensor,
    S_Z: Tensor,
    mu_S_tilde: float,
    var_S_tilde: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Closed-form Kalman update on the scalar observation S = sum_i Zi.

    Skips (returns inputs unchanged) when var_S < eps.
    """
    mu_S = mu_Z.sum()
    var_S = S_Z.sum()
    if float(var_S) < eps:
        return mu_Z, S_Z
    d_mu = (mu_S_tilde - mu_S) / var_S
    d_var = (var_S_tilde - var_S) / (var_S * var_S)
    return mu_Z + S_Z * d_mu, S_Z * (1.0 + var_S * d_var)


def _s2_projection(
    mu_Z: Tensor,
    S_Z: Tensor,
    mu_S2_tilde: float,
    var_S2_tilde: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Linearized Kalman update on S2 = sum_i Zi^2, applied after S projection.

    The per-unit Jacobian J_i = 2 mu_Zi S_Zi uses the post-S moments. The post
    variance is floored at 0 because the linearization can over-reduce.
    """
    mu_Z2 = mu_Z * mu_Z + S_Z
    S_Z2 = 2.0 * S_Z * S_Z + 4.0 * S_Z * mu_Z * mu_Z
    mu_S2 = mu_Z2.sum()
    var_S2 = S_Z2.sum()
    if float(var_S2) < eps:
        return mu_Z, S_Z
    J = 2.0 * mu_Z * S_Z
    d_mu = (mu_S2_tilde - mu_S2) / var_S2
    d_var = (var_S2_tilde - var_S2) / (var_S2 * var_S2)
    return mu_Z + J * d_mu, torch.clamp(S_Z + (J * J) * d_var, min=0.0)


def _decoupled_inverse(
    layer,
    mu_Z: Tensor,
    S_Z: Tensor,
    mu_Z_target: Tensor,
    S_Z_target: Tensor,
    eps: float,
) -> None:
    """Rescale mw, Sw, Sb and shift mb so the post-update forward matches targets.

    gamma_i = sqrt(S_Zi_target / S_Zi).  Applied per output unit.

    Linear / Conv2D have 2D parameters with the output unit on the last axis
    (mw shape (in_dim, A), mb shape (1, A)); broadcast gamma as (1, A).
    BatchNorm2D has 1D per-channel parameters of shape (A,); broadcast gamma
    directly.

    Units with S_Zi <= eps are left untouched (gamma_i = 1, delta_mu = 0).
    Modifies ``layer`` in place.
    """
    safe = S_Z > eps
    ratio = torch.where(safe, S_Z_target / S_Z.clamp(min=eps), torch.ones_like(S_Z))
    gamma = torch.sqrt(torch.clamp(ratio, min=0.0))
    gamma = torch.where(safe, gamma, torch.ones_like(gamma))
    mu_target_eff = torch.where(safe, mu_Z_target, mu_Z)

    mb_old = layer.mb.view(-1)
    tilde_mu = gamma * (mu_Z - mb_old) + mb_old
    delta_mu = mu_target_eff - tilde_mu

    if layer.mw.ndim == 2:
        g = gamma.unsqueeze(0)
        g2 = (gamma * gamma).unsqueeze(0)
    else:
        g = gamma
        g2 = gamma * gamma

    layer.mw.mul_(g)
    layer.Sw.mul_(g2)
    if layer.has_bias:
        layer.Sb.mul_(g2)
        if layer.mb.ndim == 2:
            layer.mb.add_(delta_mu.view(1, -1))
        else:
            layer.mb.add_(delta_mu)


def _calibrate_layer(
    layer,
    ma: Tensor,
    Sa: Tensor,
    sigma_m: float,
    sigma_z_fn,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Forward → project → decoupled inverse → re-forward. Returns post (ma, Sa).

    ``sigma_z_fn(layer) -> float`` returns the activation-scale prior for this
    layer (constant under global mode; probe-derived under "auto" mode).
    """
    mz, Sz = layer.forward(ma, Sa)
    mu_Z, S_Z, A = _aggregate_moments(mz, Sz, layer)

    sigma_z = sigma_z_fn(layer)
    mu_S_t, var_S_t, mu_S2_t, var_S2_t = _layer_targets(A, sigma_m, sigma_z)
    mu_Z_post, S_Z_post = _s_projection(mu_Z, S_Z, mu_S_t, var_S_t, eps)
    mu_Z_post, S_Z_post = _s2_projection(mu_Z_post, S_Z_post, mu_S2_t, var_S2_t, eps)

    _decoupled_inverse(layer, mu_Z, S_Z, mu_Z_post, S_Z_post, eps)
    return layer.forward(ma, Sa)


def _walk_resblock(
    block: ResBlock,
    ma: Tensor,
    Sa: Tensor,
    sigma_m: float,
    sigma_z_fn,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Calibrate Conv2D/BN inside a ResBlock and replicate its forward addition.

    Mirrors ResBlock.forward: save input, walk main path, walk projection path
    (or pass identity), add shortcut to main output.
    """
    mu_skip = ma.clone()
    var_skip = Sa.clone()

    mu_z, var_z = ma, Sa
    for sub in block._main_layers:
        mu_z, var_z = _walk_layer(sub, mu_z, var_z, sigma_m, sigma_z_fn, eps)

    if block.use_projection:
        mu_x, var_x = mu_skip, var_skip
        for sub in block._proj_layers:
            mu_x, var_x = _walk_layer(sub, mu_x, var_x, sigma_m, sigma_z_fn, eps)
    else:
        mu_x, var_x = mu_skip, var_skip

    triton_add_shortcut(mu_x, var_x, mu_z, var_z)
    return mu_z, var_z


def _walk_layer(
    layer,
    ma: Tensor,
    Sa: Tensor,
    sigma_m: float,
    sigma_z_fn,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Dispatch one layer: calibrate, recurse into ResBlock, or pass through."""
    if isinstance(layer, _CALIBRATABLE):
        return _calibrate_layer(layer, ma, Sa, sigma_m, sigma_z_fn, eps)
    if isinstance(layer, ResBlock):
        return _walk_resblock(layer, ma, Sa, sigma_m, sigma_z_fn, eps)
    return layer.forward(ma, Sa)


def _probe_per_layer_sigma_z(
    net: Sequential, x_probe: Tensor, eps: float
) -> dict[int, float]:
    """Probe forward through the He-init net to record per-layer sqrt(S_Z mean).

    For each Conv2D / Linear (including those nested inside ResBlock), record
    sqrt(S_Z.mean()) as the per-layer sigma_z target. Returns a dict keyed by
    ``id(layer)`` so calibration can look it up regardless of nesting.

    BN layers are forwarded normally (standalone BNs use ``_ibi_mode`` so they
    don't fire data-dep init or update running stats during the probe).
    """
    out: dict[int, float] = {}

    def visit(layer, ma, Sa):
        if isinstance(layer, _CALIBRATABLE):
            mz, Sz = layer.forward(ma, Sa)
            _, S_Z, _ = _aggregate_moments(mz, Sz, layer)
            out[id(layer)] = float(S_Z.mean().clamp(min=eps).sqrt())
            return mz, Sz
        if isinstance(layer, ResBlock):
            mu_skip = ma.clone()
            var_skip = Sa.clone()
            mu_z, var_z = ma, Sa
            for sub in layer._main_layers:
                mu_z, var_z = visit(sub, mu_z, var_z)
            if layer.use_projection:
                mu_x, var_x = mu_skip, var_skip
                for sub in layer._proj_layers:
                    mu_x, var_x = visit(sub, mu_x, var_x)
            else:
                mu_x, var_x = mu_skip, var_skip
            triton_add_shortcut(mu_x, var_x, mu_z, var_z)
            return mu_z, var_z
        return layer.forward(ma, Sa)

    ma = x_probe
    Sa = torch.zeros_like(x_probe)
    for layer in net.layers:
        ma, Sa = visit(layer, ma, Sa)
    return out


def _collect_bns(net: Sequential) -> list[BatchNorm2D]:
    """Return all BatchNorm2D layers in the network, recursing into ResBlocks."""
    out: list[BatchNorm2D] = []
    for layer in net.layers:
        if isinstance(layer, BatchNorm2D):
            out.append(layer)
        elif isinstance(layer, ResBlock):
            for sub in layer._learnable:
                if isinstance(sub, BatchNorm2D):
                    out.append(sub)
    return out


def _bn_warmup(net: Sequential, batches: list[Tensor]) -> None:
    """Populate BN running stats from a clean forward pass with calibrated params.

    For each BN: clear ``_ibi_mode``, force ``_is_initialized=False`` and zero
    running stats so the data-dep init fires on the first forward of the
    warm-up. ``preserve_var`` is NOT overridden — since IBI is pass-through on
    BN, BN's standard ``mw = sqrt(running_var + eps)`` data-dep gamma init is
    the desired behavior (matches the no-IBI baseline).
    """
    bns = _collect_bns(net)
    if not bns or not batches:
        return

    for bn in bns:
        bn._ibi_mode = False
        bn._is_initialized = False
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

    for x in batches:
        ma = x
        Sa = torch.zeros_like(x)
        for layer in net.layers:
            ma, Sa = layer.forward(ma, Sa)


@torch.no_grad()
def inference_init(
    net: Sequential,
    loader: Iterable,
    sigma_m: float,
    sigma_z: float | str,
    *,
    eps: float = 1e-8,
    warmup_batches: int = 4,
) -> None:
    """Run one epoch of IBI calibration over every calibratable layer in ``net``.

    Calibrates Linear and Conv2D — including those nested inside ResBlock. BN
    is pass-through (its parameters are left at He's data-dep init). All other
    layers (ReLU, AvgPool2D, Flatten, Remax, ...) are pass-through.

    Conv2D uses A = number of output channels and batch-aggregates over
    (N * H_out * W_out) spatial positions to produce (A,) per-channel scalars.
    ResBlock recurses into main + projection paths; the residual addition has
    no learnable parameters.

    BatchNorm running stats. During the IBI loop, BN runs in ``_ibi_mode``:
    batch stats for normalization, no running-stat updates, no data-dep gamma
    init. After IBI, the last ``warmup_batches`` batches seen are replayed in
    a clean forward to populate running_mean/running_var from the calibrated
    forward. This keeps eval-mode normalization consistent with what IBI
    calibrated against.

    Args:
        net:            Sequential network.
        loader:         Iterable over one epoch. Each item is either an input
                        Tensor or a tuple whose first element is the input
                        Tensor. Targets are ignored. Inputs are moved to
                        ``net.device`` automatically.
        sigma_m:        Global prior mean-scale hyperparameter.
        sigma_z:        Activation-scale prior. Float for global mode, or
                        ``"auto"`` to derive a per-layer sigma_z from a probe
                        forward (each Conv2D/Linear targets its own empirical
                        He-init output variance — preserves He's per-layer
                        scale so BN sees the input distribution it expects).
        eps:            Numerical floor for variances (default 1e-8).
        warmup_batches: Number of trailing batches to buffer for the post-IBI
                        BN warm-up pass (default 4). Set to 0 to skip warm-up
                        (running stats will be left at zeros/ones).
    """
    # Materialize loader if we need a probe pass — auto mode reads the first
    # batch before consuming the rest for calibration.
    auto_mode = isinstance(sigma_z, str) and sigma_z == "auto"
    if auto_mode:
        loader_list = list(loader)
        if not loader_list:
            return
        loader = loader_list

    bns = _collect_bns(net)
    for bn in bns:
        bn._ibi_mode = True
    net.train()

    if auto_mode:
        first = loader[0]
        probe_x = first[0] if isinstance(first, (tuple, list)) else first
        probe_x = probe_x.to(net.device)
        sigma_z_map = _probe_per_layer_sigma_z(net, probe_x, eps)

        def sigma_z_fn(layer, _map=sigma_z_map):
            return _map[id(layer)]
    else:
        sz_const = float(sigma_z)

        def sigma_z_fn(layer, _v=sz_const):
            return _v

    warmup_buffer: list[Tensor] = []

    try:
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(net.device)

            warmup_buffer.append(x)
            if len(warmup_buffer) > warmup_batches:
                warmup_buffer.pop(0)

            ma = x
            Sa = torch.zeros_like(x)
            for layer in net.layers:
                ma, Sa = _walk_layer(layer, ma, Sa, sigma_m, sigma_z_fn, eps)
    finally:
        for bn in bns:
            bn._ibi_mode = False

    _bn_warmup(net, warmup_buffer)
    net.train()
