"""
Observation update — compute the innovation signal at the output layer.

Given observations y, predicted mean μ_z and predicted variance S_z,
the TAGI update computes:
    δ_μ = (y − μ_z) / (S_z + σ_v²)
    δ_S = −1      / (S_z + σ_v²)

These deltas are then back-propagated through the network.
"""

import math

import torch
import triton
import triton.language as tl

BLOCK = 1024
LOGISTIC_PROBIT_LAMBDA = math.pi / 8.0
_EPS = 1e-8


# ======================================================================
#  Triton kernel
# ======================================================================


@triton.jit
def _output_innovation_kernel(
    y_ptr,
    ym_ptr,
    yS_ptr,
    sv_sq,
    dm_ptr,
    dS_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y = tl.load(y_ptr + offs, mask=valid)
    ym = tl.load(ym_ptr + offs, mask=valid)
    yS = tl.load(yS_ptr + offs, mask=valid)

    Sy = yS + sv_sq

    tl.store(dm_ptr + offs, (y - ym) / Sy, mask=valid)
    tl.store(dS_ptr + offs, -1.0 / Sy, mask=valid)


@triton.jit
def _output_innovation_kernel_heteros(
    y_ptr,
    ym_ptr,
    yS_ptr,
    dm_ptr,
    dS_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """AGVI heteroscedastic output update — 1:1 port of cuTAGI's CUDA
    ``update_delta_z_cuda_heteros`` (``src/output_updater_cuda.cu``).

    The output layer is interleaved width ``2K``: column ``2i`` is the mean
    prediction ``Z_i`` (identity activation, Jacobian = 1), column ``2i+1`` is
    the post-:class:`~triton_tagi.layers.EvenExp` aleatoric variance ``V̄²_i``
    (log-normal moments). Terminology follows cuTAGI: ``V ~ N(0, μ_V2)`` is the
    error, ``V2 = V²``, ``V2_bar`` its expectation, ``V2_bar_tilde = exp(·)`` the
    positive-domain activation output.

    Because EvenExp's Jacobian equals its mean (``jcb = μ_a``), the smoother gain
    that maps the V̄²-tilde posterior back to the pre-activation latent is
    ``jv = μ_{V̄²} / Σ_{V̄²}`` — folded in HERE, exactly as cuTAGI does. The paired
    ``EvenExp.backward`` is therefore an identity passthrough.

    Note: cuTAGI's CUDA kernel updates the mean with the *epistemic* variance
    (``var_a_col``) and the variance with the *total* variance (``var_sum``) —
    the "overfit_mu" behavior. The CPU ``compute_delta_z_heteros`` instead uses
    ``var_sum`` for both. We follow the CUDA path, which is what the cuTAGI
    regression-heteros example (``cuda=True``) actually runs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y = tl.load(y_ptr + offs, mask=valid)

    # Interleaved 2K output: even = Z (mean), odd = V̄²_tilde (post-EvenExp).
    obs_col = offs * 2

    mu_a_col = tl.load(ym_ptr + obs_col, mask=valid)
    var_a_col = tl.load(yS_ptr + obs_col, mask=valid)

    # V2_bar_tilde moments. For EvenExp, jcb = μ_a, so cov_V2_bar_tilde = μ.
    mu_v2_bar_tilde = tl.load(ym_ptr + obs_col + 1, mask=valid)
    var_v2_bar_tilde = tl.load(yS_ptr + obs_col + 1, mask=valid)
    cov_v2_bar_tilde = mu_v2_bar_tilde

    # Prior predictive for V2 (Gaussian-moment chain).
    mu_v2 = mu_v2_bar_tilde
    var_v2 = 3.0 * var_v2_bar_tilde + 2.0 * mu_v2_bar_tilde * mu_v2_bar_tilde
    cov_y_v = mu_v2

    # Total output variance: epistemic (var_a_col) + learned aleatoric (mu_v2).
    var_sum = var_a_col + mu_v2

    # ── Z (even) update: jcb_col = 1 (identity). Mean uses epistemic variance,
    #    variance uses total variance — cuTAGI's overfit_mu behavior. ──
    obs_diff = y - mu_a_col
    tmp_mu = 1.0 / var_a_col
    tmp_var = 1.0 / var_sum
    z_ok = (var_a_col > 0.0) & (var_sum > 0.0)
    delta_mu_col = tl.where(z_ok, tmp_mu * obs_diff, 0.0)
    delta_var_col = tl.where(z_ok, -tmp_var, 0.0)

    # ── V̄² (odd) AGVI update ──
    mu_v_post = cov_y_v / var_sum * obs_diff
    var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v

    mu_v2_post = mu_v_post * mu_v_post + var_v_post
    var_v2_post = 2.0 * var_v_post * var_v_post + 4.0 * var_v_post * mu_v_post * mu_v_post

    tmp_ratio = var_v2_bar_tilde / var_v2
    mu_v2_bar_tilde_post = mu_v2_bar_tilde + tmp_ratio * (mu_v2_post - mu_v2)
    var_v2_bar_tilde_post = var_v2_bar_tilde + tmp_ratio * tmp_ratio * (var_v2_post - var_v2)

    # Fold the exp Jacobian (smoother gain) into the pre-activation delta.
    jv = cov_v2_bar_tilde / var_v2_bar_tilde
    delta_mu_v2 = jv * (mu_v2_bar_tilde_post - mu_v2_bar_tilde)
    delta_var_v2 = jv * jv * (var_v2_bar_tilde_post - var_v2_bar_tilde)

    tl.store(dm_ptr + obs_col, delta_mu_col, mask=valid)
    tl.store(dS_ptr + obs_col, delta_var_col, mask=valid)

    tl.store(dm_ptr + obs_col + 1, delta_mu_v2, mask=valid)
    tl.store(dS_ptr + obs_col + 1, delta_var_v2, mask=valid)


# ======================================================================
#  Python API
# ======================================================================


def compute_innovation(y, y_pred_mu, y_pred_var, sigma_v):
    """
    Compute the output innovation (update signal) for TAGI.

    Parameters
    ----------
    y          : Tensor (B, D)  observed targets
    y_pred_mu  : Tensor (B, D)  predicted output mean
    y_pred_var : Tensor (B, D)  predicted output variance
    sigma_v    : float          observation noise std-dev

    Returns
    -------
    delta_mu  : Tensor (B, D)  mean innovation
    delta_var : Tensor (B, D)  variance innovation
    """
    n = y.numel()

    if y_pred_mu.shape[-1] == 2 * y.shape[-1]:
        delta_mu = torch.empty_like(y_pred_mu)
        delta_var = torch.empty_like(y_pred_var)
        _output_innovation_kernel_heteros[(triton.cdiv(n, BLOCK),)](
            y,
            y_pred_mu,
            y_pred_var,
            delta_mu,
            delta_var,
            n,
            BLOCK=BLOCK,
        )
    else:
        delta_mu = torch.empty_like(y)
        delta_var = torch.empty_like(y)
        _output_innovation_kernel[(triton.cdiv(n, BLOCK),)](
            y,
            y_pred_mu,
            y_pred_var,
            sigma_v**2,
            delta_mu,
            delta_var,
            n,
            BLOCK=BLOCK,
        )

    return delta_mu, delta_var


# ======================================================================
#  Dense categorical observation operator (fixed-noise-free TAGI-V)
# ======================================================================


def _center_last(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def _split_categorical_outputs(
    y_pred_mu: torch.Tensor,
    y_pred_var: torch.Tensor,
    num_classes: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    bool,
]:
    if y_pred_mu.shape != y_pred_var.shape:
        raise ValueError("y_pred_mu and y_pred_var must have the same shape")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    out_dim = y_pred_mu.shape[-1]
    if out_dim == num_classes:
        return y_pred_mu, y_pred_var, None, None, False
    if out_dim == 2 * num_classes:
        return (
            y_pred_mu[..., 0::2],
            y_pred_var[..., 0::2],
            y_pred_mu[..., 1::2],
            y_pred_var[..., 1::2],
            True,
        )
    raise ValueError("categorical output width must be num_classes or 2 * num_classes")


def tempered_categorical_probs(
    logit_mu: torch.Tensor,
    logit_var: torch.Tensor,
    v2bar: torch.Tensor | None = None,
    lambda_: float = LOGISTIC_PROBIT_LAMBDA,
) -> torch.Tensor:
    """Return uncertainty-tempered categorical probabilities.

    The fixed pi/8 logistic-probit bridge is part of the observation model,
    not a post-hoc calibration parameter.
    """

    if logit_mu.shape != logit_var.shape:
        raise ValueError("logit_mu and logit_var must have the same shape")
    if v2bar is None:
        v2bar = torch.zeros_like(logit_mu)
    elif v2bar.shape != logit_mu.shape:
        raise ValueError("v2bar must have the same shape as logit_mu")
    total_var = logit_var.clamp_min(0.0) + v2bar.clamp_min(0.0)
    attenuation = torch.rsqrt(1.0 + lambda_ * total_var)
    return torch.softmax(attenuation * _center_last(logit_mu), dim=-1)


def categorical_predictive_probs(
    y_pred_mu: torch.Tensor,
    y_pred_var: torch.Tensor,
    num_classes: int,
    lambda_: float = LOGISTIC_PROBIT_LAMBDA,
) -> torch.Tensor:
    """Predict from dense K logits or interleaved 2K TAGI-V outputs."""

    logit_mu, logit_var, v2bar, _, _ = _split_categorical_outputs(
        y_pred_mu, y_pred_var, num_classes
    )
    return tempered_categorical_probs(logit_mu, logit_var, v2bar, lambda_)


def compute_categorical_innovation(
    labels: torch.Tensor,
    y_pred_mu: torch.Tensor,
    y_pred_var: torch.Tensor,
    num_classes: int,
    lambda_: float = LOGISTIC_PROBIT_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the centered O(K) categorical TAGI innovation without sigma_v."""

    logit_mu, logit_var, v2bar, v2bar_var, has_v2bar = _split_categorical_outputs(
        y_pred_mu, y_pred_var, num_classes
    )
    leading_shape = logit_mu.shape[:-1]
    flat_mu = logit_mu.reshape(-1, num_classes)
    flat_var = logit_var.reshape(-1, num_classes).clamp_min(0.0)
    flat_labels = labels.reshape(-1).long().to(y_pred_mu.device)
    if flat_labels.numel() != flat_mu.shape[0]:
        raise ValueError("labels leading shape must match prediction leading shape")
    if bool(((flat_labels < 0) | (flat_labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    target = torch.nn.functional.one_hot(flat_labels, num_classes).to(flat_mu.dtype)

    flat_v2 = (
        torch.zeros_like(flat_mu) if v2bar is None else v2bar.reshape_as(flat_mu).clamp_min(0.0)
    )
    total_var = (flat_var + flat_v2).clamp_min(_EPS)
    attenuation = torch.rsqrt(1.0 + lambda_ * total_var)
    probabilities = torch.softmax(attenuation * _center_last(flat_mu), dim=-1)

    d = (attenuation.square() * total_var).clamp_min(_EPS)
    r = 1.0 / (1.0 / d + probabilities)
    q = r * probabilities
    denominator = (1.0 - (probabilities * q).sum(-1, keepdim=True)).clamp_min(_EPS)
    residual = target - probabilities
    delta_eta = r * residual + q * (q * residual).sum(-1, keepdim=True) / denominator
    delta_total = _center_last(delta_eta / attenuation.clamp_min(_EPS))

    diag_base = (r - d) / attenuation.square().clamp_min(_EPS)
    u = q / attenuation.clamp_min(_EPS)
    sum_u = u.sum(-1, keepdim=True)
    row_sum = diag_base + u * sum_u / denominator
    total_sum = diag_base.sum(-1, keepdim=True) + sum_u.square() / denominator
    diag_delta = (
        diag_base
        + u.square() / denominator
        - (2.0 / num_classes) * row_sum
        + total_sum / (num_classes * num_classes)
    )
    delta_mu_logit = delta_total / total_var
    delta_var_logit = diag_delta / total_var.square()

    if not has_v2bar:
        return (
            delta_mu_logit.reshape(*leading_shape, num_classes),
            delta_var_logit.reshape(*leading_shape, num_classes),
        )

    assert v2bar_var is not None
    flat_v2_var = v2bar_var.reshape_as(flat_mu).clamp_min(0.0)
    j_v = flat_v2 / total_var
    mu_v_post = j_v * delta_total
    var_v_post = (flat_v2 + j_v.square() * diag_delta).clamp_min(_EPS)
    mu_v2_post = mu_v_post.square() + var_v_post
    var_v2_post = 2.0 * var_v_post.square() + 4.0 * var_v_post * mu_v_post.square()
    var_v2 = (3.0 * flat_v2_var + 2.0 * flat_v2.square()).clamp_min(_EPS)
    ratio = flat_v2_var / var_v2
    delta_mu_v2 = ratio * (mu_v2_post - flat_v2)
    delta_var_v2 = ratio.square() * (var_v2_post - var_v2)

    delta_mu = torch.empty_like(y_pred_mu.reshape(-1, 2 * num_classes))
    delta_var = torch.empty_like(delta_mu)
    delta_mu[:, 0::2] = delta_mu_logit
    delta_var[:, 0::2] = delta_var_logit
    delta_mu[:, 1::2] = delta_mu_v2
    delta_var[:, 1::2] = delta_var_v2
    return (
        delta_mu.reshape(*leading_shape, 2 * num_classes),
        delta_var.reshape(*leading_shape, 2 * num_classes),
    )


# ======================================================================
#  Sparse (hierarchical softmax) output innovation
# ======================================================================


def compute_innovation_with_indices(
    ma: "torch.Tensor",
    Sa: "torch.Tensor",
    y_obs: "torch.Tensor",
    var_obs: "torch.Tensor",
    selected_idx: "torch.Tensor",
    *,
    mask: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Sparse output innovation for hierarchical softmax classification.

    For each sample b and encoded bit c, updates only the selected tree node::

        node = selected_idx[b, c] - 1          (0-indexed)
        denom = Sa[b, node] + var_obs[b, c]
        delta_mu[b, node] = (y_obs[b, c] - ma[b, node]) / denom
        delta_Sa[b, node] = -1 / denom

    All other positions in delta_mu and delta_Sa are zero.

    Replicates cuTAGI's ``compute_selected_delta_z_output()`` from
    ``src/base_output_updater.cpp``.

    Args:
        ma:           Output means, shape (B, n_total_nodes).
        Sa:           Output variances, shape (B, n_total_nodes).
        y_obs:        Encoded ±1 observations, shape (B, n_obs).
        var_obs:      Observation variance, shape (B, n_obs).
        selected_idx: 1-indexed node positions, shape (B, n_obs).
        mask:         Optional (B, n_obs) indicator that is zero on padded path
                      entries of a variable-depth tree.

    Returns:
        delta_mu: Mean innovations, shape (B, n_total_nodes), sparse.
        delta_Sa: Variance innovations, shape (B, n_total_nodes), sparse.
    """
    delta_mu = torch.zeros_like(ma)
    delta_Sa = torch.zeros_like(Sa)

    # Convert 1-indexed to 0-indexed: (B, n_obs)
    node_idx = selected_idx.long() - 1

    # Gather predicted mean and variance at selected nodes
    ma_sel = torch.gather(ma, 1, node_idx)  # (B, n_obs)
    Sa_sel = torch.gather(Sa, 1, node_idx)  # (B, n_obs)

    # Innovation formula (same as dense case, evaluated at selected nodes only)
    denom = Sa_sel + var_obs
    dm = (y_obs - ma_sel) / denom
    dS = -1.0 / denom
    if mask is not None:
        if mask.shape != y_obs.shape:
            raise ValueError("mask must match the observation shape")
        keep = mask.to(ma.dtype)
        dm = dm * keep
        dS = dS * keep

    # Scatter innovations back into the full output buffers.
    # Each class uses distinct tree nodes at every depth level, so no
    # within-sample index collision occurs for valid HRC trees.
    delta_mu.scatter_add_(1, node_idx, dm)
    delta_Sa.scatter_add_(1, node_idx, dS)

    return delta_mu, delta_Sa


def _inverse_mills_ratio(gamma: torch.Tensor) -> torch.Tensor:
    """Return ``phi(gamma) / Phi(gamma)`` without left-tail underflow."""

    original_dtype = gamma.dtype
    work = (
        gamma.double() if gamma.dtype in (torch.float16, torch.bfloat16, torch.float32) else gamma
    )
    # Phi(gamma) = 0.5 * erfc(-gamma / sqrt(2)). In the left tail, erfcx
    # avoids the 0 / 0 encountered by a direct density/CDF ratio.
    left = math.sqrt(2.0 / math.pi) / torch.special.erfcx(-work / math.sqrt(2.0))
    right = torch.exp(
        -0.5 * work.square() - 0.5 * math.log(2.0 * math.pi) - torch.special.log_ndtr(work)
    )
    return torch.where(work <= 0.0, left, right).to(original_dtype)


def compute_probit_innovation_with_indices(
    ma: torch.Tensor,
    Sa: torch.Tensor,
    y_obs: torch.Tensor,
    var_obs: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    latent_shift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse exact-moment innovation for hierarchical probit observations.

    Each signed observation represents the half-space event
    ``y_obs * (Z + b + V) > 0`` rather than a continuous +/-1 target. For a
    selected node, ``r^2 = Sa + var_obs``, ``gamma = y_obs * (ma + b) / r``,
    and ``Lambda = phi(gamma) / Phi(gamma)``. TAGI's backward recursion
    consumes posterior moment changes standardized by the output variance,
    giving ``delta_mu = y_obs * Lambda / r`` and
    ``delta_var = -Lambda * (Lambda + gamma) / r^2``.

    Unselected nodes have zero innovation. The corresponding output moments
    are exact for the extended skew-normal posterior; the later Gaussian
    moment projection remains a TAGI approximation.

    Args:
        ma: Node means, shape (batch, nodes).
        Sa: Node variances, shape (batch, nodes).
        y_obs: Signed branch observations, shape (batch, n_obs).
        var_obs: Latent probit variance ``tau^2``, shape (batch, n_obs).
        selected_idx: 1-indexed node positions, shape (batch, n_obs).
        mask: Optional (batch, n_obs) indicator that is zero on padded path
            entries of a variable-depth tree. Padded entries contribute no
            innovation.
        latent_shift: Optional (batch, n_obs) deterministic branch offset ``b``
            in latent units, i.e. ``tau * offset`` for a prior-corrected tree.
    """

    if ma.shape != Sa.shape or ma.dim() != 2:
        raise ValueError("HRC output moments must have matching two-dimensional shapes")
    if y_obs.shape != var_obs.shape or y_obs.shape != selected_idx.shape:
        raise ValueError("HRC observations, variances, and indices must have matching shapes")
    if y_obs.dim() != 2 or y_obs.shape[0] != ma.shape[0]:
        raise ValueError("HRC observations must match the prediction batch")
    if bool(((y_obs != 1) & (y_obs != -1)).any()):
        raise ValueError("probit observations must be signed +/-1 values")
    if bool((var_obs <= 0).any()):
        raise ValueError("probit observation variances must be positive")

    node_idx = selected_idx.long() - 1
    if bool(((node_idx < 0) | (node_idx >= ma.shape[1])).any()):
        raise ValueError("selected HRC node index is out of range")

    ma_sel = torch.gather(ma, 1, node_idx)
    Sa_sel = torch.gather(Sa, 1, node_idx).clamp_min(0.0)
    r2 = (Sa_sel + var_obs.to(Sa.dtype)).clamp_min(_EPS)
    r = torch.sqrt(r2)
    signs = y_obs.to(ma.dtype)
    if latent_shift is not None:
        if latent_shift.shape != y_obs.shape:
            raise ValueError("latent_shift must match the observation shape")
        ma_sel = ma_sel + latent_shift.to(ma.dtype)
    gamma = signs * ma_sel / r
    mills = _inverse_mills_ratio(gamma)
    # The exact contraction factor lies in [0, 1]; clamp round-off only.
    contraction = (mills * (mills + gamma)).clamp(0.0, 1.0)

    delta_mu_selected = signs * mills / r
    delta_var_selected = -contraction / r2
    if mask is not None:
        if mask.shape != y_obs.shape:
            raise ValueError("mask must match the observation shape")
        keep = mask.to(ma.dtype)
        delta_mu_selected = delta_mu_selected * keep
        delta_var_selected = delta_var_selected * keep
    delta_mu = torch.zeros_like(ma)
    delta_var = torch.zeros_like(Sa)
    delta_mu.scatter_add_(1, node_idx, delta_mu_selected)
    delta_var.scatter_add_(1, node_idx, delta_var_selected)
    return delta_mu, delta_var


def compute_hrc_tagiv_innovation(
    ma: torch.Tensor,
    Sa: torch.Tensor,
    y_obs: torch.Tensor,
    selected_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse heteroscedastic innovation for interleaved HRC node outputs.

    The moments use [z_0, Vbar2_0, z_1, Vbar2_1, ...] layout.
    Only nodes on each target class path receive an update.
    """

    if ma.shape != Sa.shape or ma.dim() != 2 or ma.shape[1] % 2:
        raise ValueError("HRC TAGI-V moments must have shape (batch, 2 * nodes)")
    if y_obs.shape != selected_idx.shape or y_obs.shape[0] != ma.shape[0]:
        raise ValueError("HRC observations and indices must match the prediction batch")

    node_idx = selected_idx.long() - 1
    mean_idx = 2 * node_idx
    variance_idx = mean_idx + 1
    z_mu = torch.gather(ma, 1, mean_idx)
    z_var = torch.gather(Sa, 1, mean_idx).clamp_min(0.0)
    v2bar = torch.gather(ma, 1, variance_idx).clamp_min(_EPS)
    v2bar_var = torch.gather(Sa, 1, variance_idx).clamp_min(0.0)
    total_var = (z_var + v2bar).clamp_min(_EPS)
    residual = y_obs.to(ma.dtype) - z_mu

    delta_mu_z = residual / total_var
    delta_var_z = -1.0 / total_var

    j_v = v2bar / total_var
    mu_v_post = j_v * residual
    var_v_post = (v2bar - v2bar.square() / total_var).clamp_min(_EPS)
    mu_v2_post = mu_v_post.square() + var_v_post
    var_v2_post = 2.0 * var_v_post.square() + 4.0 * var_v_post * mu_v_post.square()
    var_v2 = (3.0 * v2bar_var + 2.0 * v2bar.square()).clamp_min(_EPS)
    ratio = v2bar_var / var_v2
    delta_mu_v2 = ratio * (mu_v2_post - v2bar)
    delta_var_v2 = ratio.square() * (var_v2_post - var_v2)

    delta_mu = torch.zeros_like(ma)
    delta_var = torch.zeros_like(Sa)
    delta_mu.scatter_add_(1, mean_idx, delta_mu_z)
    delta_var.scatter_add_(1, mean_idx, delta_var_z)
    delta_mu.scatter_add_(1, variance_idx, delta_mu_v2)
    delta_var.scatter_add_(1, variance_idx, delta_var_v2)
    return delta_mu, delta_var
