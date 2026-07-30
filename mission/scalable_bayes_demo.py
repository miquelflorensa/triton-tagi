"""Closed-form Bayesian demonstrations behind mission/scale_to_98_lowrank_smoother.md.

These are deliberately tiny, exact, CPU-only experiments (float64) whose ground
truth is a *closed-form* Bayesian posterior — NOT a Monte-Carlo or linearized
reference. They isolate the four claims the mission makes:

  T1  The per-(linear-)layer TAGI update is EXACT Gaussian conditioning — the
      sequential gain/covariance recursion reproduces the batch Bayesian-linear-
      regression posterior to machine precision. No Jacobian, no linearization.

  T2  The ONLY approximation that limits accuracy is diagonalizing the posterior.
      On correlated inputs the diagonal posterior MIS-CALIBRATES the predictive
      (innovation) variance — exactly the quantity condition (III) controls — and
      a diagonal+low-rank correction restores calibration monotonically in rank,
      recovering the exact posterior at full rank.

  T3  Calibration is a condition on the output INNOVATION COVARIANCE S = Σ_ŷ + R.
      A scalar budget only matches its trace, leaving the gain matrix J = Σ_ŷ S⁻¹
      far from ½·I in correlated directions; the matrix condition Σ_ŷ = R makes
      J = ½·I exactly (balanced learning in every output direction).

  T4  Depth is time: a forward filter alone cannot assign credit to early layers
      from a label seen only at the output. The closed-form (RTS) smoother — which
      is what TAGI's backward pass already is — propagates that information back
      and matches the exact full-data posterior. This is why cross-layer
      covariance must NOT be dropped.

Run:  python mission/scalable_bayes_demo.py
"""

from __future__ import annotations

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


# ======================================================================
#  Exact references (closed form)
# ======================================================================


def exact_blr(A, y, P0, m0, r):
    """Exact Bayesian linear regression posterior N(m_n, P_n).

    y_i = a_iᵀ w + ε,  ε~N(0,r),  w~N(m0, P0).
        P_n = (P0⁻¹ + AᵀA/r)⁻¹ ,  m_n = P_n (P0⁻¹ m0 + Aᵀy/r)
    """
    P0inv = torch.linalg.inv(P0)
    Pn = torch.linalg.inv(P0inv + A.T @ A / r)
    mn = Pn @ (P0inv @ m0 + A.T @ y / r)
    return mn, Pn


def kalman_sequential(A, y, P0, m0, r):
    """Sequential EXACT Gaussian conditioning, one observation at a time.

    This is precisely TAGI's linear-layer weight update: the observation
    z = aᵀw is *linear* in w, so the update is closed-form and exact — there is
    nothing to linearize.
        S_i = a_iᵀ P a_i + r        (scalar innovation variance)
        k_i = P a_i / S_i           (gain)
        m  += k_i (y_i − a_iᵀ m)
        P  -= (P a_i)(P a_i)ᵀ / S_i
    """
    P, m = P0.clone(), m0.clone()
    for i in range(A.shape[0]):
        a = A[i]
        Pa = P @ a
        s = a @ Pa + r
        m = m + Pa * (y[i] - a @ m) / s
        P = P - torch.outer(Pa, Pa) / s
    return m, P


def diag_plus_lowrank(P, k):
    """Project a full covariance to diagonal + rank-k (the mission's posterior form).

    D = diag(P) is TAGI's kept part; we add the top-k eigatoms of the OFF-diagonal
    residual R = P − D. k = 0 reproduces the pure diagonal (current TAGI); k at the
    matrix rank reproduces P exactly.
    """
    D = torch.diag(torch.diag(P))
    if k == 0:
        return D
    resid = P - D
    evals, evecs = torch.linalg.eigh(resid)
    idx = torch.argsort(evals.abs(), descending=True)[:k]
    low = (evecs[:, idx] * evals[idx]) @ evecs[:, idx].T
    return D + low


# ======================================================================
#  T1 — the linear-layer update is exact closed-form Gaussian conditioning
# ======================================================================


def t1_update_is_exact() -> None:
    d, n, r = 8, 40, 0.3
    A = torch.randn(n, d)
    w_true = torch.randn(d)
    y = A @ w_true + r**0.5 * torch.randn(n)
    P0 = torch.eye(d) * 2.0
    m0 = torch.zeros(d)

    m_seq, P_seq = kalman_sequential(A, y, P0, m0, r)
    m_ex, P_ex = exact_blr(A, y, P0, m0, r)

    dm = (m_seq - m_ex).abs().max().item()
    dP = (P_seq - P_ex).abs().max().item()
    print("  [T1] sequential closed-form update vs exact batch posterior")
    print(f"       max |Δmean| = {dm:.2e}   max |Δcov| = {dP:.2e}   (no linearization)")
    assert dm < 1e-9 and dP < 1e-9, "sequential Gaussian update must be exact"


# ======================================================================
#  T2 — diagonal mis-calibrates predictive variance; low-rank fixes it
# ======================================================================


def _predictive_calibration(P_approx, P_exact, A_test, r):
    """Mean ratio S_approx/S_exact of predictive innovation variances on test dirs.

    S = aᵀ P a + r is the predictive (innovation) variance — the Bayes-calibrated
    denominator of the Kalman gain. Ratio → 1 means correctly calibrated.
    """
    s_ap = torch.einsum("ij,jk,ik->i", A_test, P_approx, A_test) + r
    s_ex = torch.einsum("ij,jk,ik->i", A_test, P_exact, A_test) + r
    ratio = (s_ap / s_ex)
    return ratio.mean().item(), ratio.std().item()


def t2_diagonal_miscalibration() -> None:
    d, n, r = 12, 60, 0.2
    # Strongly CORRELATED inputs ⇒ exact posterior has large off-diagonals.
    L = torch.randn(d, d)
    Sigma = L @ L.T / d + 0.05 * torch.eye(d)
    chol = torch.linalg.cholesky(Sigma)
    A = torch.randn(n, d) @ chol.T
    w_true = torch.randn(d)
    y = A @ w_true + r**0.5 * torch.randn(n)
    P0, m0 = torch.eye(d), torch.zeros(d)

    _, P_ex = exact_blr(A, y, P0, m0, r)
    A_test = torch.randn(400, d) @ chol.T  # held-out directions from same dist

    print("  [T2] predictive-variance calibration (correlated inputs)")
    print("       rank k | mean S_approx/S_exact | mean|ratio-1| | ‖P̂−P‖_F (cov error)")
    prev_cov_err = None
    for k in [0, 1, 2, 4, 8, d]:
        Pk = diag_plus_lowrank(P_ex, k)
        mean, _ = _predictive_calibration(Pk, P_ex, A_test, r)
        s_ap = torch.einsum("ij,jk,ik->i", A_test, Pk, A_test) + r
        s_ex = torch.einsum("ij,jk,ik->i", A_test, P_ex, A_test) + r
        cal_err = (s_ap / s_ex - 1.0).abs().mean().item()
        cov_err = torch.linalg.norm(Pk - P_ex).item()      # Eckart–Young: ↓ in k
        tag = "  diagonal=TAGI" if k == 0 else ("  exact" if k == d else "")
        print(f"        {k:5d} |        {mean:7.4f}        |    {cal_err:7.4f}    |   {cov_err:8.2e}{tag}")
        if prev_cov_err is not None:
            assert cov_err <= prev_cov_err + 1e-9, "covariance error must decrease with rank"
        prev_cov_err = cov_err
    # full rank must be exact; diagonal must be materially miscalibrated
    mean0, _ = _predictive_calibration(diag_plus_lowrank(P_ex, 0), P_ex, A_test, r)
    meanf, _ = _predictive_calibration(diag_plus_lowrank(P_ex, d), P_ex, A_test, r)
    assert abs(meanf - 1.0) < 1e-9, "full rank must reproduce exact predictive variance"
    assert abs(mean0 - 1.0) > 0.05, "diagonal must visibly mis-calibrate on correlated inputs"


# ======================================================================
#  T3 — calibration is on the innovation COVARIANCE, not its trace
# ======================================================================


def t3_matrix_innovation_calibration() -> None:
    m = 4
    # Anisotropic epistemic output covariance Σ_ŷ (correlated outputs): build it
    # with an explicit spectrum so the gain imbalance is unambiguous.
    Qrot, _ = torch.linalg.qr(torch.randn(m, m))
    spec = torch.tensor([0.2, 0.5, 2.0, 5.0])          # cond ≈ 25
    Sigma_ep = Qrot @ torch.diag(spec) @ Qrot.T
    R = torch.eye(m)                                    # observation noise

    def gain_eigs(S_ep):
        J = S_ep @ torch.linalg.inv(S_ep + R)  # output Kalman gain matrix
        return torch.linalg.eigvalsh(0.5 * (J + J.T))

    # Scalar calibration: match the TRACE,  trace(c·Σ_ep) = trace(R).
    c = R.trace() / Sigma_ep.trace()
    eig_scalar = gain_eigs(c * Sigma_ep)

    # Matrix calibration: shape the budget so Σ_ŷ = R exactly (Lyapunov/whitening).
    eig_matrix = gain_eigs(R.clone())

    print("  [T3] output Kalman-gain eigenvalues  J = Σ_ŷ(Σ_ŷ+R)⁻¹   (target all 0.5)")
    print(f"       scalar (trace-matched) : min={eig_scalar.min():.3f}  max={eig_scalar.max():.3f}")
    print(f"       matrix (Σ_ŷ = R)       : min={eig_matrix.min():.3f}  max={eig_matrix.max():.3f}")
    assert (eig_matrix - 0.5).abs().max() < 1e-9, "matrix calibration ⇒ J = ½·I exactly"
    assert (eig_scalar - 0.5).abs().max() > 0.1, "scalar calibration leaves gain unbalanced"


# ======================================================================
#  T4 — depth-as-time: the closed-form smoother assigns credit; the filter can't
# ======================================================================


def t4_smoother_vs_filter() -> None:
    """Linear-Gaussian chain x_{l+1}=f·x_l+w, label observed ONLY at layer L.

    Filter forward leaves x_0 at its prior (the late label never reaches it).
    The RTS smoother (= TAGI's backward pass, exact for linear-Gaussian) recovers
    the exact posterior of x_0 given the output label.
    """
    L, f, q, p0, robs = 6, 0.9, 0.1, 1.0, 0.05
    y = 0.8  # observed output label

    # Forward filter (predict only until L, update at L).
    mf = [0.0]
    Pf = [p0]
    for _ in range(L):
        mf.append(f * mf[-1])
        Pf.append(f * f * Pf[-1] + q)
    # update at l = L with the observation y = x_L + v
    S = Pf[L] + robs
    K = Pf[L] / S
    mL = mf[L] + K * (y - mf[L])
    PL = (1 - K) * Pf[L]
    ms = mf.copy(); Ps = Pf.copy()
    ms[L], Ps[L] = mL, PL
    # RTS backward smoother
    for l in range(L - 1, -1, -1):
        Ppred = f * f * Pf[l] + q          # P_{l+1}^-
        G = f * Pf[l] / Ppred              # smoother gain
        ms[l] = mf[l] + G * (ms[l + 1] - f * mf[l])
        Ps[l] = Pf[l] + G * G * (Ps[l + 1] - Ppred)

    # Exact closed-form posterior of x_0 given y (full joint Gaussian).
    var_xL = f ** (2 * L) * p0 + q * sum(f ** (2 * j) for j in range(L))
    cov_0L = f ** L * p0
    denom = var_xL + robs
    m0_ex = cov_0L / denom * y
    P0_ex = p0 - cov_0L ** 2 / denom

    print("  [T4] credit assignment to layer 0 from a label at layer L")
    print(f"       filter   Var[x0|y] = {Pf[0]:.4f}   (= prior {p0:.4f}: no credit)")
    print(f"       smoother Var[x0|y] = {Ps[0]:.4f}   mean = {ms[0]:.4f}")
    print(f"       exact    Var[x0|y] = {P0_ex:.4f}   mean = {m0_ex:.4f}")
    assert abs(Pf[0] - p0) < 1e-12, "filter-only cannot update an early layer"
    assert abs(Ps[0] - P0_ex) < 1e-9 and abs(ms[0] - m0_ex) < 1e-9, "smoother must equal exact"
    assert Ps[0] < Pf[0] - 1e-6, "smoother must reduce early-layer uncertainty"


if __name__ == "__main__":
    print("=" * 70)
    print("  Closed-form Bayesian demonstrations  (float64, exact references)")
    print("=" * 70)
    t1_update_is_exact()
    print()
    t2_diagonal_miscalibration()
    print()
    t3_matrix_innovation_calibration()
    print()
    t4_smoother_vs_filter()
    print()
    print("  all demonstrations passed ✓")
