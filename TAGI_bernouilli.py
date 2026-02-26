"""
TAGI Bernoulli — Max-indicator moments for independent Gaussians
================================================================

Problem
-------
Given n independent Gaussians  X_i ~ N(mu_i, sigma_i^2),
define the hard Bernoulli indicator:

    f_i = 1  if  X_i = max(X_1, ..., X_n),  else 0

We compute three statistics for every component i:

    P_i = P(X_i = max)         [probability of being the winner]
    V_i = Var(f_i)             [= P_i*(1-P_i)  — Bernoulli identity]
    C_i = Cov(f_i, X_i)       [how much X_i co-moves with winning]

Analytical formula
------------------
Because the X_j are independent:

    P_i = integral  phi_i(x) * prod_{j != i} Phi_j(x)  dx

where phi_i is the pdf of X_i and Phi_j is the cdf of X_j.

Change of variable  x = sqrt(2)*sigma_i*t + mu_i  gives a
Gauss-Hermite sum — O(n) per component, exact to machine precision:

    P_i = (1/sqrt(pi)) * sum_k  w_k * prod_{j != i} Phi_j(x_k^(i))

The same nodes give  E[X_i * f_i]  by weighting with x_k,
and then  C_i = E[X_i * f_i] - mu_i * P_i.
"""

import os
import numpy as np
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# 1.  ANALYTICAL — Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

def hermite_moments(mu, sigma, n_gh=50):
    """
    Compute P_i, Var(f_i), Cov(f_i, X_i) for every component i.

    Parameters
    ----------
    mu    : array (n,)   means of the n Gaussians
    sigma : array (n,)   standard deviations
    n_gh  : int          quadrature order (32 is more than sufficient)

    Returns
    -------
    P : array (n,)   P(X_i = max)
    V : array (n,)   Var(f_i)  =  P_i * (1 - P_i)
    C : array (n,)   Cov(f_i, X_i)

    Complexity:  O(n * n_gh)  in time and memory.
    """
    mu    = np.asarray(mu,    dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n     = len(mu)

    # Gauss-Hermite nodes t_k and weights w_k  (for int f(t) exp(-t^2) dt)
    nodes, weights = hermgauss(n_gh)

    # Map nodes to original space for each component i
    # x[i, k] = sqrt(2)*sigma_i*t_k + mu_i       shape (n, n_gh)
    x = np.sqrt(2.0) * sigma[:, None] * nodes[None, :] + mu[:, None]

    # log Phi_j(x[i, k])  for all (j, i, k)      shape (n_j, n_i, n_gh)
    logcdf = norm.logcdf(
        x[None, :, :],
        loc=mu[:, None, None],
        scale=sigma[:, None, None],
    )

    # prod_{j != i} Phi_j  via log-sum then removing the self-term j==i
    total_logprod = logcdf.sum(axis=0)                        # (n, n_gh)
    self_logcdf   = logcdf[np.arange(n), np.arange(n), :]    # (n, n_gh)
    prod_excl     = np.exp(total_logprod - self_logcdf)       # (n, n_gh)

    # (1/sqrt(pi)) * sum_k w_k * f(t_k)
    w   = weights[None, :] / np.sqrt(np.pi)
    P   = np.sum(w * prod_excl,     axis=1)   # P(X_i = max)
    EXf = np.sum(w * x * prod_excl, axis=1)   # E[X_i * f_i]

    # Normalize P
    P = P / np.sum(P)
    V = P * (1.0 - P)    # Var(f_i)   — Bernoulli identity
    C = EXf - mu * P     # Cov(f_i, X_i)

    return P, V, C


# ---------------------------------------------------------------------------
# 2.  MONTE CARLO — ground truth for validation
# ---------------------------------------------------------------------------

def mc_moments(mu, sigma, n_samples=100_000, seed=0):
    """
    Estimate P_i, Var(f_i), Cov(f_i, X_i) by Monte Carlo.

    f_i is the HARD 0/1 indicator (NOT the soft ratio X_i / sum(X_j)).

    Returns
    -------
    P, V, C : arrays (n,)
    ci95    : dict  — 95% confidence half-widths for P, V, C
    """
    mu    = np.asarray(mu,    dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n     = len(mu)
    rng   = np.random.default_rng(seed)

    # Draw n_samples joint vectors, identify the winner each time
    samples = rng.normal(mu, sigma, size=(n_samples, n))   # (S, n)
    winners = np.argmax(samples, axis=1)                   # (S,)

    # Sufficient statistics
    counts   = np.bincount(winners, minlength=n).astype(float)
    win_vals = samples[np.arange(n_samples), winners]
    sum_xw   = np.zeros(n);  np.add.at(sum_xw,  winners, win_vals)
    sum_xw2  = np.zeros(n);  np.add.at(sum_xw2, winners, win_vals ** 2)

    P   = counts / n_samples
    EXf = sum_xw  / n_samples
    V   = P * (1.0 - P)
    C   = EXf - mu * P

    # 95% CI half-widths via delta method
    se_P   = np.sqrt(np.maximum(P * (1.0 - P), 0.0) / n_samples)
    se_EXf = np.sqrt(np.maximum(sum_xw2 / n_samples - EXf ** 2, 0.0) / n_samples)
    ci95 = {
        "P": 1.96 * se_P,
        "V": 1.96 * np.abs(1.0 - 2.0 * P) * se_P,
        "C": 1.96 * np.sqrt(se_EXf ** 2 + (mu * se_P) ** 2),
    }

    return P, V, C, ci95


# ---------------------------------------------------------------------------
# 3.  PRINT — side-by-side table Hermite vs MC
# ---------------------------------------------------------------------------

def print_comparison(mu, sigma, n_gh=50, n_samples=200_000, seed=0):
    """Print a clean comparison table for one (mu, sigma) point."""
    P_a, V_a, C_a       = hermite_moments(mu, sigma, n_gh)
    P_m, V_m, C_m, ci95 = mc_moments(mu, sigma, n_samples, seed)

    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  mu    = {np.round(mu, 3)}")
    print(f"  sigma = {np.round(sigma, 3)}")
    print(f"{sep}")
    hdr = (f"{'i':>3}  {'P_ana':>8}  {'P_mc':>8}  "
           f"{'V_ana':>8}  {'V_mc':>8}  {'C_ana':>8}  {'C_mc':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i in range(len(mu)):
        print(
            f"{i:>3}  {P_a[i]:>8.5f}  {P_m[i]:>8.5f}  "
            f"{V_a[i]:>8.5f}  {V_m[i]:>8.5f}  "
            f"{C_a[i]:>8.5f}  {C_m[i]:>8.5f}"
        )
    print(f"\n  Sum P  Hermite={P_a.sum():.8f}   MC={P_m.sum():.8f}")
    print(f"\n  Max |Hermite - MC|  (MC 95% CI half-width in parentheses):")
    print(f"    P : {np.max(np.abs(P_a - P_m)):.2e}   (CI95={ci95['P'].max():.2e})")
    print(f"    V : {np.max(np.abs(V_a - V_m)):.2e}   (CI95={ci95['V'].max():.2e})")
    print(f"    C : {np.max(np.abs(C_a - C_m)):.2e}   (CI95={ci95['C'].max():.2e})")


# ---------------------------------------------------------------------------
# 5.  ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("plots", "comparison")

    # Numerical check: print table of values and errors at one point
    print_comparison(
        mu    = np.array([ 3,  0.0, -0.1,  3.1]),
        sigma = np.array([ 0.01,  1.0,  0.01,  30]),
        n_samples=1_000_000,
    )

