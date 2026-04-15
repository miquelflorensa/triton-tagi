"""
Inference-based initialization — MNIST, first linear layer.

Top row   : 5 selected units as single Gaussians
            N(E_batch[mz_i], sqrt(Var_batch[mz_i] + E_batch[Sz_i]))
            Left: He init       Right: after inference_init

Bottom row: TAGI aggregate constraint moments, drawn as Gaussians.
  The y-axis represents the probability density of the aggregate statistic.
  Left  : S  / A  =  (1/A) Σ_i Z_i         (avg unit mean)
  Right : S2 / A  =  (1/A) Σ_i Z_i²        (avg unit 2nd moment)
  Each panel: before (blue) / after (green) / target (red dashed).

TAGI moment definitions (matching inference_init verbose output):
  μ_S   = mean_batch( Σ_i mz_i )
  σ²_S  = mean_batch( Σ_i Sz_i )                    ← sum of epistemic vars
  μ_S2  = mean_batch( Σ_i (mz_i² + Sz_i) )
  σ²_S2 = mean_batch( Σ_i (2Sz_i² + 4Sz_i mz_i²) ) ← GMA approximation

Usage:
    conda run -n cuTAGI python posterior_geometry_mnist_layer1.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from torchvision import datasets

from triton_tagi import Sequential, inference_init
from triton_tagi.layers import Linear, ReLU, Remax
from triton_tagi.kernels.common import triton_fused_var_forward

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 5          # large batch → stable GMA estimates
N_UNITS     = 5
N_INIT_ITER = 1

SIGMA_M = 1.0      # σ_M² = 0.5
SIGMA_Z = 1.0      # σ_Z² = 0.5   →  σ²_total = 1
A       = 5               # first-layer width

C_BEFORE = '#2E86AB'
C_AFTER  = '#1A8754'
C_TARGET = '#E94F37'

RIDGE_X   = (-5.0, 5.0)
RIDGE_SPACING = 1.0
PEAK_HEIGHT   = 0.7

pt = 1.0 / 72.27
FIG_WIDTH  = 246.0 * pt
FIG_HEIGHT = FIG_WIDTH * 1.2

plt.rc('font', family='serif', size=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)


# ── Data & network ────────────────────────────────────────────────────────────

def load_batch(n=BATCH_SIZE, data_dir="data"):
    ds   = datasets.MNIST(data_dir, train=True, download=True)
    x    = ds.data.float().view(-1, 784) / 255.0
    pmu  = x.mean(dim=0, keepdim=True)
    pstd = x.std(dim=0, keepdim=True).clamp(min=1e-7)
    return ((x - pmu) / pstd)[:n].to(DEVICE)


def build_net():
    return Sequential([
        Linear(784, A, device=DEVICE),
        ReLU(),
        Linear(A,   A, device=DEVICE),
        ReLU(),
        Linear(A, 10,  device=DEVICE),
        Remax(),
    ], device=DEVICE)


def first_layer_moments(net, x):
    layer = net.layers[0]
    with torch.no_grad():
        mz = torch.matmul(x, layer.mw) + layer.mb
        Sz = triton_fused_var_forward(x, torch.zeros_like(x),
                                      layer.mw, layer.Sw, layer.Sb)
    return mz.cpu().float().numpy(), Sz.cpu().float().numpy()


# ── TAGI moments (match inference_init verbose output) ────────────────────────

def tagi_moments(mz, Sz):
    """
    Returns (μ_S, σ²_S, μ_S2, σ²_S2) — batch-averaged aggregate statistics.
    Identical to the quantities printed by inference_init verbose.
    """
    # S constraint
    mu_S   = float(mz.sum(1).mean())
    var_S  = float(Sz.sum(1).mean())                       # σ²_S = Σ Sz avg

    # S2 constraint (GMA)
    mu_Z2  = mz**2 + Sz
    sig2_Z2 = 2.0 * Sz**2 + 4.0 * Sz * mz**2
    mu_S2  = float(mu_Z2.sum(1).mean())
    var_S2 = float(sig2_Z2.sum(1).mean())                  # σ²_S2 (GMA)

    return mu_S, var_S, mu_S2, var_S2


def target_moments(sigma_M, sigma_Z, n_units):
    sM2, sZ2 = sigma_M**2, sigma_Z**2
    mu_S_t   = 0.0
    var_S_t  = n_units * sZ2
    mu_S2_t  = n_units * (sM2 + sZ2)
    var_S2_t = n_units * (2 * sZ2**2 + 4 * sM2 * sZ2)
    return mu_S_t, var_S_t, mu_S2_t, var_S2_t


# ── Per-unit parameters (for ridge plot) ──────────────────────────────────────

def per_unit_params(mz, Sz):
    mu    = mz.mean(axis=0)
    sigma = np.sqrt(np.maximum(mz.var(axis=0) + Sz.mean(axis=0), 1e-12))
    return mu, sigma


def pick_units(sigma_all, n=N_UNITS):
    order = np.argsort(sigma_all)
    picks = np.round(np.linspace(0, len(order) - 1, n)).astype(int)
    return sorted(order[picks].tolist())


# ── Drawing ───────────────────────────────────────────────────────────────────

def _gauss(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


def draw_ridge(ax, mu_all, sigma_all, unit_ids, color):
    x = np.linspace(*RIDGE_X, 500)
    offsets = []
    for k, uid in enumerate(unit_ids):
        offset = (N_UNITS - 1 - k) * RIDGE_SPACING
        offsets.append(offset)
        y = _gauss(x, mu_all[uid], sigma_all[uid])
        y = y * (PEAK_HEIGHT / y.max())
        ax.plot(x, y + offset, color=color, lw=1.0)
        ax.fill_between(x, offset, y + offset, color=color, alpha=0.2)
    ax.axvline(0, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlim(*RIDGE_X)
    ax.set_ylim(-0.15, (N_UNITS - 1) * RIDGE_SPACING + PEAK_HEIGHT + 0.15)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_yticks([o + PEAK_HEIGHT * 0.3 for o in offsets])
    ax.set_yticklabels([rf'$Z_{{{k+1}}}$' for k in range(N_UNITS)], fontsize=6)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.set_xticks([])


def draw_constraint_panel(ax, before, after, target, xlabel):
    """
    before / after / target are (mu, sigma) pairs.
    x-range is set to cover all three Gaussians ± 3.5 sigma (before dominates).
    """
    mu_b, std_b = before
    mu_a, std_a = after
    mu_t, std_t = target

    # Let before set the range (it's wider), centred on the action
    lo = min(mu_b - 3.5*std_b, mu_a - 3.5*std_a, mu_t - 3.5*std_b)
    hi = max(mu_b + 3.5*std_b, mu_a + 3.5*std_a, mu_t + 3.5*std_b)
    x  = np.linspace(lo, hi, 600)

    for mu, std, c, ls in [
        (mu_b, std_b, C_BEFORE, '-'),
        (mu_a, std_a, C_AFTER,  '-'),
        (mu_t, std_t, C_TARGET, '--'),
    ]:
        y = _gauss(x, mu, std)
        ax.plot(x, y, color=c, lw=1.0, ls=ls)
        if ls == '-':
            ax.fill_between(x, y, color=c, alpha=0.15)

    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    x   = load_batch()
    net = build_net()

    # ── Before ───────────────────────────────────────────────────────────
    mz_b, Sz_b = first_layer_moments(net, x)
    mu_b_all, sigma_b_all = per_unit_params(mz_b, Sz_b)
    unit_ids = pick_units(sigma_b_all)

    mu_S_b, var_S_b, mu_S2_b, var_S2_b = tagi_moments(mz_b, Sz_b)

    # ── Inference init ───────────────────────────────────────────────────
    inference_init(net, x, sigma_M=SIGMA_M, sigma_Z=SIGMA_Z,
                   n_iter=N_INIT_ITER, verbose=True)

    # ── After ────────────────────────────────────────────────────────────
    mz_a, Sz_a = first_layer_moments(net, x)
    mu_a_all, sigma_a_all = per_unit_params(mz_a, Sz_a)

    mu_S_a, var_S_a, mu_S2_a, var_S2_a = tagi_moments(mz_a, Sz_a)

    # ── Target ───────────────────────────────────────────────────────────
    mu_S_t, var_S_t, mu_S2_t, var_S2_t = target_moments(SIGMA_M, SIGMA_Z, A)

    # Normalise by A so both panels are on a per-unit scale
    def norm_by_A(mu, var):
        return mu, np.sqrt(max(var, 1e-12))

    print("\n─── S constraint  (normalised by A) ───")
    print(f"  Before : N({mu_S_b:.4f}, {np.sqrt(var_S_b):.4f})")
    print(f"  After  : N({mu_S_a:.4f}, {np.sqrt(var_S_a):.4f})")
    print(f"  Target : N({mu_S_t:.4f}, {np.sqrt(var_S_t):.4f})")

    print("─── S2 constraint (normalised by A) ───")
    print(f"  Before : N({mu_S2_b:.4f}, {np.sqrt(var_S2_b):.4f})")
    print(f"  After  : N({mu_S2_a:.4f}, {np.sqrt(var_S2_a):.4f})")
    print(f"  Target : N({mu_S2_t:.4f}, {np.sqrt(var_S2_t):.4f})")

    # ── Figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1.4],
                           hspace=0.30, wspace=0.25)

    ax_tl = fig.add_subplot(gs[0, 0])
    draw_ridge(ax_tl, mu_b_all, sigma_b_all, unit_ids, C_BEFORE)
    ax_tl.set_title('(a) He init', fontsize=8, pad=3)

    ax_tr = fig.add_subplot(gs[0, 1])
    draw_ridge(ax_tr, mu_a_all, sigma_a_all, unit_ids, C_AFTER)
    ax_tr.set_title('(b) Inferenced-based init', fontsize=8, pad=3)

    ax_bl = fig.add_subplot(gs[1, 0])
    draw_constraint_panel(
        ax_bl,
        before=norm_by_A(mu_S_b, var_S_b),
        after =norm_by_A(mu_S_a, var_S_a),
        target=norm_by_A(mu_S_t, var_S_t),
        xlabel=r'$S = \sum_i Z_i$',
    )

    ax_br = fig.add_subplot(gs[1, 1])
    draw_constraint_panel(
        ax_br,
        before=norm_by_A(mu_S2_b, var_S2_b),
        after =norm_by_A(mu_S2_a, var_S2_a),
        target=norm_by_A(mu_S2_t, var_S2_t),
        xlabel=r'$S2 = \sum_i Z_i^2$',
    )

    legend_handles = [
        Line2D([0], [0], color=C_BEFORE, lw=1.5, label='He init'),
        Line2D([0], [0], color=C_AFTER,  lw=1.5, label='Inference-based init'),
        Line2D([0], [0], color=C_TARGET, lw=1.5, ls='--', label='Target'),
    ]
    bottom_of_top = min(ax_tl.get_position().y0, ax_tr.get_position().y0)
    top_of_bottom = max(ax_bl.get_position().y1, ax_br.get_position().y1)
    legend_y = top_of_bottom + 0.5 * (bottom_of_top - top_of_bottom)

    fig.legend(
        handles=legend_handles,
        loc='center',
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        fontsize=6,
        handlelength=1.8,
        frameon=False,
        borderaxespad=0.0,
        borderpad=0.0,
        labelspacing=0.2,
        columnspacing=1.2,
    )

    os.makedirs("figures", exist_ok=True)
    out = "figures/inference_init_mnist_layer1.png"
    fig.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    out_pdf = out.replace('.png', '.pdf')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved → {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
