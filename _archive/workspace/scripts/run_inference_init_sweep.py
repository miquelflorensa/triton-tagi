"""
Inference-init sweep on MNIST.

Systematically evaluates inference-based initialization across:
  - Network depths:  3, 5, 7 hidden layers
  - Observation noise: sigma_v in {0.01, 0.05}
  - Target moments:   (var_M, var_Z) grid

Each configuration trains for 10 epochs and records per-epoch test accuracy.
Results are saved to JSON, then plotted as publication-ready figures.

Usage:
    conda run -n cuTAGI python run_inference_init_sweep.py
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

from triton_tagi import Sequential, inference_init
from triton_tagi.layers import Linear, ReLU, Remax

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 128
N_EPOCHS   = 10
HIDDEN_DIM = 512
OUT_DIM    = 10
IN_DIM     = 784

DEPTHS     = [1, 3, 5, 7]                        # number of hidden layers
SIGMA_VS   = [0.01, 0.05]
VAR_PAIRS  = [                                   # (sigma_M, sigma_Z)
    (None,  None),                               # He init baseline
    (1.0,   1.0),
    (1.0,   0.5),
    (0.5,   1.0),
    (0.5,   0.5),
    (1.5,   1.0),
    (1.0,   1.5),
]

RESULTS_PATH = Path("results_inference_init_sweep.json")

# ============================================================================
# DATA
# ============================================================================

def load_mnist(data_dir="data"):
    train_ds = datasets.MNIST(data_dir, train=True,  download=True)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test  = test_ds.data.float().view(-1, 784)  / 255.0

    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test  = ((x_test  - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels  = test_ds.targets.to(DEVICE)

    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train_labels, x_test, y_test_labels


# ============================================================================
# NETWORK BUILDER
# ============================================================================

def build_net(n_hidden):
    """MLP with n_hidden hidden layers of width HIDDEN_DIM."""
    layers = []
    dims = [IN_DIM] + [HIDDEN_DIM] * n_hidden + [OUT_DIM]
    for i in range(len(dims) - 2):
        layers += [Linear(dims[i], dims[i+1], device=DEVICE), ReLU()]
    layers += [Linear(dims[-2], dims[-1], device=DEVICE), Remax()]
    return Sequential(layers, device=DEVICE)


# ============================================================================
# TRAINING
# ============================================================================

def evaluate(net, x_test, y_labels, batch_size=1024):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        mu, _ = net.forward(x_test[i:i+batch_size])
        correct += (mu.argmax(1) == y_labels[i:i+batch_size]).sum().item()
    return correct / len(x_test)


def train_one(net, x_train, y_train_oh, x_test, y_test_labels, sigma_v):
    accs = []
    for epoch in range(1, N_EPOCHS + 1):
        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s, y_s = x_train[perm], y_train_oh[perm]
        for i in range(0, len(x_s), BATCH_SIZE):
            net.step(x_s[i:i+BATCH_SIZE], y_s[i:i+BATCH_SIZE], sigma_v)
        torch.cuda.synchronize()
        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)
    return accs


# ============================================================================
# SWEEP
# ============================================================================

def run_sweep():
    print("Loading MNIST...")
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()
    print(f"Train: {x_train.shape[0]:,}  Test: {x_test.shape[0]:,}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    configs = list(itertools.product(DEPTHS, SIGMA_VS, VAR_PAIRS))
    total = len(configs)

    for run_i, (depth, sigma_v, (sigma_M, sigma_Z)) in enumerate(configs):
        label = "He" if sigma_M is None else f"sM={sigma_M},sZ={sigma_Z}"
        print(f"\n[{run_i+1}/{total}] depth={depth}  sigma_v={sigma_v}  {label}")

        torch.manual_seed(42)
        np.random.seed(42)
        net = build_net(depth)

        if sigma_M is not None:
            inference_init(net, x_train, sigma_M=sigma_M, sigma_Z=sigma_Z,
                           verbose=False)

        t0 = time.perf_counter()
        accs = train_one(net, x_train, y_train_oh, x_test, y_test_labels,
                         sigma_v)
        dt = time.perf_counter() - t0

        best = max(accs)
        print(f"  best={best*100:.2f}%  final={accs[-1]*100:.2f}%  "
              f"time={dt:.1f}s")

        results.append({
            "depth": depth,
            "sigma_v": sigma_v,
            "sigma_M": sigma_M,
            "sigma_Z": sigma_Z,
            "label": label,
            "accs": accs,
            "best": best,
            "time_s": dt,
        })

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    return results


# ============================================================================
# FIGURES
# ============================================================================

def make_figures(results=None):
    import matplotlib.pyplot as plt

    # -- LaTeX / style setup (STYLE_GUIDE.md) --------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=6)

    # -- dimensions -----------------------------------------------------------
    pt = 1.0 / 72.27
    TWO_COL = 510.0 * pt          # PRD two-column width
    golden  = (1 + 5**0.5) / 2

    # -- colors ---------------------------------------------------------------
    COLORS = {
        "He":               "#2C3E50",
        "sM=1.0,sZ=1.0":   "#2E86AB",
        "sM=1.0,sZ=0.5":   "#E94F37",
        "sM=0.5,sZ=1.0":   "#F39C12",
        "sM=0.5,sZ=0.5":   "#9B59B6",
        "sM=1.5,sZ=1.0":   "#2E7D32",
        "sM=1.0,sZ=1.5":   "#C62828",
    }
    MARKERS = {
        "He":               "D",
        "sM=1.0,sZ=1.0":   "o",
        "sM=1.0,sZ=0.5":   "s",
        "sM=0.5,sZ=1.0":   "^",
        "sM=0.5,sZ=0.5":   "v",
        "sM=1.5,sZ=1.0":   "P",
        "sM=1.0,sZ=1.5":   "X",
    }
    LATEX_LABELS = {
        "He":               r"He init",
        "sM=1.0,sZ=1.0":   r"$\sigma_M\!=\!1.0,\;\sigma_Z\!=\!1.0$",
        "sM=1.0,sZ=0.5":   r"$\sigma_M\!=\!1.0,\;\sigma_Z\!=\!0.5$",
        "sM=0.5,sZ=1.0":   r"$\sigma_M\!=\!0.5,\;\sigma_Z\!=\!1.0$",
        "sM=0.5,sZ=0.5":   r"$\sigma_M\!=\!0.5,\;\sigma_Z\!=\!0.5$",
        "sM=1.5,sZ=1.0":   r"$\sigma_M\!=\!1.5,\;\sigma_Z\!=\!1.0$",
        "sM=1.0,sZ=1.5":   r"$\sigma_M\!=\!1.0,\;\sigma_Z\!=\!1.5$",
    }

    if results is None:
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 1: Training curves  (depth x sigma_v grid)
    #   rows = sigma_v values,  cols = depths
    # ================================================================
    n_sv = len(SIGMA_VS)
    n_d  = len(DEPTHS)
    fig, axes = plt.subplots(n_sv, n_d, figsize=(TWO_COL, TWO_COL / golden),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    epochs = np.arange(1, N_EPOCHS + 1)

    for row, sigma_v in enumerate(SIGMA_VS):
        for col, depth in enumerate(DEPTHS):
            ax = axes[row, col]
            subset = [r for r in results
                      if r["depth"] == depth and r["sigma_v"] == sigma_v]

            for r in subset:
                lbl = r["label"]
                ax.plot(epochs, np.array(r["accs"]) * 100,
                        color=COLORS[lbl], marker=MARKERS[lbl],
                        markersize=3, markevery=2, linewidth=1.2,
                        label=LATEX_LABELS[lbl])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, N_EPOCHS)

            if row == 0:
                ax.set_title(rf'\textbf{{{depth} hidden layers}}',
                             fontsize=9, pad=3)
            if row == n_sv - 1:
                ax.set_xlabel('Epoch')
            if col == 0:
                ax.set_ylabel(rf'Test accuracy (\%)')
                ax.annotate(rf'$\sigma_v\!=\!{sigma_v}$',
                            xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-42, 0), textcoords='offset points',
                            fontsize=8, ha='right', va='center',
                            fontweight='bold')

    # single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(VAR_PAIRS), framealpha=0.9, fontsize=6,
               bbox_to_anchor=(0.5, 1.04))

    for fmt in ("pdf", "png"):
        fig.savefig(figdir / f"inference_init_curves.{fmt}",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved figures/inference_init_curves.pdf|png")

    # ================================================================
    # Figure 2: Best accuracy heatmap  (rows = methods, cols = configs)
    #   One subplot per sigma_v
    # ================================================================
    methods = [p for p in VAR_PAIRS]
    method_labels = [LATEX_LABELS["He" if m is None else
                     f"sM={m},sZ={z}"]
                     for m, z in methods]
    # mathtt{L} for LaTeX monospace L in depth labels
    depth_labels = [rf'$\mathtt{{L}}={d}$' for d in DEPTHS]

    fig, axes_hm = plt.subplots(1, n_sv, figsize=(TWO_COL, 1.8),
                                constrained_layout=True)

    for sv_i, sigma_v in enumerate(SIGMA_VS):
        ax = axes_hm[sv_i]
        mat = np.zeros((len(methods), n_d))
        for mi, (sM, sZ) in enumerate(methods):
            for di, depth in enumerate(DEPTHS):
                match = [r for r in results
                         if r["depth"] == depth
                         and r["sigma_v"] == sigma_v
                         and r["sigma_M"] == sM
                         and r["sigma_Z"] == sZ]
                mat[mi, di] = match[0]["best"] * 100 if match else 0.0

        im = ax.imshow(mat, aspect='auto', cmap='viridis',
                       vmin=10, vmax=100)

        # annotate cells
        for mi in range(len(methods)):
            for di in range(n_d):
                val = mat[mi, di]
                color = 'white' if val < 60 else 'black'
                ax.text(di, mi, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
                
        # change sigma_M by sigma_{\manthtt{M}} in labels
        for mi, (sM, sZ) in enumerate(methods):
            lbl = "He" if sM is None else f"sM={sM},sZ={sZ}"
            method_labels[mi] = LATEX_LABELS[lbl].replace(
                r'\sigma_M', r'\sigma_{{M}}').replace(r'\sigma_Z', r'\sigma_{{Z}}')
            

        ax.set_xticks(range(n_d))
        ax.set_xticklabels(depth_labels)
        ax.set_xlabel('Depth')
        if sv_i == 0:
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(method_labels)
        else:
            ax.set_yticks([])
        ax.set_title(rf'\textbf{{$\sigma_V = {sigma_v}$}}',
                     fontsize=9, pad=3)

    fig.colorbar(im, ax=axes_hm.tolist(), fraction=0.03, pad=0.04,
                 label=r'Best test accuracy (\%)', shrink=0.9)

    for fmt in ("pdf", "png"):
        fig.savefig(figdir / f"inference_init_heatmap.{fmt}",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved figures/inference_init_heatmap.pdf|png")

    # ================================================================
    # Figure 3: Depth scaling  (accuracy at epoch 5 and 10 vs depth)
    #   One subplot per sigma_v, lines per method
    # ================================================================
    fig, axes_ds = plt.subplots(1, n_sv, figsize=(TWO_COL, TWO_COL / golden / 1.6),
                                sharey=True, constrained_layout=True)

    for sv_i, sigma_v in enumerate(SIGMA_VS):
        ax = axes_ds[sv_i]
        for sM, sZ in VAR_PAIRS:
            lbl = "He" if sM is None else f"sM={sM},sZ={sZ}"
            bests = []
            for depth in DEPTHS:
                match = [r for r in results
                         if r["depth"] == depth
                         and r["sigma_v"] == sigma_v
                         and r["sigma_M"] == sM
                         and r["sigma_Z"] == sZ]
                bests.append(match[0]["best"] * 100 if match else 0.0)

            ax.plot(DEPTHS, bests, color=COLORS[lbl], marker=MARKERS[lbl],
                    markersize=5, linewidth=1.5, label=LATEX_LABELS[lbl])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Hidden layers')
        ax.set_xticks(DEPTHS)
        if sv_i == 0:
            ax.set_ylabel(r'Best test accuracy (\%)')
        ax.set_title(rf'\textbf{{$\sigma_v = {sigma_v}$}}',
                     fontsize=9, pad=3)

    handles, labels = axes_ds[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(VAR_PAIRS), framealpha=0.9, fontsize=6,
               bbox_to_anchor=(0.5, 1.07))

    for fmt in ("pdf", "png"):
        fig.savefig(figdir / f"inference_init_depth_scaling.{fmt}",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved figures/inference_init_depth_scaling.pdf|png")


# ============================================================================
# INCREMENTAL SWEEP (only run missing configs)
# ============================================================================

def run_missing():
    """Load existing results, run only missing (depth, sigma_v, var_pair)
    configs, merge, save, and return the full result list."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {RESULTS_PATH}")
    else:
        existing = []

    # index what we already have
    done = {(r["depth"], r["sigma_v"], r["sigma_M"], r["sigma_Z"])
            for r in existing}

    all_configs = list(itertools.product(DEPTHS, SIGMA_VS, VAR_PAIRS))
    missing = [(d, sv, sM, sZ)
               for d, sv, (sM, sZ) in all_configs
               if (d, sv, sM, sZ) not in done]

    if not missing:
        print("All configs already computed. Generating figures only.")
        return existing

    print(f"{len(missing)} missing configs to run.")
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()

    for run_i, (depth, sigma_v, sigma_M, sigma_Z) in enumerate(missing):
        label = "He" if sigma_M is None else f"sM={sigma_M},sZ={sigma_Z}"
        print(f"\n[{run_i+1}/{len(missing)}] depth={depth}  "
              f"sigma_v={sigma_v}  {label}")

        torch.manual_seed(42)
        np.random.seed(42)
        net = build_net(depth)

        if sigma_M is not None:
            inference_init(net, x_train, sigma_M=sigma_M, sigma_Z=sigma_Z,
                           verbose=False)

        t0 = time.perf_counter()
        accs = train_one(net, x_train, y_train_oh, x_test, y_test_labels,
                         sigma_v)
        dt = time.perf_counter() - t0

        best = max(accs)
        print(f"  best={best*100:.2f}%  final={accs[-1]*100:.2f}%  "
              f"time={dt:.1f}s")

        existing.append({
            "depth": depth,
            "sigma_v": sigma_v,
            "sigma_M": sigma_M,
            "sigma_Z": sigma_Z,
            "label": label,
            "accs": accs,
            "best": best,
            "time_s": dt,
        })

    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH} ({len(existing)} total)")
    return existing


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_missing()
    make_figures(results)
