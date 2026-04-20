"""
posterior_geometry_var_scaled.py — Visualize TAGI's Variance-Scaled Geometry
=============================================================================

After training a model on CIFAR-10/100, this script draws a 2D cross-section
through the posterior weight distribution. 

Instead of tracking historical epochs, this uses the TAGI-Native method:
  1. Stand at the final converged posterior means (μ).
  2. Generate two random orthogonal directions in Z-space (z1, z2).
  3. Scale those directions element-wise by the posterior standard deviations (σ).
  4. Build a grid: w(α, β) = μ + α·(z1 ⊙ σ) + β·(z2 ⊙ σ).
  
The α and β axes represent Standard Deviations (Z-scores) from the mean,
allowing us to directly visualize how well TAGI's Gaussian approximation
matches the true underlying loss landscape.

Usage
-----
    conda run -n cuTAGI python posterior_geometry_var_scaled.py --epochs 100 --num-std 3.0
    conda run -n cuTAGI python posterior_geometry_var_scaled.py --arch resnet18 --ckpt-dir run_logs/checkpoints --epochs 139
    conda run -n cuTAGI python posterior_geometry_var_scaled.py --arch resnet18_adam --epochs 100
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering — no display needed
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cifar10 import build_simple_3cnn, load_cifar10, train, evaluate, DEVICE
from run_resnet18 import (build_resnet18, load_checkpoint as resnet_load_checkpoint,
                          load_cifar10 as resnet_load_cifar10,
                          evaluate as resnet_evaluate)
from run_resnet18_cifar100_adam import (
    build_resnet18 as build_resnet18_c100,
    load_cifar100,
    load_checkpoint as c100_load_checkpoint,
    evaluate as c100_evaluate,
    NUM_CLASSES as C100_NUM_CLASSES,
)
from triton_tagi.init import init_residual_aware
from triton_tagi.monitor import TAGIMonitor


# ══════════════════════════════════════════════════════════════════════════
#  Weight-space helpers (Means & Variances)
# ══════════════════════════════════════════════════════════════════════════

def _learnable_layers(net):
    return [l for l in net.layers if hasattr(l, 'mw')]

def _has_bias(layer):
    if hasattr(layer, 'has_bias'):
        return layer.has_bias
    return getattr(layer, 'mb', None) is not None

def get_flat_means(net):
    parts = []
    for layer in _learnable_layers(net):
        parts.append(layer.mw.detach().flatten())
        if _has_bias(layer):
            parts.append(layer.mb.detach().flatten())
    return torch.cat(parts)

def get_flat_vars(net):
    parts = []
    for layer in _learnable_layers(net):
        parts.append(layer.Sw.detach().flatten())
        if _has_bias(layer):
            parts.append(layer.Sb.detach().flatten())
    return torch.cat(parts)

def set_flat_means(net, w_flat):
    idx = 0
    for layer in _learnable_layers(net):
        n = layer.mw.numel()
        layer.mw.data.copy_(w_flat[idx:idx + n].view_as(layer.mw))
        idx += n
        if _has_bias(layer):
            n = layer.mb.numel()
            layer.mb.data.copy_(w_flat[idx:idx + n].view_as(layer.mb))
            idx += n


# ══════════════════════════════════════════════════════════════════════════
#  Landscape evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_point(net, x, y_oh, y_lbl, batch_size=64):
    """
    Evaluate the *current* weight state on the given data.
    """
    net.eval()
    total_nll = 0.0
    correct   = 0
    n         = len(x)
    eps       = 1e-7

    for i in range(0, n, batch_size):
        xb  = x[i:i + batch_size]
        yb  = y_oh[i:i + batch_size]
        lb  = y_lbl[i:i + batch_size]
        
        # In TAGI, the forward pass inherently uses the injected Sw/Sb
        mu, _ = net.forward(xb)
        total_nll += -(yb * torch.log(mu.clamp(min=eps))).sum(dim=1).sum().item()
        correct   += (mu.argmax(1) == lb).sum().item()
        del mu
    torch.cuda.empty_cache()

    return total_nll / n, correct / n


# ══════════════════════════════════════════════════════════════════════════
#  Core geometry routine (Variance-Scaled Method)
# ══════════════════════════════════════════════════════════════════════════

def compute_landscape(net, x_eval, y_oh, y_lbl, grid_size=31, num_std=3.0):
    """
    Build Log-Likelihood and accuracy landscapes on a 2D grid defined by
    random directions scaled precisely by TAGI's posterior standard deviations.
    """
    # 1. Extract Final Converged Means and Variances
    m1 = get_flat_means(net).clone()
    S1 = get_flat_vars(net).clone()
    
    # Calculate standard deviation (σ)
    sigma = torch.sqrt(S1.clamp(min=1e-10))
    D = m1.numel()

    # 2. Generate random orthogonal directions in Z-space
    z1 = torch.randn_like(m1)
    z2 = torch.randn_like(m1)
    
    # Gram-Schmidt to ensure z2 is perfectly orthogonal to z1
    z2 = z2 - (torch.dot(z1, z2) / torch.dot(z1, z1)) * z1
    
    # 3. Scale directions by the posterior standard deviations
    dir1 = z1 * sigma
    dir2 = z2 * sigma

    # Grid directly represents number of Standard Deviations
    coords = torch.linspace(-num_std, num_std, grid_size, device=m1.device)

    print(f"  [geometry] Flat weight dim D = {D:,}")
    print(f"  [geometry] Grid: {grid_size}×{grid_size}, range ±{num_std} std devs")
    print(f"  [geometry] Evaluating {grid_size**2} grid points on {len(x_eval):,} samples …")

    nll_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    acc_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i, alpha in enumerate(coords):
        for j, beta in enumerate(coords):
            # Shift the mean. Variances remain fixed at their converged state.
            m_pt = m1 + alpha * dir1 + beta * dir2
            set_flat_means(net, m_pt)
            
            nll, acc = eval_point(net, x_eval, y_oh, y_lbl)
            nll_grid[i, j] = nll
            acc_grid[i, j] = acc
            
            del m_pt

        torch.cuda.empty_cache()
        if (i % 5 == 0) or (i == grid_size - 1):
            pct = 100 * (i + 1) / grid_size
            print(f"    row {i+1:3d}/{grid_size}  ({pct:.0f}%)", flush=True)

    # Restore the original posterior means
    set_flat_means(net, m1)
    print("  [geometry] Original weights restored.")

    return nll_grid, acc_grid, coords.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
#  Plotting (1x2 Clean Layout)
# ══════════════════════════════════════════════════════════════════════════

def plot_landscape(nll_grid, acc_grid, coords, n_eval,
                   title_suffix="", save_path="run_logs/posterior_geometry_var_scaled.png",
                   arch_label="3-CNN"):
    """
    Produce a clean 1x2 contour plot showing Log-Likelihood and Accuracy.
    """
    log_lik_grid = -nll_grid * n_eval

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], log_lik_grid, "RdYlGn", "Log-Likelihood (higher is better)", "Log-Likelihood"),
        (axes[1], acc_grid,     "RdYlGn", "Accuracy [0–1]", "Accuracy"),
    ]

    for ax, grid, cmap, cbar_label, title in panels:
        cf = ax.contourf(coords, coords, grid.T, levels=30, cmap=cmap)
        ax.contour(coords, coords, grid.T, levels=30, colors='k', linewidths=0.25, alpha=0.35)
        plt.colorbar(cf, ax=ax, label=cbar_label)

        # Mark w1 (Center)
        ax.scatter([0], [0], c='royalblue', s=180, zorder=6, marker='*', label='μ (Final Converged Mean)')

        ax.set_xlabel('α  (Standard Deviations along Z₁)', fontsize=9)
        ax.set_ylabel('β  (Standard Deviations along Z₂)', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='lower right')

    plt.suptitle(
        f'TAGI Variance-Scaled Geometry ({arch_label})\n'
        f'{title_suffix}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="TAGI Variance-Scaled Geometry visualizer")
    ap.add_argument("--arch",       type=str,   default="3cnn",
                    choices=["3cnn", "resnet18", "resnet18_adam", "resnet18_cifar100", "resnet18_cifar100_adam"])
    ap.add_argument("--epochs",     type=int,   default=100,
                    help="The specific epoch checkpoint to load (default 100)")
    ap.add_argument("--grid",       type=int,   default=31,
                    help="Grid resolution N×N (default 31 → 961 evals)")
    ap.add_argument("--num-std",    type=float, default=3.0,
                    help="How many standard deviations to explore from the mean (default 3.0)")
    ap.add_argument("--eval-n",     type=int,   default=2000,
                    help="# test samples used for landscape evaluation (default 2000)")
    ap.add_argument("--sigma-v",    type=float, default=0.05,
                    help="Observation noise σ_v for 3cnn training (default 0.05)")
    ap.add_argument("--batch-size", type=int,   default=128,
                    help="Mini-batch size for training (default 128)")
    ap.add_argument("--ckpt-dir",   type=str,   default="run_logs/checkpoints",
                    help="Checkpoint directory for resnet18")
    ap.add_argument("--save",       type=str,   default=None,
                    help="Output path for the figure (auto-generated if not provided)")
    args = ap.parse_args()

    if args.save is None:
        args.save = f"run_logs/posterior_geometry_{args.arch}_var_scaled.png"
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    if args.arch == "resnet18_adam":
        arch_label = "ResNet-18 (CIFAR-10 Adam)"
        dataset_label = "CIFAR-10"
        num_classes = 10
        if args.ckpt_dir == "run_logs/checkpoints":
            args.ckpt_dir = "run_logs_adam_resnet/checkpoints"
    elif args.arch == "resnet18_cifar100_adam":
        arch_label = "ResNet-18 (CIFAR-100 Adam)"
        dataset_label = "CIFAR-100"
        num_classes = C100_NUM_CLASSES
        if args.ckpt_dir == "run_logs/checkpoints":
            args.ckpt_dir = "run_logs_adam_cifar100/checkpoints"
    elif args.arch == "resnet18_cifar100":
        arch_label = "ResNet-18 (CIFAR-100)"
        dataset_label = "CIFAR-100"
        num_classes = C100_NUM_CLASSES
        if args.ckpt_dir == "run_logs/checkpoints":
            args.ckpt_dir = "run_logs_cifar100/checkpoints"
    elif args.arch == "resnet18":
        arch_label = "ResNet-18"
        dataset_label = "CIFAR-10"
        num_classes = 10
    else:
        arch_label = "3-CNN"
        dataset_label = "CIFAR-10"
        num_classes = 10

    print("=" * 60)
    print(f"  TAGI Variance-Scaled Geometry — {arch_label} on {dataset_label}")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  Arch        : {args.arch}")
    print(f"  Target Epoch: {args.epochs}")
    print(f"  Grid        : {args.grid}×{args.grid} (±{args.num_std} std devs)")
    print(f"  Eval samples: {args.eval_n:,}")

    # ── Data ──
    if args.arch in ("resnet18_cifar100", "resnet18_cifar100_adam"):
        print("\n  Loading CIFAR-100 …")
        x_train, y_train_oh, y_train_lbl, x_test, y_test_lbl = load_cifar100()
    elif args.arch in ("resnet18", "resnet18_adam"):
        print("\n  Loading CIFAR-10 …")
        x_train, y_train_oh, y_train_lbl, x_test, y_test_lbl = resnet_load_cifar10()
    else:
        print("\n  Loading CIFAR-10 …")
        x_train, y_train_oh, y_train_lbl, x_test, y_test_lbl = load_cifar10()

    # ── Model ──
    if args.arch in ("resnet18_cifar100", "resnet18_cifar100_adam", "resnet18", "resnet18_adam"):
        print(f"\n  Building {arch_label} …")
        if "cifar100" in args.arch:
            net = build_resnet18_c100(num_classes=num_classes, head="remax", device=DEVICE, g_min=0.1, g_max=0.1)
            init_residual_aware(net, eta=0.125, verbose=False)
            eval_fn = c100_evaluate
        else:
            net = build_resnet18(num_classes=num_classes, head="remax", device=DEVICE, g_min=0.1, g_max=0.1)
            init_residual_aware(net, eta=0.5, verbose=False)
            eval_fn = resnet_evaluate

        # Print statistics about the final converged posterior weights
        mean_means = get_flat_means(net).mean().item()
        mean_vars  = get_flat_vars(net).mean().item()
        print(f"  [init] Mean of means (μ) before loading checkpoint: {mean_means:.6f}")
        print(f"  [init] Mean of variances (σ²) before loading checkpoint: {mean_vars:.6f}")
        std_means = get_flat_means(net).std().item()
        std_vars  = get_flat_vars(net).std().item()
        print(f"  [init] Std of means (μ) before loading checkpoint: {std_means:.6f}")
        print(f"  [init] Std of variances (σ²) before loading checkpoint: {std_vars:.6f}")
        max_means = get_flat_means(net).max().item()
        max_vars  = get_flat_vars(net).max().item()
        print(f"  [init] Max of means (μ) before loading checkpoint: {max_means:.6f}")
        print(f"  [init] Max of variances (σ²) before loading checkpoint: {max_vars:.6f}")
        min_means = get_flat_means(net).min().item()
        min_vars  = get_flat_vars(net).min().item()
        print(f"  [init] Min of means (μ) before loading checkpoint: {min_means:.6f}")
        print(f"  [init] Min of variances (σ²) before loading checkpoint: {min_vars:.6f}")
            
        net.train()
        net.forward(x_train[:32])
        net.eval()
        torch.cuda.empty_cache()

        ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{args.epochs:04d}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resnet_load_checkpoint(net, ckpt_path)
        print(f"  [Checkpoint] Loaded epoch {args.epochs} from {ckpt_path}")

    else:
        print(f"\n  Building 3-CNN …")
        net = build_simple_3cnn(num_classes=10, head="remax", device=DEVICE, gain_w=0.1, gain_b=0.1)

        print(f"\n  Training for {args.epochs} epoch(s) to reach final posterior …")
        monitor = TAGIMonitor(net, log_dir="run_logs", probe_size=256)
        
        for epoch in range(1, args.epochs + 1):
            train(net, x_train, y_train_oh, y_train_lbl, x_test, y_test_lbl,
                  batch_size=args.batch_size, initial_sigma_v=args.sigma_v,
                  n_epochs=1, monitor=monitor, monitor_every=1)
        eval_fn = evaluate


    # Print statistics about the final converged posterior weights
    mean_means = get_flat_means(net).mean().item()
    mean_vars  = get_flat_vars(net).mean().item()
    print(f"  [final] Mean of means (μ) after loading checkpoint: {mean_means:.6f}")    
    print(f"  [final] Mean of variances (σ²) after loading checkpoint: {mean_vars:.6f}")
    std_means = get_flat_means(net).std().item()
    std_vars  = get_flat_vars(net).std().item()
    print(f"  [final] Std of means (μ) after loading checkpoint: {std_means:.6f}")
    print(f"  [final] Std of variances (σ²) after loading checkpoint: {std_vars:.6f}")
    max_means = get_flat_means(net).max().item()
    max_vars  = get_flat_vars(net).max().item()
    print(f"  [final] Max of means (μ) after loading checkpoint: {max_means:.6f}")
    print(f"  [final] Max of variances (σ²) after loading checkpoint: {max_vars:.6f}")
    min_means = get_flat_means(net).min().item()
    min_vars  = get_flat_vars(net).min().item()
    print(f"  [final] Min of means (μ) after loading checkpoint: {min_means:.6f}")
    print(f"  [final] Min of variances (σ²) after loading checkpoint: {min_vars:.6f}")

    # ── Geometry ──
    print("\n  Computing variance-scaled geometry …")
    n_eval     = min(args.eval_n, len(x_test))
    x_eval     = x_test[:n_eval]
    y_eval_lbl = y_test_lbl[:n_eval]
    y_eval_oh  = torch.zeros(n_eval, num_classes, device=DEVICE)
    y_eval_oh.scatter_(1, y_eval_lbl.unsqueeze(1), 1.0)

    test_acc = eval_fn(net, x_test, y_test_lbl)
    print(f"  Test accuracy at centre (μ): {test_acc * 100:.2f}%")

    nll_grid, acc_grid, coords = compute_landscape(
        net, x_eval, y_eval_oh, y_eval_lbl,
        grid_size=args.grid,
        num_std=args.num_std
    )

    # ── Plot ──
    print("\n  Plotting …")
    title = f"Scaled by Posterior $\\sigma$  ·  Center Test Acc: {test_acc*100:.1f}%"
    plot_landscape(
        nll_grid, acc_grid, coords, n_eval,
        title_suffix=title,
        save_path=args.save,
        arch_label=arch_label,
    )

    # ── Summary ──
    cx, cy = args.grid // 2, args.grid // 2
    log_lik_grid = -nll_grid * n_eval
    print(f"\n  NLL  range        : [{nll_grid.min():.3f},  {nll_grid.max():.3f}]")
    print(f"  Acc  range        : [{acc_grid.min():.3f},  {acc_grid.max():.3f}]")
    print(f"  Log-Lik range     : [{log_lik_grid.min():.1f},  {log_lik_grid.max():.1f}]")
    print(f"  Centre NLL        : {nll_grid[cx, cy]:.3f}")
    print(f"  Centre Acc        : {acc_grid[cx, cy]:.3f}")
    print("\nDone.")

if __name__ == "__main__":
    main()