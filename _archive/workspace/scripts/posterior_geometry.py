"""
posterior_geometry.py — Visualize TAGI's Posterior Weight-Space Geometry
=========================================================================

After training a model on CIFAR-10, this script draws a 2D cross-section
through the posterior weight distribution using the Garipov et al. (2018)
subspace projection method.

Supports two architectures:
  --arch 3cnn     (default) trains a 3-block CNN from scratch
  --arch resnet18           loads pre-trained ResNet-18 checkpoints

This implementation interpolates BOTH the weight means (mw) and weight 
variances (Sw) using Barycentric coordinates mapped from the 3 checkpoints.

Usage
-----
    conda run -n cuTAGI python posterior_geometry.py --epochs 100 --interval 10
    conda run -n cuTAGI python posterior_geometry.py --arch resnet18 --ckpt-dir run_logs/checkpoints --epochs 139 --interval 10
    conda run -n cuTAGI python posterior_geometry.py --arch resnet18_adam --epochs 100 --interval 10
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
    """Return all layers that store posterior weight means (mw)."""
    return [l for l in net.layers if hasattr(l, 'mw')]

def _has_bias(layer):
    """True if this layer has a learnable bias (mb is a live tensor)."""
    if hasattr(layer, 'has_bias'):
        return layer.has_bias
    return getattr(layer, 'mb', None) is not None

def get_flat_means(net):
    """Concatenate all posterior weight means into a single 1-D vector."""
    parts = []
    for layer in _learnable_layers(net):
        parts.append(layer.mw.detach().flatten())
        if _has_bias(layer):
            parts.append(layer.mb.detach().flatten())
    return torch.cat(parts)

def get_flat_vars(net):
    """Concatenate all posterior weight variances into a single 1-D vector."""
    parts = []
    for layer in _learnable_layers(net):
        parts.append(layer.Sw.detach().flatten())
        if _has_bias(layer):
            parts.append(layer.Sb.detach().flatten())
    return torch.cat(parts)

def set_flat_means(net, w_flat):
    """Write a flat mean vector back into the network's mw / mb tensors."""
    idx = 0
    for layer in _learnable_layers(net):
        n = layer.mw.numel()
        layer.mw.data.copy_(w_flat[idx:idx + n].view_as(layer.mw))
        idx += n
        if _has_bias(layer):
            n = layer.mb.numel()
            layer.mb.data.copy_(w_flat[idx:idx + n].view_as(layer.mb))
            idx += n

def set_flat_vars(net, v_flat):
    """Write a flat variance vector back into the network's Sw / Sb tensors."""
    idx = 0
    for layer in _learnable_layers(net):
        n = layer.Sw.numel()
        layer.Sw.data.copy_(v_flat[idx:idx + n].view_as(layer.Sw))
        idx += n
        if _has_bias(layer):
            n = layer.Sb.numel()
            layer.Sb.data.copy_(v_flat[idx:idx + n].view_as(layer.Sb))
            idx += n


# ══════════════════════════════════════════════════════════════════════════
#  Landscape evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_point(net, x, y_oh, y_lbl, batch_size=64):
    """
    Evaluate the *current* weight state (means and variances) on the given data.
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
        
        # In TAGI, the forward pass inherently uses the injected Sw/Sb to generate mu
        mu, _ = net.forward(xb)
        total_nll += -(yb * torch.log(mu.clamp(min=eps))).sum(dim=1).sum().item()
        correct   += (mu.argmax(1) == lb).sum().item()
        del mu
    torch.cuda.empty_cache()

    return total_nll / n, correct / n


# ══════════════════════════════════════════════════════════════════════════
#  Core geometry routine (Barycentric Interpolation)
# ══════════════════════════════════════════════════════════════════════════

def compute_landscape(net, x_eval, y_oh, y_lbl, checkpoints_mean, checkpoints_var, grid_size=21):
    """
    Build Log-Likelihood and accuracy landscapes on a 2D grid, interpolating 
    BOTH weight means and variances.
    """
    epochs_saved = sorted(list(checkpoints_mean.keys()))
    if len(epochs_saved) < 3:
        raise ValueError("Need exactly 3 checkpoints to define the 2D plane!")

    print(f"  [geometry] Using checkpoints from epochs: {epochs_saved}")
    
    # Extract Means
    m3 = checkpoints_mean[epochs_saved[0]]  # Oldest checkpoint
    m2 = checkpoints_mean[epochs_saved[1]]  # Middle checkpoint
    m1 = checkpoints_mean[epochs_saved[2]]  # Final checkpoint (Center)
    
    # Extract Variances
    S3 = checkpoints_var[epochs_saved[0]]
    S2 = checkpoints_var[epochs_saved[1]]
    S1 = checkpoints_var[epochs_saved[2]]

    # ── Gram-Schmidt orthonormalization on the Means ─────────────────────
    u_orig = m2 - m1
    u_norm = u_orig.norm()
    u_hat  = u_orig / u_norm

    v_orig = m3 - m1
    v_proj = (u_hat @ v_orig)
    v_perp = v_orig - v_proj * u_hat
    v_perp_norm = v_perp.norm()
    v_hat  = v_perp / v_perp_norm

    # Adaptive grid range
    alpha_w2 = u_norm.item()                     
    beta_w3  = v_perp_norm.item()
    grid_range = max(alpha_w2, beta_w3) * 1.3
    if grid_range == 0: grid_range = 1.0

    coords = torch.linspace(-grid_range, grid_range, grid_size, device=m1.device)

    print(f"  [geometry] ||m2 − m1|| = {alpha_w2:.4f}")
    print(f"  [geometry] ||m3⊥||     = {beta_w3:.4f}")
    print(f"  [geometry] Grid: {grid_size}×{grid_size}, range ±{grid_range:.3f}")
    print(f"  [geometry] Evaluating {grid_size**2} grid points on {len(x_eval):,} samples …")

    nll_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    acc_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Pre-compute logs of variances to safely interpolate in log-space
    log_S1 = torch.log(S1.clamp(min=1e-10))
    log_S2 = torch.log(S2.clamp(min=1e-10))
    log_S3 = torch.log(S3.clamp(min=1e-10))

    for i, alpha in enumerate(coords):
        for j, beta in enumerate(coords):
            # 1. Map Coordinates & Set Means
            m_pt = m1 + alpha * u_hat + beta * v_hat
            set_flat_means(net, m_pt)
            
            # 2. Compute Barycentric coords (c1, c2, c3) to interpolate variances
            c3 = (beta / v_perp_norm).item()
            c2 = ((alpha - c3 * v_proj) / u_norm).item()
            c1 = 1.0 - c2 - c3
            
            # 3. Interpolate Variances (in log space) & Set
            log_S_pt = c1 * log_S1 + c2 * log_S2 + c3 * log_S3
            S_pt = torch.exp(log_S_pt)
            set_flat_vars(net, S_pt)

            # 4. Evaluate network (now carrying blended means AND variances)
            nll, acc = eval_point(net, x_eval, y_oh, y_lbl)
            nll_grid[i, j] = nll
            acc_grid[i, j] = acc
            
            del m_pt, S_pt

        torch.cuda.empty_cache()
        if (i % 5 == 0) or (i == grid_size - 1):
            pct = 100 * (i + 1) / grid_size
            print(f"    row {i+1:3d}/{grid_size}  ({pct:.0f}%)", flush=True)

    # Restore the original posterior means and variances
    set_flat_means(net, m1)
    set_flat_vars(net, S1)
    print("  [geometry] Original weights restored.")

    return nll_grid, acc_grid, coords.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
#  Plotting (1x2 Clean Layout)
# ══════════════════════════════════════════════════════════════════════════

def plot_landscape(nll_grid, acc_grid, coords, n_eval,
                   title_suffix="", save_path="run_logs/posterior_geometry.png",
                   arch_label="3-CNN"):
    """
    Produce a clean 1x2 contour plot showing Log-Likelihood and Accuracy.
    """
    # log-likelihood = -NLL * N
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
        ax.scatter([0], [0], c='royalblue', s=180, zorder=6, marker='*', label='w₁ (Final Epoch)')

        ax.set_xlabel('α  (û direction)', fontsize=9)
        ax.set_ylabel('β  (v̂ direction)', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='lower right')

    plt.suptitle(
        f'TAGI-Triton · Posterior Geometry ({arch_label})\n'
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
    ap = argparse.ArgumentParser(description="TAGI posterior geometry visualizer")
    ap.add_argument("--arch",       type=str,   default="3cnn",
                    choices=["3cnn", "resnet18", "resnet18_adam", "resnet18_cifar100", "resnet18_cifar100_adam"],
                    help="Architecture: '3cnn', 'resnet18' (CIFAR-10), 'resnet18_adam', 'resnet18_cifar100', or 'resnet18_cifar100_adam'")
    ap.add_argument("--epochs",     type=int,   default=100,
                    help="Total training epochs / final checkpoint epoch (default 100)")
    ap.add_argument("--interval",   type=int,   default=10,
                    help="Epoch interval between the 3 checkpoints (default 10)")
    ap.add_argument("--grid",       type=int,   default=31,
                    help="Grid resolution N×N (default 31 → 961 evals)")
    ap.add_argument("--eval-n",     type=int,   default=2000,
                    help="# test samples used for landscape evaluation (default 2000)")
    ap.add_argument("--sigma-v",    type=float, default=0.05,
                    help="Observation noise σ_v for 3cnn training (default 0.05)")
    ap.add_argument("--batch-size", type=int,   default=128,
                    help="Mini-batch size for training (default 128)")
    ap.add_argument("--ckpt-dir",   type=str,   default="run_logs/checkpoints",
                    help="Checkpoint directory for resnet18 (default run_logs/checkpoints)")
    ap.add_argument("--save",       type=str,   default=None,
                    help="Output path for the figure (auto-generated if not provided)")
    args = ap.parse_args()

    if args.save is None:
        args.save = f"run_logs/posterior_geometry_{args.arch}.png"
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
    print(f"  TAGI Posterior Geometry — {arch_label} on {dataset_label}")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  Arch        : {args.arch}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Grid        : {args.grid}×{args.grid} = {args.grid**2} points")
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
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    # ── Checkpoint epochs to use ──
    target_epochs = [args.epochs - 2 * args.interval,
                     args.epochs - args.interval,
                     args.epochs]
    target_epochs = [max(1, e) for e in target_epochs]

    trajectory_checkpoints_mean = {}
    trajectory_checkpoints_var  = {}

    if args.arch in ("resnet18_cifar100", "resnet18_cifar100_adam", "resnet18", "resnet18_adam"):
        # ── ResNet-18 variations ──
        adam_tag = " (Adam)" if "adam" in args.arch else ""
        print(f"\n  Building {arch_label} …")
        
        if "cifar100" in args.arch:
            net = build_resnet18_c100(num_classes=num_classes, head="remax", device=DEVICE, g_min=0.1, g_max=0.1)
            init_residual_aware(net, eta=0.125, verbose=False)
            eval_fn = c100_evaluate
        else:
            net = build_resnet18(num_classes=num_classes, head="remax", device=DEVICE, g_min=0.1, g_max=0.1)
            init_residual_aware(net, eta=0.5, verbose=False)
            eval_fn = resnet_evaluate
            
        # Warm-up BN
        net.train()
        net.forward(x_train[:32])
        net.eval()
        torch.cuda.empty_cache()

        print(f"  Parameters: {net.num_parameters():,}")
        
        for ep in target_epochs:
            ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{ep:04d}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nAdjust --epochs and --interval.")
            resnet_load_checkpoint(net, ckpt_path)
            
            trajectory_checkpoints_mean[ep] = get_flat_means(net).clone()
            trajectory_checkpoints_var[ep]  = get_flat_vars(net).clone()
            print(f"  [Checkpoint] Loaded epoch {ep} from {ckpt_path}")

        # Leave final checkpoint loaded
        resnet_load_checkpoint(net, os.path.join(args.ckpt_dir, f"checkpoint_epoch_{target_epochs[-1]:04d}.pt"))

    else:
        # ── 3-CNN: train from scratch ──
        print(f"\n  Building 3-CNN …")
        net = build_simple_3cnn(num_classes=10, head="remax", device=DEVICE, gain_w=0.1, gain_b=0.1)
        print(f"  Parameters: {net.num_parameters():,}")

        print(f"\n  Training for {args.epochs} epoch(s) and tracking trajectory …")
        monitor = TAGIMonitor(net, log_dir="run_logs", probe_size=256)
        monitor.record(epoch=0, x_probe=x_train[:256], tag="init")

        for epoch in range(1, args.epochs + 1):
            train(net, x_train, y_train_oh, y_train_lbl, x_test, y_test_lbl,
                  batch_size=args.batch_size, initial_sigma_v=args.sigma_v,
                  n_epochs=1, monitor=monitor, monitor_every=1)
                  
            if epoch in target_epochs:
                print(f"  [Checkpoint] Saving weights & variances at epoch {epoch}")
                trajectory_checkpoints_mean[epoch] = get_flat_means(net).clone()
                trajectory_checkpoints_var[epoch]  = get_flat_vars(net).clone()

        eval_fn = evaluate

    # ── Geometry ──
    print("\n  Computing posterior geometry …")
    n_eval     = min(args.eval_n, len(x_test))
    x_eval     = x_test[:n_eval]
    y_eval_lbl = y_test_lbl[:n_eval]
    y_eval_oh  = torch.zeros(n_eval, num_classes, device=DEVICE)
    y_eval_oh.scatter_(1, y_eval_lbl.unsqueeze(1), 1.0)

    test_acc = eval_fn(net, x_test, y_test_lbl)
    print(f"  Test accuracy at centre (w1): {test_acc * 100:.2f}%")

    nll_grid, acc_grid, coords = compute_landscape(
        net, x_eval, y_eval_oh, y_eval_lbl,
        checkpoints_mean=trajectory_checkpoints_mean,
        checkpoints_var=trajectory_checkpoints_var,
        grid_size=args.grid
    )

    # ── Plot ──
    print("\n  Plotting …")
    title = f"Tracking epochs {target_epochs}  ·  Final Test Acc: {test_acc*100:.1f}%"
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