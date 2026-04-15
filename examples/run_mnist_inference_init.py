"""
MNIST Experiment: He Init vs Inference-Based Init
==================================================

Compares initialization strategies on a 784→512→512→256→10 MLP:

  1. He init (baseline)
  2. Inference init (σ_M=1.0, σ_Z=1.0)  — balanced epistemic + aleatoric
  3. Inference init (σ_M=1.0, σ_Z=0.5)  — more epistemic, less aleatoric
  4. Inference init (σ_M=0.5, σ_Z=1.0)  — less epistemic, more aleatoric

Usage:
    conda run -n cuTAGI python run_mnist_inference_init.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from torchvision import datasets, transforms

from triton_tagi import Sequential, inference_init
from triton_tagi.layers import Linear, ReLU, Remax

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 128
SIGMA_V    = 0.1
N_EPOCHS   = 10
PROBE_SIZE = None  # use full training set


# ======================================================================
#  Data
# ======================================================================

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


# ======================================================================
#  Network builder
# ======================================================================

def build_net():
    return Sequential([
        Linear(784, 512, device=DEVICE),
        ReLU(),
        Linear(512, 512, device=DEVICE),
        ReLU(),
        Linear(512, 512, device=DEVICE),
        ReLU(),
        Linear(512, 512, device=DEVICE),
        ReLU(),
        Linear(512, 512, device=DEVICE),
        ReLU(),
        Linear(512, 512, device=DEVICE),
        ReLU(),
        Linear(512, 512, device=DEVICE),
        ReLU(),
        Linear(512, 256, device=DEVICE),
        ReLU(),
        Linear(256, 10,  device=DEVICE),
        Remax(),
    ], device=DEVICE)


# ======================================================================
#  Pre-training diagnostics
# ======================================================================

def pre_train_stats(net, x_probe):
    """Print E[Z] and E[Z²] at each Linear layer before training."""
    ma = x_probe
    Sa = torch.zeros_like(ma)
    print("  Pre-training hidden unit statistics:")
    with torch.no_grad():
        for i, layer in enumerate(net.layers):
            if isinstance(layer, Linear):
                from triton_tagi.kernels.common import triton_fused_var_forward
                mz = torch.matmul(ma, layer.mw) + layer.mb
                Sz = triton_fused_var_forward(ma, Sa, layer.mw, layer.Sw, layer.Sb)
                Ez  = mz.mean().item()
                Ez2 = (mz**2 + Sz).mean().item()
                print(f"    Layer {i}: E[Z]={Ez:+.4f}  E[Z²]={Ez2:.4f}")
                ma, Sa = mz, Sz
            else:
                ma, Sa = layer.forward(ma, Sa)


# ======================================================================
#  Training
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=1024):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        mu, _ = net.forward(x_test[i:i+batch_size])
        correct += (mu.argmax(1) == y_labels[i:i+batch_size]).sum().item()
    return correct / len(x_test)


def train(net, x_train, y_train_oh, x_test, y_test_labels, label):
    print(f"\n  {'Ep':>3s}  {'Acc':>7s}  {'Time':>7s}")
    print("  " + "─" * 23)

    accs = []
    best = 0.0
    t_total = time.perf_counter()

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s, y_s = x_train[perm], y_train_oh[perm]

        for i in range(0, len(x_s), BATCH_SIZE):
            net.step(x_s[i:i+BATCH_SIZE], y_s[i:i+BATCH_SIZE], SIGMA_V)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)
        best = max(best, acc)
        print(f"  {epoch:3d}  {acc*100:6.2f}%  {dt:6.2f}s")

    total = time.perf_counter() - t_total
    print("  " + "─" * 23)
    print(f"  Best: {best*100:.2f}%  |  Total: {total:.1f}s")
    return best, accs


# ======================================================================
#  Main
# ======================================================================

def main():
    print("=" * 64)
    print("  MNIST: He Init vs Inference-Based Init")
    print(f"  MLP: 784→512→512→256→10,  epochs={N_EPOCHS}")
    print("=" * 64)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading MNIST...")
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    x_probe = x_train  # full training set

    experiments = [
        ("He init",             None,  None, None),
        ("InfInit σ_M=1.0 σ_Z=1.0", 1.0, 1.0, None),
        ("InfInit σ_M=1.0 σ_Z=0.5", 1.0, 0.5, None),
        ("InfInit σ_M=0.5 σ_Z=1.0", 0.5, 1.0, None),
    ]

    results = {}
    all_accs = {}

    for i, (name, sigma_M, sigma_Z, _) in enumerate(experiments):
        torch.manual_seed(42)
        np.random.seed(42)

        print(f"\n{'=' * 64}")
        print(f"  [{i+1}/{len(experiments)}] {name}")
        print("=" * 64)

        net = build_net()

        if sigma_M is not None:
            inference_init(net, x_probe, sigma_M=sigma_M, sigma_Z=sigma_Z,
                           verbose=True)

        pre_train_stats(net, x_probe)
        best, accs = train(net, x_train, y_train_oh, x_test, y_test_labels,
                           label=name)
        results[name] = best
        all_accs[name] = accs

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  Summary — Best Test Accuracy")
    print("=" * 64)
    for name, best in results.items():
        print(f"  {name:<30s}: {best*100:.2f}%")

    print()
    print("  Accuracy @ epoch 5:")
    for name, accs in all_accs.items():
        ep5 = accs[4] if len(accs) >= 5 else accs[-1]
        print(f"  {name:<30s}: {ep5*100:.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
