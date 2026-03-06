"""
MNIST Classification with TAGI: PyTorch vs Triton
===================================================
Architecture: 784 → 4096 → 4096 → 10 (FNN with ReLU)
Plots test accuracy per epoch for both implementations.
Now uses exact batch processing in the Triton version.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from tagi_triton import TritonTAGINet

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda")


# ====================================================================
# PyTorch TAGI  (pure-tensor, no autograd — identical logic to Triton)
# ====================================================================

class PTLayer:
    def __init__(self, in_f, out_f, dev):
        # He initialization: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / in_f)
        self.mw = torch.randn(in_f, out_f, device=dev) * scale
        self.Sw = torch.full((in_f, out_f), scale ** 2, device=dev)
        self.mb = torch.zeros(1, out_f, device=dev)
        self.Sb = torch.full((1, out_f), scale ** 2, device=dev)

    def forward(self, ma, Sa):
        self.ma_in = ma
        self.mz = ma @ self.mw + self.mb
        self.Sz = (ma ** 2) @ self.Sw + Sa @ (self.mw ** 2) + Sa @ self.Sw + self.Sb
        return self.mz, self.Sz

    def backward(self, dmz, dSz):
        bs = dmz.shape[0]
        # Mean: averaged
        gm = (self.ma_in.T @ dmz) / bs
        gbm = dmz.mean(0, keepdim=True)
        # Variance: summed (each obs reduces uncertainty)
        gS = (self.ma_in ** 2).T @ dSz
        gbS = dSz.sum(0, keepdim=True)
        self.mw += self.Sw * gm
        self.mb += self.Sb * gbm
        self.Sw = torch.clamp(self.Sw + self.Sw ** 2 * gS, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb ** 2 * gbS, min=1e-6)
        return dmz @ self.mw.T, dSz @ (self.mw ** 2).T


class PTNet:
    def __init__(self, struct, dev):
        self.layers = [PTLayer(struct[i], struct[i + 1], dev)
                       for i in range(len(struct) - 1)]

    @staticmethod
    def _bayesian_relu(mz, Sz):
        Sz_safe = torch.clamp(Sz, min=1e-12)
        sigma_z = torch.sqrt(Sz_safe)
        alpha = mz / sigma_z
        pdf = torch.exp(-0.5 * alpha ** 2) * 0.3989422804014327
        cdf = 0.5 * (1.0 + torch.erf(alpha * 0.7071067811865476))
        mu_m = sigma_z * pdf + mz * cdf
        var_m = torch.clamp(
            -mu_m ** 2 + 2 * mu_m * mz - mz * sigma_z * pdf
            + (Sz_safe - mz ** 2) * cdf, min=1e-12)
        return mu_m, var_m, cdf

    def forward(self, x):
        ma, Sa = x, torch.zeros_like(x)
        self.jacobians = []
        for i, L in enumerate(self.layers):
            mz, Sz = L.forward(ma, Sa)
            if i < len(self.layers) - 1:
                ma, Sa, J = self._bayesian_relu(mz, Sz)
                self.jacobians.append(J)
            else:
                ma, Sa = mz, Sz
                self.jacobians.append(torch.ones_like(mz))
        return ma, Sa

    def step(self, xb, yb, sv):
        ym, yS = self.forward(xb)
        Sy = yS + sv ** 2
        dm, ds = (yb - ym) / Sy, -1.0 / Sy
        for i in reversed(range(len(self.layers))):
            J = self.jacobians[i]
            dm, ds = self.layers[i].backward(dm * J, ds * J * J)


# ====================================================================
# Shared helpers
# ====================================================================

def evaluate(net, x_test, y_labels, batch_size=1024):
    """Compute classification accuracy."""
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v):
    """Shuffle + mini-batch training for one epoch."""
    perm = torch.randperm(x_train.size(0), device=x_train.device)
    x_s, y_s = x_train[perm], y_train_oh[perm]
    for i in range(0, len(x_s), batch_size):
        xb = x_s[i:i + batch_size]
        yb = y_s[i:i + batch_size]
        with torch.no_grad():
            # ------------------------------------------------------------
            # Use exact batch update for Triton, standard step for PyTorch
            # ------------------------------------------------------------
            if isinstance(net, TritonTAGINet):
                net.exact_step(xb, yb, sigma_v, cg_tol=1e-6, cg_max_iter=50, m_rand=20)
            else:
                net.step(xb, yb, sigma_v)


def run_training(label, NetClass, struct, device,
                 x_train, y_train_oh, x_test, y_test_labels,
                 batch_size, sigma_v, n_epochs, warmup=False):
    """Full training loop, returns (accuracies, wall-times, total_time)."""
    # Deterministic init
    torch.manual_seed(42)
    net = NetClass(struct, device)

    # Triton JIT warm-up (compile kernels, excluded from timing)
    if warmup and NetClass == TritonTAGINet:
        tmp_net = NetClass(struct, device)
        for _ in range(3):
            tmp_net.exact_step(x_train[:batch_size], y_train_oh[:batch_size], sigma_v,
                               cg_max_iter=10, m_rand=5)  # faster warmup
        torch.cuda.synchronize()
        del tmp_net
        # Re-init with same seed
        torch.manual_seed(42)
        net = NetClass(struct, device)

    accs, wall = [], []
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        ep_start = time.perf_counter()
        train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v)
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_start
        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)
        wall.append(time.perf_counter() - t0)
        print(f"  [{label}] Epoch {epoch + 1:>2}/{n_epochs}  "
              f"Acc: {acc * 100:5.2f}%  ({ep_time:.2f}s)")

    total = time.perf_counter() - t0
    return accs, wall, total


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 66)
    print("  MNIST Classification with TAGI")
    print("  FNN: 784 → 4096 → 4096 → 10   (ReLU activation)")
    print("=" * 66)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load MNIST ---
    print("Loading MNIST...", flush=True)
    train_ds = datasets.MNIST("data", train=True,  download=True)
    test_ds  = datasets.MNIST("data", train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test  = test_ds.data.float().view(-1, 784)  / 255.0

    # Standardize using training statistics
    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test  = ((x_test  - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels  = test_ds.targets.to(DEVICE)

    # One-hot encode (use -1/1 encoding as in original script)
    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE) - 3.0
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 3.0)

    # --- Hyperparameters ---
    struct     = [784, 4096, 4096, 10]
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 100

    print(f"  Batch: {batch_size}  |  σ_v: {sigma_v}  |  Epochs: {n_epochs}\n")

    # --- PyTorch (optional – uncomment if you want to compare) ---
    # accs_pt, wall_pt, t_pt = run_training(
    #     "PyTorch", PTNet, struct, DEVICE,
    #     x_train, y_train_oh, x_test, y_test_labels,
    #     batch_size, sigma_v, n_epochs,
    # )
    # print()

    # --- Triton with exact step ---
    accs_tr, wall_tr, t_tr = run_training(
        "Triton ", TritonTAGINet, struct, DEVICE,
        x_train, y_train_oh, x_test, y_test_labels,
        batch_size, sigma_v, n_epochs, warmup=True,
    )

    # --- Summary ---
    print()
    print("=" * 66)
    print(f"  {'':30s} {'PyTorch':>12s} {'Triton':>12s}")
    # Use dummy values if PyTorch is commented out
    final_pt = accs_pt[-1] if 'accs_pt' in locals() else 0.0
    best_pt  = max(accs_pt) if 'accs_pt' in locals() else 0.0
    t_pt     = t_pt if 't_pt' in locals() else 0.0
    print(f"  {'Final Accuracy':30s} {final_pt*100:11.2f}% {accs_tr[-1]*100:11.2f}%")
    print(f"  {'Best Accuracy':30s} {best_pt*100:11.2f}% {max(accs_tr)*100:11.2f}%")
    print(f"  {'Total Time':30s} {t_pt:11.2f}s {t_tr:11.2f}s")
    if t_pt > 0:
        print(f"  {'Speedup':30s} {'':>12s} {t_pt/t_tr:11.2f}x")
    print("=" * 66)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = np.arange(1, n_epochs + 1)

    # Left: Accuracy vs Epoch
    if 'accs_pt' in locals():
        ax1.plot(epochs, [a * 100 for a in accs_pt], "o-",
                 color="#3b82f6", lw=2, ms=5, label=f"PyTorch  ({t_pt:.1f}s)")
    ax1.plot(epochs, [a * 100 for a in accs_tr], "s--",
             color="#ef4444", lw=2, ms=5, label=f"Triton   ({t_tr:.1f}s)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("MNIST — TAGI  [784 → 4096 → 4096 → 10]", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, n_epochs)

    # Right: Accuracy vs Wall-Clock Time
    if 'wall_pt' in locals():
        ax2.plot(wall_pt, [a * 100 for a in accs_pt], "o-",
                 color="#3b82f6", lw=2, ms=5, label="PyTorch")
    ax2.plot(wall_tr, [a * 100 for a in accs_tr], "s--",
             color="#ef4444", lw=2, ms=5, label="Triton")
    ax2.set_xlabel("Wall-Clock Time (s)", fontsize=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Training Time", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "mnist_tagi_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


if __name__ == "__main__":
    main()