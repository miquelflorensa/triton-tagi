"""
==========================================================================
  TAGI Batch Update — Mathematical Analysis & Strategy Comparison
==========================================================================

MATHEMATICAL BACKGROUND
========================

In TAGI, each parameter θ (weight or bias) has a Gaussian belief:
    p(θ) = N(m_θ, S_θ)     (mean m, variance S)

After observing a single data point x with observation model y=f(x,θ)+ε,
the TAGI update for parameter θ is (Kalman-style):

    m_θ ← m_θ + S_θ · δ_μ        ... (1)  mean update
    S_θ ← S_θ + S_θ² · δ_S       ... (2)  variance update (additive)

where:
    δ_μ = ∂z/∂θ · (y - ŷ) / S_y      (innovation signal for the mean)
    δ_S = -(∂z/∂θ)² / S_y              (always negative → variance shrinks)

THE BATCH PROBLEM
==================

Sequential vs. Parallel Processing of B observations:

SEQUENTIAL (exact Bayesian):
  For i = 1..B:
    τ_i = τ_{i-1} + |δ_S_i|          (precision grows)
    m_i = m_{i-1} + S_{i-1} · δ_μ_i  (mean shifts with current S)
  
  After all B:  τ_B ≈ τ_0 + Σ|δ_S_i|,  but each mean step used a
  different (shrinking) S, giving a "decaying step size" effect.

PARALLEL (batch approximation):
  We compute ALL δ_μ_i, δ_S_i with the SAME S_old (frozen prior).
  Then apply ONE combined update. But how to combine?

  ─── Mean: always average (1/B · Σ δ_μ_i) ───
  Averaging is correct: the direction is the mean gradient,
  and the step size comes from S_old (which is shared).
  Summing would give a B-times-too-large step.

  ─── Variance: the key question ───
  
  Option A: SUM δ_S_i → absorb ALL information from B samples
    τ_new = τ_old + Σ|δ_S_i|
    This is the "pure Bayesian" answer: each sample independently
    contributes information. But with large B, variance collapses
    too fast — after a few epochs, S → ε (the floor), and the 
    mean can no longer adapt. This contradicts the sequential case
    where S gradually shrinks between updates.
  
  Option B: AVERAGE δ_S_i → treat batch as "one effective observation"
    τ_new = τ_old + (1/B)·Σ|δ_S_i|
    Conservative: each batch step adds exactly as much precision
    as a single sample would. Very stable but very slow convergence.
  
  Option C: DAMPED SUM → γ · Σ δ_S_i  with 0 < γ < 1
    Interpolates between Options A and B.
    With γ = 1/√B, we get a "geometric mean" between full info and
    per-sample averaging. This is analogous to how SGD with momentum
    effectively damps the noise while retaining signal.

  ─── Variance update formula ───
  
  Additive:   S_new = S_old + S_old² · δ_S_combined
    Can go negative → needs clamping → loses information.
  
  Precision:  S_new = S_old / (1 - S_old · δ_S_combined)
    Mathematically guaranteed S_new > 0 when δ_S < 0.
    Equivalent to: τ_new = τ_old - δ_S_combined.
    
    WARNING: If δ_S_combined is POSITIVE (numerical error or bad gradients),
    the denominator can approach zero → S_new → ∞ → NaN explosion!
    This is what killed S2 (BASELINE_BATCH) at epoch 9.

This script tests 8 strategies head-to-head on MNIST.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import copy
from torchvision import datasets
from tagi_triton import (
    triton_fused_var_forward,
    triton_relu_moments,
    triton_fused_backward_delta,
    triton_output_innovation,
    param_update_kernel,
    BLOCK_EW,
    triton,
)

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")


# ====================================================================
# Strategy Implementations 
# ====================================================================

class TAGILayerBase:
    """Base layer with shared forward + init."""

    def __init__(self, in_features, out_features, device, gain_mean=2.0, gain_var=0.1):
        self.device = device
        # He initialization: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / in_features)
        self.mw = torch.randn(in_features, out_features, device=device) * scale
        self.Sw = torch.full((in_features, out_features), (gain_var * scale) ** 2, device=device)
        self.mb = torch.zeros(1, out_features, device=device)
        self.Sb = torch.full((1, out_features), (gain_var * scale) ** 2, device=device)

    def forward(self, ma, Sa):
        self.ma_in = ma
        self.mz = torch.matmul(ma, self.mw) + self.mb
        self.Sz = triton_fused_var_forward(ma, Sa, self.mw, self.Sw, self.Sb)
        return self.mz, self.Sz


# ─────────────────────────────────────────────────────────────────────
# Strategy 1: Original PyTorch Reference
#   Mean: /B  |  Var: sum  |  Update: additive S + S²·δ
# ─────────────────────────────────────────────────────────────────────
class Layer_S1_OrigPT(TAGILayerBase):
    name = "S1: PT-ref (avg+sum+add)"

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) / bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz)
        grad_Sb = delta_Sz.sum(0, keepdim=True)

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw + self.Sw ** 2 * grad_Sw, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb ** 2 * grad_Sb, min=1e-6)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 2: Current Triton (BASELINE_BATCH=32 + precision-space)
# ─────────────────────────────────────────────────────────────────────
class Layer_S2_CurrentTriton(TAGILayerBase):
    name = "S2: Triton-cur (base32+prec)"

    def backward(self, delta_mz, delta_Sz):
        BASELINE_BATCH = 32.0
        scale_mean = 1.0 / BASELINE_BATCH

        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * scale_mean
        grad_mb = delta_mz.sum(0, keepdim=True) * scale_mean
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz)
        grad_Sb = delta_Sz.sum(0, keepdim=True)

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw / (1.0 - self.Sw * grad_Sw), min=1e-6)
        self.Sb = torch.clamp(self.Sb / (1.0 - self.Sb * grad_Sb), min=1e-6)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 3: Full Average (both mean and var /B) + additive
# ─────────────────────────────────────────────────────────────────────
class Layer_S3_FullAvg(TAGILayerBase):
    name = "S3: Full-avg (both /B+add)"

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        inv_bs = 1.0 / bs
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * inv_bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz) * inv_bs
        grad_Sb = delta_Sz.sum(0, keepdim=True) * inv_bs

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw + self.Sw ** 2 * grad_Sw, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb ** 2 * grad_Sb, min=1e-6)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 4: Avg + precision-space
# ─────────────────────────────────────────────────────────────────────
class Layer_S4_AvgPrecision(TAGILayerBase):
    name = "S4: Avg + Precision"

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        inv_bs = 1.0 / bs
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * inv_bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz) * inv_bs
        grad_Sb = delta_Sz.sum(0, keepdim=True) * inv_bs

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw / (1.0 - self.Sw * grad_Sw), min=1e-6)
        self.Sb = torch.clamp(self.Sb / (1.0 - self.Sb * grad_Sb), min=1e-6)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 5: Damped precision (γ=0.1)
# ─────────────────────────────────────────────────────────────────────
class Layer_S5_DampedPrecision(TAGILayerBase):
    name = "S5: Damped prec (g=0.1)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = 0.1

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        inv_bs = 1.0 / bs
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * inv_bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz) * self.gamma
        grad_Sb = delta_Sz.sum(0, keepdim=True) * self.gamma

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw / (1.0 - self.Sw * grad_Sw), min=1e-6)
        self.Sb = torch.clamp(self.Sb / (1.0 - self.Sb * grad_Sb), min=1e-6)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 6: √B-scaled precision
# ─────────────────────────────────────────────────────────────────────
class Layer_S6_SqrtScale(TAGILayerBase):
    name = "S6: sqrtB precision"

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        inv_bs = 1.0 / bs
        inv_sqrt_bs = 1.0 / np.sqrt(bs)

        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * inv_bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz) * inv_sqrt_bs
        grad_Sb = delta_Sz.sum(0, keepdim=True) * inv_sqrt_bs

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw / (1.0 - self.Sw * grad_Sw), min=1e-6)
        self.Sb = torch.clamp(self.Sb / (1.0 - self.Sb * grad_Sb), min=1e-6)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 7: Safe precision with negative-only clamping
#   Key insight from S2 failure: the precision-space formula
#     S_new = S_old / (1 - S_old * δ_S)
#   explodes when δ_S > 0 (numerical error makes variance "increase").
#   Fix: clamp δ_S to be strictly ≤ 0 before applying precision update.
#   This makes the formula unconditionally stable.
#   Combined with sum (full Bayesian info absorption).
# ─────────────────────────────────────────────────────────────────────
class Layer_S7_SafePrecision(TAGILayerBase):
    name = "S7: Safe prec (sum+clamp)"

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        inv_bs = 1.0 / bs
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * inv_bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        # Sum the variance info (full Bayesian)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz)
        grad_Sb = delta_Sz.sum(0, keepdim=True)
        # SAFETY: clamp to be non-positive (precision can only increase)
        grad_Sw = torch.clamp(grad_Sw, max=0.0)
        grad_Sb = torch.clamp(grad_Sb, max=0.0)

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        # Now this is unconditionally stable: denom ≥ 1
        self.Sw = self.Sw / (1.0 - self.Sw * grad_Sw)
        self.Sb = self.Sb / (1.0 - self.Sb * grad_Sb)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ─────────────────────────────────────────────────────────────────────
# Strategy 8: Per-epoch annealing — sum B·(epoch_fraction) samples worth
#   Early epochs: absorb less info (explore), late: absorb more (refine).
#   α(t) = t/T where T = total epochs.
#   Effective info = α(t) * Σ δ_S_i
# ─────────────────────────────────────────────────────────────────────
class Layer_S8_Annealing(TAGILayerBase):
    name = "S8: Annealed precision"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.total_epochs = 20  # will be set externally

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        inv_bs = 1.0 / bs
        # Anneal from averaging (1/B) to full sum (1.0)
        alpha = max(0.01, (self.epoch + 1) / self.total_epochs)
        var_scale = alpha  # range [0.05, 1.0]

        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * inv_bs
        grad_mb = delta_mz.mean(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz) * var_scale
        grad_Sb = delta_Sz.sum(0, keepdim=True) * var_scale
        # Safety clamp
        grad_Sw = torch.clamp(grad_Sw, max=0.0)
        grad_Sb = torch.clamp(grad_Sb, max=0.0)

        self.mw = self.mw + self.Sw * grad_mw
        self.mb = self.mb + self.Sb * grad_mb
        self.Sw = self.Sw / (1.0 - self.Sw * grad_Sw)
        self.Sb = self.Sb / (1.0 - self.Sb * grad_Sb)

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ====================================================================
# Network Wrapper (parameterized by layer class)
# ====================================================================

class TAGINet:
    def __init__(self, layer_cls, layers_struct, device):
        self.layers = []
        self.device = device
        for i in range(len(layers_struct) - 1):
            self.layers.append(layer_cls(layers_struct[i], layers_struct[i + 1], device))

    def forward(self, x):
        ma, Sa = x, torch.zeros_like(x)
        self.jacobians = []
        for i, layer in enumerate(self.layers):
            mz, Sz = layer.forward(ma, Sa)
            if i < len(self.layers) - 1:
                ma, Sa, J = triton_relu_moments(mz, Sz)
                self.jacobians.append(J)
            else:
                ma, Sa = mz, Sz
                self.jacobians.append(torch.ones_like(mz))
        return ma, Sa

    def step(self, x_batch, y_batch, sigma_v):
        y_pred_m, y_pred_S = self.forward(x_batch)
        delta_mz, delta_Sz = triton_output_innovation(y_batch, y_pred_m, y_pred_S, sigma_v)

        for i in reversed(range(len(self.layers))):
            J = self.jacobians[i]
            dm = delta_mz * J
            ds = delta_Sz * J * J
            delta_mz, delta_Sz = self.layers[i].backward(dm, ds)

    def set_epoch(self, epoch, total_epochs):
        """For strategies that need epoch info (S8)."""
        for layer in self.layers:
            if hasattr(layer, 'epoch'):
                layer.epoch = epoch
                layer.total_epochs = total_epochs


# ====================================================================
# Evaluation & Training
# ====================================================================

def evaluate(net, x_test, y_labels, batch_size=1024):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v):
    perm = torch.randperm(x_train.size(0), device=x_train.device)
    x_s, y_s = x_train[perm], y_train_oh[perm]
    for i in range(0, len(x_s), batch_size):
        xb = x_s[i:i + batch_size]
        yb = y_s[i:i + batch_size]
        with torch.no_grad():
            net.step(xb, yb, sigma_v)


def measure_variance_stats(net):
    """Return median weight variance across all layers."""
    all_Sw = []
    for layer in net.layers:
        all_Sw.append(layer.Sw.cpu().numpy().ravel())
    all_Sw = np.concatenate(all_Sw)
    return float(np.nanmedian(all_Sw)), float(np.nanmean(all_Sw)), float(np.nanstd(all_Sw))


# ====================================================================
# Main Experiment
# ====================================================================

def main():
    print("=" * 70)
    print("  TAGI Batch Update Strategy Comparison")
    print("  MNIST: 784 → 512 → 512 → 10  (compact for speed)")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load MNIST ---
    print("Loading MNIST...", flush=True)
    train_ds = datasets.MNIST("data", train=True, download=True)
    test_ds = datasets.MNIST("data", train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test = test_ds.data.float().view(-1, 784) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test = ((x_test - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels = test_ds.targets.to(DEVICE)

    # One-hot ±3 encoding (matching current train_mnist.py)
    y_train_oh = torch.full((len(y_train_labels), 10), -3.0, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 3.0)

    # --- Architecture ---
    struct = [784, 512, 512, 10]
    batch_size = 128
    sigma_v = 0.01
    n_epochs = 20

    print(f"  Arch: {struct}  |  BS: {batch_size}  |  σ_v: {sigma_v}  |  Epochs: {n_epochs}\n")

    # --- Strategies to test ---
    strategies = [
        Layer_S1_OrigPT,
        Layer_S2_CurrentTriton,
        Layer_S3_FullAvg,
        Layer_S4_AvgPrecision,
        Layer_S5_DampedPrecision,
        Layer_S6_SqrtScale,
        Layer_S7_SafePrecision,
        Layer_S8_Annealing,
    ]

    results = {}

    for strat_cls in strategies:
        label = strat_cls.name
        print(f"\n{'─' * 70}")
        print(f"  Testing: {label}")
        print(f"{'─' * 70}")

        # Same seed for each strategy
        torch.manual_seed(42)
        np.random.seed(42)

        net = TAGINet(strat_cls, struct, DEVICE)

        # Warmup for JIT compilation
        for _ in range(2):
            net.step(x_train[:batch_size], y_train_oh[:batch_size], sigma_v)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        np.random.seed(42)
        net = TAGINet(strat_cls, struct, DEVICE)

        accs = []
        var_medians = []
        var_means = []
        t0 = time.perf_counter()

        for epoch in range(n_epochs):
            net.set_epoch(epoch, n_epochs)
            train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v)
            torch.cuda.synchronize()
            acc = evaluate(net, x_test, y_test_labels)
            med, mn, sd = measure_variance_stats(net)
            accs.append(acc)
            var_medians.append(med)
            var_means.append(mn)
            print(f"    Epoch {epoch+1:>2}/{n_epochs}  Acc: {acc*100:5.2f}%  "
                  f"Var(med/mean): {med:.2e}/{mn:.2e}")

        total_time = time.perf_counter() - t0
        results[label] = {
            "accs": accs,
            "var_medians": var_medians,
            "var_means": var_means,
            "time": total_time,
            "best_acc": max(accs),
        }
        print(f"  → Best: {max(accs)*100:.2f}%  Time: {total_time:.1f}s")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print(f"  {'Strategy':<40s} {'Best Acc':>10s}  {'@Ep20':>8s}  {'Time':>8s}")
    print("=" * 70)
    for label, r in results.items():
        print(f"  {label:<40s} {r['best_acc']*100:9.2f}%  "
              f"{r['accs'][-1]*100:7.2f}%  {r['time']:7.1f}s")
    print("=" * 70)

    # ====================================================================
    # Plot
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    epochs = np.arange(1, n_epochs + 1)

    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', 
              '#ec4899', '#14b8a6', '#f97316']

    # Accuracy
    ax = axes[0]
    for i, (label, r) in enumerate(results.items()):
        ax.plot(epochs, [a * 100 for a in r["accs"]], "o-", lw=2, ms=3,
                color=colors[i % len(colors)], label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Variance Median (log scale)
    ax = axes[1]
    for i, (label, r) in enumerate(results.items()):
        # Filter out NaN values for plotting
        valid = [v for v in r["var_medians"] if not np.isnan(v)]
        valid_epochs = [e for e, v in zip(epochs, r["var_medians"]) if not np.isnan(v)]
        if valid:
            ax.semilogy(valid_epochs, valid, "s-", lw=2, ms=3,
                         color=colors[i % len(colors)], label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Median Weight Variance", fontsize=12)
    ax.set_title("Variance Evolution (Median)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Variance Mean (log scale)
    ax = axes[2]
    for i, (label, r) in enumerate(results.items()):
        valid = [v for v in r["var_means"] if not np.isnan(v)]
        valid_epochs = [e for e, v in zip(epochs, r["var_means"]) if not np.isnan(v)]
        if valid:
            ax.semilogy(valid_epochs, valid, "^-", lw=2, ms=3,
                         color=colors[i % len(colors)], label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Weight Variance", fontsize=12)
    ax.set_title("Variance Evolution (Mean)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "batch_update_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")

    # ====================================================================
    # Batch Size Sensitivity for top strategies
    # ====================================================================
    best_strategies = [Layer_S5_DampedPrecision, Layer_S7_SafePrecision]
    
    for strat_cls in best_strategies:
        print(f"\n{'=' * 70}")
        print(f"  Batch Size Sensitivity — {strat_cls.name}")
        print("=" * 70)

        batch_sizes = [16, 32, 64, 128, 256]
        bs_results = {}

        for bs_val in batch_sizes:
            torch.manual_seed(42)
            np.random.seed(42)
            net = TAGINet(strat_cls, struct, DEVICE)

            accs = []
            for epoch in range(n_epochs):
                net.set_epoch(epoch, n_epochs)
                train_one_epoch(net, x_train, y_train_oh, bs_val, sigma_v)
                torch.cuda.synchronize()
                acc = evaluate(net, x_test, y_test_labels)
                accs.append(acc)
                if (epoch + 1) % 5 == 0:
                    print(f"    BS={bs_val:>3d}  Epoch {epoch+1:>2}/{n_epochs}  Acc: {acc*100:5.2f}%")

            bs_results[bs_val] = accs
            print(f"  → BS={bs_val}: Best {max(accs)*100:.2f}%\n")

        fig_bs, ax = plt.subplots(1, 1, figsize=(8, 5))
        for i, (bs_val, accs) in enumerate(bs_results.items()):
            ax.plot(epochs, [a * 100 for a in accs], "o-", lw=2, ms=4,
                    color=colors[i % len(colors)], label=f"BS={bs_val}")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(f"{strat_cls.name} — Batch Size Sensitivity", 
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_bs = f"bs_sensitivity_{strat_cls.name.split(':')[0].strip()}.png"
        plt.savefig(out_bs, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {out_bs}")


if __name__ == "__main__":
    main()
