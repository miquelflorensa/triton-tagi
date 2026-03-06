"""
Benchmark: PyTorch vs Triton TAGI Implementation
=================================================
Compares wall-clock time for training TAGI on regression data.
Tests multiple network widths and batch sizes.
"""

import torch
import numpy as np
import time
from tagi_triton import TritonTAGINet

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda")
WARMUP_ITERS = 5       # JIT compilation warm-up


# ====================================================================
# PyTorch reference (GPU, no nn.Module overhead — raw tensors)
# ====================================================================

class PyTorchTAGILayer:
    def __init__(self, in_f, out_f, device):
        # He initialization: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / in_f)
        self.mw = torch.randn(in_f, out_f, device=device) * scale
        self.Sw = torch.full((in_f, out_f), scale ** 2, device=device)
        self.mb = torch.zeros(1, out_f, device=device)
        self.Sb = torch.full((1, out_f), scale ** 2, device=device)

    def forward(self, ma, Sa):
        self.ma_in, self.Sa_in = ma, Sa
        self.mz = ma @ self.mw + self.mb
        self.Sz = (ma ** 2) @ self.Sw + Sa @ (self.mw ** 2) + Sa @ self.Sw + self.Sb
        return self.mz, self.Sz

    def backward(self, dmz, dSz):
        bs = dmz.shape[0]
        grad_mw = (self.ma_in.T @ dmz) / bs
        grad_mb = dmz.mean(0, keepdim=True)
        grad_Sw = ((self.ma_in ** 2).T @ dSz) / bs
        grad_Sb = dSz.mean(0, keepdim=True)

        self.mw += self.Sw * grad_mw
        self.mb += self.Sb * grad_mb
        self.Sw = torch.clamp(self.Sw + self.Sw ** 2 * grad_Sw, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb ** 2 * grad_Sb, min=1e-6)

        return dmz @ self.mw.T, dSz @ (self.mw ** 2).T


class PyTorchTAGINet:
    def __init__(self, struct, device):
        self.layers = [PyTorchTAGILayer(struct[i], struct[i + 1], device)
                       for i in range(len(struct) - 1)]

    def forward(self, x):
        ma, Sa = x, torch.zeros_like(x)
        self.masks = []
        for i, layer in enumerate(self.layers):
            mz, Sz = layer.forward(ma, Sa)
            if i < len(self.layers) - 1:
                mask = (mz > 0).float()
                self.masks.append(mask)
                ma, Sa = mz * mask, Sz * mask
            else:
                ma, Sa = mz, Sz
                self.masks.append(torch.ones_like(mz))
        return ma, Sa

    def step(self, xb, yb, sigma_v):
        ym, yS = self.forward(xb)
        Sy = yS + sigma_v ** 2
        dmz = (yb - ym) / Sy
        dSz = -1.0 / Sy
        for i in reversed(range(len(self.layers))):
            m = self.masks[i]
            dmz, dSz = self.layers[i].backward(dmz * m, dSz * m)


# ====================================================================
# Benchmark Harness
# ====================================================================

def make_data(n, in_dim, device):
    """Generate synthetic regression data."""
    x = torch.randn(n, in_dim, device=device)
    y = torch.randn(n, 1, device=device)
    return x, y


def time_training(net_class, struct, device, x, y, batch_size, n_epochs, sigma_v):
    """Time full training loop. Returns (net, elapsed_seconds)."""
    net = net_class(struct, device)

    # Warm-up (JIT compilation for Triton)
    for _ in range(WARMUP_ITERS):
        net.step(x[:batch_size], y[:batch_size], sigma_v)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for epoch in range(n_epochs):
        perm = torch.randperm(x.size(0), device=device)
        x_s, y_s = x[perm], y[perm]
        for i in range(0, len(x_s), batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            with torch.no_grad():
                net.step(xb, yb, sigma_v)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return net, elapsed


def run_benchmark():
    print("=" * 72)
    print("       TAGI Benchmark: PyTorch  vs  Triton (fused kernels)")
    print("=" * 72)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        # (struct,       batch_size, n_data, n_epochs, label)
        ([1,   64,  1],       32,    1000,  50, "Small  [1→64→1]"),
        ([1,  256,  1],       64,    2000,  50, "Medium [1→256→1]"),
        ([1,  256, 256, 1],  128,    4000,  30, "Deep   [1→256→256→1]"),
        ([1,  512, 512, 1],  256,    8000,  20, "Wide   [1→512→512→1]"),
        ([1, 1024,1024, 1],  512,   16000,  10, "Large  [1→1024→1024→1]"),
    ]

    print(f"{'Config':<30} {'Batch':>6} {'Data':>6} {'Epochs':>6} "
          f"{'PyTorch (s)':>12} {'Triton (s)':>12} {'Speedup':>8}")
    print("-" * 96)

    for struct, bs, n_data, n_epochs, label in configs:
        sigma_v = 0.5
        x, y = make_data(n_data, struct[0], DEVICE)

        # --- PyTorch ---
        torch.manual_seed(42)
        _, t_pt = time_training(PyTorchTAGINet, struct, DEVICE, x, y, bs, n_epochs, sigma_v)

        # --- Triton ---
        torch.manual_seed(42)
        _, t_tr = time_training(TritonTAGINet, struct, DEVICE, x, y, bs, n_epochs, sigma_v)

        speedup = t_pt / t_tr if t_tr > 0 else float('inf')
        print(f"  {label:<28} {bs:>6} {n_data:>6} {n_epochs:>6} "
              f"{t_pt:>11.4f}s {t_tr:>11.4f}s {speedup:>7.2f}x")

    print()
    print("=" * 72)
    print("  Notes:")
    print("  • Speedup > 1 means Triton is faster")
    print("  • Triton wins come from fusing 3 matmuls into 1 (var forward)")
    print("    and fusing 2 matmuls into 1 (backward delta propagation)")
    print("  • Small networks are dominated by kernel launch overhead")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
