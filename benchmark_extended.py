"""
Extended Benchmark: Multi-dimensional TAGI
==========================================
This tests with multi-dimensional inputs/outputs where
the matmul tile utilization is much better, giving Triton
fusion a real chance to outperform PyTorch.
"""

import torch
import numpy as np
import time
from tagi_triton import TritonTAGINet

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")
WARMUP = 5


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
        self.Sz = (ma**2) @ self.Sw + Sa @ (self.mw**2) + Sa @ self.Sw + self.Sb
        return self.mz, self.Sz

    def backward(self, dmz, dSz):
        bs = dmz.shape[0]
        gm = (self.ma_in.T @ dmz) / bs
        gS = ((self.ma_in**2).T @ dSz) / bs
        gb_m = dmz.mean(0, keepdim=True)
        gb_S = dSz.mean(0, keepdim=True)
        self.mw += self.Sw * gm
        self.mb += self.Sb * gb_m
        self.Sw = torch.clamp(self.Sw + self.Sw**2 * gS, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb**2 * gb_S, min=1e-6)
        return dmz @ self.mw.T, dSz @ (self.mw**2).T


class PyTorchTAGINet:
    def __init__(self, struct, device):
        self.layers = [PyTorchTAGILayer(struct[i], struct[i+1], device)
                       for i in range(len(struct)-1)]

    def forward(self, x):
        ma, Sa = x, torch.zeros_like(x)
        self.masks = []
        for i, L in enumerate(self.layers):
            mz, Sz = L.forward(ma, Sa)
            if i < len(self.layers)-1:
                m = (mz > 0).float(); self.masks.append(m)
                ma, Sa = mz*m, Sz*m
            else:
                ma, Sa = mz, Sz; self.masks.append(torch.ones_like(mz))
        return ma, Sa

    def step(self, xb, yb, sv):
        ym, yS = self.forward(xb)
        Sy = yS + sv**2
        dm, ds = (yb - ym)/Sy, -1.0/Sy
        for i in reversed(range(len(self.layers))):
            m = self.masks[i]
            dm, ds = self.layers[i].backward(dm*m, ds*m)


def bench(NetClass, struct, device, x, y, bs, n_ep, sv):
    net = NetClass(struct, device)
    for _ in range(WARMUP):
        net.step(x[:bs], y[:bs], sv)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_ep):
        perm = torch.randperm(x.size(0), device=device)
        xs, ys = x[perm], y[perm]
        for i in range(0, len(xs), bs):
            with torch.no_grad():
                net.step(xs[i:i+bs], ys[i:i+bs], sv)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("=" * 80)
    print("   Multi-Dimensional TAGI Benchmark:  PyTorch vs Triton")
    print("=" * 80)
    print(f"   GPU: {torch.cuda.get_device_name(0)}\n")

    # 1D regression (original use case)
    print("── 1D Regression ──")
    configs_1d = [
        ([1,  64,   1],     32,  1000, 50, "Tiny"),
        ([1, 256,   1],     64,  2000, 50, "Small"),
        ([1, 512, 512, 1], 256,  8000, 20, "Deep"),
    ]

    print(f"  {'Config':<35} {'BS':>4} {'N':>6} {'Ep':>4} "
          f"{'PyTorch':>10} {'Triton':>10} {'Ratio':>7}")
    print("  " + "-" * 78)

    for struct, bs, nd, ne, lbl in configs_1d:
        x = torch.randn(nd, struct[0], device=DEVICE)
        y = torch.randn(nd, struct[-1], device=DEVICE)
        t_pt = bench(PyTorchTAGINet,  struct, DEVICE, x, y, bs, ne, 0.5)
        t_tr = bench(TritonTAGINet,   struct, DEVICE, x, y, bs, ne, 0.5)
        r = t_pt / t_tr
        print(f"  {lbl + ' ' + str(struct):<35} {bs:>4} {nd:>6} {ne:>4} "
              f"{t_pt:>9.3f}s {t_tr:>9.3f}s {r:>6.2f}x")

    # Multi-dimensional (better GPU utilization)
    print("\n── Multi-Dimensional (classif-like) ──")
    configs_md = [
        ([32,  128,  10],     64,  4000, 30, "32→128→10"),
        ([64,  256,  10],    128,  8000, 20, "64→256→10"),
        ([128, 256, 256, 10], 256,  8000, 15, "128→256²→10"),
        ([256, 512, 512, 10], 256, 16000, 10, "256→512²→10"),
        ([256,1024,1024, 10], 512, 16000,  5, "256→1024²→10"),
    ]

    print(f"  {'Config':<35} {'BS':>4} {'N':>6} {'Ep':>4} "
          f"{'PyTorch':>10} {'Triton':>10} {'Ratio':>7}")
    print("  " + "-" * 78)

    for struct, bs, nd, ne, lbl in configs_md:
        x = torch.randn(nd, struct[0], device=DEVICE)
        y = torch.randn(nd, struct[-1], device=DEVICE)
        t_pt = bench(PyTorchTAGINet,  struct, DEVICE, x, y, bs, ne, 0.5)
        t_tr = bench(TritonTAGINet,   struct, DEVICE, x, y, bs, ne, 0.5)
        r = t_pt / t_tr
        print(f"  {lbl:<35} {bs:>4} {nd:>6} {ne:>4} "
              f"{t_pt:>9.3f}s {t_tr:>9.3f}s {r:>6.2f}x")

    print("\n" + "=" * 80)
    print("  Ratio > 1 → Triton faster  |  Ratio < 1 → PyTorch faster")
    print("=" * 80)


if __name__ == "__main__":
    main()
