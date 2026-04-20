"""Does pytagi's CUDA Conv forward differ from its CPU Conv forward?

Loads the SAME weights into pytagi; runs one forward on CPU, then (new net)
one forward on CUDA, compares outputs.  Also compares both to triton-GPU.
"""
from __future__ import annotations

import math
import numpy as np
import torch

import pytagi
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda"
C_IN, C_OUT = 3, 32
K = 5
H = W = 32
BATCH = 4


def _cmp(label, a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel = diff / denom
    print(
        f"  {label:<24s}  n={a.size:>6d}  "
        f"|Δ|max={diff.max():.3e}  rel_max={rel.max():.3e}  "
        f"|a|mean={np.abs(a).mean():.3e}"
    )


def main():
    torch.manual_seed(0); pytagi.manual_seed(0)
    fan_in = C_IN * K * K
    scale = math.sqrt(1.0 / fan_in)
    Kdim = fan_in
    mw = torch.randn(Kdim, C_OUT) * scale
    Sw = torch.full((Kdim, C_OUT), scale ** 2)
    mb = torch.zeros(1, C_OUT)
    Sb = torch.full((1, C_OUT), scale ** 2)

    # Build triton (GPU)
    layer_tri = TConv2D(C_IN, C_OUT, K, padding=2, device=DEVICE)
    layer_tri.mw = mw.to(DEVICE); layer_tri.Sw = Sw.to(DEVICE)
    layer_tri.mb = mb.to(DEVICE); layer_tri.Sb = Sb.to(DEVICE)
    net_tri = TSequential([layer_tri], device=DEVICE)

    # pytagi CPU
    net_cpu = PSequential(PConv2d(C_IN, C_OUT, K, padding=2, in_width=W, in_height=H))
    net_cpu.preinit_layer()
    k_key = list(net_cpu.state_dict().keys())[0]
    net_cpu.load_state_dict({
        k_key: (
            mw.T.cpu().numpy().flatten().tolist(),
            Sw.T.cpu().numpy().flatten().tolist(),
            mb.squeeze().cpu().numpy().tolist(),
            Sb.squeeze().cpu().numpy().tolist(),
        )
    })

    # pytagi GPU (fresh net, same weights, then to_device)
    net_gpu = PSequential(PConv2d(C_IN, C_OUT, K, padding=2, in_width=W, in_height=H))
    net_gpu.preinit_layer()
    k_key_gpu = list(net_gpu.state_dict().keys())[0]
    net_gpu.load_state_dict({
        k_key_gpu: (
            mw.T.cpu().numpy().flatten().tolist(),
            Sw.T.cpu().numpy().flatten().tolist(),
            mb.squeeze().cpu().numpy().tolist(),
            Sb.squeeze().cpu().numpy().tolist(),
        )
    })
    net_gpu.to_device("cuda")

    # Fwd
    torch.manual_seed(1)
    xb = torch.randn(BATCH, C_IN, H, W)
    x_np = xb.numpy().reshape(-1).astype(np.float32)

    mu_tri, var_tri = net_tri.forward(xb.to(DEVICE))
    mu_tri_np = mu_tri.detach().cpu().numpy()
    var_tri_np = var_tri.detach().cpu().numpy()

    mu_cpu_flat, var_cpu_flat = net_cpu(x_np)
    mu_cpu = np.array(mu_cpu_flat[: BATCH * C_OUT * H * W]).reshape(BATCH, C_OUT, H, W)
    var_cpu = np.array(var_cpu_flat[: BATCH * C_OUT * H * W]).reshape(BATCH, C_OUT, H, W)

    mu_gpu_flat, var_gpu_flat = net_gpu(x_np)
    mu_gpu = np.array(mu_gpu_flat[: BATCH * C_OUT * H * W]).reshape(BATCH, C_OUT, H, W)
    var_gpu = np.array(var_gpu_flat[: BATCH * C_OUT * H * W]).reshape(BATCH, C_OUT, H, W)

    print("=== triton GPU  vs  pytagi CPU ===")
    _cmp("mu  tri_GPU vs cut_CPU", mu_tri_np, mu_cpu)
    _cmp("var tri_GPU vs cut_CPU", var_tri_np, var_cpu)

    print("\n=== triton GPU  vs  pytagi GPU ===")
    _cmp("mu  tri_GPU vs cut_GPU", mu_tri_np, mu_gpu)
    _cmp("var tri_GPU vs cut_GPU", var_tri_np, var_gpu)

    print("\n=== pytagi CPU  vs  pytagi GPU ===")
    _cmp("mu  cut_CPU vs cut_GPU", mu_cpu, mu_gpu)
    _cmp("var cut_CPU vs cut_GPU", var_cpu, var_gpu)


if __name__ == "__main__":
    main()
