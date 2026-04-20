"""Direct comparison of Conv layer outputs (no readout), triton vs cuTAGI."""
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
IN_C, H, W = 3, 32, 32
C_OUT = 32
K = 5
BATCH = 4


def _he_conv(C_in, C_out, k):
    fan_in = C_in * k * k
    s = math.sqrt(1.0 / fan_in)
    Kdim = C_in * k * k
    return (torch.randn(Kdim, C_out) * s, torch.full((Kdim, C_out), s**2),
            torch.zeros(1, C_out), torch.full((1, C_out), s**2))


torch.manual_seed(0); pytagi.manual_seed(0)
conv_p = _he_conv(IN_C, C_OUT, K)
mw, Sw, mb, Sb = conv_p

# triton
c_tri = TConv2D(IN_C, C_OUT, K, padding=2, device=DEVICE)
c_tri.mw, c_tri.Sw = mw.to(DEVICE), Sw.to(DEVICE)
c_tri.mb, c_tri.Sb = mb.to(DEVICE), Sb.to(DEVICE)
net_tri = TSequential([c_tri], device=DEVICE)

# pytagi
net_cut = PSequential(PConv2d(IN_C, C_OUT, K, padding=2, in_width=W, in_height=H))
net_cut.preinit_layer()
k_key = list(net_cut.state_dict().keys())[0]
net_cut.load_state_dict({
    k_key: (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )
})
net_cut.to_device("cuda")

# forward
torch.manual_seed(1)
xb = torch.randn(BATCH, IN_C, H, W)

mu_tri, var_tri = net_tri.forward(xb.to(DEVICE))
print(f"triton mu shape: {mu_tri.shape}")

x_np = xb.numpy().reshape(-1).astype(np.float32)
mu_flat, var_flat = net_cut(x_np)
print(f"pytagi mu len:   {len(mu_flat)}  (expected {BATCH*C_OUT*H*W})")


def cmp(label, a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel = diff / denom
    print(
        f"  {label}:  n={a.size}  |Δ|max={diff.max():.3e}  "
        f"rel_max={rel.max():.3e}  |a|mean={np.abs(a).mean():.3e}  bias={(a-b).mean():+.3e}"
    )


# Try multiple layout interpretations for pytagi's output
mu_tri_np = mu_tri.detach().cpu().numpy()  # shape: (B, C_out, H, W)
var_tri_np = var_tri.detach().cpu().numpy()

n = BATCH * C_OUT * H * W
mu_cut = np.array(mu_flat[:n])
var_cut = np.array(var_flat[:n])

# Layout A: (B, C_out, H, W)
cmp("A (B,Co,H,W)", mu_tri_np.reshape(-1), mu_cut.reshape(BATCH, C_OUT, H, W).reshape(-1))
# Layout B: (B, H, W, C_out)
cmp("B (B,H,W,Co)", mu_tri_np.transpose(0, 2, 3, 1).reshape(-1), mu_cut.reshape(BATCH, H, W, C_OUT).reshape(-1))
# Layout C: (C_out, B, H, W)
cmp("C (Co,B,H,W)", mu_tri_np.transpose(1, 0, 2, 3).reshape(-1), mu_cut.reshape(C_OUT, BATCH, H, W).reshape(-1))
