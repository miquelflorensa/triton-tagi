"""Test which K-axis layout makes triton Conv match pytagi Conv.

Uses pytagi's own init (bit-for-bit identical weights), then tries several
interpretations of pytagi's flat mw_py layout when building triton:

  L1: (C_out, C_in, kH, kW)  — standard PyTorch layout
  L2: (C_out, C_in, kW, kH)  — spatial axes swapped
  L3: (C_out, kH, kW, C_in)  — channel last
  L4: (C_out, kW, kH, C_in)  — channel last + spatial swap

Triton's im2col produces K = [C_in, kH, kW] row-major (see im2col kernel
line 68-71).  So for each interpretation we rearrange to match triton's K
order.  Report which layout gives Conv.mu parity.
"""

from __future__ import annotations

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


def build_pytagi_and_read():
    torch.manual_seed(0); pytagi.manual_seed(0)
    net = PSequential(PConv2d(C_IN, C_OUT, K, padding=2, in_width=W, in_height=H))
    net.preinit_layer()
    ck = [k for k in net.state_dict() if k.startswith("Conv2d")][0]
    mw_flat, Sw_flat, mb_flat, Sb_flat = net.state_dict()[ck]
    net.to_device("cuda")
    return net, mw_flat, Sw_flat, mb_flat, Sb_flat


def triton_conv_with_mw(mw_tri_KC, Sw_tri_KC, mb, Sb):
    layer = TConv2D(C_IN, C_OUT, K, padding=2, device=DEVICE)
    layer.mw = torch.tensor(mw_tri_KC, dtype=torch.float32).to(DEVICE)
    layer.Sw = torch.tensor(Sw_tri_KC, dtype=torch.float32).to(DEVICE)
    layer.mb = torch.tensor(mb, dtype=torch.float32).reshape(1, C_OUT).to(DEVICE)
    layer.Sb = torch.tensor(Sb, dtype=torch.float32).reshape(1, C_OUT).to(DEVICE)
    return TSequential([layer], device=DEVICE)


def layout_interp(mw_flat, Sw_flat, layout):
    """Reshape pytagi's flat into (C_out, C_in, kH, kW) under 'layout',
    then reorder to triton's K = [C_in, kH, kW] row-major.
    Returns (Kdim, C_out) arrays ready to load into triton."""
    arr = np.array(mw_flat, dtype=np.float32)
    sw = np.array(Sw_flat, dtype=np.float32)
    if layout == "C,Ci,kH,kW":
        mw_4d = arr.reshape(C_OUT, C_IN, K, K)
        sw_4d = sw.reshape(C_OUT, C_IN, K, K)
    elif layout == "C,Ci,kW,kH":
        mw_4d = arr.reshape(C_OUT, C_IN, K, K).transpose(0, 1, 3, 2)
        sw_4d = sw.reshape(C_OUT, C_IN, K, K).transpose(0, 1, 3, 2)
    elif layout == "C,kH,kW,Ci":
        mw_4d = arr.reshape(C_OUT, K, K, C_IN).transpose(0, 3, 1, 2)
        sw_4d = sw.reshape(C_OUT, K, K, C_IN).transpose(0, 3, 1, 2)
    elif layout == "C,kW,kH,Ci":
        mw_4d = arr.reshape(C_OUT, K, K, C_IN).transpose(0, 3, 2, 1)
        sw_4d = sw.reshape(C_OUT, K, K, C_IN).transpose(0, 3, 2, 1)
    else:
        raise ValueError(layout)
    # Flatten triton's K = [C_in, kH, kW] row-major, shape (Kdim, C_out)
    mw_KC = mw_4d.reshape(C_OUT, -1).T.copy()
    sw_KC = sw_4d.reshape(C_OUT, -1).T.copy()
    return mw_KC, sw_KC


def main():
    net_cut, mw_flat, Sw_flat, mb_flat, Sb_flat = build_pytagi_and_read()
    torch.manual_seed(1)
    xb = torch.randn(BATCH, C_IN, H, W)
    x_np = xb.numpy().reshape(-1).astype(np.float32)

    mu_flat, var_flat = net_cut(x_np)
    n = BATCH * C_OUT * H * W
    mu_cut = np.array(mu_flat[:n])
    var_cut = np.array(var_flat[:n])

    for layout in ["C,Ci,kH,kW", "C,Ci,kW,kH", "C,kH,kW,Ci", "C,kW,kH,Ci"]:
        mw_KC, Sw_KC = layout_interp(mw_flat, Sw_flat, layout)
        net_tri = triton_conv_with_mw(mw_KC, Sw_KC, mb_flat, Sb_flat)
        mu_tri, var_tri = net_tri.forward(xb.to(DEVICE))
        mu_tri = mu_tri.detach().cpu().numpy().reshape(-1)

        diff = np.abs(mu_tri - mu_cut.astype(np.float64))
        rel = diff / (np.maximum(np.abs(mu_tri), np.abs(mu_cut)) + 1e-30)
        print(f"  layout={layout:<13s}  mu |Δ|max={diff.max():.3e}  rel_max={rel.max():.3e}")


if __name__ == "__main__":
    main()
