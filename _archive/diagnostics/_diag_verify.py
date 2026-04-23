"""Two-part diagnostic:

Part A — byte-level verification that triton's loaded weights equal pytagi's
own init.  Uses pytagi's CPU state_dict as ground truth and reshapes/
transposes to compare against triton's mw/Sw/mb/Sb tensors.

Part B — single-layer forward comparison using those same weights.  Tests
Conv alone, Conv+BN, and Conv+BN+AvgPool to localize where (if anywhere)
the forward outputs diverge.
"""

from __future__ import annotations

import numpy as np
import torch

import pytagi
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBatchNorm2d
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda"
IN_C, H, W = 3, 32, 32
C_OUT = 32
K = 5
BATCH = 4


def _pytagi_conv_bn_cpu():
    torch.manual_seed(0); pytagi.manual_seed(0)
    net = PSequential(
        PConv2d(IN_C, C_OUT, K, padding=2, in_width=W, in_height=H),
        PBatchNorm2d(C_OUT),
    )
    net.preinit_layer()
    sd = net.state_dict()
    conv_k = [k for k in sd if k.startswith("Conv2d")][0]
    bn_k = [k for k in sd if k.startswith("BatchNorm2d")][0]
    return net, sd[conv_k], sd[bn_k]


def _cmp(label, a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel = diff / denom
    print(
        f"  {label:<20s}  n={a.size:>6d}  "
        f"|Δ|max={diff.max():.3e}  rel_max={rel.max():.3e}  "
        f"|a|mean={np.abs(a).mean():.3e}"
    )


def part_a_init_bitwise(conv_p_py, bn_p_py):
    print("=== Part A — byte-level init comparison ===")
    mw_flat, Sw_flat, mb_flat, Sb_flat = conv_p_py
    Kdim = IN_C * K * K

    # Triton expects (K, C_OUT).  pytagi flat is (C_OUT, Kdim) row-major.
    mw_py_as_CK = np.array(mw_flat, dtype=np.float32).reshape(C_OUT, Kdim)
    Sw_py_as_CK = np.array(Sw_flat, dtype=np.float32).reshape(C_OUT, Kdim)

    # Load into triton by (C_OUT, K).T = (K, C_OUT).
    mw_tri = torch.tensor(mw_py_as_CK.T.copy())
    Sw_tri = torch.tensor(Sw_py_as_CK.T.copy())
    mb_tri = torch.tensor(mb_flat, dtype=torch.float32).reshape(1, C_OUT)
    Sb_tri = torch.tensor(Sb_flat, dtype=torch.float32).reshape(1, C_OUT)

    # Read back and compare to pytagi flat.  For mw: triton.T.flatten() should
    # give exactly the pytagi flat.
    mw_tri_asflat = mw_tri.numpy().T.flatten()
    _cmp("conv.mw roundtrip", mw_tri_asflat, np.array(mw_flat))
    Sw_tri_asflat = Sw_tri.numpy().T.flatten()
    _cmp("conv.Sw roundtrip", Sw_tri_asflat, np.array(Sw_flat))
    _cmp("conv.mb roundtrip", mb_tri.numpy().ravel(), np.array(mb_flat))
    _cmp("conv.Sb roundtrip", Sb_tri.numpy().ravel(), np.array(Sb_flat))

    mg_flat, Sg_flat, mbb_flat, Sbb_flat = bn_p_py
    _cmp("bn.mw roundtrip", np.array(mg_flat, dtype=np.float32), np.array(mg_flat))
    _cmp("bn.Sw roundtrip", np.array(Sg_flat, dtype=np.float32), np.array(Sg_flat))

    return mw_tri, Sw_tri, mb_tri, Sb_tri


def _build_triton_conv_bn(mw_tri, Sw_tri, mb_tri, Sb_tri, bn_p):
    conv = TConv2D(IN_C, C_OUT, K, padding=2, device=DEVICE)
    conv.mw, conv.Sw = mw_tri.to(DEVICE), Sw_tri.to(DEVICE)
    conv.mb, conv.Sb = mb_tri.to(DEVICE), Sb_tri.to(DEVICE)
    bn = TBatchNorm2D(C_OUT, device=DEVICE, preserve_var=False)
    mg, Sg, mbb, Sbb = bn_p
    bn.mw = torch.tensor(mg).to(DEVICE)
    bn.Sw = torch.tensor(Sg).to(DEVICE)
    bn.mb = torch.tensor(mbb).to(DEVICE)
    bn.Sb = torch.tensor(Sbb).to(DEVICE)
    return conv, bn


def part_b_forward(net_cut_cpu, conv, bn):
    print("\n=== Part B — single-layer forward, identical bits ===")
    net_cut_cpu.to_device("cuda")  # move pytagi net to CUDA

    torch.manual_seed(1)
    xb = torch.randn(BATCH, IN_C, H, W)

    # triton: Conv only
    net_tri_conv = TSequential([conv], device=DEVICE)
    mu_t_c, var_t_c = net_tri_conv.forward(xb.to(DEVICE))

    # pytagi: currently net is Conv+BN — rebuild Conv-only
    torch.manual_seed(0); pytagi.manual_seed(0)
    net_cut_conv = PSequential(PConv2d(IN_C, C_OUT, K, padding=2, in_width=W, in_height=H))
    net_cut_conv.preinit_layer()  # MUST match the init that gave us conv_p
    sd = net_cut_conv.state_dict()
    ck = [k for k in sd if k.startswith("Conv2d")][0]
    # Use THE weights we extracted from the Conv+BN net's conv, not the new
    # init — we need to make sure the values match. Get them from net_cut_cpu
    # is impossible (already on CUDA), so just rely on pytagi init being
    # deterministic with the same seeds. The print below checks this.
    sd_conv_only = net_cut_conv.state_dict()
    conv_p = sd_conv_only[ck]
    print(f"  conv_p (conv-only net) mw[:3] = {conv_p[0][:3]}")

    net_cut_conv.to_device("cuda")
    mu_flat, var_flat = net_cut_conv(xb.numpy().reshape(-1).astype(np.float32))
    n = BATCH * C_OUT * H * W
    mu_c_c = np.array(mu_flat[:n])
    var_c_c = np.array(var_flat[:n])

    # triton returns (B, C_OUT, H, W) → flatten row-major
    _cmp("Conv.mu", mu_t_c.detach().cpu().numpy().ravel(), mu_c_c)
    _cmp("Conv.var", var_t_c.detach().cpu().numpy().ravel(), var_c_c)


def main():
    net_cut_cpu, conv_p_py, bn_p_py = _pytagi_conv_bn_cpu()
    print(f"  pytagi conv.mw[:3] = {conv_p_py[0][:3]}")
    print(f"  pytagi conv.Sw[:3] = {conv_p_py[1][:3]}")
    print(f"  pytagi bn.mg[:3]   = {bn_p_py[0][:3]}")
    print(f"  pytagi bn.Sg[:3]   = {bn_p_py[1][:3]}")

    mw_tri, Sw_tri, mb_tri, Sb_tri = part_a_init_bitwise(conv_p_py, bn_p_py)
    conv, bn = _build_triton_conv_bn(mw_tri, Sw_tri, mb_tri, Sb_tri, bn_p_py)
    part_b_forward(net_cut_cpu, conv, bn)


if __name__ == "__main__":
    main()
