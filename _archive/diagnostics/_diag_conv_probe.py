"""Probe Conv weight layout using a single-nonzero mw.

For each K-slot s:
  1. Build both nets with mw[0, s] = 1, all others = 0, Sw = tiny positive.
  2. Feed an input grid where pixel (c, h, w) = c*100 + h*10 + w  (unique ids).
  3. Read the center pixel output.  With stride=1, pad=1, 3x3 kernel, that
     output equals mu_a at the single input position selected by slot s.
  4. Decode which (c, h, w) triton picked vs which pytagi picked.

If the mappings differ, the two libraries interpret the K-slot layout
differently.  Any mismatch pinpoints the bug.
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
C_IN, C_OUT = 3, 1
K = 3
H = W = 3
BATCH = 1
SW_VAL = 1e-6  # tiny but nonzero


def make_distinct_input():
    x = torch.zeros(BATCH, C_IN, H, W, dtype=torch.float32)
    for c in range(C_IN):
        for h in range(H):
            for w in range(W):
                x[0, c, h, w] = c * 100 + h * 10 + w
    return x


def probe_triton(slot):
    Kdim = C_IN * K * K
    mw = torch.zeros(Kdim, C_OUT); mw[slot, 0] = 1.0
    Sw = torch.full((Kdim, C_OUT), SW_VAL)
    mb = torch.zeros(1, C_OUT); Sb = torch.zeros(1, C_OUT)
    conv = TConv2D(C_IN, C_OUT, K, padding=1, device=DEVICE)
    conv.mw, conv.Sw = mw.to(DEVICE), Sw.to(DEVICE)
    conv.mb, conv.Sb = mb.to(DEVICE), Sb.to(DEVICE)
    net = TSequential([conv], device=DEVICE)

    x = make_distinct_input().to(DEVICE)
    mu, _ = net.forward(x)
    return mu.detach().cpu().numpy()[0, 0, 1, 1]  # center pixel


def probe_pytagi(slot):
    Kdim = C_IN * K * K
    net = PSequential(PConv2d(C_IN, C_OUT, K, padding=1, in_width=W, in_height=H))
    net.preinit_layer()
    ck = [k for k in net.state_dict() if k.startswith("Conv2d")][0]

    mw_py = [0.0] * (C_OUT * Kdim); mw_py[0 * Kdim + slot] = 1.0
    Sw_py = [SW_VAL] * (C_OUT * Kdim)
    mb_py = [0.0] * C_OUT; Sb_py = [0.0] * C_OUT
    net.load_state_dict({ck: (mw_py, Sw_py, mb_py, Sb_py)})

    # Verify load actually took effect
    sd_after = net.state_dict()[ck]
    loaded_mw = np.array(sd_after[0])
    assert loaded_mw[slot] == 1.0, f"load_state_dict didn't take: slot={slot}"

    net.to_device("cuda")

    x = make_distinct_input()
    x_np = x.numpy().reshape(-1).astype(np.float32)
    mu_flat, _ = net(x_np)
    mu = np.array(mu_flat[: BATCH * C_OUT * H * W]).reshape(BATCH, C_OUT, H, W)
    return mu[0, 0, 1, 1]


def decode(v):
    iv = int(round(v))
    if abs(v - iv) > 1e-3:
        return "(non-integer)"
    if iv == 0:
        return "(zero)"
    return f"c={iv // 100}, h={(iv // 10) % 10}, w={iv % 10}"


def main():
    Kdim = C_IN * K * K
    print(f"Kdim = {Kdim}")
    print(f"{'slot':>4s}  {'tri center':>14s}  {'cut center':>14s}  "
          f"{'tri decode':>14s}  {'cut decode':>14s}  {'match?':>6s}")
    mismatches = 0
    for slot in range(Kdim):
        t = probe_triton(slot)
        c = probe_pytagi(slot)
        td = decode(t); cd = decode(c)
        match = "OK" if abs(t - c) < 1e-3 else "FAIL"
        if match == "FAIL":
            mismatches += 1
        print(f"{slot:>4d}  {t:>14.4f}  {c:>14.4f}  {td:>14s}  {cd:>14s}  {match:>6s}")
    print(f"\n{mismatches}/{Kdim} slots disagree")


if __name__ == "__main__":
    main()
