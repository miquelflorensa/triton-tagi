"""
Old vs. new methodology on CIFAR-10 ResNet-18 — triton-tagi.

Same deep ResNet-18 + Remax architecture for both arms; the ONLY difference is
the calibration. Goal: maximise classification accuracy toward 98%.

    baseline (old)                       calibrated (new)
    ---------------------------------    -----------------------------------------
    He init + gain_w/gain_b              operator T init: signal=1, K_b/K_w=1, J=½
    no re-inflation → variance collapse  surprise-driven re-inflation (process Q)
    μ_W drifts → outputs correlate       Muon exact-SVD polar projection
    fixed σ_v                            online homoscedastic σ_v (method-of-moments)
    Remax output head                    Remax output head           ← identical

Reuses ``build_resnet18``, ``gpu_augment``, and ``load_cifar10`` from
``cifar10_resnet18_calibrated.py``. See ``docs/calibration.md`` for the math.

Usage:
    python examples/compare_calibration_resnet18.py
    python examples/compare_calibration_resnet18.py --n_epochs 30 --no_augment
"""

from __future__ import annotations

import argparse
import time

import torch
from cifar10_resnet18_calibrated import build_resnet18, gpu_augment, load_cifar10

from triton_tagi import (
    OnlineCalibration,
    Sequential,
    calibrate,
)
from triton_tagi.base import LearnableLayer


def evaluate(net: Sequential, x_test, y_labels, batch_size: int = 256) -> float:
    """Return test accuracy (Remax head: 10 class outputs, direct argmax)."""
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu.argmax(dim=1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


def deep_layer_weight_var(net: Sequential) -> float:
    """Mean weight variance σ²_W of a middle learnable layer (collapse probe)."""
    learn = [layer for layer in _flatten_learnable(net) if getattr(layer, "Sw", None) is not None]
    mid = learn[len(learn) // 2]
    return float(mid.Sw.mean().item())


def _flatten_learnable(net: Sequential) -> list:
    out: list = []
    for layer in net.layers:
        sub = getattr(layer, "_learnable", None)
        if sub is not None:
            out.extend(sub)
        elif isinstance(layer, LearnableLayer):
            out.append(layer)
    return out


def train_one(net, online, x_train, y_oh, x_test, y_labels,
              n_epochs, batch_size, sigma_v, augment, device,
              label: str = "", patience: int = 15):
    """Train one arm with early stopping on test accuracy.

    Returns (best_acc, best_epoch, per-epoch acc, per-epoch deep-layer σ²_W).
    For the calibrated arm sigma_v is overridden by online.sigma_v (adapts);
    for the baseline arm it is used directly as the fixed observation noise.
    """
    accs: list[float] = []
    wvar: list[float] = []
    best = 0.0
    best_epoch = 0
    wait = 0
    t0_total = time.perf_counter()

    print(f"\n  [{label}]  {'Epoch':>5}  {'Test Acc':>9}  {'Best':>9}  "
          f"{'σ_v':>8}  {'σ²_W ratio':>10}  {'Time':>7}  {'Patience':>8}")
    print(f"  [{label}]  " + "─" * 70)

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_oh[perm]
        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            net.step(xb, y_s[i : i + batch_size], sigma_v, online=online)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_labels)
        accs.append(acc)
        wv = deep_layer_weight_var(net)
        wvar.append(wv)

        improved = acc > best
        if improved:
            best = acc
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        wv0 = wvar[0] if wvar[0] != 0 else 1.0
        current_sv = online.sigma_v if online is not None else sigma_v
        marker = " *" if improved else ""
        print(f"  [{label}]  {epoch:5d}  {acc*100:8.2f}%  {best*100:8.2f}%  "
              f"{current_sv:8.4f}  {wv / wv0:10.3f}  {wall:6.2f}s  {wait:3d}/{patience}{marker}")

        if wait >= patience:
            print(f"  [{label}]  Early stopping: no improvement for {patience} epochs "
                  f"(best was epoch {best_epoch}: {best*100:.2f}%)")
            break

    wall_total = time.perf_counter() - t0_total
    print(f"  [{label}]  " + "─" * 70)
    print(f"  [{label}]  Best: {best*100:.2f}% (epoch {best_epoch})  —  {wall_total:.1f}s total")
    return best, best_epoch, accs, wvar


def main(
    n_epochs: int = 30,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    sigma2_obs: float = 0.001,
    augment: bool = True,
    data_dir: str = "data",
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 15,
) -> dict:
    """Run the baseline vs calibrated comparison (both arms: ResNet-18 + Remax).

    sigma_v is the fixed observation noise for the baseline arm; for the
    calibrated arm it seeds the online estimate which then adapts each step.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 70)
    print("  CIFAR-10 ResNet-18 — baseline vs calibrated (Remax, toward 98%)")
    print("=" * 70)
    if dev.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, x_test, y_test_labels = load_cifar10(data_dir, dev)

    # ── Both arms: identical ResNet-18 + Remax, same seed ──
    torch.manual_seed(seed)
    net_base = build_resnet18(dev)
    torch.manual_seed(seed)
    net_cal = build_resnet18(dev)

    # Calibrated arm: operator T init + online homoscedastic σ_v adaptation.
    calibrate(net_cal, sigma2_obs=sigma2_obs, metric="analytic")
    online = OnlineCalibration(mode="surprise", sigma_v_mode="homoscedastic",
                               sigma_v2=sigma2_obs, project=True)

    print(f"\n  Params: {net_cal.num_parameters():,}  |  epochs={n_epochs} batch={batch_size}"
          f"  |  patience={patience}")
    print("  baseline = He init, no calibration   |   calibrated = operator T + online")

    t0 = time.perf_counter()
    base_best, base_best_ep, base_acc, base_wv = train_one(
        net_base, None, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, augment, dev,
        label="baseline", patience=patience)
    cal_best, cal_best_ep, cal_acc, cal_wv = train_one(
        net_cal, online, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, augment, dev,
        label="calibrated", patience=patience)
    wall = time.perf_counter() - t0

    n_base = len(base_acc)
    n_cal = len(cal_acc)
    n_max = max(n_base, n_cal)
    wv0_b = base_wv[0] if base_wv[0] != 0 else 1.0
    wv0_c = cal_wv[0] if cal_wv[0] != 0 else 1.0

    print(f"\n  {'Epoch':>5} │ {'base acc':>9}  {'calib acc':>9} │"
          f" {'base σ²_W↓':>11}  {'calib σ²_W↓':>11}")
    print("  ──────┼" + "─" * 22 + "┼" + "─" * 26)
    for ep in range(n_max):
        b_acc_s = f"{base_acc[ep]*100:8.2f}%" if ep < n_base else "    ---"
        c_acc_s = f"{cal_acc[ep]*100:8.2f}%" if ep < n_cal else "    ---"
        b_wv_s = f"{base_wv[ep]/wv0_b:11.3f}" if ep < n_base else "       ---"
        c_wv_s = f"{cal_wv[ep]/wv0_c:11.3f}" if ep < n_cal else "       ---"
        print(f"  {ep + 1:5d} │ {b_acc_s}  {c_acc_s} │"
              f" {b_wv_s}  {c_wv_s}")

    print("\n" + "=" * 70)
    print(f"  best test acc   baseline = {base_best*100:6.2f}% (ep {base_best_ep})   "
          f"calibrated = {cal_best*100:6.2f}% (ep {cal_best_ep})")
    print(f"  deep-layer σ²_W  baseline ×{base_wv[-1]/wv0_b:.3f}   "
          f"calibrated ×{cal_wv[-1]/wv0_c:.3f}   (1.0 = no collapse)")
    print(f"  total wall = {wall:.1f}s")
    print("=" * 70)
    return {"base_best": base_best, "cal_best": cal_best,
            "base_acc": base_acc, "cal_acc": cal_acc}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 baseline vs calibrated (TAGI-V)")
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32, help="Training batch size (reduce if OOM)")
    p.add_argument("--sigma_v", type=float, default=0.05, help="baseline seed (ignored by TAGI-V)")
    p.add_argument("--sigma2_obs", type=float, default=0.001, help="calibration σ_v² seed")
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    p.set_defaults(augment=True)
    args = p.parse_args()
    main(**vars(args))
