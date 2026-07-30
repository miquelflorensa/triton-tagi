"""
CIFAR-10 — ResNet-18 — Unified parameter-free online calibration (triton-tagi).

This is the calibrated counterpart of ``cifar10_resnet18.py``. It removes every
init/training hyperparameter the baseline relies on:

    baseline                         calibrated (this file)
    ----------------------------     -----------------------------------------
    init_method="He" / gain_w/gain_b unified operator T  (calibrate(net))
    sigma_v = 0.05 (+ decay)         learned online (TAGI-V / AGVI heteroscedastic head)
    no re-inflation (collapse)       online operator T: surprise-driven λ_t
                                     re-inflation + Muon mean projection

The network is built with default layer init, then ``calibrate(net)`` overwrites
every parameter so that, at every layer: signal = 1, the epistemic budget equals
σ_v² (output Kalman gain J = ½), and per-parameter gains are balanced
(K_b/K_w = 1). During training the online operator keeps those invariants alive
— the only mechanism that prevents TAGI's monotone variance collapse — with the
forgetting rate set automatically from the batch surprise χ².

Usage:
    python examples/cifar10_resnet18_calibrated.py
    python examples/cifar10_resnet18_calibrated.py --n_epochs 30 --no_augment
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    EvenSoftplus,
    Flatten,
    Linear,
    OnlineCalibration,
    ReLU,
    Remax,
    ResBlock,
    Sequential,
    calibrate,
)
from triton_tagi.checkpoint import RunDir

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Single source of truth for the seed σ_v²: it is BOTH the calibration output
# budget (Sz_final = σ_v² → J=½) and the AGVI noise-head init (E[V²]_init = σ_v²),
# so it must be defined once and shared by main() and the CLI default.
_SIGMA2_OBS_DEFAULT = 0.001


def load_cifar10(data_dir: str, device: torch.device):
    """Load CIFAR-10 as normalized (N,3,32,32) tensors on ``device``."""
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    )
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)

    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=device)

    y_train_oh = torch.zeros(len(y_train), 10, device=device)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, x_test, y_test


def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip + random crop on-device."""
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)
    rows = top.unsqueeze(1) + torch.arange(H, device=x.device).unsqueeze(0)
    cols = left.unsqueeze(1) + torch.arange(W, device=x.device).unsqueeze(0)
    return x_pad[
        torch.arange(B, device=x.device)[:, None, None, None],
        torch.arange(C, device=x.device)[None, :, None, None],
        rows[:, None, :, None].expand(B, C, H, W),
        cols[:, None, None, :].expand(B, C, H, W),
    ]


def evaluate(net: Sequential, x_test, y_labels, batch_size: int = 256) -> float:
    """Return test accuracy. Even columns are class means (TAGI-V / AGVI head)."""
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu[:, 0::2].argmax(dim=1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


def build_resnet18(device: torch.device) -> Sequential:
    """CIFAR-10 ResNet-18 (default init; calibrate() overwrites all parameters)."""
    kw = {"device": device}
    return Sequential(
        [
            Conv2D(3, 64, 3, stride=1, padding=1, **kw),
            ReLU(),
            BatchNorm2D(64, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            AvgPool2D(4),
            Flatten(),
            Linear(512, 10, **kw),
            Remax(),
        ],
        device=device,
    )


def build_resnet18_agvi(device: torch.device) -> Sequential:
    """CIFAR-10 ResNet-18 with TAGI-V / AGVI head (2·K outputs).

    Reuses build_resnet18 and replaces Linear(512, 10) + Remax with
    Linear(512, 20) + EvenSoftplus(half_width=10): interleaved [mean, noise-var]
    per class. EvenSoftplus ensures the odd outputs (V²) stay positive.
    The AGVI kernel (_output_innovation_kernel_heteros) is auto-selected by
    compute_innovation when output_width == 2 * target_width.
    """
    net = build_resnet18(device)
    in_features = net.layers[-2].in_features  # 512 — final Linear's fan-in
    net.layers = net.layers[:-2] + [
        Linear(in_features, 20, device=device),
        EvenSoftplus(half_width=10),
    ]
    return net


def _init_noise_head(linear_layer, sigma2_obs: float) -> None:
    """Align the AGVI noise head so initial mu_v2 ≈ sigma2_obs → Kalman J = 0.5.

    Without this, softplus(0) ≈ 0.693 >> sigma2_obs (e.g. 0.001), giving
    J = var_a / (var_a + mu_v2) ≈ 0.001 / 0.694 ≈ 0.0014 instead of 0.5.

    Fix: zero the odd-output weight columns (noise head starts input-independent)
    and set the odd biases to  b = softplus_inv(sigma2_obs)  so that
    softplus(b) = sigma2_obs, aligning mu_v2 with the calibration budget.
    """
    # softplus_inv(x) = log(exp(x) - 1)  ≈  log(x)  for small x
    bias_val = math.log(math.expm1(max(sigma2_obs, 1e-6)))
    with torch.no_grad():
        linear_layer.mw[:, 1::2].zero_()  # noise columns: no input dependency at init
        # mb has shape (1, out_features) — index the columns, not the rows
        linear_layer.mb[0, 1::2] = bias_val  # softplus(b) = sigma2_obs


def print_calibration_properties(
    net: Sequential, online: OnlineCalibration, sigma2_obs: float
) -> None:
    """Print the three calibration invariants and OnlineCalibration config.

    Mirrors the checks in tests/validation/test_calibrate.py against the three
    conditions in docs/calibration.md:
      (I)   signal = v · Σ_i mw[i,o]²  ≈ 1   per output
      (II)  K_b / K_w                   ≈ 1   balanced per-parameter gain
      (III) budget AW + B               = c_global·budget_denom_ℓ  (per-layer LOCAL
            budget — now VARIES by layer; condition III is a GLOBAL output condition,
            not a per-layer one: the OUTPUT budget Sz_final = σ_v² gives J = ½)
    """

    def _iter_layers(layers):
        for layer in layers:
            sub = getattr(layer, "_learnable", None)
            if sub is not None:
                yield from sub
            else:
                yield layer

    signals, ratios, budgets, gains = [], [], [], []
    for layer in _iter_layers(net.layers):
        meta = getattr(layer, "_calib", None)
        if meta is None or meta["kind"] != "dense":
            continue
        v = meta["v_analytic"]
        g = meta["g_diag"]
        sw = layer.Sw[:, 0]
        # (I)
        signals.extend((v * (layer.mw**2).sum(dim=0)).tolist())
        # (II)
        kw = float((g.sqrt() * sw).mean().item())
        kb = float(layer.Sb.flatten()[0].item())
        ratios.append(kb / max(kw, 1e-12))
        # (III)
        budget = float((g * sw).sum().item()) + kb
        budgets.append(budget)
        gains.append(budget / (budget + sigma2_obs))

    def _fmt(vals: list) -> str:
        if not vals:
            return "n/a"
        mn, mx = min(vals), max(vals)
        mu = sum(vals) / len(vals)
        return f"min={mn:.5f}  mean={mu:.5f}  max={mx:.5f}"

    # (III-global) Forward Sz sensitivity pass: with every layer at c_global, the
    # OUTPUT budget Sz_final = c_global·A(net) = σ_v², giving J = ½ at the head.
    from triton_tagi.calibrate import _forward_sz_analytic

    A = getattr(net, "_calib_A", 1.0)
    c_global = sigma2_obs / A if A > 0 else sigma2_obs
    sz_final = _forward_sz_analytic(net, c=c_global)
    j_global = sz_final / (sz_final + sigma2_obs)

    w = 68
    print("\n  " + "─" * w)
    print("  Calibration invariants (post-init, analytic metric) — docs/calibration.md")
    print("  " + "─" * w)
    print(f"  (I)    signal = v·Σmw²  {_fmt(signals)}  target 1.0")
    print(f"  (II)   K_b / K_w        {_fmt(ratios)}  target 1.0")
    print(f"  (III)  budget AW+B      {_fmt(budgets)}  per-layer LOCAL (varies)")
    print(f"         (local J)         {_fmt(gains)}  per-layer (not the output J)")
    print("  " + "─" * w)
    print(f"  (III-global) A(net) amplification : {A:.4g}")
    print(f"               c_global = σ_v²/A     : {c_global:.6g}")
    print(f"               OUTPUT Sz_final       : {sz_final:.6f}  target σ_v²={sigma2_obs:.5f}")
    print(f"               OUTPUT Kalman gain J  : {j_global:.6f}  target 0.5")
    print("  " + "─" * w)
    print(f"  OnlineCalibration")
    print(f"    mode         : {online.mode}")
    print(f"    sigma_v_mode : {online.sigma_v_mode}")
    print(f"    sigma_v² seed: {online.sigma_v2:.5f}  →  σ_v = {online.sigma_v:.5f}")
    print(f"    project(Muon): {online.project}")
    print(f"    lam_max      : {online.lam:.4f}")
    print(f"    tau          : {online.tau:.4f}")
    print(f"    sigma_v2_rho : {online.sigma_v2_rho:.4f}  (EMA rate for budget update)")
    print("  " + "─" * w)


def train(
    net,
    online,
    x_train,
    y_train_oh,
    x_test,
    y_test_labels,
    n_epochs,
    batch_size,
    augment,
    device,
    run,
    config,
    patience: int = 15,
) -> float:
    """Parameter-free online training loop with early stopping on test accuracy.

    Returns best test accuracy. Stops if no improvement for ``patience`` epochs.
    """
    print(
        f"\n  {'Epoch':>5}  {'Test Acc':>9}  {'Best':>9}  {'σ_v(AGVI)':>10}  {'mean λ':>8}"
        f"  {'Time':>7}  {'Patience':>8}"
    )
    print("  " + "─" * 68)
    best_acc = 0.0
    best_epoch = 0
    wait = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]
        ep_lam_sum, ep_sv_sum, ep_lam_n = 0.0, 0.0, 0

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            # sigma_v=0.0 ignored — AGVI kernel uses predicted noise from odd outputs
            y_pred_mu, _ = net.step(xb, y_s[i : i + batch_size], 0.0, online=online)
            ep_lam_sum += online.last_lambda
            ep_sv_sum += float(y_pred_mu[:, 1::2].mean().item() ** 0.5)  # mean AGVI σ_v
            ep_lam_n += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_test_labels)
        mean_lam = ep_lam_sum / max(ep_lam_n, 1)
        mean_sv = ep_sv_sum / max(ep_lam_n, 1)

        improved = acc > best_acc
        if improved:
            best_acc = acc
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        marker = " *" if improved else ""
        print(
            f"  {epoch:5d}  {acc * 100:8.2f}%  {best_acc * 100:8.2f}%  "
            f"{mean_sv:10.4f}  {mean_lam:8.4f}  {wall:6.2f}s  "
            f"{wait:3d}/{patience}{marker}"
        )
        run.append_metrics(
            epoch, test_acc=acc, sigma_v_agvi=mean_sv, mean_lambda=mean_lam, wall_s=wall
        )

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

        if wait >= patience:
            print(f"\n  Early stopping: no accuracy improvement for {patience} epochs")
            print(f"  (best was epoch {best_epoch}: {best_acc * 100:.2f}%)")
            break

    print("  " + "─" * 68)
    print(f"  Best test accuracy: {best_acc * 100:.2f}%  (epoch {best_epoch})")
    return best_acc


def main(
    n_epochs: int = 100,
    batch_size: int = 128,
    sigma2_obs: float = _SIGMA2_OBS_DEFAULT,  # seed σ_v² (also AGVI noise-head init → J=½)
    augment: bool = True,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 15,
) -> float:
    """CIFAR-10 ResNet-18 with unified parameter-free online calibration.

    Args:
        sigma2_obs: Seed observation-noise variance (learned online thereafter).
        augment:    Random flip + crop augmentation each batch.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 60)
    print("  CIFAR-10 — ResNet-18 — unified parameter-free calibration")
    print("=" * 60)
    if dev.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, x_test, y_test_labels = load_cifar10(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    config: dict = {
        "dataset": "cifar10",
        "arch": "resnet18_agvi",
        "optimizer": "tagi-calibrated-agvi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma2_obs_seed": sigma2_obs,
        "augment": augment,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "patience": patience,
    }
    run = RunDir("cifar10", "resnet18_agvi_calibrated", "tagi")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    net = build_resnet18_agvi(dev)

    # ── Unified operator T (init): signal=1, balanced gains, global budget=σ_v² ──
    # Analytic closed-form solution — tractable regardless of output head size.
    # calibrate() sets the GLOBAL gain c=σ_v²/A(net) so the network OUTPUT epistemic
    # variance Sz_final = σ_v² for ANY σ_v² (A(net) is architectural, σ_v²-independent).
    calibrate(net, sigma2_obs=sigma2_obs, metric="analytic")

    # Align the AGVI noise head to the same budget: E[V²]_init = σ_v². Without this
    # the odd outputs default to softplus(0)=log(2), so for σ_v²≠log(2) the OUTPUT
    # Kalman gain J = Var[z]/(Var[z]+E[V²]) collapses (e.g. σ_v²=0.001 → J≈0.002).
    _init_noise_head(net.layers[-2], sigma2_obs)
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Online operator T: surprise-driven λ + Muon projection ──
    # sigma_v2 kept as the re-inflation budget seed; AGVI head predicts noise directly.
    online = OnlineCalibration(
        mode="surprise",
        sigma_v_mode="heteroscedastic",
        sigma_v2=sigma2_obs,
        project=True,
        calib_A=net._calib_A,  # global budget: online c = σ_v² / A(net)
    )
    print_calibration_properties(net, online, sigma2_obs)

    best_acc = train(
        net,
        online,
        x_train,
        y_train_oh,
        x_test,
        y_test_labels,
        n_epochs,
        batch_size,
        augment,
        dev,
        run,
        config,
        patience=patience,
    )
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 — parameter-free calibration")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size (reduce if OOM)"
    )
    parser.add_argument(
        "--sigma2_obs",
        type=float,
        default=_SIGMA2_OBS_DEFAULT,
        help="σ_v² seed (noise head aligned to it via _init_noise_head → output J=½ for any value)",
    )
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    main(**vars(args))
