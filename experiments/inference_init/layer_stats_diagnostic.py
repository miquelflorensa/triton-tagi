"""
Diagnostic: S and S2 summary statistics at each Linear layer.

Builds a 9-layer MLP (784 → [128]×8 → 10) with default He init,
runs one forward pass on the last MNIST batch, and prints the
layer-aggregate first and second moment statistics that IBI targets:

    S  = Σ_i Z_i           (sum of pre-activations)
    S2 = Σ_i Z_i²          (sum of squared pre-activations)

For each statistic we print its expected value and variance as seen
from the Gaussian moment propagation (mu_Z, S_Z from the forward pass):

    E[S]    = Σ mu_Z_i
    Var[S]  = Σ S_Z_i
    E[S2]   = Σ (mu_Z_i² + S_Z_i)
    Var[S2] = Σ (2 S_Z_i² + 4 S_Z_i mu_Z_i²)

Usage:
    python experiments/inference_init/layer_stats_diagnostic.py
    python experiments/inference_init/layer_stats_diagnostic.py --depth 7 --hidden 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triton_tagi import Linear, ReLU, Remax, Sequential, inference_init


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_first_and_last_batch(data_dir: Path, batch_size: int, device: torch.device):
    ds = datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    first = None
    for x, _ in loader:
        if first is None:
            first = x.view(-1, 784).to(device)
        last = x.view(-1, 784).to(device)
    return first, last


def build_mlp(depth: int, hidden: int, device: torch.device) -> Sequential:
    layers = []
    in_feat = 784
    for _ in range(depth - 1):
        layers.append(Linear(in_feat, hidden, device=device))
        layers.append(ReLU())
        in_feat = hidden
    layers.append(Linear(in_feat, 10, device=device))
    layers.append(Remax())
    return Sequential(layers, device=device)


@torch.no_grad()
def collect_stats(net: Sequential, x: torch.Tensor) -> list[dict]:
    """Forward pass collecting S/S2 moments at each Linear layer."""
    ma, Sa = x, torch.zeros_like(x)
    stats = []
    layer_idx = 0
    for layer in net.layers:
        ma, Sa = layer.forward(ma, Sa)
        if isinstance(layer, Linear):
            A = ma.shape[-1]
            mu_Z = ma.reshape(-1, A).mean(dim=0)
            S_Z  = Sa.reshape(-1, A).mean(dim=0)

            E_S    = mu_Z.sum().item()
            Var_S  = S_Z.sum().item()
            E_S2   = (mu_Z ** 2 + S_Z).sum().item()
            Var_S2 = (2 * S_Z ** 2 + 4 * S_Z * mu_Z ** 2).sum().item()

            stats.append({
                "layer": layer_idx,
                "width": A,
                "E[S]":    E_S,
                "Var[S]":  Var_S,
                "E[S2]":   E_S2,
                "Var[S2]": Var_S2,
            })
            layer_idx += 1
    return stats


@torch.no_grad()
def print_remax_probs(net: Sequential, x: torch.Tensor, n_classes: int = 10) -> None:
    ma, Sa = x, torch.zeros_like(x)
    for layer in net.layers:
        ma, Sa = layer.forward(ma, Sa)
    mean_probs = ma.mean(dim=0)
    print(f"  Remax mean probs: {[f'{p:.3f}' for p in mean_probs.tolist()]}")
    print(f"  Expected uniform: {1/n_classes:.3f}  |  max deviation: {(mean_probs - 1/n_classes).abs().max().item():.4f}")


def print_stats(stats: list[dict], title: str) -> None:
    w = 10
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"{'Layer':>5}  {'Width':>5}  {'E[S]':>{w}}  {'Var[S]':>{w}}  {'E[S2]':>{w}}  {'Var[S2]':>{w}}")
    print("-" * (5 + 2 + 5 + 2 + 4 * (w + 2)))
    for s in stats:
        print(
            f"{s['layer']:>5}  {s['width']:>5}"
            f"  {s['E[S]']:>{w}.3f}"
            f"  {s['Var[S]']:>{w}.3f}"
            f"  {s['E[S2]']:>{w}.3f}"
            f"  {s['Var[S2]']:>{w}.3f}"
        )


def main(depth: int = 9, hidden: int = 128, batch_size: int = 512,
         sigma_m: float = 1.0, sigma_z: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  depth={depth}  hidden={hidden}  batch={batch_size}")
    print(f"IBI targets: sigma_m={sigma_m}  sigma_z={sigma_z}")

    flat_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=flat_transform),
        batch_size=batch_size, shuffle=False,
    )

    x_first, x_last = load_first_and_last_batch(DATA_DIR, batch_size, device)
    net = build_mlp(depth, hidden, device)

    target_str = (f"targets: E[S]=0, Var[S]={hidden*sigma_z**2:.1f}, "
                  f"E[S2]={hidden*(sigma_m**2+sigma_z**2):.1f}, "
                  f"Var[S2]={hidden*(2*sigma_z**4 + 4*sigma_m**2*sigma_z**2):.1f}")

    print_stats(collect_stats(net, x_first), "He init — first batch")
    print_stats(collect_stats(net, x_last),  "He init — last batch")

    inference_init(net, loader, sigma_m=sigma_m, sigma_z=sigma_z)

    print(f"\n  ({target_str})")
    print_stats(collect_stats(net, x_first), "After IBI — first batch")
    print_stats(collect_stats(net, x_last),  "After IBI — last batch")

    print(f"\n{'─' * 60}")
    print("  Remax output (expect ~0.100 per class)")
    print(f"{'─' * 60}")
    print("  first batch:")
    print_remax_probs(net, x_first)
    print("  last batch:")
    print_remax_probs(net, x_last)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth",      type=int,   default=9)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--sigma_m",    type=float, default=1.0)
    parser.add_argument("--sigma_z",    type=float, default=0.5)
    args = parser.parse_args()
    main(args.depth, args.hidden, args.batch_size, args.sigma_m, args.sigma_z)
