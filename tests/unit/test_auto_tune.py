"""Smoke test for auto_tune on CIFAR-10 3-CNN."""

import numpy as np
import torch
from torchvision import datasets, transforms

from triton_tagi import Sequential
from triton_tagi.auto_tune import auto_tune, find_best_gain
from triton_tagi.layers import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    Remax,
)

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Network builder (2-arg signature) ──
def build_net(gain_w, gain_b):
    layers = [
        Conv2D(3, 32, 5, stride=1, padding=2, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(32, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),
        Conv2D(32, 64, 5, stride=1, padding=2, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),
        Conv2D(64, 64, 5, stride=1, padding=2, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),
        Flatten(),
        Linear(64 * 4 * 4, 256, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        Linear(256, 10, device=DEVICE, gain_w=gain_w, gain_b=gain_b),
        # Bernoulli(n_gh=32),
        Remax(),
    ]
    return Sequential(layers, device=DEVICE)


# ── Load a subset of CIFAR-10 ──
print("Loading CIFAR-10 subset...")
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
train_ds = datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10("data", train=False, download=True, transform=transform)

# Use small subsets for speed
N_TRAIN = 1024
N_TEST = 500

x_train = torch.stack([train_ds[i][0] for i in range(N_TRAIN)]).to(DEVICE)
y_labels_train = torch.tensor([train_ds[i][1] for i in range(N_TRAIN)], device=DEVICE)
y_train_oh = torch.zeros(N_TRAIN, 10, device=DEVICE)
y_train_oh.scatter_(1, y_labels_train.unsqueeze(1), 1.0)

x_test = torch.stack([test_ds[i][0] for i in range(N_TEST)]).to(DEVICE)
y_test = torch.tensor([test_ds[i][1] for i in range(N_TEST)], device=DEVICE)

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# ── Test 1: find_best_gain only ──
print("\n" + "=" * 60)
print("  TEST 1: find_best_gain (forward-only)")
print("=" * 60)
gr = find_best_gain(
    builder_fn=build_net,
    x_probe=x_train[:256],
    gains_w=[0.01, 0.1, 0.5, 1.0, 2.0],
    bias_factors=[1.0],
    target_var=1.0,
    refine=False,
    verbose=True,
)
print(f"  Result: gain_w={gr.best_gain_w}, gain_b={gr.best_gain_b}")
assert gr.best_score.verdict in ("GOOD", "OK"), f"Unexpected verdict: {gr.best_score.verdict}"

# ── Test 2: Full auto_tune ──
print("\n" + "=" * 60)
print("  TEST 2: Full auto_tune")
print("=" * 60)
result = auto_tune(
    builder_fn=build_net,
    x_probe=x_train,
    y_probe=y_train_oh,
    gains_w=[0.1, 0.5, 1.0, 2.0],
    bias_factors=[0.5, 1.0],
    sigma_vs=[0.001, 0.01, 0.1, 1.0],
    n_steps=20,
    batch_size=64,
    x_eval=x_test,
    y_eval=y_test,
    probe_size=256,
    verbose=True,
    plot_filename="auto_tune_test.png",
)

print(
    f"\n  Final: gain_w={result.gain_w:.4f}, gain_b={result.gain_b:.4f}, "
    f"sigma_v={result.sigma_v:.4e}"
)

print("\n  All tests passed!")
