"""
1-D Regression with epistemic uncertainty — triton-tagi example.

Trains a 2-layer MLP on 1-D data, then plots mean predictions and 3σ
predictive uncertainty bands.  Training data is sampled from two disjoint
intervals, leaving a gap where epistemic uncertainty grows visibly.

Usage:
    python examples/regression.py
    python examples/regression.py --n_epochs 100 --sigma_v 0.1
    python examples/regression.py --data_dir /path/to/cuTAGI/data/toy_example
    python examples/regression.py --help
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from triton_tagi import Linear, ReLU, Sequential
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def _true_fn(x: np.ndarray) -> np.ndarray:
    """Underlying function: y = x * sin(x)."""
    return x * np.sin(x)


def generate_data(
    n_train: int = 40,
    n_test: int = 200,
    noise_std: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 1-D regression data with a gap region.

    Training data is drawn from [0, 2] ∪ [4, 2π], leaving [2, 4] unobserved
    so that epistemic uncertainty visibly grows in that interval.

    Args:
        n_train:   Number of training points (split evenly across two intervals).
        n_test:    Number of test points (uniformly over [0, 2π]).
        noise_std: Standard deviation of additive Gaussian noise.
        seed:      Random seed for reproducibility.

    Returns:
        x_train, y_train, x_test, y_test as float32 arrays of shape (N, 1).
    """
    rng = np.random.default_rng(seed)
    half = n_train // 2
    x1 = rng.uniform(0.0, 2.0, size=half)
    x2 = rng.uniform(4.0, 2 * math.pi, size=n_train - half)
    x_tr = np.concatenate([x1, x2])
    y_tr = _true_fn(x_tr) + rng.normal(0.0, noise_std, size=len(x_tr))

    x_te = np.linspace(0.0, 2 * math.pi, n_test)
    y_te = _true_fn(x_te) + rng.normal(0.0, noise_std, size=n_test)

    def col(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1, 1).astype(np.float32)

    return col(x_tr), col(y_tr), col(x_te), col(y_te)


def load_csv_data(
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cuTAGI-format 1-D regression CSVs (header row, one value per line).

    Args:
        data_dir: Directory containing x_train_1D.csv, y_train_1D.csv,
                  x_test_1D.csv, y_test_1D.csv.

    Returns:
        x_train, y_train, x_test, y_test as float32 arrays of shape (N, 1).
    """
    def _read(path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32).reshape(-1, 1)

    d = Path(data_dir)
    return (
        _read(d / "x_train_1D.csv"),
        _read(d / "y_train_1D.csv"),
        _read(d / "x_test_1D.csv"),
        _read(d / "y_test_1D.csv"),
    )


def normalise(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Z-score standardisation using training statistics.

    Returns:
        Normalised arrays plus a stats tuple (x_mean, x_std, y_mean, y_std).
    """
    x_mean, x_std = x_tr.mean(), x_tr.std() + 1e-8
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    return (
        (x_tr - x_mean) / x_std,
        (y_tr - y_mean) / y_std,
        (x_te - x_mean) / x_std,
        (y_te - y_mean) / y_std,
        (x_mean, x_std, y_mean, y_std),
    )


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def log_likelihood(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_std: np.ndarray,
) -> float:
    """Mean Gaussian log-likelihood over the test set."""
    ll = -0.5 * np.log(2 * math.pi * pred_std**2) - 0.5 * ((y_true - y_pred) / pred_std) ** 2
    return float(ll.mean())


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train(
    net: Sequential,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    n_epochs: int,
    batch_size: int,
    sigma_v: float,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> None:
    rng = np.random.default_rng(42)
    n = len(x_tr)
    print(f"\n  {'Epoch':>5}  {'Train MSE':>10}  {'Time':>7}")
    print("  " + "─" * 28)

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        x_s, y_s = x_tr[perm], y_tr[perm]

        sq_errs = []
        for i in range(0, n, batch_size):
            xb = torch.tensor(x_s[i : i + batch_size], device=device)
            yb = torch.tensor(y_s[i : i + batch_size], device=device)
            mu_pred, _ = net.step(xb, yb, sigma_v)
            sq_errs.append(((mu_pred.cpu().numpy() - y_s[i : i + batch_size]) ** 2).mean())

        train_mse = float(np.mean(sq_errs))
        wall = time.perf_counter() - t0
        print(f"  {epoch:5d}  {train_mse:10.4f}  {wall:6.3f}s")
        run.append_metrics(epoch, train_mse=train_mse, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)


# ---------------------------------------------------------------------------
#  Evaluation and figure
# ---------------------------------------------------------------------------

def evaluate(
    net: Sequential,
    x_te: np.ndarray,
    y_te: np.ndarray,
    sigma_v: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return predicted means and stds (in normalised space)."""
    net.eval()
    mu_list, std_list = [], []
    for i in range(0, len(x_te), batch_size):
        xb = torch.tensor(x_te[i : i + batch_size], device=device)
        with torch.no_grad():
            mu, Sy = net.forward(xb)
        # Predictive variance = epistemic (Sy) + aleatoric (sigma_v^2)
        pred_var = Sy.cpu().numpy() + sigma_v**2
        mu_list.append(mu.cpu().numpy())
        std_list.append(np.sqrt(np.maximum(pred_var, 0.0)))
    net.train()
    return np.concatenate(mu_list), np.concatenate(std_list)


def save_figure(
    x_tr_orig: np.ndarray,
    y_tr_orig: np.ndarray,
    x_te_orig: np.ndarray,
    y_te_orig: np.ndarray,
    mu_pred_orig: np.ndarray,
    std_pred_orig: np.ndarray,
    run: RunDir,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure (pip install matplotlib)")
        return

    sort_idx = np.argsort(x_te_orig.ravel())
    x_p = x_te_orig.ravel()[sort_idx]
    mu_p = mu_pred_orig.ravel()[sort_idx]
    std_p = std_pred_orig.ravel()[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(x_p, mu_p - 3 * std_p, mu_p + 3 * std_p, alpha=0.15, color="C0", label="±3σ")
    ax.fill_between(x_p, mu_p - std_p, mu_p + std_p, alpha=0.3, color="C0", label="±1σ")
    ax.plot(x_p, mu_p, color="C0", linewidth=1.5, label="Prediction mean")
    ax.scatter(x_tr_orig.ravel(), y_tr_orig.ravel(), s=25, color="k", zorder=5, label="Training data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("TAGI 1-D Regression — epistemic uncertainty")
    ax.legend(fontsize=8)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(run.figures / f"predictions.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {run.figures}/")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 50,
    batch_size: int = 10,
    sigma_v: float = 0.2,
    hidden: int = 50,
    gain_w: float = 1.0,
    gain_b: float = 1.0,
    data_dir: str | None = None,
    checkpoint_interval: int = 10,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    print("=" * 56)
    print("  TAGI 1-D Regression — triton-tagi")
    print("=" * 56)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    if data_dir is not None:
        print(f"\n  Loading data from {data_dir}")
        x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = load_csv_data(data_dir)
    else:
        print("\n  Generating synthetic data (y = x·sin(x), gap in [2, 4])")
        x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = generate_data(seed=seed)

    x_tr, y_tr, x_te, y_te, stats = normalise(x_tr_raw, y_tr_raw, x_te_raw, y_te_raw)
    x_mean, x_std, y_mean, y_std = stats
    print(f"  Train: {len(x_tr)}  |  Test: {len(x_te)}")

    # ── Config ──
    config: dict = {
        "dataset": "regression_1d",
        "arch": f"mlp_1-{hidden}-1",
        "optimizer": "tagi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "hidden": hidden,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "triton_tagi_version": "0.1.0",
    }

    # ── RunDir ──
    run = RunDir("regression_1d", f"mlp_1-{hidden}-1", "tagi")
    run.save_config(config)
    print(f"\n  Run directory: {run.path}")

    # ── Network ──
    net = Sequential(
        [
            Linear(1, hidden, device=dev, gain_w=gain_w, gain_b=gain_b),
            ReLU(),
            Linear(hidden, 1, device=dev, gain_w=gain_w, gain_b=gain_b),
        ],
        device=dev,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  σ_v: {sigma_v}")

    # ── Train ──
    train(net, x_tr, y_tr, n_epochs, batch_size, sigma_v, dev, run, config)

    # ── Evaluate ──
    mu_norm, std_norm = evaluate(net, x_te, y_te, sigma_v, dev, batch_size)

    # Unstandardise
    mu_orig = mu_norm * y_std + y_mean
    std_orig = std_norm * y_std

    test_mse = mse(mu_orig, y_te_raw.ravel())
    test_ll = log_likelihood(mu_orig, y_te_raw.ravel(), std_orig)

    print("\n  " + "─" * 40)
    print(f"  Test MSE           : {test_mse:.4f}")
    print(f"  Test log-likelihood: {test_ll:.4f}")
    print("  " + "─" * 40)

    # ── Figure ──
    save_figure(x_tr_raw, y_tr_raw, x_te_raw, y_te_raw, mu_orig, std_orig, run)

    print(f"\n  Results in: {run.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAGI 1-D regression demo with epistemic uncertainty"
    )
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sigma_v", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--gain_b", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to cuTAGI toy_example/ directory (optional)")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(**vars(args))
