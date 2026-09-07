"""
Hierarchical probit calibration on frozen CIFAR-10 features, two panels.

Left: test NLL against the held-out calibration size, for one global gain, one
gain per tree level, and one per internal node, each shown with the gain
uncertainty retained (solid) and with the fitted mean read as a point value
(dashed). The uncalibrated unit-gain head is the horizontal reference.

Right: the fitted posterior standard deviation of every log-gain group against
the number of calibration node-visits it received, over every sharing structure
and calibration size, with the Fisher slope for reference.

Reads runs/last_layer/hsm_calibration/cifar10/records.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=8)
plt.rc("axes", labelsize=8)
plt.rc("xtick", labelsize=7)
plt.rc("ytick", labelsize=7)
plt.rc("legend", fontsize=6)
plt.rcParams["text.latex.preamble"] = r"\usepackage{times}"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECORDS = REPOSITORY_ROOT / "runs/last_layer/hsm_calibration/cifar10/records.json"
FIGURE_DIR = Path(__file__).resolve().parent / "report_assets"
FIGURE_STEM = "hsm_calibration_cifar10"

FIG_WIDTH = 6.75
FIG_HEIGHT = 2.5

SHARING_COLORS = {"global": "#2E86AB", "level": "#F39C12", "node": "#E94F37"}
SHARING_LABELS = {"global": "global gain", "level": "per level", "node": "per node"}
REFERENCE_COLOR = "#2C3E50"
FISHER_COLOR = "#0B5345"
MARKER = "o"
MARKER_SIZE = 3.0
LINE_WIDTH = 1.1
CAP_SIZE = 2.0

# The point readouts of the finer groupings overshoot by a factor of two at the
# smallest calibration size. Clipping keeps the resolved region legible; the
# off-scale values are annotated instead of dropped.
NLL_LIMITS = (0.183, 0.245)

# ============================================================================
# END OF CONFIGURATION
# ============================================================================


def load_records() -> list[dict]:
    if not RECORDS.exists():
        raise SystemExit(f"run the study first: {RECORDS} is missing")
    return json.loads(RECORDS.read_text())


def nll_curve(records: list[dict], arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return calibration sizes with the mean and spread of the test NLL."""

    grouped: dict[int, list[float]] = {}
    for record in records:
        if record["arm"] == arm:
            grouped.setdefault(record["calibration_size"], []).append(record["test_nll"])
    sizes = np.array(sorted(grouped), dtype=float)
    centres = np.array([np.mean(grouped[int(size)]) for size in sizes])
    spreads = np.array([np.std(grouped[int(size)], ddof=1) if len(grouped[int(size)]) > 1 else 0.0
                        for size in sizes])
    return sizes, centres, spreads


def group_identifiability(records: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-visit posterior deviations for every sharing structure."""

    collected: dict[str, tuple[list[float], list[float]]] = {}
    for record in records:
        if record["method"] != "grid" or not record["gain_uncertainty"]:
            continue
        visits, variances = record.get("visits"), record.get("log_gain_variance")
        if not visits or not variances:
            continue
        bucket = collected.setdefault(record["sharing"], ([], []))
        bucket[0].extend(float(count) for count in visits)
        bucket[1].extend(float(value) ** 0.5 for value in variances)
    return {
        name: (np.array(counts), np.array(deviations))
        for name, (counts, deviations) in collected.items()
    }


def annotate_off_scale(
    ax, sizes: np.ndarray, centres: np.ndarray, color: str, rank: int
) -> None:
    """Mark points above the clipped range with an arrow and their value.

    Several arms overshoot at the same calibration size, so the label is
    staggered by the arm's position in the drawing order.
    """

    span = NLL_LIMITS[1] - NLL_LIMITS[0]
    for size, centre in zip(sizes, centres):
        if centre <= NLL_LIMITS[1]:
            continue
        ax.annotate(
            f"{centre:.2f}",
            xy=(size, NLL_LIMITS[1]),
            xytext=(size * 1.3, NLL_LIMITS[1] - (0.03 + 0.07 * rank) * span),
            color=color,
            fontsize=6,
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 0.8,
                "shrinkA": 0.0,
                "shrinkB": 1.0,
            },
        )


def draw_calibration_curves(ax, records: list[dict]) -> None:
    """Plot test NLL against calibration size for every sharing structure."""

    baseline = nll_curve(records, "uncalibrated")[1]
    ax.axhline(
        float(baseline.mean()),
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=LINE_WIDTH,
        label="uncalibrated",
        zorder=1,
    )
    for rank, (sharing, color) in enumerate(SHARING_COLORS.items()):
        for arm, style in ((f"hsm_{sharing}", "-"), (f"hsm_{sharing}_point", "--")):
            sizes, centres, spreads = nll_curve(records, arm)
            if sizes.size == 0:
                continue
            ax.errorbar(
                sizes,
                centres,
                yerr=spreads,
                color=color,
                linestyle=style,
                linewidth=LINE_WIDTH,
                marker=MARKER,
                markersize=MARKER_SIZE,
                capsize=CAP_SIZE,
                label=f"{SHARING_LABELS[sharing]}" + ("" if style == "-" else ", point"),
                zorder=3,
            )
            annotate_off_scale(ax, sizes, centres, color, rank)
    ax.set_xscale("log")
    ax.set_ylim(*NLL_LIMITS)
    ax.set_xlabel(r"calibration examples $n_{\mathrm{cal}}$")
    ax.set_ylabel(r"test NLL (nats)")
    ax.set_title(r"\textbf{Keeping the gain uncertainty}", fontsize=9)
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=1.0)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_identifiability(ax, records: list[dict]) -> None:
    """Plot the posterior deviation of every gain group against its visits."""

    for sharing, (counts, deviations) in group_identifiability(records).items():
        keep = counts > 0
        ax.scatter(
            counts[keep],
            deviations[keep],
            s=6.0,
            color=SHARING_COLORS[sharing],
            alpha=0.55,
            edgecolors="none",
            label=SHARING_LABELS[sharing],
            zorder=3,
        )
    reference = np.array([30.0, 4.0e4])
    anchor = 3.0
    ax.plot(
        reference,
        anchor / np.sqrt(reference),
        color=FISHER_COLOR,
        linestyle="-.",
        linewidth=LINE_WIDTH,
        label=r"$\propto V_r^{-1/2}$",
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"calibration node-visits $|\mathcal{V}_r|$")
    ax.set_ylabel(r"posterior $\sqrt{q_r}$")
    ax.set_title(r"\textbf{Identifiability per gain group}", fontsize=9)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figure() -> None:
    """Assemble and save both panels."""

    records = load_records()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    draw_calibration_curves(axes[0], records)
    draw_identifiability(axes[1], records)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"{FIGURE_STEM}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)
    print(f"wrote {FIGURE_DIR / FIGURE_STEM}.pdf and .png")


if __name__ == "__main__":
    create_figure()
