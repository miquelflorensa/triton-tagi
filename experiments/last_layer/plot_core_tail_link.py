"""Figure: the Core-Tail link against softmax and the variance-matched probit.

All three links are visually indistinguishable when plotted directly, so the
left panel plots the deviation from softmax instead: Core-Tail follows the
probit through the competitive core and returns to softmax outside it, while
the probit does not. The middle panel is the binary slope, where the match at
a tie is exact by construction and not fitted. The right panel is the far
tail, where Core-Tail coincides with softmax while the probit's quadratic
decay diverges from both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import A_STAR, core_tail_log_probabilities  # noqa: E402

# ---------------------------------------------------------------- configuration
PT = 1.0 / 72.27
JOUR_SIZES = {"PRD": {"onecol": 246.0 * PT, "twocol": 510.0 * PT}}
FIGURE_WIDTH = JOUR_SIZES["PRD"]["twocol"]
FIGURE_HEIGHT = 1.85

COLOR_SOFTMAX = "#4A90E2"
COLOR_PROBIT = "#F5A623"
COLOR_CORE_TAIL = "#9B59B6"
COLOR_GUIDE = "#2C3E50"

LINE_WIDTH = 1.3
GUIDE_WIDTH = 0.7
TITLE_SIZE = 8
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 6.5

CORE_RANGE = 6.0
TAIL_RANGE = 25.0
NUM_POINTS = 2001
NUM_ITERATIONS = 64
TAIL_CLASSES = 4

OUTPUT_DIR = Path(__file__).resolve().parent / "report_assets"
STEM = "core_tail_link"

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=LABEL_SIZE)
plt.rcParams["text.latex.preamble"] = r"\usepackage{times}\usepackage{amsmath}"
plt.rc("axes", labelsize=LABEL_SIZE)
plt.rc("xtick", labelsize=TICK_SIZE)
plt.rc("ytick", labelsize=TICK_SIZE)
plt.rc("legend", fontsize=LEGEND_SIZE)


# ---------------------------------------------------------------------- links
def binary_core_tail(difference: np.ndarray) -> np.ndarray:
    """Return the Core-Tail probability of the leading class in a binary tie."""

    logits = torch.stack(
        [
            torch.from_numpy(difference).double(),
            torch.zeros(difference.shape[0], dtype=torch.float64),
        ],
        dim=-1,
    )
    return (
        core_tail_log_probabilities(logits, num_iterations=NUM_ITERATIONS)[:, 0]
        .exp()
        .numpy()
    )


def binary_softmax(difference: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-difference))


def binary_probit(difference: np.ndarray) -> np.ndarray:
    """Gaussian decision noise matched to the Gumbel difference variance."""

    from scipy.special import ndtr

    return ndtr(np.sqrt(3.0) / np.pi * difference)


def multiclass_trailing_nll(margin: np.ndarray) -> dict[str, np.ndarray]:
    """Return the trailing class's negative log probability against its margin."""

    logits = np.zeros((margin.shape[0], TAIL_CLASSES))
    logits[:, 0] = margin
    tensor = torch.from_numpy(logits).double()
    core_tail = -core_tail_log_probabilities(tensor, num_iterations=NUM_ITERATIONS)[
        :, 1
    ].numpy()
    softmax = -torch.log_softmax(tensor, dim=-1)[:, 1].numpy()
    # The Gaussian argmax event decays quadratically; the standard-normal
    # log-tail -log Phi(-d) ~ d^2 / 2 with the variance-matched scale.
    scaled = np.sqrt(3.0) / np.pi * margin
    from scipy.special import log_ndtr

    probit = -log_ndtr(-scaled)
    return {"softmax": softmax, "core_tail": core_tail, "probit": probit}


# --------------------------------------------------------------------- panels
def draw_core(axis) -> None:
    difference = np.linspace(-CORE_RANGE, CORE_RANGE, NUM_POINTS)
    softmax = binary_softmax(difference)
    axis.axhline(0.0, color=COLOR_SOFTMAX, lw=LINE_WIDTH, label=r"softmax (Gumbel)")
    axis.plot(
        difference,
        binary_probit(difference) - softmax,
        color=COLOR_PROBIT,
        lw=LINE_WIDTH,
        ls="--",
        label=r"probit, variance matched",
    )
    axis.plot(
        difference,
        binary_core_tail(difference) - softmax,
        color=COLOR_CORE_TAIL,
        lw=LINE_WIDTH,
        label=r"Core--Tail",
    )
    axis.set_xlabel(r"logit difference $d$")
    axis.set_ylabel(r"$p_1(d)-\mathrm{softmax}(d)_1$")
    axis.set_title(r"\textbf{Deviation from softmax}", fontsize=TITLE_SIZE)
    axis.legend(loc="lower right", frameon=False)


def draw_slope(axis) -> None:
    difference = np.linspace(-CORE_RANGE, CORE_RANGE, NUM_POINTS)
    step = difference[1] - difference[0]
    for values, color, style, label in (
        (binary_softmax(difference), COLOR_SOFTMAX, "-", r"softmax (Gumbel)"),
        (binary_probit(difference), COLOR_PROBIT, "--", r"probit, variance matched"),
        (binary_core_tail(difference), COLOR_CORE_TAIL, "-", r"Core--Tail"),
    ):
        axis.plot(
            difference,
            np.gradient(values, step),
            color=color,
            lw=LINE_WIDTH,
            ls=style,
            label=label,
        )
    tie_slope = 1.0 / (4.0 + 2.0 * A_STAR)
    axis.axhline(tie_slope, color=COLOR_GUIDE, lw=GUIDE_WIDTH, ls=":")
    axis.annotate(
        r"$\dfrac{1}{4+2a_\star}=\dfrac{\sqrt3}{\pi\sqrt{2\pi}}$",
        xy=(-CORE_RANGE * 0.92, tie_slope),
        xytext=(-CORE_RANGE * 0.92, tie_slope * 1.12),
        fontsize=LEGEND_SIZE,
        color=COLOR_GUIDE,
        va="bottom",
    )
    axis.set_ylim(top=0.345)
    axis.set_xlabel(r"logit difference $d$")
    axis.set_ylabel(r"$\mathrm{d}p_1/\mathrm{d}d$")
    axis.set_title(r"\textbf{Sensitivity in the core}", fontsize=TITLE_SIZE)


def draw_tail(axis) -> None:
    margin = np.linspace(0.0, TAIL_RANGE, NUM_POINTS)
    curves = multiclass_trailing_nll(margin)
    axis.plot(
        margin,
        curves["softmax"],
        color=COLOR_SOFTMAX,
        lw=3.0 * LINE_WIDTH,
        alpha=0.55,
        label=r"softmax (Gumbel)",
    )
    axis.plot(
        margin,
        curves["probit"],
        color=COLOR_PROBIT,
        lw=LINE_WIDTH,
        ls="--",
        label=r"probit, variance matched",
    )
    axis.plot(
        margin,
        curves["core_tail"],
        color=COLOR_CORE_TAIL,
        lw=LINE_WIDTH,
        label=r"Core--Tail",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"margin $\Delta=z_w-z_c$")
    axis.set_ylabel(r"$-\log p_c$")
    axis.set_title(r"\textbf{Tail of the trailing class}", fontsize=TITLE_SIZE)
    axis.legend(loc="upper left", frameon=False)
    inset = axis.inset_axes([0.52, 0.14, 0.44, 0.38])
    inset.plot(
        margin, curves["core_tail"] - curves["softmax"], color=COLOR_CORE_TAIL, lw=LINE_WIDTH
    )
    inset.axhline(0.0, color=COLOR_GUIDE, lw=GUIDE_WIDTH, ls=":")
    inset.set_title(r"Core--Tail $-$ softmax", fontsize=LEGEND_SIZE, pad=2)
    inset.set_xlabel(r"$\Delta$", fontsize=LEGEND_SIZE, labelpad=1)
    inset.tick_params(labelsize=LEGEND_SIZE - 1)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)


def main() -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True
    )
    draw_core(axes[0])
    draw_slope(axes[1])
    draw_tail(axes[2])
    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.grid(True, lw=0.3, alpha=0.35)
        axis.set_axisbelow(True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT_DIR / f"{STEM}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_DIR / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
