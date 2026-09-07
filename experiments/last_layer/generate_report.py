#!/usr/bin/env python3
"""Generate the CIFAR frozen-last-layer research report and figures.

The script is intentionally read-only with respect to trained artifacts.  It
reads the confirmation histories and final evaluation reports, then writes the
human-readable report and its figures next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((EXPERIMENT_ROOT / "study.json").read_text())
ARTIFACT_ROOT = (
    REPOSITORY_ROOT
    / MANIFEST["paths"]["artifacts"]
    / MANIFEST["study_id"]
)
REPORT_PATH = EXPERIMENT_ROOT / "REPORT.md"
ASSET_ROOT = EXPERIMENT_ROOT / "report_assets"

METHODS = [
    "pytorch_softmax",
    "probit_ovr",
    "remax_lognormal",
    "remax_laplace_diag",
    "hrc",
    "categorical_tagiv",
    "hrc_tagiv",
]
TRAINED_METHODS = [method for method in METHODS if method != "pytorch_softmax"]
LABELS = {
    "pytorch_softmax": "PyTorch softmax",
    "probit_ovr": "Probit OVR",
    "remax_lognormal": "ReMax moment matching",
    "remax_laplace_diag": "ReMax Laplace (diagonal)",
    "hrc": "Hierarchical classifier (HRC)",
    "categorical_tagiv": "Categorical TAGI-V",
    "hrc_tagiv": "Hierarchical TAGI-V",
    "hrc_probit": "Unit-probit HRC",
}
SHORT_LABELS = {
    "pytorch_softmax": "Softmax",
    "probit_ovr": "Probit OVR",
    "remax_lognormal": "MM-ReMax",
    "remax_laplace_diag": "Laplace-ReMax",
    "hrc": "HRC",
    "categorical_tagiv": "Categorical TAGI-V",
    "hrc_tagiv": "HRC TAGI-V",
}
COLORS = {
    "pytorch_softmax": "#4C78A8",
    "probit_ovr": "#F5A623",
    "remax_lognormal": "#9B59B6",
    "remax_laplace_diag": "#7D3C98",
    "hrc": "#2CA02C",
    "categorical_tagiv": "#E45756",
    "hrc_tagiv": "#17A2B8",
}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def mean_std(
    summary: pd.DataFrame,
    head: str,
    metric: str,
    *,
    dataset: str = "cifar10",
    evaluation_kind: str | None = None,
) -> tuple[float, float, int]:
    kind = evaluation_kind or (
        "fixed_baseline" if head == "pytorch_softmax" else "validation_selected"
    )
    row = summary[
        (summary["dataset"] == dataset)
        & (summary["head"] == head)
        & (summary["evaluation_kind"] == kind)
        & (summary["metric"] == metric)
    ]
    if row.empty:
        return np.nan, np.nan, 0
    if len(row) != 1:
        raise ValueError(
            f"Expected one summary row for {dataset}/{head}/{kind}/{metric}, "
            f"got {len(row)}"
        )
    item = row.iloc[0]
    return float(item["mean"]), float(item["std"]), int(item["n"])


def format_metric(
    summary: pd.DataFrame,
    head: str,
    metric: str,
    *,
    dataset: str = "cifar10",
    evaluation_kind: str | None = None,
    percent: bool = False,
    digits: int = 3,
) -> str:
    mean, std, n = mean_std(
        summary,
        head,
        metric,
        dataset=dataset,
        evaluation_kind=evaluation_kind,
    )
    scale = 100.0 if percent else 1.0
    if np.isnan(mean):
        return "—"
    if n == 1 or np.isnan(std) or std == 0.0:
        return f"{scale * mean:.{digits}f}"
    return f"{scale * mean:.{digits}f} ± {scale * std:.{digits}f}"


def selected_evaluations(
    report: pd.DataFrame, dataset: str = "cifar10"
) -> pd.DataFrame:
    return report[
        (report["dataset"] == dataset)
        & report["evaluation_kind"].isin(["validation_selected", "fixed_baseline"])
    ].copy()


def load_histories(dataset: str = "cifar10") -> pd.DataFrame:
    root = ARTIFACT_ROOT / "heads" / "confirm" / dataset
    rows: list[dict[str, float | int | str]] = []
    for history_path in sorted(root.glob("*/*/history.json")):
        config = json.loads((history_path.parent / "config.json").read_text())
        for record in json.loads(history_path.read_text()):
            rows.append({"head": config["head"], "seed": config["seed"], **record})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No confirmation histories found under {root}")
    return frame


def load_corruption_rows(
    report: pd.DataFrame, dataset: str = "cifar10"
) -> pd.DataFrame:
    evaluation_root = ARTIFACT_ROOT / "evaluations" / dataset
    rows: list[dict[str, float | int | str]] = []
    for item in selected_evaluations(report, dataset).itertuples():
        if item.head == "pytorch_softmax":
            path = evaluation_root / "pytorch_softmax.json"
            seed = -1
        else:
            checkpoint = Path(item.checkpoint)
            path = (
                evaluation_root
                / item.head
                / checkpoint.parents[1].name
                / f"{checkpoint.stem}.json"
            )
            seed = int(item.seed)
        payload = json.loads(path.read_text())
        for name, result in payload["corruptions"].items():
            corruption, severity_text = name.rsplit("_s", 1)
            classification = result["shift_classification"]
            ood = result["ood"]
            rows.append(
                {
                    "head": item.head,
                    "seed": seed,
                    "corruption": corruption,
                    "severity": int(severity_text),
                    "accuracy": classification["accuracy"],
                    "nll": classification["nll"],
                    "brier": classification["brier"],
                    "ece": classification["ece"],
                    "entropy_auroc": ood["entropy"]["auroc"],
                    "entropy_fpr95": ood["entropy"]["fpr95"],
                    "maxprob_auroc": ood["negative_max_probability"]["auroc"],
                    "native_auroc": ood.get("native_epistemic", {}).get("auroc", np.nan),
                }
            )
    return pd.DataFrame(rows)


def plot_histories(histories: pd.DataFrame, selected_epochs: dict[str, list[int]]) -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )

    normalized = histories.copy()
    initial = normalized[normalized["epoch"] == 0][
        ["head", "seed", "epistemic_mean"]
    ].rename(columns={"epistemic_mean": "epistemic_initial"})
    normalized = normalized.merge(initial, on=["head", "seed"], how="left")
    normalized["epistemic_relative"] = (
        normalized["epistemic_mean"] / normalized["epistemic_initial"]
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for head in TRAINED_METHODS:
        group = histories[histories["head"] == head]
        stats = group.groupby("epoch")["epistemic_mean"].agg(["mean", "std"])
        x = stats.index.to_numpy()
        y = stats["mean"].to_numpy()
        sd = stats["std"].fillna(0.0).to_numpy()
        color = COLORS[head]
        ax.plot(x, y, color=color, linewidth=2, label=SHORT_LABELS[head])
        ax.fill_between(x, np.maximum(y - sd, 1e-12), y + sd, color=color, alpha=0.12)
        selection = int(round(float(np.median(selected_epochs[head]))))
        selected_y = float(stats.loc[selection, "mean"])
        ax.scatter([selection], [selected_y], color=color, edgecolor="white", s=44, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Last-layer training epoch")
    ax.set_ylabel("Mean predictive epistemic variance")
    ax.set_title("Absolute epistemic uncertainty during last-layer training")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "epistemic_uncertainty_absolute.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for head in TRAINED_METHODS:
        group = normalized[normalized["head"] == head]
        stats = group.groupby("epoch")["epistemic_relative"].agg(["mean", "std"])
        x = stats.index.to_numpy()
        y = stats["mean"].to_numpy()
        sd = stats["std"].fillna(0.0).to_numpy()
        color = COLORS[head]
        ax.plot(x, y, color=color, linewidth=2, label=SHORT_LABELS[head])
        ax.fill_between(x, np.maximum(y - sd, 1e-12), y + sd, color=color, alpha=0.12)
        selection = int(round(float(np.median(selected_epochs[head]))))
        selected_y = float(stats.loc[selection, "mean"])
        ax.scatter([selection], [selected_y], color=color, edgecolor="white", s=44, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Last-layer training epoch")
    ax.set_ylabel(r"Relative epistemic variance $U_t/U_0$")
    ax.set_title("Posterior contraction relative to initialization")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "epistemic_uncertainty_relative.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for head in TRAINED_METHODS:
        group = histories[(histories["head"] == head) & (histories["epoch"] >= 1)]
        color = COLORS[head]
        for ax, metric, label in (
            (axes[0], "val_nll", "Validation NLL"),
            (axes[1], "val_ece", "Validation ECE"),
        ):
            stats = group.groupby("epoch")[metric].agg(["mean", "std"])
            x = stats.index.to_numpy()
            y = stats["mean"].to_numpy()
            sd = stats["std"].fillna(0.0).to_numpy()
            ax.plot(x, y, color=color, linewidth=1.8, label=SHORT_LABELS[head])
            ax.fill_between(x, np.maximum(y - sd, 0.0), y + sd, color=color, alpha=0.10)
            selection = int(round(float(np.median(selected_epochs[head]))))
            ax.scatter(
                [selection],
                [float(stats.loc[selection, "mean"])],
                color=color,
                edgecolor="white",
                s=38,
                zorder=3,
            )
            ax.set_xlabel("Last-layer training epoch")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.22)
    axes[0].set_ylim(0.14, 0.34)
    axes[1].set_ylim(0.0, 0.06)
    axes[0].set_title("Proper-score dynamics")
    axes[1].set_title("Calibration dynamics")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(ASSET_ROOT / "validation_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def plot_aleatoric_histories(
    histories: pd.DataFrame, selected_epochs: dict[str, list[int]]
) -> None:
    """Plot the learned positive variance channels of the two TAGI-V heads."""

    heads = ["categorical_tagiv", "hrc_tagiv"]
    frame = histories[histories["head"].isin(heads)].dropna(subset=["aleatoric_mean"])
    initial = frame[frame["epoch"] == 0][
        ["head", "seed", "aleatoric_mean"]
    ].rename(columns={"aleatoric_mean": "aleatoric_initial"})
    frame = frame.merge(initial, on=["head", "seed"], how="left")
    frame["aleatoric_relative"] = (
        frame["aleatoric_mean"] / frame["aleatoric_initial"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for head in heads:
        group = frame[frame["head"] == head]
        color = COLORS[head]
        for ax, metric, ylabel in (
            (axes[0], "aleatoric_mean", "Mean learned aleatoric variance"),
            (axes[1], "aleatoric_relative", r"Relative aleatoric variance $A_t/A_0$"),
        ):
            stats = group.groupby("epoch")[metric].agg(["mean", "std"])
            x = stats.index.to_numpy()
            y = stats["mean"].to_numpy()
            sd = stats["std"].fillna(0.0).to_numpy()
            ax.plot(x, y, color=color, linewidth=2.2, label=SHORT_LABELS[head])
            ax.fill_between(x, np.maximum(y - sd, 0.0), y + sd, color=color, alpha=0.14)
            selection = int(round(float(np.median(selected_epochs[head]))))
            ax.scatter(
                [selection],
                [float(stats.loc[selection, "mean"])],
                color=color,
                edgecolor="white",
                s=44,
                zorder=3,
            )
            ax.set_xlabel("Last-layer training epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22)
    axes[0].set_title("Absolute TAGI-V aleatoric uncertainty")
    axes[1].set_title("Aleatoric change from initialization")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(ASSET_ROOT / "tagiv_aleatoric_uncertainty.png", bbox_inches="tight")
    plt.close(fig)


def _predictive_entropy(probabilities):
    probabilities = probabilities.clamp_min(1e-12)
    return -(probabilities * probabilities.log()).sum(dim=1)


def plot_svhn_entropy_histograms(
    report: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    device: str = "cuda",
    batch_size: int = 2048,
) -> None:
    """Compare clean and SVHN predictive-entropy distributions.

    For trained heads, entropy is averaged per example across the five
    validation-selected last-layer seeds before plotting.
    """

    import torch

    from triton_tagi.cifar_study import load_feature_shard
    from triton_tagi.classification import TAGILastLayerClassifier

    feature_root = ARTIFACT_ROOT / "features" / "cifar10"
    clean = load_feature_shard(feature_root / "test.pt")
    svhn = load_feature_shard(feature_root / "svhn.pt")

    def classifier_entropy(classifier, features):
        parts = []
        for start in range(0, features.shape[0], batch_size):
            prediction = classifier.predict(
                features[start : start + batch_size], sigma_v=classifier.sigma_v
            )
            parts.append(_predictive_entropy(prediction.probabilities).cpu())
        return torch.cat(parts).numpy()

    score_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    score_pairs["pytorch_softmax"] = (
        _predictive_entropy(torch.softmax(clean["logits"], dim=1)).numpy(),
        _predictive_entropy(torch.softmax(svhn["logits"], dim=1)).numpy(),
    )

    chosen = selected_evaluations(report)
    for head in TRAINED_METHODS:
        clean_seeds = []
        svhn_seeds = []
        rows = chosen[chosen["head"] == head].sort_values("seed")
        for row in rows.itertuples():
            print(f"entropy histogram {head} seed={int(row.seed)} epoch={int(row.epoch)}")
            classifier, _ = TAGILastLayerClassifier.load(row.checkpoint, device=device)
            clean_seeds.append(classifier_entropy(classifier, clean["features"]))
            svhn_seeds.append(classifier_entropy(classifier, svhn["features"]))
            del classifier
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        score_pairs[head] = (
            np.mean(np.stack(clean_seeds), axis=0),
            np.mean(np.stack(svhn_seeds), axis=0),
        )

    bins = np.linspace(0.0, np.log(10.0), 61)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0), sharex=True, sharey=True)
    for ax, head in zip(axes.flat, METHODS, strict=False):
        clean_entropy, svhn_entropy = score_pairs[head]
        ax.hist(
            clean_entropy,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.24,
            color="#4C78A8",
            label="CIFAR-10",
        )
        ax.hist(
            svhn_entropy,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.24,
            color="#E45756",
            label="SVHN",
        )
        ax.hist(clean_entropy, bins=bins, density=True, histtype="step", linewidth=1.5, color="#4C78A8")
        ax.hist(svhn_entropy, bins=bins, density=True, histtype="step", linewidth=1.5, color="#E45756")
        auroc, _, _ = mean_std(summary, head, "clean_svhn_ood_entropy_auroc")
        ax.set_title(f"{SHORT_LABELS[head]}\nmean AUROC {100 * auroc:.2f}%", fontsize=10)
        ax.grid(True, alpha=0.18)
    axes.flat[-1].axis("off")
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Predictive entropy (nats)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.12), frameon=False)
    fig.suptitle("Clean CIFAR-10 versus SVHN predictive-entropy distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ASSET_ROOT / "svhn_entropy_histograms.png", bbox_inches="tight")
    plt.close(fig)



def build_dataset_tables(
    summary: pd.DataFrame, report: pd.DataFrame, dataset: str
) -> dict[str, str]:
    """Build the complete table set for one dataset without mixing datasets."""

    chosen = selected_evaluations(report, dataset)
    histories = load_histories(dataset)
    corruptions = load_corruption_rows(report, dataset)
    selected_configs = json.loads(
        (ARTIFACT_ROOT / "heads" / "confirm" / dataset / "selection.json").read_text()
    )["selected"]
    methods = METHODS + [head for head in selected_configs if head not in METHODS]
    trained_methods = [head for head in methods if head != "pytorch_softmax"]
    selected_epochs = {
        head: [int(value) for value in chosen[chosen["head"] == head]["epoch"]]
        for head in trained_methods
    }
    config_rows = []
    for head in trained_methods:
        config = selected_configs[head]["config"]
        if head in {"categorical_tagiv", "hrc_tagiv"}:
            noise = (
                f"learned; $\\bar v^2_0={config['v2bar_init']}$, "
                f"$S_{{b,v}}={config['v2bar_bias_var']}$"
            )
        elif head == "hrc_probit":
            noise = "unit probit scale"
        else:
            noise = f"$\\sigma_v={config['sigma_v']}$"
        epochs = selected_epochs[head]
        epoch_text = str(epochs[0]) if len(set(epochs)) == 1 else ", ".join(map(str, epochs))
        config_rows.append(
            [LABELS[head], noise, f"{config['gain_w']}/{config['gain_b']}", epoch_text]
        )

    def metric_table(
        columns: list[tuple[str, str, bool, int]],
    ) -> str:
        rows = []
        for head in methods:
            rows.append(
                [LABELS[head]]
                + [
                    format_metric(
                        summary,
                        head,
                        metric,
                        dataset=dataset,
                        percent=percent,
                        digits=digits,
                    )
                    for _, metric, percent, digits in columns
                ]
            )
        return markdown_table(["Method"] + [label for label, *_ in columns], rows)

    clean = metric_table(
        [
            ("Accuracy ↑", "clean_svhn_classification_accuracy", True, 2),
            ("NLL ↓", "clean_svhn_classification_nll", False, 3),
            ("Brier ↓", "clean_svhn_classification_brier", False, 3),
            ("ECE ↓", "clean_svhn_classification_ece", True, 2),
            ("Adaptive ECE ↓", "clean_svhn_classification_adaptive_ece", True, 2),
        ]
    )
    svhn = metric_table(
        [
            ("Entropy AUROC ↑", "clean_svhn_ood_entropy_auroc", True, 2),
            ("Entropy AUPR-OOD ↑", "clean_svhn_ood_entropy_aupr_ood", True, 2),
            ("Entropy FPR95 ↓", "clean_svhn_ood_entropy_fpr95", True, 2),
            (r"$1-\max p$ AUROC ↑", "clean_svhn_ood_negative_max_probability_auroc", True, 2),
            ("Native epi. AUROC ↑", "clean_svhn_ood_native_epistemic_auroc", True, 2),
            ("Native epi. FPR95 ↓", "clean_svhn_ood_native_epistemic_fpr95", True, 2),
        ]
    )
    corruption_classification = metric_table(
        [
            ("Accuracy ↑", "corruption_macro_shift_classification_accuracy", True, 2),
            ("NLL ↓", "corruption_macro_shift_classification_nll", False, 3),
            ("Brier ↓", "corruption_macro_shift_classification_brier", False, 3),
            ("ECE ↓", "corruption_macro_shift_classification_ece", True, 2),
            ("Adaptive ECE ↓", "corruption_macro_shift_classification_adaptive_ece", True, 2),
        ]
    )
    corruption_ood = metric_table(
        [
            ("Entropy AUROC ↑", "corruption_macro_ood_entropy_auroc", True, 2),
            ("Entropy AUPR-OOD ↑", "corruption_macro_ood_entropy_aupr_ood", True, 2),
            ("Entropy FPR95 ↓", "corruption_macro_ood_entropy_fpr95", True, 2),
            (r"$1-\max p$ AUROC ↑", "corruption_macro_ood_negative_max_probability_auroc", True, 2),
            ("Native epi. AUROC ↑", "corruption_macro_ood_native_epistemic_auroc", True, 2),
            ("Native epi. FPR95 ↓", "corruption_macro_ood_native_epistemic_fpr95", True, 2),
        ]
    )

    by_severity = (
        corruptions.groupby(["head", "seed", "severity"], as_index=False)
        .mean(numeric_only=True)
        .groupby(["head", "severity"], as_index=False)
        .mean(numeric_only=True)
    )

    def severity_table(metric: str) -> str:
        rows = []
        for head in methods:
            group = by_severity[by_severity["head"] == head].set_index("severity")
            rows.append(
                [LABELS[head]]
                + [f"{100 * float(group.loc[level, metric]):.2f}" for level in range(1, 6)]
            )
        return markdown_table(
            ["Method"] + [f"Severity {level}" for level in range(1, 6)], rows
        )

    def secondary_classification(prefix: str) -> str:
        return metric_table(
            [
                ("Top-5 acc. ↑", f"{prefix}_top5_accuracy", True, 2),
                ("Mean conf.", f"{prefix}_mean_confidence", True, 2),
                ("Classwise ECE ↓", f"{prefix}_classwise_ece", True, 2),
                ("AURC ↓", f"{prefix}_aurc", True, 2),
                ("Risk@80 ↓", f"{prefix}_risk_at_80_coverage", True, 2),
                ("Risk@90 ↓", f"{prefix}_risk_at_90_coverage", True, 2),
                ("Risk@95 ↓", f"{prefix}_risk_at_95_coverage", True, 2),
            ]
        )

    def complete_ood(prefix: str) -> str:
        score_names = {
            "entropy": "Entropy",
            "negative_max_probability": r"$1-\max p$",
            "native_epistemic": "Native epistemic",
        }
        rows = []
        for head in methods:
            for score, score_label in score_names.items():
                metric_prefix = f"{prefix}_{score}"
                mean, _, _ = mean_std(
                    summary, head, f"{metric_prefix}_auroc", dataset=dataset
                )
                if np.isnan(mean):
                    continue
                rows.append(
                    [
                        LABELS[head],
                        score_label,
                        format_metric(summary, head, f"{metric_prefix}_auroc", dataset=dataset, percent=True, digits=2),
                        format_metric(summary, head, f"{metric_prefix}_aupr_ood", dataset=dataset, percent=True, digits=2),
                        format_metric(summary, head, f"{metric_prefix}_aupr_id", dataset=dataset, percent=True, digits=2),
                        format_metric(summary, head, f"{metric_prefix}_fpr95", dataset=dataset, percent=True, digits=2),
                    ]
                )
        return markdown_table(
            ["Method", "Score", "AUROC ↑", "AUPR-OOD ↑", "AUPR-ID ↑", "FPR95 ↓"],
            rows,
        )

    def selected_vs_epoch200(prefix: str) -> str:
        rows = []
        metrics = [f"{prefix}_accuracy", f"{prefix}_nll", f"{prefix}_ece"]
        for head in trained_methods:
            values = []
            for metric in metrics:
                selected_mean, _, _ = mean_std(summary, head, metric, dataset=dataset)
                final_mean, _, _ = mean_std(
                    summary,
                    head,
                    metric,
                    dataset=dataset,
                    evaluation_kind="epoch_200",
                )
                if metric.endswith(("accuracy", "ece")):
                    values.extend([f"{100 * selected_mean:.2f}", f"{100 * final_mean:.2f}"])
                else:
                    values.extend([f"{selected_mean:.3f}", f"{final_mean:.3f}"])
            rows.append([LABELS[head], *values])
        return markdown_table(
            [
                "Method",
                "Selected acc.",
                "Epoch-200 acc.",
                "Selected NLL",
                "Epoch-200 NLL",
                "Selected ECE",
                "Epoch-200 ECE",
            ],
            rows,
        )

    convergence_rows = []
    for head in trained_methods:
        group = histories[histories["head"] == head]
        initial = group[group["epoch"] == 0]["epistemic_mean"]
        final = group[group["epoch"] == 200]["epistemic_mean"]
        ratios = []
        required = []
        statuses = []
        root = ARTIFACT_ROOT / "heads" / "confirm" / dataset / head
        for complete_path in sorted(root.glob("*/complete.json")):
            complete = json.loads(complete_path.read_text())
            ratios.append(complete["epistemic_convergence"]["u_final_over_u_initial"])
            statuses.append(complete["epistemic_convergence"]["status"])
            if complete.get("training_required_epoch") is not None:
                required.append(complete["training_required_epoch"])
        required_text = (
            "not reached"
            if not required
            else f"{np.mean(required):.1f} ({min(required)}–{max(required)})"
        )
        status = max(set(statuses), key=statuses.count)
        convergence_rows.append(
            [
                LABELS[head],
                required_text,
                f"{initial.mean():.3e}",
                f"{final.mean():.3e}",
                f"{np.mean(ratios):.3f}",
                status.replace("_", " "),
            ]
        )

    aleatoric_rows = []
    for head in ("categorical_tagiv", "hrc_tagiv"):
        group = histories[histories["head"] == head]
        initial = group[group["epoch"] == 0]["aleatoric_mean"]
        final = group[group["epoch"] == 200]["aleatoric_mean"]
        selected_values = []
        for row in chosen[chosen["head"] == head].itertuples():
            value = group[
                (group["seed"] == int(row.seed)) & (group["epoch"] == int(row.epoch))
            ]["aleatoric_mean"]
            selected_values.append(float(value.iloc[0]))
        ratio = float((final.to_numpy() / initial.to_numpy()).mean())
        initial_fraction = group[group["epoch"] == 0]["epistemic_fraction"].mean()
        final_fraction = group[group["epoch"] == 200]["epistemic_fraction"].mean()
        aleatoric_rows.append(
            [
                LABELS[head],
                f"{initial.mean():.3e}",
                f"{np.mean(selected_values):.3e}",
                f"{final.mean():.3e}",
                f"{ratio:.3f}",
                f"{100 * initial_fraction:.2f} → {100 * final_fraction:.2f}",
            ]
        )

    return {
        "config": markdown_table(
            ["Method", "Observation variance", "Gain W/b", "Selected epochs (5 seeds)"],
            config_rows,
        ),
        "clean": clean,
        "svhn": svhn,
        "corruption_classification": corruption_classification,
        "corruption_ood": corruption_ood,
        "severity_accuracy": severity_table("accuracy"),
        "severity_ece": severity_table("ece"),
        "severity_auroc": severity_table("entropy_auroc"),
        "clean_selected_vs_final": selected_vs_epoch200("clean_svhn_classification"),
        "corruption_selected_vs_final": selected_vs_epoch200(
            "corruption_macro_shift_classification"
        ),
        "convergence": markdown_table(
            ["Method", "Required epoch, mean (range)", "$U_0$", "$U_{200}$", "$U_{200}/U_0$", "Epoch 170–200"],
            convergence_rows,
        ),
        "aleatoric": markdown_table(
            ["Method", "$A_0$", "$A_{selected}$", "$A_{200}$", "$A_{200}/A_0$", "Epistemic fraction"],
            aleatoric_rows,
        ),
        "clean_secondary": secondary_classification("clean_svhn_classification"),
        "corruption_secondary": secondary_classification(
            "corruption_macro_shift_classification"
        ),
        "svhn_complete_ood": complete_ood("clean_svhn_ood"),
        "corruption_complete_ood": complete_ood("corruption_macro_ood"),
    }

def generate_report() -> None:
    summary = pd.read_csv(ARTIFACT_ROOT / "report_summary.csv")
    report = pd.read_csv(ARTIFACT_ROOT / "report.csv")
    histories = load_histories()
    corruptions = load_corruption_rows(report)

    selected = selected_evaluations(report)
    selected_epochs = {
        head: [int(value) for value in selected[selected["head"] == head]["epoch"]]
        for head in TRAINED_METHODS
    }
    plot_histories(histories, selected_epochs)
    plot_aleatoric_histories(histories, selected_epochs)
    plot_svhn_entropy_histograms(report, summary)

    refine = json.loads(
        (ARTIFACT_ROOT / "heads" / "refine" / "cifar10" / "selection.json").read_text()
    )["selected"]
    tagiv = json.loads(
        (ARTIFACT_ROOT / "heads" / "tagiv" / "cifar10" / "selection.json").read_text()
    )["selected"]
    selected_configs = {**refine, **tagiv}

    config_rows = []
    for head in TRAINED_METHODS:
        config = selected_configs[head]["config"]
        if head in {"categorical_tagiv", "hrc_tagiv"}:
            noise = (
                f"learned; $\\bar v^2_0={config['v2bar_init']}$, "
                f"$S_{{b,v}}={config['v2bar_bias_var']}$"
            )
        else:
            noise = f"$\\sigma_v={config['sigma_v']}$"
        epochs = selected_epochs[head]
        epoch_text = str(epochs[0]) if len(set(epochs)) == 1 else ", ".join(map(str, epochs))
        config_rows.append(
            [
                LABELS[head],
                noise,
                f"{config['gain_w']}/{config['gain_b']}",
                epoch_text,
            ]
        )
    config_table = markdown_table(
        ["Method", "Observation variance", "Gain W/b", "Selected epochs (5 seeds)"],
        config_rows,
    )

    clean_rows = []
    for head in METHODS:
        clean_rows.append(
            [
                LABELS[head],
                format_metric(summary, head, "clean_svhn_classification_accuracy", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_classification_nll", digits=3),
                format_metric(summary, head, "clean_svhn_classification_brier", digits=3),
                format_metric(summary, head, "clean_svhn_classification_ece", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_classification_adaptive_ece", percent=True, digits=2),
            ]
        )
    clean_table = markdown_table(
        ["Method", "Accuracy ↑", "NLL ↓", "Brier ↓", "ECE ↓", "Adaptive ECE ↓"],
        clean_rows,
    )

    svhn_rows = []
    for head in METHODS:
        svhn_rows.append(
            [
                LABELS[head],
                format_metric(summary, head, "clean_svhn_ood_entropy_auroc", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_ood_entropy_aupr_ood", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_ood_entropy_fpr95", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_ood_negative_max_probability_auroc", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_ood_native_epistemic_auroc", percent=True, digits=2),
                format_metric(summary, head, "clean_svhn_ood_native_epistemic_fpr95", percent=True, digits=2),
            ]
        )
    svhn_table = markdown_table(
        [
            "Method",
            "Entropy AUROC ↑",
            "Entropy AUPR-OOD ↑",
            "Entropy FPR95 ↓",
            "$1-\\max p$ AUROC ↑",
            "Native epi. AUROC ↑",
            "Native epi. FPR95 ↓",
        ],
        svhn_rows,
    )

    corruption_classification_rows = []
    for head in METHODS:
        corruption_classification_rows.append(
            [
                LABELS[head],
                format_metric(summary, head, "corruption_macro_shift_classification_accuracy", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_shift_classification_nll", digits=3),
                format_metric(summary, head, "corruption_macro_shift_classification_brier", digits=3),
                format_metric(summary, head, "corruption_macro_shift_classification_ece", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_shift_classification_adaptive_ece", percent=True, digits=2),
            ]
        )
    corruption_classification_table = markdown_table(
        ["Method", "Accuracy ↑", "NLL ↓", "Brier ↓", "ECE ↓", "Adaptive ECE ↓"],
        corruption_classification_rows,
    )

    corruption_ood_rows = []
    for head in METHODS:
        corruption_ood_rows.append(
            [
                LABELS[head],
                format_metric(summary, head, "corruption_macro_ood_entropy_auroc", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_ood_entropy_aupr_ood", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_ood_entropy_fpr95", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_ood_negative_max_probability_auroc", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_ood_native_epistemic_auroc", percent=True, digits=2),
                format_metric(summary, head, "corruption_macro_ood_native_epistemic_fpr95", percent=True, digits=2),
            ]
        )
    corruption_ood_table = markdown_table(
        [
            "Method",
            "Entropy AUROC ↑",
            "Entropy AUPR-OOD ↑",
            "Entropy FPR95 ↓",
            "$1-\\max p$ AUROC ↑",
            "Native epi. AUROC ↑",
            "Native epi. FPR95 ↓",
        ],
        corruption_ood_rows,
    )

    by_severity = (
        corruptions.groupby(["head", "seed", "severity"], as_index=False)
        .mean(numeric_only=True)
        .groupby(["head", "severity"], as_index=False)
        .mean(numeric_only=True)
    )

    def severity_table(metric: str, percent: bool = True) -> str:
        rows = []
        for head in METHODS:
            group = by_severity[by_severity["head"] == head].set_index("severity")
            values = []
            for severity in range(1, 6):
                value = float(group.loc[severity, metric])
                values.append(f"{100 * value:.2f}" if percent else f"{value:.3f}")
            rows.append([LABELS[head], *values])
        return markdown_table(["Method", "Severity 1", "Severity 2", "Severity 3", "Severity 4", "Severity 5"], rows)

    severity_accuracy_table = severity_table("accuracy")
    severity_ece_table = severity_table("ece")
    severity_auroc_table = severity_table("entropy_auroc")

    convergence_rows = []
    for head in TRAINED_METHODS:
        group = histories[histories["head"] == head]
        first = group[group["epoch"] == 0]["epistemic_mean"]
        last = group[group["epoch"] == 200]["epistemic_mean"]
        ratios = []
        required = []
        statuses = []
        complete_root = ARTIFACT_ROOT / "heads" / "confirm" / "cifar10" / head
        for complete_path in sorted(complete_root.glob("*/complete.json")):
            complete = json.loads(complete_path.read_text())
            ratios.append(complete["epistemic_convergence"]["u_final_over_u_initial"])
            statuses.append(complete["epistemic_convergence"]["status"])
            if complete.get("training_required_epoch") is not None:
                required.append(complete["training_required_epoch"])
        required_text = "not reached" if not required else f"{np.mean(required):.1f} ({min(required)}–{max(required)})"
        status = max(set(statuses), key=statuses.count)
        convergence_rows.append(
            [
                LABELS[head],
                required_text,
                f"{first.mean():.3e}",
                f"{last.mean():.3e}",
                f"{np.mean(ratios):.3f}",
                status.replace("_", " "),
            ]
        )
    convergence_table = markdown_table(
        ["Method", "Required epoch, mean (range)", "$U_0$", "$U_{200}$", "$U_{200}/U_0$", "Epoch 170–200"],
        convergence_rows,
    )

    aleatoric_rows = []
    for head in ("categorical_tagiv", "hrc_tagiv"):
        group = histories[histories["head"] == head]
        initial = group[group["epoch"] == 0]["aleatoric_mean"]
        final = group[group["epoch"] == 200]["aleatoric_mean"]
        selected_values = []
        for row in selected[selected["head"] == head].itertuples():
            value = group[
                (group["seed"] == int(row.seed)) & (group["epoch"] == int(row.epoch))
            ]["aleatoric_mean"]
            selected_values.append(float(value.iloc[0]))
        initial_fraction = group[group["epoch"] == 0]["epistemic_fraction"].mean()
        final_fraction = group[group["epoch"] == 200]["epistemic_fraction"].mean()
        ratio = float((final.to_numpy() / initial.to_numpy()).mean())
        behavior = (
            "constant / inactive channel"
            if np.isclose(ratio, 1.0, atol=1e-8)
            else "increasing; near late plateau"
        )
        aleatoric_rows.append(
            [
                LABELS[head],
                f"{initial.mean():.3e}",
                f"{np.mean(selected_values):.3e}",
                f"{final.mean():.3e}",
                f"{ratio:.3f}",
                f"{100 * initial_fraction:.2f} → {100 * final_fraction:.2f}",
                behavior,
            ]
        )
    aleatoric_table = markdown_table(
        [
            "Method",
            "$A_0$",
            "$A_{selected}$",
            "$A_{200}$",
            "$A_{200}/A_0$",
            "Epistemic fraction",
            "Observed behavior",
        ],
        aleatoric_rows,
    )

    selected_vs_final_rows = []
    for head in TRAINED_METHODS:
        selected_nll, _, _ = mean_std(summary, head, "corruption_macro_shift_classification_nll")
        selected_ece, _, _ = mean_std(summary, head, "corruption_macro_shift_classification_ece")
        final_nll_row = summary[
            (summary["head"] == head)
            & (summary["evaluation_kind"] == "epoch_200")
            & (summary["metric"] == "corruption_macro_shift_classification_nll")
        ].iloc[0]
        final_ece_row = summary[
            (summary["head"] == head)
            & (summary["evaluation_kind"] == "epoch_200")
            & (summary["metric"] == "corruption_macro_shift_classification_ece")
        ].iloc[0]
        selected_vs_final_rows.append(
            [
                LABELS[head],
                f"{selected_nll:.3f}",
                f"{float(final_nll_row['mean']):.3f}",
                f"{100 * selected_ece:.2f}",
                f"{100 * float(final_ece_row['mean']):.2f}",
            ]
        )
    selected_vs_final_table = markdown_table(
        ["Method", "Selected NLL", "Epoch-200 NLL", "Selected ECE", "Epoch-200 ECE"],
        selected_vs_final_rows,
    )

    def secondary_classification_table(prefix: str) -> str:
        rows = []
        for head in METHODS:
            rows.append(
                [
                    LABELS[head],
                    format_metric(summary, head, f"{prefix}_top5_accuracy", percent=True, digits=2),
                    format_metric(summary, head, f"{prefix}_mean_confidence", percent=True, digits=2),
                    format_metric(summary, head, f"{prefix}_classwise_ece", percent=True, digits=2),
                    format_metric(summary, head, f"{prefix}_aurc", percent=True, digits=2),
                    format_metric(summary, head, f"{prefix}_risk_at_80_coverage", percent=True, digits=2),
                    format_metric(summary, head, f"{prefix}_risk_at_90_coverage", percent=True, digits=2),
                    format_metric(summary, head, f"{prefix}_risk_at_95_coverage", percent=True, digits=2),
                ]
            )
        return markdown_table(
            [
                "Method",
                "Top-5 acc. ↑",
                "Mean conf.",
                "Classwise ECE ↓",
                "AURC ↓",
                "Risk@80 ↓",
                "Risk@90 ↓",
                "Risk@95 ↓",
            ],
            rows,
        )

    def complete_ood_table(prefix: str) -> str:
        score_names = {
            "entropy": "Entropy",
            "negative_max_probability": r"$1-\max p$",
            "native_epistemic": "Native epistemic",
        }
        rows = []
        for head in METHODS:
            for score, score_label in score_names.items():
                metric_prefix = f"{prefix}_{score}"
                mean, _, _ = mean_std(summary, head, f"{metric_prefix}_auroc")
                if np.isnan(mean):
                    continue
                rows.append(
                    [
                        LABELS[head],
                        score_label,
                        format_metric(summary, head, f"{metric_prefix}_auroc", percent=True, digits=2),
                        format_metric(summary, head, f"{metric_prefix}_aupr_ood", percent=True, digits=2),
                        format_metric(summary, head, f"{metric_prefix}_aupr_id", percent=True, digits=2),
                        format_metric(summary, head, f"{metric_prefix}_fpr95", percent=True, digits=2),
                    ]
                )
        return markdown_table(
            ["Method", "Score", "AUROC ↑", "AUPR-OOD ↑", "AUPR-ID ↑", "FPR95 ↓"],
            rows,
        )

    clean_secondary_table = secondary_classification_table("clean_svhn_classification")
    corruption_secondary_table = secondary_classification_table(
        "corruption_macro_shift_classification"
    )
    svhn_complete_ood_table = complete_ood_table("clean_svhn_ood")
    corruption_complete_ood_table = complete_ood_table("corruption_macro_ood")
    cifar100_tables = build_dataset_tables(summary, report, "cifar100")

    report_text = f"""# Frozen-feature TAGI last-layer classification study

## Executive summary

This study asks whether analytic Bayesian last-layer inference can improve uncertainty quantification while preserving the accuracy of a strong deterministic classifier. Separate CIFAR ResNet-18 backbones were trained on deterministic 40,000/10,000 train/validation splits for CIFAR-10 and CIFAR-100 and then frozen. Six shared TAGI last-layer heads were trained from each backbone's 512-dimensional features; CIFAR-100 additionally evaluates unit-probit HRC. The original PyTorch softmax classifier is the baseline. No method uses temperature scaling or any other post-hoc calibration.

On CIFAR-10, the answer is positive for calibration and promising but mixed for OOD detection. Clean accuracy stays within 0.20 percentage points of the 95.00% baseline. Clean ECE improves from 2.78% to 0.53% with HRC, while clean NLL improves from 0.194 to 0.173 with categorical TAGI-V. On CIFAR-10-C, hierarchical TAGI-V reduces macro ECE from 19.68% to 12.39% and NLL from 1.358 to 1.090 at essentially unchanged accuracy. Entropy-based SVHN detection improves from 91.18% AUROC for softmax to 92.48% for HRC; the corresponding CIFAR-10-C improvement is smaller, 70.76% to 71.33%.

CIFAR-100 is more qualified. Categorical TAGI-V gives the strongest overall Bayesian tradeoff: 76.61% clean accuracy versus 76.67% for softmax, and on CIFAR-100-C it improves NLL from 2.452 to 2.411, Brier from 0.691 to 0.678, and ECE from 12.86% to 10.29% with a 0.28-point accuracy decrease. It also slightly improves entropy AUROC on SVHN (86.05% versus 85.49%) and CIFAR-100-C (69.72% versus 69.39%). The hierarchical heads lose 3.3–4.7 clean-accuracy points, so their lower shifted ECE cannot be read as a calibration-only improvement.

Native epistemic uncertainty is the main unresolved issue. ReMax variance is useful for OOD detection, reaching about 90% AUROC on SVHN and 70% on CIFAR-10-C, while the native variance of the other heads is anti-correlated with OOD examples. Moreover, HRC, probit OVR, and hierarchical TAGI-V continue contracting through epoch 200 even after accuracy and calibration have stabilized. Posterior contraction therefore cannot by itself be treated as evidence that the resulting scalar is a valid OOD score.

The learned-noise result is also asymmetric: hierarchical TAGI-V increases its aleatoric variance and becomes aleatoric-dominated, whereas dense categorical TAGI-V leaves its nominally trainable variance channel at initialization. The latter should be treated as an inactive-channel result until its update parameterization is corrected or shown to learn under another controlled configuration.

## Scope and experimental protocol

- Dataset split: deterministic stratified 40,000 train / 10,000 validation from each official CIFAR training set; each official 10,000-example test set is held out until final evaluation.
- Backbones: separate CIFAR ResNet-18 models trained for 200 epochs with SGD, momentum 0.9, weight decay $5\\times10^{{-4}}$, cosine learning-rate schedule, and initial learning rate 0.1. CIFAR-10 reached 95.44% best validation accuracy at epoch 195 and 95.39% final accuracy. CIFAR-100 reached its best and final validation accuracy of 76.36% at epoch 200.
- Last-layer scope: the backbone and all 512-dimensional feature vectors are frozen. Only the feature-to-output layer is changed.
- Search: fixed-noise heads screen $\\sigma_v$, weight gain, and bias gain. TAGI-V heads instead search the prior for learned observation variance. Hyperparameters are selected by minimum validation NLL among configurations within one percentage point of the best validation accuracy, breaking ties with Brier score and ECE.
- Confirmation: five last-layer seeds, 200 epochs, with validation-selected and epoch-200 checkpoints both retained.
- OOD: SVHN is semantic OOD. CIFAR-10-C and CIFAR-100-C use the 15 canonical corruptions at severities 1–5 and are better described as covariate shift; OOD metrics compare each clean CIFAR dataset against its corresponding corrupted set.
- Uncertainty scores: predictive entropy, $1-\\max_k p_k$, and the mean native predictive epistemic variance across outputs. AUROC/AUPR-OOD are higher-is-better; FPR95 is lower-is-better.
- Replication boundary: within each dataset, the five seeds share one trained backbone, so intervals measure last-layer variability, not backbone-training variability. Reported $\\pm$ values are sample standard deviations; each baseline is one fixed run.

### Selected configurations

{config_table}

The per-seed validation selector chose epoch 10 for probit OVR, epoch 1 for moment-matched ReMax, epochs 10–20 for diagonal Laplace ReMax, epoch 200 for fixed HRC, epoch 75 for categorical TAGI-V, and epochs 20 or 200 for hierarchical TAGI-V.

## Methods and formulations

### Common TAGI last-layer model

For frozen feature vector $h\\in\\mathbb{{R}}^D$, each weight and bias is represented by an independent Gaussian posterior,

$$W_{{dk}}\\sim\\mathcal N(m_{{W,dk}},S_{{W,dk}}),\\qquad b_k\\sim\\mathcal N(m_{{b,k}},S_{{b,k}}).$$

Because $h$ is deterministic, a linear output has diagonal moments

$$m_{{z,k}}=\\sum_d h_d m_{{W,dk}}+m_{{b,k}},\\qquad S_{{z,k}}=\\sum_d h_d^2 S_{{W,dk}}+S_{{b,k}}.$$

The regression-style Gaussian observation model is

$$y_k=z_k+v_k,\\qquad v_k\\sim\\mathcal N(0,\\sigma_v^2),$$

which yields the output innovations

$$\\Delta m_{{z,k}}=\\frac{{y_k-m_{{z,k}}}}{{S_{{z,k}}+\\sigma_v^2}},\\qquad
\\Delta S_{{z,k}}=-\\frac{{1}}{{S_{{z,k}}+\\sigma_v^2}}.$$

TAGI propagates these two terms analytically through the last layer and performs capped Gaussian parameter updates. There is no gradient optimizer for these heads. Initial gains determine the prior parameter variances; $\\sigma_v$ determines how strongly observations update that prior.

### Probit one-versus-rest

The true class receives target $+1$ and every other class receives $-1$. Each logit is updated with the Gaussian regression observation above. Its binary predictive probability is

$$\\tilde p_k=\\Phi\\left(\\frac{{m_{{z,k}}}}{{\\sqrt{{S_{{z,k}}+\\sigma_v^2}}}}\\right),\\qquad
p_k=\\frac{{\\tilde p_k}}{{\\sum_j\\tilde p_j}}.$$

This is an independent one-versus-rest construction followed by normalization; it is not the external NormCDF/Laplace method discussed elsewhere.

### ReMax probability-space observations

ReMax maps Gaussian logits to a simplex without exponentiation. Let $M_k=\\max(0,Z_k)$ and

$$A_k=\\frac{{M_k}}{{\\sum_j M_j}}.$$

The class label is a one-hot vector in probability space and the same Gaussian observation update is applied after propagating moments through $A$. Two approximations were evaluated:

1. **Moment-matched ReMax:** truncated-Gaussian moments of $M_k$ are converted to log-normal moments; moments of $\\log M_k-\\log\\sum_jM_j$ give $E[A_k]$, $\\operatorname{{Var}}(A_k)$, and a diagonal statistical Jacobian $\\operatorname{{Cov}}(A_k,Z_k)/S_{{z,k}}$.
2. **Laplace ReMax, diagonal:** fixed Gaussian quadrature evaluates the Laplace identities for $1/\\sum_jM_j$ and its square. The implementation can form a full cross-class Jacobian, but this experiment deliberately uses only its diagonal.

### Fixed-noise hierarchical classification

For $K$ classes, each class is assigned a binary code of length $L=\\lceil\\log_2K\\rceil$. CIFAR-10 uses four decisions and 11 unique tree nodes. A class $c$ has path nodes $j_{{c,\\ell}}$ and signs $s_{{c,\\ell}}\\in\\{{-1,+1\\}}$. Only those four nodes receive an update for each training example. Node probabilities are

$$q_j=\\Phi\\left(\\frac{{m_{{z,j}}}}{{\\sqrt{{(1/3)^2+S_{{z,j}}}}}}\\right),$$

and the class probability is the normalized path product

$$\\tilde p(c)=\\prod_{{\\ell=1}}^L
q_{{j_{{c,\\ell}}}}^{{\\mathbb 1[s_{{c,\\ell}}=+1]}}
(1-q_{{j_{{c,\\ell}}}})^{{\\mathbb 1[s_{{c,\\ell}}=-1]}}.$$

This makes the observation update sparse and changes the multiclass geometry: examples update only their path rather than all $K$ outputs. CIFAR-100 paths contain seven decisions rather than four.

### Unit-probit HRC

The CIFAR-100 replication also evaluates an exact half-space probit update on the same sparse hierarchy. It fixes the structural link variance to one, accepts no $\\sigma_v$, and predicts each positive branch with

$$q_j=\\Phi\\left(\\frac{{m_{{z,j}}}}{{\\sqrt{{1+S_{{z,j}}}}}}\\right).$$

This separates the effect of a unit-probit observation model from the tuned fixed-noise HRC model. Its gains are searched, but its link scale is neither inferred nor used as a prediction-time temperature.

### Categorical TAGI-V

TAGI-V removes manual $\\sigma_v$. The output width is $2K$, interleaving a logit latent and a positive learned variance latent for every class. Denote the epistemic logit variance by $S_{{z,k}}$ and the learned aleatoric mean by $\\bar v_k^2$. The predictive probability uses a logistic–probit moment bridge,

$$a_k=\\left(1+\\frac{{\\pi}}{{8}}(S_{{z,k}}+\\bar v_k^2)\\right)^{{-1/2}},\\qquad
p_k=\\operatorname{{softmax}}_k\\left(a_k[m_{{z,k}}-\\bar m_z]\\right).$$

Training uses the integer categorical label directly. A centered $O(K)$ Gaussian approximation to the categorical innovation couples the classes through the simplex constraint. The variance channel is updated by matching the first two moments of the squared residual, and an even softplus activation keeps $\\bar v_k^2$ positive. Thus $S_z$ is reported as epistemic uncertainty and $\\bar v^2$ as learned aleatoric uncertainty; neither is fitted post hoc.

### Hierarchical TAGI-V

Hierarchical TAGI-V combines the sparse binary tree with a learned variance channel at every tree node, giving $2J$ interleaved outputs for $J$ nodes. For each selected node,

$$S_{{\\text{{total}},j}}=S_{{z,j}}+\\bar v_j^2,$$

$$\\Delta m_{{z,j}}=\\frac{{s_j-m_{{z,j}}}}{{S_{{\\text{{total}},j}}}},\\qquad
\\Delta S_{{z,j}}=-\\frac{{1}}{{S_{{\\text{{total}},j}}}}.$$

The learned variance is again updated by squared-residual moment matching. At prediction time, $S_z+\\bar v^2$ enters each node CDF before multiplying probabilities along the class path. This is the no-manual-noise hierarchical classifier proposed in this study.

## Clean CIFAR-10 classification and calibration

{clean_table}

All heads preserve accuracy. The largest decrease from the 95.00% baseline is 0.20 percentage points. Categorical TAGI-V has the best NLL and Brier score, while fixed HRC has the best standard ECE. Hierarchical TAGI-V is close to fixed HRC without requiring a manually selected observation variance.

## SVHN semantic OOD detection

{svhn_table}

Entropy and $1-\\max p$ give the same ranking. HRC is strongest, improving entropy AUROC by about 1.30 percentage points and reducing FPR95 by about 3.40 points versus softmax. ReMax's native epistemic variance is qualitatively different: it reaches approximately 90% AUROC, while native variances from probit, HRC, and categorical TAGI-V are strongly anti-correlated with semantic OOD inputs. Consequently, predictive entropy is currently the reliable score for HRC/TAGI-V; native variance cannot yet be used indiscriminately.

![Clean CIFAR-10 and SVHN entropy histograms](report_assets/svhn_entropy_histograms.png)

Every panel uses the same entropy bins and axes. For learned heads, each example's entropy is averaged across the five validation-selected last-layer seeds before histogramming. HRC and hierarchical TAGI-V move more SVHN mass toward higher entropy while retaining a concentrated low-entropy clean distribution, explaining their stronger entropy AUROC. The overlap remains substantial, consistent with FPR95 values around 20–22% rather than near-perfect separation.

## CIFAR-10-C covariate shift

Each number below is a macro-average over 15 corruptions and five severities (75 equally sized conditions).

### Shifted classification and calibration

{corruption_classification_table}

Accuracy is effectively tied, showing that the output head cannot repair corruption errors already embedded in frozen features. Calibration does improve materially. Hierarchical TAGI-V reduces ECE by 7.29 percentage points, NLL by 0.268, and Brier by 0.044 relative to softmax.

### Clean-versus-corrupted detection

{corruption_ood_table}

The entropy-AUROC gains are modest: fixed HRC improves from 70.76% to 71.33%. ReMax is the only family whose native epistemic variance has the correct direction, at about 70% AUROC; the native scores of the other heads remain below 50%.

### Severity dependence

Corrupted accuracy (%):

{severity_accuracy_table}

ECE (%):

{severity_ece_table}

Clean-versus-corrupted entropy AUROC (%):

{severity_auroc_table}

All heads degrade similarly in accuracy, from about 87% at severity 1 to 54% at severity 5. Hierarchical TAGI-V is consistently better calibrated, although its ECE still rises from 3.40% to 24.11%. Detection becomes easier as severity grows because corrupted representations move farther from the clean distribution.

The easiest corruption is brightness (about 93.5% accuracy averaged across severities); the hardest are Gaussian noise (about 37%), impulse noise (about 49.5%), and shot noise (about 51%).

## CIFAR-100 replication

The locked experiment was repeated on CIFAR-100 with the same deterministic stratified 40,000/10,000 train/validation split, official 10,000-example test set, frozen 512-dimensional ResNet-18 features, search rule, five last-layer confirmation seeds, and 200-epoch confirmation horizon. A separate CIFAR-100 ResNet-18 was trained from scratch for 200 epochs; its best and final validation accuracy was 76.36% at epoch 200. CIFAR-100-C contains the same 15 corruption families and five severities as CIFAR-10-C. The CIFAR-100 study additionally includes unit-probit HRC, whose latent probit scale is fixed to one and therefore has no searched observation-noise parameter.

The five-seed intervals below again measure last-layer variability under one shared backbone, not variability from retraining the backbone. Primary comparisons use each seed's validation-selected checkpoint; explicit epoch-200 tables show the effect of continued training.

### CIFAR-100 selected configurations

{cifar100_tables['config']}

### Clean CIFAR-100 classification and calibration

{cifar100_tables['clean']}

The 100-class replication is materially harder than CIFAR-10. Categorical TAGI-V nearly matches softmax accuracy (76.61% versus 76.67%) but does not improve clean NLL or ECE. Probit OVR also retains most accuracy but is less calibrated. All hierarchical variants lose 3.3–4.7 clean-accuracy points; unit-probit HRC's 5.33% ECE is therefore paired with substantially worse accuracy and NLL.

### CIFAR-100 versus SVHN semantic OOD

{cifar100_tables['svhn']}

Categorical TAGI-V is the only head to improve on the softmax entropy AUROC, reaching 86.05% versus 85.49% and reducing FPR95 from 43.94% to 43.08%. The gain is modest relative to its seed variability. Native epistemic variance is anti-correlated for probit and hierarchical heads; ReMax is positively oriented but much weaker than entropy on CIFAR-100 SVHN.

### CIFAR-100-C shifted classification and calibration

Each result is a macro-average over the 75 corruption/severity conditions.

{cifar100_tables['corruption_classification']}

Categorical TAGI-V provides the clearest shifted-data result: versus softmax it gives 47.83% rather than 48.11% accuracy while improving NLL, Brier, ECE, and adaptive ECE. Probit OVR also lowers ECE and Brier but worsens NLL. The hierarchical heads' still-lower ECE comes with 3.7–4.9 points less shifted accuracy and substantially worse NLL, so it is not an unqualified calibration win.

### CIFAR-100 clean-versus-corrupted detection

{cifar100_tables['corruption_ood']}

Categorical TAGI-V again gives the only improvement over softmax entropy AUROC (69.72% versus 69.39%) and FPR95 (76.37% versus 76.65%); the differences are small. ReMax native variance remains correctly oriented but does not beat softmax entropy, while the other native epistemic scores are anti-correlated.

### CIFAR-100-C severity dependence

Corrupted accuracy (%):

{cifar100_tables['severity_accuracy']}

ECE (%):

{cifar100_tables['severity_ece']}

Clean-versus-corrupted entropy AUROC (%):

{cifar100_tables['severity_auroc']}

### CIFAR-100 checkpoint selection and uncertainty dynamics

Clean metrics at validation-selected and epoch-200 checkpoints (accuracy and ECE in %):

{cifar100_tables['clean_selected_vs_final']}

CIFAR-100-C macro metrics at validation-selected and epoch-200 checkpoints (accuracy and ECE in %):

{cifar100_tables['corruption_selected_vs_final']}

Validation selection is crucial for both ReMax variants and unit-probit HRC. For example, continuing unit-probit HRC from epoch 1 to epoch 200 lowers clean accuracy from 71.96% to 70.78% and raises CIFAR-100-C NLL from 3.204 to 5.252. Categorical TAGI-V, probit OVR, and fixed HRC select epoch 200 and therefore have identical selected/final rows.

Posterior-contraction diagnostics over the 200-epoch confirmation run:

{cifar100_tables['convergence']}

TAGI-V aleatoric-channel diagnostics:

{cifar100_tables['aleatoric']}

### Complete CIFAR-100 metric appendix

Clean secondary classification metrics (%):

{cifar100_tables['clean_secondary']}

CIFAR-100-C secondary shifted-classification metrics (%):

{cifar100_tables['corruption_secondary']}

Complete SVHN OOD metrics (%):

{cifar100_tables['svhn_complete_ood']}

Complete CIFAR-100-C clean-versus-corrupted metrics (%):

{cifar100_tables['corruption_complete_ood']}

## CIFAR-10 training duration and epistemic contraction

The diagnostic cohort is the first 2,048 validation features. At every epoch, $U$ is the mean predictive epistemic variance over examples and output dimensions. The "required epoch" is the earliest epoch for which validation NLL remains within 1% of its best value and accuracy remains within 0.5 percentage points of its best value for three consecutive records. The convergence label examines relative changes over epochs 170–180, 180–190, and 190–200.

{convergence_table}

![Absolute epistemic uncertainty across epochs](report_assets/epistemic_uncertainty_absolute.png)

The filled regions show one sample standard deviation across five last-layer seeds. White-edged markers show the median validation-selected epoch. Absolute scales differ by orders of magnitude because gains and output transformations define different priors; cross-method comparisons should therefore emphasize contraction ratios rather than raw values.

![Epistemic uncertainty relative to initialization](report_assets/epistemic_uncertainty_relative.png)

Categorical TAGI-V rapidly reaches a high uncertainty floor and retains about 94% of its initial predictive epistemic variance. ReMax contracts to roughly 8–10% and plateaus or slightly rebounds late. Probit and fixed HRC also reach about 10% but are still shrinking by approximately 3% per ten epochs at epoch 200. Hierarchical TAGI-V contracts most strongly, to about 3.9%, and is also still shrinking. Thus some posteriors do not reach a finite empirical floor over the observed horizon.

### TAGI-V aleatoric uncertainty

Here $A$ is the mean positive variance-channel output over the same 2,048 validation examples and all class or tree-node outputs. The epistemic fraction is the mean $S_z/(S_z+\\bar v^2)$.

{aleatoric_table}

![TAGI-V aleatoric uncertainty across epochs](report_assets/tagiv_aleatoric_uncertainty.png)

The two TAGI-V variants behave very differently. Hierarchical TAGI-V increases mean aleatoric variance from $1.50\\times10^{{-3}}$ to $7.01\\times10^{{-3}}$ (4.67×) while its epistemic component contracts; by epoch 200 only about 0.77% of total node variance is epistemic. Its aleatoric growth slows below 1% per ten epochs after epoch 170, indicating a near plateau.

Dense categorical TAGI-V does **not** adapt its aleatoric mean in this experiment: it remains exactly $1.049965\\times10^{{-3}}$ at every recorded epoch and seed. Inspection of saved checkpoints confirms that variance-channel weights remain numerically negligible (about $10^{{-13}}$ mean absolute size at epoch 200) and the variance-channel bias mean remains at initialization. It is therefore more accurate to describe this configuration as having a trainable-but-empirically-inactive aleatoric channel. Its calibration improvement cannot be attributed to learned heteroscedastic variance; it comes from the coupled categorical update and uncertainty-tempered predictive. This is an implementation or parameterization issue to investigate, not evidence that dense TAGI-V has successfully learned aleatoric uncertainty.

![Validation proper-score and calibration dynamics](report_assets/validation_dynamics.png)

Continued contraction is not uniformly beneficial. The next table compares validation-selected and epoch-200 CIFAR-10-C calibration:

{selected_vs_final_table}

Moment-matched ReMax is the clearest failure mode: epoch 1 is selected, while continued training drives corrupted NLL from 1.281 to 2.346 and ECE from 15.64% to 23.80% without a meaningful accuracy gain. Diagonal Laplace ReMax and categorical TAGI-V show the same pattern more mildly. HRC is selected at epoch 200, while hierarchical TAGI-V is mixed across seeds. Validation-based checkpointing is therefore essential even when only a tiny last layer is trained.

## Feasibility and conclusion

This research direction is feasible within the deliberately narrow frozen-last-layer scope.

1. **The CIFAR-10 result is strong but does not transfer uniformly.** All six heads preserve CIFAR-10 accuracy and several materially improve calibration. On CIFAR-100, only categorical TAGI-V preserves nearly all clean and corrupted accuracy while improving the principal CIFAR-100-C calibration metrics.
2. **Categorical TAGI-V is the most robust 100-class choice.** It reaches 76.61% clean accuracy, improves CIFAR-100-C NLL/Brier/ECE, and gives the best entropy OOD results among the evaluated heads. Its clean calibration still trails softmax, so the benefit is shift-specific rather than universal.
3. **Hierarchical scaling is the main negative result.** Fixed HRC, hierarchical TAGI-V, and unit-probit HRC lose several accuracy points on CIFAR-100. Their low ECE values partly reflect reduced confidence and must be judged with NLL and accuracy. Unit-probit HRC also overtrains sharply after its epoch-1 selection.
4. **The uncertainty decomposition is not yet generally trustworthy.** Native epistemic variance fails as an OOD ranking score for most heads; ReMax is positively oriented but weaker on CIFAR-100 and its predictive calibration is highly sensitive to overtraining. Hierarchical TAGI-V learns an aleatoric channel, whereas the dense categorical channel remains inactive on both datasets.

The most defensible next step is to investigate why hierarchical output geometry loses accuracy at 100 classes and why dense TAGI-V's variance channel is inactive, while retaining categorical TAGI-V as the strongest frozen-feature CIFAR-100 candidate. Independently retrained backbones are still necessary before making architecture-independent claims.

## Complete metric appendix

The main text emphasizes accuracy, proper scores, marginal calibration, AUROC, AUPR-OOD, and FPR95. The following tables preserve the remaining metrics computed by the study. AURC is area under the risk–coverage curve; Risk@X is the error rate among the X% most confident predictions. Classwise ECE is the mean one-versus-rest calibration error. For OOD metrics, corrupted or SVHN examples are positive for AUPR-OOD and clean CIFAR-10 examples are positive for AUPR-ID.

### Clean CIFAR-10 secondary classification metrics (%)

{clean_secondary_table}

### CIFAR-10-C secondary shifted-classification metrics (%)

{corruption_secondary_table}

### Complete SVHN OOD metrics (%)

{svhn_complete_ood_table}

### Complete CIFAR-10-C clean-versus-corrupted metrics (%)

{corruption_complete_ood_table}

## Reproducibility and artifacts

- Study manifest: [`study.json`](study.json)
- Runner: [`run_study.py`](run_study.py)
- Report generator: [`generate_report.py`](generate_report.py)
- Per-run machine-readable report: [`../../runs/last_layer/{MANIFEST['study_id']}/report.csv`](../../runs/last_layer/{MANIFEST['study_id']}/report.csv)
- Aggregated five-seed report: [`../../runs/last_layer/{MANIFEST['study_id']}/report_summary.csv`](../../runs/last_layer/{MANIFEST['study_id']}/report_summary.csv)
- Full JSON report: [`../../runs/last_layer/{MANIFEST['study_id']}/report.json`](../../runs/last_layer/{MANIFEST['study_id']}/report.json)

Regenerate this document and its figures with:

```bash
python experiments/last_layer/generate_report.py
```

The entropy histogram regeneration loads the selected TAGI checkpoints and therefore requires the cached clean/SVHN features and a CUDA-capable environment for the Triton heads.

The report combines the completed CIFAR-10 and CIFAR-100 studies. Every aggregate can be traced to the per-run CSV/JSON artifacts above; the CIFAR-100 backbone checkpoint and all validation-selected and epoch-200 last-layer checkpoints are retained under the same study artifact root.
"""

    REPORT_PATH.write_text(report_text)
    print(f"wrote {REPORT_PATH}")
    for path in sorted(ASSET_ROOT.glob("*.png")):
        print(f"wrote {path}")


if __name__ == "__main__":
    generate_report()
