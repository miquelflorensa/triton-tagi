"""Smoke test for the resumable last-layer study runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import experiments.last_layer.run_study as runner
from experiments.last_layer.run_study import (
    evaluation_kinds,
    fixed_screen_configs,
    run_configuration,
)
from triton_tagi.cifar_study import save_feature_shard


class _IndexedDataset(Dataset):
    def __init__(self, targets, *, fail_on_access=False):
        self.targets = list(targets)
        self.fail_on_access = fail_on_access
        self.accessed = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.fail_on_access:
            raise AssertionError("the official test set was accessed during training")
        self.accessed.append(index)
        return torch.tensor([float(index)]), self.targets[index]


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, inputs):
        return self.classifier(inputs.float().reshape(inputs.shape[0], -1))


def test_unit_probit_is_not_in_observation_noise_sweep():
    configs = fixed_screen_configs(
        {
            "fixed_noise": {
                "heads": ["hrc"],
                "sigma_v": [0.3, 1.0],
                "tied_gains": [0.1],
            },
            "unit_probit": {
                "heads": ["hrc_probit"],
                "tied_gains": [0.1, 0.3],
            },
        }
    )
    probit = [config for config in configs if config["head"] == "hrc_probit"]
    assert probit == [
        {"head": "hrc_probit", "gain_w": 0.1, "gain_b": 0.1},
        {"head": "hrc_probit", "gain_w": 0.3, "gain_b": 0.3},
    ]
    assert all("sigma_v" not in config for config in probit)


def test_final_checkpoint_can_also_be_validation_selected():
    assert evaluation_kinds(20, 20, 200) == ["validation_selected"]
    assert evaluation_kinds(200, 20, 200) == ["epoch_200"]
    assert evaluation_kinds(200, 200, 200) == ["validation_selected", "epoch_200"]


def test_backbone_holds_out_validation_before_training_and_never_reads_test(
    tmp_path, monkeypatch
):
    official_test_sets = []

    def fake_clean_datasets(*args, **kwargs):
        training = _IndexedDataset([0, 0, 0, 1, 1, 1])
        official_test = _IndexedDataset([0, 1], fail_on_access=True)
        official_test_sets.append(official_test)
        return training, official_test

    monkeypatch.setattr(runner, "clean_datasets", fake_clean_datasets)
    monkeypatch.setattr(runner, "CifarResNet18", _TinyClassifier)
    monkeypatch.setattr(
        runner,
        "get_spec",
        lambda dataset: SimpleNamespace(num_classes=2, backbone_accuracy_gate=0.0),
    )
    manifest = {
        "study_id": "split-smoke",
        "paths": {"artifacts": str(tmp_path / "artifacts"), "data": str(tmp_path)},
        "split": {"seed": 7, "validation_size": 2},
        "backbone": {
            "seed": 11,
            "epochs": 1,
            "batch_size": 2,
            "evaluation_batch_size": 2,
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
    }

    runner.train_backbone(
        SimpleNamespace(
            dataset="cifar10", device="cpu", workers=0, no_download=True
        ),
        manifest,
    )

    checkpoint = torch.load(
        runner.backbone_path(manifest, "cifar10"),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["split"]["train_size"] == 4
    assert checkpoint["split"]["validation_size"] == 2
    assert "validation_accuracy" in checkpoint["history"][0]
    assert "test_accuracy" not in checkpoint["history"][0]
    assert all(not dataset.accessed for dataset in official_test_sets)


@pytest.mark.cuda
def test_single_configuration_runner_writes_resumable_artifacts(tmp_path):
    manifest = {
        "study_id": "smoke",
        "paths": {"artifacts": str(tmp_path)},
        "last_layer": {"batch_size": 8, "diagnostic_cohort_size": 8},
    }
    feature_root = tmp_path / "smoke" / "features" / "cifar10"
    generator = torch.Generator().manual_seed(1)
    for split, samples in (("train", 24), ("validation", 12)):
        save_feature_shard(
            feature_root / f"{split}.pt",
            {
                "features": torch.randn(samples, 5, generator=generator),
                "logits": torch.randn(samples, 10, generator=generator),
                "labels": torch.arange(samples) % 10,
            },
            {"fingerprint": split},
        )

    run_configuration(
        manifest,
        "cifar10",
        "screen",
        {
            "head": "probit_ovr",
            "sigma_v": 0.05,
            "gain_w": 0.1,
            "gain_b": 0.1,
        },
        seed=0,
        epochs=1,
        checkpoints=[0, 1],
        args=SimpleNamespace(device="cuda", force=False),
    )
    run_root = tmp_path / "smoke" / "heads" / "screen" / "cifar10" / "probit_ovr"
    runs = list(run_root.iterdir())
    assert len(runs) == 1
    assert (runs[0] / "complete.json").exists()
    assert (runs[0] / "history.json").exists()
    assert (runs[0] / "checkpoints" / "epoch_0000.pt").exists()
    assert (runs[0] / "checkpoints" / "epoch_0001.pt").exists()


def _write_evaluation(root, *, head, mean_init, seed, accuracy, nll):
    """Write the minimum evaluation payload that report_study reads."""

    payload = {
        "config": {
            "dataset": "cifar10",
            "head": head,
            "seed": seed,
            "epochs": 200,
            "stage": "init_confirm",
            "mean_init": mean_init,
            "gain_w": 0.3,
            "gain_b": 0.3,
            "sigma_v": 0.1,
        },
        "checkpoint": f"{head}/{mean_init}/seed{seed}",
        "metadata": {"epoch": 200},
        "selection_epoch": 200,
        "clean_and_svhn": {
            "classification": {"accuracy": accuracy, "nll": nll},
        },
        "corruptions": {},
    }
    destination = root / head / f"{mean_init}_seed{seed}" / "epoch_0200.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload))


def test_report_keeps_initialization_arms_apart(tmp_path):
    """Two init arms of one head must not average into a single summary row.

    ``init_confirm`` evaluates the selected initialization arm *and* its
    ``random`` counterpart for the same (dataset, head), so a summary grouped
    only by (dataset, head, evaluation_kind) would report their mean and hide
    the very delta the study measures.
    """

    manifest = {"study_id": "smoke", "paths": {"artifacts": str(tmp_path)}}
    evaluations = tmp_path / "smoke" / "evaluations" / "cifar10"
    for seed in (0, 1):
        _write_evaluation(
            evaluations, head="remax_lognormal", mean_init="zero",
            seed=seed, accuracy=0.75, nll=2.0,
        )
        _write_evaluation(
            evaluations, head="remax_lognormal", mean_init="random",
            seed=seed, accuracy=0.53, nll=2.5,
        )

    runner.report_study(SimpleNamespace(), manifest)

    summary = json.loads(
        (tmp_path / "smoke" / "report_summary.json").read_text()
    )
    accuracy = {
        row["mean_init"]: row
        for row in summary
        if row["metric"] == "clean_svhn_classification_accuracy"
    }
    assert set(accuracy) == {"zero", "random"}
    assert accuracy["zero"]["mean"] == pytest.approx(0.75)
    assert accuracy["random"]["mean"] == pytest.approx(0.53)
    assert accuracy["zero"]["n"] == 2
    assert accuracy["random"]["n"] == 2

    report = json.loads((tmp_path / "smoke" / "report.json").read_text())
    assert {row["mean_init"] for row in report} == {"zero", "random"}
    assert {row["gain_w"] for row in report} == {0.3}
    assert {row["stage"] for row in report} == {"init_confirm"}


def test_report_leaves_unrecorded_axes_blank(tmp_path):
    """A run predating an axis reports it blank, not as an invented value."""

    manifest = {"study_id": "smoke", "paths": {"artifacts": str(tmp_path)}}
    evaluations = tmp_path / "smoke" / "evaluations" / "cifar10"
    destination = evaluations / "hrc" / "legacy" / "epoch_0200.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": "cifar10",
                    "head": "hrc",
                    "seed": 0,
                    "epochs": 200,
                    "gain_w": 0.3,
                    "gain_b": 0.3,
                    "sigma_v": None,
                },
                "checkpoint": "legacy",
                "metadata": {"epoch": 200},
                "clean_and_svhn": {"classification": {"accuracy": 0.95}},
                "corruptions": {},
            }
        )
    )

    runner.report_study(SimpleNamespace(), manifest)

    report = json.loads((tmp_path / "smoke" / "report.json").read_text())
    # The final checkpoint is also the validation-selected one, so it reports
    # under both evaluation kinds; every row must leave the axes blank.
    assert {row["evaluation_kind"] for row in report} == {
        "validation_selected",
        "epoch_200",
    }
    assert {row["mean_init"] for row in report} == {""}
    assert {row["sigma_v"] for row in report} == {""}
    assert {row["stage"] for row in report} == {""}
    assert {row["gain_w"] for row in report} == {0.3}
