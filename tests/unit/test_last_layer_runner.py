"""Smoke test for the resumable last-layer study runner."""

from __future__ import annotations

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
