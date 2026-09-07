"""Smoke test for the resumable last-layer study runner."""

from __future__ import annotations

import json
import pathlib
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


def test_evaluate_reads_the_requested_confirmation_stage(tmp_path, monkeypatch):
    """``evaluate --stage init_confirm`` must read the init_confirm tree.

    The stage was hardcoded to ``confirm``, so the initialization study's runs
    were invisible to evaluation and every deliverable table would have come
    back empty.
    """

    manifest = {
        "study_id": "smoke",
        "paths": {"artifacts": str(tmp_path)},
        "confirmation": {"checkpoints": [0, 200]},
        "init_study": {"confirm": {"checkpoints": [0, 7]}},
    }
    root = tmp_path / "smoke"
    for stage, epoch in (("confirm", 200), ("init_confirm", 7)):
        run_dir = root / "heads" / stage / "cifar10" / "hrc" / "hash_seed0"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "dataset": "cifar10",
                    "head": "hrc",
                    "seed": 0,
                    "epochs": epoch,
                    "stage": stage,
                    "mean_init": "zero" if stage == "init_confirm" else None,
                }
            )
        )
        (run_dir / "history.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": epoch,
                        "val_accuracy": 0.9,
                        "val_nll": 0.3,
                        "val_brier": 0.1,
                        "val_ece": 0.02,
                    }
                ]
            )
        )
        (run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt").write_text("stub")

    visited = []

    def _fake_load(checkpoint, device=None):
        visited.append(pathlib.Path(checkpoint))
        return object(), {"epoch": int(pathlib.Path(checkpoint).stem.split("_")[1])}

    features = root / "features" / "cifar10"
    generator = torch.Generator().manual_seed(3)
    for name in ("test", "svhn"):
        save_feature_shard(
            features / f"{name}.pt",
            {
                "features": torch.randn(8, 5, generator=generator),
                "logits": torch.randn(8, 10, generator=generator),
                "labels": torch.arange(8) % 10,
            },
            {"fingerprint": name},
        )

    monkeypatch.setattr(
        runner.TAGILastLayerClassifier, "load", staticmethod(_fake_load)
    )
    monkeypatch.setattr(
        runner,
        "predict_batches",
        lambda classifier, inputs, batch_size: (
            torch.full((inputs.shape[0], 10), 0.1),
            torch.full((inputs.shape[0],), 0.5),
        ),
    )

    runner.evaluate_study(
        SimpleNamespace(dataset="cifar10", stage="init_confirm", device="cpu", batch_size=4),
        manifest,
    )

    assert [path.stem for path in visited] == ["epoch_0007"]
    evaluations = root / "evaluations" / "cifar10"
    assert (evaluations / "pytorch_softmax.json").exists()
    written = sorted(
        path.relative_to(evaluations).as_posix()
        for path in evaluations.glob("**/epoch_*.json")
    )
    assert written == ["init_confirm/hrc/hash_seed0/epoch_0007.json"]


def _init_manifest(tmp_path):
    return {
        "study_id": "smoke",
        "paths": {"artifacts": str(tmp_path)},
        "last_layer": {"batch_size": 8, "diagnostic_cohort_size": 8},
        "init_study": {
            "fixed_noise_heads": ["hrc", "remax_lognormal"],
            "logit_heads": ["logit_tagiv"],
            "hrc_trees": ["padded", "full"],
            "mean_init": ["random", "zero", "backbone"],
            "tied_gains": [0.1, 0.3],
            "sigma_v": [0.1],
            "screen": {"seeds": [0], "epochs": 20, "checkpoints": [0, 20]},
            "confirm": {"seeds": [0], "epochs": 200, "checkpoints": [0, 200]},
        },
    }


def test_hrc_is_crossed_with_both_trees_and_nothing_else_is(tmp_path):
    manifest = _init_manifest(tmp_path)
    configs = runner.init_screen_configs(manifest)

    hrc = [c for c in configs if c["head"] == "hrc"]
    remax = [c for c in configs if c["head"] == "remax_lognormal"]
    assert len(hrc) == 12  # 3 arms x 2 gains x 1 sigma_v x 2 trees
    assert len(remax) == 6  # no tree to vary
    assert all("hrc_tree" not in c for c in remax)
    assert {c.get("hrc_tree") for c in hrc} == {None, "full"}


def test_the_default_tree_arm_keeps_the_run_hash_of_the_cells_already_on_disk(tmp_path):
    """The padded arm must not carry the key, or 116 finished cells re-run.

    hrc_tree="auto" resolves to padded for the hrc head, so the cells already
    computed recorded no hrc_tree at all. Emitting it explicitly would change
    stable_hash and silently recompute an identical grid.
    """

    manifest = _init_manifest(tmp_path)
    configs = runner.init_screen_configs(manifest)
    padded = [c for c in configs if c["head"] == "hrc" and "hrc_tree" not in c]
    assert len(padded) == 6
    for config in padded:
        assert set(config) == {"head", "sigma_v", "gain_w", "gain_b", "mean_init"}


def test_selection_arm_separates_the_two_trees(tmp_path):
    assert runner.selection_arm({"head": "hrc"}) == "hrc"
    assert runner.selection_arm({"head": "hrc", "hrc_tree": "padded"}) == "hrc"
    assert runner.selection_arm({"head": "hrc", "hrc_tree": "full"}) == "hrc:full"
    # the probit head's default is the full tree, so that one keeps its name
    assert runner.selection_arm({"head": "hrc_probit", "hrc_tree": "full"}) == "hrc_probit"
    assert runner.selection_arm({"head": "remax_lognormal"}) == "remax_lognormal"


def test_select_keeps_a_winner_per_tree_and_carries_the_tree_forward(tmp_path):
    """Both trees must survive selection, with hrc_tree in the chosen config.

    Grouping by head alone made the two trees compete for one slot, and the
    config projection dropped hrc_tree, so a winning full-tree cell would have
    been confirmed on the padded tree -- and silently, since padded is what
    hrc_tree="auto" resolves to.
    """

    manifest = _init_manifest(tmp_path)
    root = runner.stage_root(manifest, "init_screen", "cifar10")
    cells = [
        ("padded_win", None, 0.90, 0.40),
        ("padded_lose", None, 0.89, 0.55),
        ("full_win", "full", 0.88, 0.30),
        ("full_lose", "full", 0.87, 0.61),
    ]
    for name, tree, accuracy, nll in cells:
        run_dir = root / "hrc" / name
        run_dir.mkdir(parents=True)
        config = {
            "head": "hrc", "seed": 0, "sigma_v": 0.1,
            "gain_w": 0.3, "gain_b": 0.3, "mean_init": "zero",
            "dataset": "cifar10", "stage": "init_screen", "epochs": 20,
        }
        if tree:
            config["hrc_tree"] = tree
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "history.json").write_text(
            json.dumps(
                [{"epoch": 20, "val_accuracy": accuracy, "val_nll": nll,
                  "val_brier": 0.2, "val_ece": 0.03}]
            )
        )

    runner.select_stage(
        SimpleNamespace(stage="init_screen", dataset="cifar10"), manifest
    )
    selected = json.loads((root / "selection.json").read_text())["selected"]

    assert set(selected) == {"hrc", "hrc:full"}
    assert selected["hrc"]["run_dir"].endswith("padded_win")
    assert selected["hrc:full"]["run_dir"].endswith("full_win")
    assert "hrc_tree" not in selected["hrc"]["config"]
    assert selected["hrc:full"]["config"]["hrc_tree"] == "full"

    # ...and the confirm grid therefore carries the tree into training.
    confirm = runner.init_confirm_configs(manifest, "cifar10")
    full_arms = [c for c in confirm if c.get("hrc_tree") == "full"]
    assert len(full_arms) == 2  # the selected arm and its random counterpart
    assert {c["mean_init"] for c in full_arms} == {"zero", "random"}


def test_calibrate_refuses_when_no_full_tree_run_exists(tmp_path):
    """The padded tree cannot be gain-calibrated, and the error must say so."""

    manifest = _init_manifest(tmp_path)
    root = runner.stage_root(manifest, "init_confirm", "cifar10")
    run_dir = root / "hrc" / "padded_only"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {"head": "hrc", "seed": 0, "epochs": 200, "dataset": "cifar10",
             "stage": "init_confirm", "mean_init": "zero"}
        )
    )
    (run_dir / "history.json").write_text(json.dumps([]))

    features = tmp_path / "smoke" / "features" / "cifar10"
    generator = torch.Generator().manual_seed(5)
    for name in ("validation", "test", "svhn"):
        save_feature_shard(
            features / f"{name}.pt",
            {
                "features": torch.randn(8, 5, generator=generator),
                "logits": torch.randn(8, 10, generator=generator),
                "labels": torch.arange(8) % 10,
            },
            {"fingerprint": name},
        )

    with pytest.raises(FileNotFoundError, match="hrc_tree=full"):
        runner.calibrate_stage(
            SimpleNamespace(
                dataset="cifar10", stage="init_confirm", sharing=None,
                method="grid", skip_corruptions=True, force=False,
                batch_size=4, device="cpu",
            ),
            manifest,
        )


def test_select_never_picks_the_untrained_prior(tmp_path):
    """Epoch 0 is the prior and must not win, however good it scores.

    For the backbone arm epoch 0 is the backbone's own trained fc, so it
    scores like the softmax baseline; selecting it would report the study's
    own baseline as a TAGI result and would confirm an untrained classifier
    at 200 epochs.
    """

    manifest = _init_manifest(tmp_path)
    root = runner.stage_root(manifest, "init_screen", "cifar10")
    run_dir = root / "remax_lognormal" / "warm_start"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "head": "remax_lognormal", "seed": 0, "sigma_v": 0.1,
                "gain_w": 1.0, "gain_b": 1.0, "mean_init": "backbone",
                "dataset": "cifar10", "stage": "init_screen", "epochs": 20,
            }
        )
    )
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                # the warm start: the best NLL in the run, and untrained
                {"epoch": 0, "val_accuracy": 0.9539, "val_nll": 0.1913,
                 "val_brier": 0.07, "val_ece": 0.0476},
                {"epoch": 20, "val_accuracy": 0.9512, "val_nll": 0.3943,
                 "val_brier": 0.08, "val_ece": 0.0346},
            ]
        )
    )

    runner.select_stage(
        SimpleNamespace(stage="init_screen", dataset="cifar10"), manifest
    )
    selected = json.loads((root / "selection.json").read_text())["selected"]
    assert selected["remax_lognormal"]["record"]["epoch"] == 20
