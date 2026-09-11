"""Specification checks for the exact-K direct-probit ``ProbiTree``."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.probitree import (
    ProbiTree,
    gate_update,
    label_updates,
    predict_log_probs,
    probit_stats,
    probitree_label_update,
    probitree_log_probs,
    uniform_reference_means,
)


def test_one_gate_reference_value() -> None:
    update = gate_update(0.0, 1.0, 1, r=1.0)
    assert update["logp"] == pytest.approx(math.log(0.5), abs=1e-15)
    assert update["m_post"] == pytest.approx(1.0 / math.sqrt(math.pi), abs=1e-14)
    assert update["v_post"] == pytest.approx(1.0 - 1.0 / math.pi, abs=1e-14)


@pytest.mark.parametrize(
    ("m", "v", "r", "b"),
    [(0.0, 0.0, 1.0, 1), (0.7, 0.2, 0.4, 1), (-4.0, 3.0, 2.0, -1)],
)
def test_sign_symmetry_and_variance_bounds(m: float, v: float, r: float, b: int) -> None:
    update = gate_update(m, v, b, r)
    reflected = gate_update(-m, v, -b, r)
    assert update["m_post"] == pytest.approx(-reflected["m_post"], abs=1e-13)
    assert update["v_post"] == pytest.approx(reflected["v_post"], abs=1e-13)
    assert update["logp"] == pytest.approx(reflected["logp"], abs=1e-13)
    assert v * r / (r + v) - 1e-14 <= update["v_post"] <= v + 1e-14


def test_deterministic_gate_has_zero_moment_change_and_finite_messages() -> None:
    update = gate_update(3.0, 0.0, -1, r=0.25)
    assert update["dm"] == 0.0
    assert update["dv"] == 0.0
    assert update["m_post"] == 3.0
    assert update["v_post"] == 0.0
    assert math.isfinite(update["g"])
    assert math.isfinite(update["hess"])


def test_k4_preorder_paths_and_path_mask() -> None:
    tree = ProbiTree(4)
    assert tree.left == (1, -1, -3)
    assert tree.right == (2, -2, -4)
    assert tree.paths == (
        ((0, 1), (1, 1)),
        ((0, 1), (1, -1)),
        ((0, -1), (2, 1)),
        ((0, -1), (2, -1)),
    )
    updates, evidence = label_updates(tree, [0.0] * 3, [1.0] * 3, 2)
    assert set(updates) == {0, 2}
    assert updates[0]["m_post"] < 0.0
    assert updates[2]["m_post"] > 0.0
    assert evidence == pytest.approx(math.log(0.25), abs=1e-14)


@pytest.mark.parametrize("num_classes", [2, 3, 5, 10, 31, 100])
def test_exact_k_tree_shape_depth_and_normalization(num_classes: int) -> None:
    tree = ProbiTree(num_classes)
    assert tree.num_gates == num_classes - 1
    assert tree.max_depth == math.ceil(math.log2(num_classes))
    means = [math.sin(node + 0.25) * 2.0 for node in range(tree.num_gates)]
    variances = [0.1 + (node % 7) / 3.0 for node in range(tree.num_gates)]
    log_probabilities = predict_log_probs(tree, means, variances, r=0.7)
    assert math.fsum(math.exp(value) for value in log_probabilities) == pytest.approx(
        1.0, abs=2e-14
    )


@pytest.mark.parametrize("num_classes", [3, 5, 10, 17])
def test_non_power_of_two_paths_only_update_active_gates(num_classes: int) -> None:
    tree = ProbiTree(num_classes)
    means = torch.linspace(-1.0, 1.0, tree.num_gates).reshape(1, -1)
    variances = torch.linspace(0.1, 1.0, tree.num_gates).reshape(1, -1)
    for label in range(num_classes):
        update = probitree_label_update(tree, means, variances, torch.tensor([label]), r=1.3)
        active = {node for node, _ in tree.paths[label]}
        nonzero = set(torch.nonzero(update.g[0], as_tuple=False).reshape(-1).tolist())
        assert nonzero == active
        for node in set(range(tree.num_gates)) - active:
            assert update.delta_mean[0, node] == 0.0
            assert update.delta_variance[0, node] == 0.0


@pytest.mark.parametrize("num_classes", [3, 5, 10])
def test_evidence_matches_pre_update_predictive_probability(num_classes: int) -> None:
    generator = torch.Generator().manual_seed(num_classes)
    tree = ProbiTree(num_classes)
    means = torch.randn(4, tree.num_gates, generator=generator, dtype=torch.float64)
    variances = torch.rand(4, tree.num_gates, generator=generator, dtype=torch.float64) * 3.0
    labels = torch.tensor([0, 1, num_classes - 2, num_classes - 1])
    update = probitree_label_update(tree, means, variances, labels, r=0.8)
    log_probabilities = probitree_log_probs(tree, means, variances, r=0.8)
    expected = log_probabilities.gather(1, labels[:, None]).squeeze(1)
    torch.testing.assert_close(update.log_evidence, expected, rtol=0.0, atol=2e-14)


@pytest.mark.parametrize("num_classes", [3, 5, 10, 31])
def test_reference_means_give_uniform_class_probabilities(num_classes: int) -> None:
    tree = ProbiTree(num_classes)
    variances = [0.05 + (node % 5) * 0.4 for node in range(tree.num_gates)]
    means = uniform_reference_means(tree, variances, r=1.7)
    probabilities = [math.exp(value) for value in predict_log_probs(tree, means, variances, 1.7)]
    assert probabilities == pytest.approx([1.0 / num_classes] * num_classes, abs=3e-15)


@pytest.mark.parametrize("margin", [-34.999, -35.001, -50.0, -100.0, -1000.0])
def test_rare_label_updates_remain_finite_and_nonnegative(margin: float) -> None:
    m = margin * math.sqrt(2.0)
    logp, lam, kappa = probit_stats(margin)
    update = gate_update(m, 1.0, 1, r=1.0)
    assert all(math.isfinite(value) for value in update.values())
    assert logp < 0.0 and lam > 0.0 and 0.0 <= kappa <= 1.0
    assert 0.0 <= update["v_post"] <= 1.0
    # erfcx is an independent scaled-integral implementation of the ratio.
    reference = math.sqrt(2.0 / math.pi) / float(
        torch.special.erfcx(torch.tensor(-margin / math.sqrt(2.0), dtype=torch.float64))
    )
    assert lam == pytest.approx(reference, rel=3e-9)


def test_tensor_and_scalar_gate_updates_agree() -> None:
    tree = ProbiTree(2)
    means = torch.tensor([[0.7], [-5.0]], dtype=torch.float64)
    variances = torch.tensor([[1.2], [0.3]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    update = probitree_label_update(tree, means, variances, labels, r=0.6)
    for row, sign in enumerate((1, -1)):
        expected = gate_update(float(means[row, 0]), float(variances[row, 0]), sign, 0.6)
        assert update.posterior_mean[row, 0] == pytest.approx(expected["m_post"], abs=1e-14)
        assert update.posterior_variance[row, 0] == pytest.approx(expected["v_post"], abs=1e-14)
        assert update.g[row, 0] == pytest.approx(expected["g"], abs=1e-14)
        assert update.h[row, 0] == pytest.approx(expected["hess"], abs=1e-14)


def test_message_adapter_matches_dense_conditional_moment_update() -> None:
    tree = ProbiTree(2)
    mean_f = torch.tensor([[0.4]], dtype=torch.float64)
    variance_f = torch.tensor([[1.7]], dtype=torch.float64)
    update = probitree_label_update(tree, mean_f, variance_f, torch.tensor([0]), r=0.9)
    covariance_xf = 0.65
    prior_mean_x = -0.2
    prior_variance_x = 2.3

    message_mean = prior_mean_x + covariance_xf * float(update.g[0, 0])
    message_variance = prior_variance_x + covariance_xf**2 * float(update.h[0, 0])

    gain = covariance_xf / float(variance_f[0, 0])
    dense_mean = prior_mean_x + gain * float(update.delta_mean[0, 0])
    dense_variance = prior_variance_x + gain**2 * float(update.delta_variance[0, 0])
    assert message_mean == pytest.approx(dense_mean, abs=1e-15)
    assert message_variance == pytest.approx(dense_variance, abs=1e-15)


def test_tree_mapping_round_trip_and_classifier_checkpoint_config(tmp_path) -> None:
    tree = ProbiTree(10)
    assert ProbiTree.from_dict(tree.to_dict()) == tree
    classifier = TAGILastLayerClassifier(
        4,
        10,
        head="probitree",
        device="cpu",
        probitree_r=0.75,
        mean_init="zero",
    )
    assert classifier.linear.out_features == 9
    assert classifier.probitree == tree
    config = classifier.config()
    assert config["probitree_tree"] == tree.to_dict()
    assert config["probitree_r"] == 0.75
    path = classifier.save(tmp_path / "probitree.pt")
    restored, _ = TAGILastLayerClassifier.load(path, device="cpu")
    assert restored.probitree == tree
    assert restored.probitree_r == 0.75
    torch.testing.assert_close(restored.linear.mb, classifier.linear.mb)


@pytest.mark.cuda
def test_cuda_classifier_runs_one_direct_probit_backward_update() -> None:
    torch.manual_seed(4)
    classifier = TAGILastLayerClassifier(3, 5, head="probitree", device="cuda", probitree_r=0.8)
    sample = torch.tensor([[0.2, -0.5, 1.1]])
    label = torch.tensor([3])
    prediction = classifier.predict(sample)
    expected_loss = -float(prediction.diagnostics["class_log_probabilities"][0, 3].cpu())
    old_mean = classifier.linear.mw.clone()

    classifier.train_step(sample, label)

    assert classifier.last_pre_update_loss == pytest.approx(expected_loss, abs=1e-7)
    assert not torch.equal(classifier.linear.mw, old_mean)
    assert bool(torch.isfinite(classifier.linear.mw).all())
    assert bool(torch.isfinite(classifier.linear.Sw).all())
    assert bool((classifier.linear.Sw >= 0.0).all())
    updated = classifier.predict(sample)
    assert float(updated.diagnostics["class_probability_sum_deviation"]) < 1e-8
    torch.testing.assert_close(
        updated.probabilities.sum(1), torch.ones(1, device="cuda"), rtol=1e-6, atol=1e-6
    )
    history = classifier.fit(sample, label, epochs=1, batch_size=1, record_initial=False)
    assert history.records[0]["train_pre_update_nll"] > 0.0


def test_classifier_default_initialization_is_uniform_at_its_reference() -> None:
    classifier = TAGILastLayerClassifier(3, 5, head="probitree", device="cpu")
    assert classifier.probitree is not None
    means = classifier.linear.mb.detach().double().reshape(-1).tolist()
    variances = classifier.probitree_reference_variances
    assert variances is not None
    probabilities = [
        math.exp(value)
        for value in predict_log_probs(
            classifier.probitree, means, variances, classifier.probitree_r
        )
    ]
    assert probabilities == pytest.approx([0.2] * 5, abs=2e-8)


def test_backend_rejects_labels_that_do_not_match_the_batch() -> None:
    classifier = TAGILastLayerClassifier(3, 4, head="probitree", device="cpu")
    with pytest.raises(ValueError, match="labels must have shape"):
        classifier.fit(torch.zeros(2, 3), torch.tensor([0, 1, 2]), epochs=1, batch_size=2)


@pytest.mark.cuda
def test_batched_backward_sums_the_per_sample_parameter_deltas() -> None:
    """A batch update must equal the sum of its samples' individual deltas.

    That summation *is* the TAGI batch approximation the other heads use, so
    pinning it here keeps the ProbiTree adapter honest about what batching
    means: cross-sample covariance between gate messages is neglected.
    """

    from triton_tagi.layers import Linear
    from triton_tagi.network import Sequential

    tree = ProbiTree(10)
    inputs = torch.randn(4, 6, device="cuda")
    labels = torch.tensor([0, 3, 7, 9], device="cuda")

    def fresh() -> tuple[Sequential, Linear]:
        torch.manual_seed(11)
        head = Linear(6, tree.num_gates, device="cuda")
        return Sequential([head], device="cuda"), head

    batched, batched_head = fresh()
    _, _, log_evidence = batched.step_probitree(inputs, labels, tree, r=0.7)
    assert log_evidence.shape == (4,)

    summed, summed_head = fresh()
    accumulated = torch.zeros_like(summed_head.mw)
    for row in range(4):
        summed.forward(inputs[row : row + 1])
        one = probitree_label_update(
            tree, *summed.forward(inputs[row : row + 1]), labels[row : row + 1], r=0.7
        )
        for layer in reversed(summed.layers):
            layer.backward(one.g.to(inputs.dtype), one.h.to(inputs.dtype))
        accumulated += summed_head.delta_mw

    torch.testing.assert_close(batched_head.delta_mw, accumulated, rtol=1e-5, atol=1e-6)
