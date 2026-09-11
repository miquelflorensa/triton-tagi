"""Unit tests for triton_tagi.hrc_softmax.

No cuTAGI dependency — purely tests the Python logic.
"""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.hrc_softmax import (
    HierarchicalSoftmax,
    class_to_obs,
    get_predicted_labels,
    labels_to_hrc,
    obs_to_class_probs,
    obs_to_class_probs_probit,
)
from triton_tagi.update.observation import (
    compute_innovation_with_indices,
    compute_probit_innovation_with_indices,
)


# ──────────────────────────────────────────────────────────────────────────────
#  class_to_obs
# ──────────────────────────────────────────────────────────────────────────────


class TestClassToObs:
    def test_n_obs_is_ceil_log2(self):
        for n in [2, 4, 8, 10, 16, 100]:
            hrc = class_to_obs(n)
            assert hrc.n_obs == math.ceil(math.log2(n))

    def test_obs_shape(self):
        hrc = class_to_obs(10)
        assert hrc.obs.shape == (10, hrc.n_obs)

    def test_idx_shape(self):
        hrc = class_to_obs(10)
        assert hrc.idx.shape == (10, hrc.n_obs)

    def test_obs_values_are_plus_minus_one(self):
        hrc = class_to_obs(10)
        assert torch.all((hrc.obs == 1.0) | (hrc.obs == -1.0))

    def test_idx_values_are_positive(self):
        hrc = class_to_obs(10)
        assert torch.all(hrc.idx >= 1)

    def test_idx_max_equals_len(self):
        hrc = class_to_obs(10)
        assert hrc.idx.max().item() == hrc.len

    def test_10classes_len_is_11(self):
        """cuTAGI uses Linear(hidden, 11) for MNIST — len must be 11."""
        hrc = class_to_obs(10)
        assert hrc.len == 11

    def test_10classes_n_obs_is_4(self):
        """ceil(log2(10)) = 4."""
        hrc = class_to_obs(10)
        assert hrc.n_obs == 4

    def test_2classes_len_is_1(self):
        """Binary classification: 1 output node, 1 bit."""
        hrc = class_to_obs(2)
        assert hrc.n_obs == 1
        assert hrc.len == 1

    def test_root_node_index_is_always_1(self):
        """All classes share the root (idx[:, 0] == 1)."""
        hrc = class_to_obs(10)
        assert torch.all(hrc.idx[:, 0] == 1)

    def test_obs_class0_all_positive(self):
        """Class 0 is encoded as all zeros → all observations are +1."""
        hrc = class_to_obs(10)
        assert torch.all(hrc.obs[0] == 1.0)

    def test_deterministic(self):
        """Two calls with same n_classes produce identical tensors."""
        hrc1 = class_to_obs(10)
        hrc2 = class_to_obs(10)
        torch.testing.assert_close(hrc1.obs, hrc2.obs)
        torch.testing.assert_close(hrc1.idx.float(), hrc2.idx.float())


# ──────────────────────────────────────────────────────────────────────────────
#  labels_to_hrc
# ──────────────────────────────────────────────────────────────────────────────


class TestLabelsToHrc:
    def setup_method(self):
        self.hrc = class_to_obs(10)

    def test_obs_shape(self):
        labels = torch.arange(10)
        y_obs, _ = labels_to_hrc(labels, self.hrc)
        assert y_obs.shape == (10, self.hrc.n_obs)

    def test_idx_shape(self):
        labels = torch.arange(10)
        _, y_idx = labels_to_hrc(labels, self.hrc)
        assert y_idx.shape == (10, self.hrc.n_obs)

    def test_matches_hrc_obs(self):
        labels = torch.arange(10)
        y_obs, _ = labels_to_hrc(labels, self.hrc)
        torch.testing.assert_close(y_obs, self.hrc.obs)

    def test_matches_hrc_idx(self):
        labels = torch.arange(10)
        _, y_idx = labels_to_hrc(labels, self.hrc)
        torch.testing.assert_close(y_idx.float(), self.hrc.idx.float())

    def test_single_label(self):
        label = torch.tensor([3])
        y_obs, y_idx = labels_to_hrc(label, self.hrc)
        torch.testing.assert_close(y_obs[0], self.hrc.obs[3])
        torch.testing.assert_close(y_idx[0].float(), self.hrc.idx[3].float())

    def test_repeated_labels(self):
        labels = torch.tensor([0, 0, 0])
        y_obs, _ = labels_to_hrc(labels, self.hrc)
        assert y_obs.shape[0] == 3
        assert torch.all(y_obs[0] == y_obs[1])


# ──────────────────────────────────────────────────────────────────────────────
#  obs_to_class_probs
# ──────────────────────────────────────────────────────────────────────────────


class TestObsToClassProbs:
    def setup_method(self):
        torch.manual_seed(42)
        self.hrc = class_to_obs(10)
        self.B = 8

    def test_output_shape(self):
        ma = torch.randn(self.B, self.hrc.len)
        Sa = torch.rand(self.B, self.hrc.len) * 0.1
        P = obs_to_class_probs(ma, Sa, self.hrc)
        assert P.shape == (self.B, 10)

    def test_probs_non_negative(self):
        ma = torch.randn(self.B, self.hrc.len)
        Sa = torch.rand(self.B, self.hrc.len) * 0.1
        P = obs_to_class_probs(ma, Sa, self.hrc)
        assert torch.all(P >= 0)

    def test_probs_at_most_one(self):
        ma = torch.randn(self.B, self.hrc.len)
        Sa = torch.rand(self.B, self.hrc.len) * 0.1
        P = obs_to_class_probs(ma, Sa, self.hrc)
        assert torch.all(P <= 1.0 + 1e-6)

    def test_high_mean_prefers_class0(self):
        """All nodes pushed strongly positive → class 0 (all +1) wins."""
        ma = torch.full((1, self.hrc.len), 10.0)
        Sa = torch.zeros(1, self.hrc.len)
        P = obs_to_class_probs(ma, Sa, self.hrc)
        assert P.argmax(dim=1).item() == 0

    def test_deterministic(self):
        torch.manual_seed(7)
        ma = torch.randn(self.B, self.hrc.len)
        Sa = torch.rand(self.B, self.hrc.len) * 0.1
        P1 = obs_to_class_probs(ma, Sa, self.hrc)
        P2 = obs_to_class_probs(ma, Sa, self.hrc)
        torch.testing.assert_close(P1, P2)

    def test_zero_variance_sum_to_one_approx(self):
        """With zero variance, probabilities sum to approximately 1
        (not exactly due to unused binary codes beyond n_classes)."""
        ma = torch.zeros(1, self.hrc.len)
        Sa = torch.zeros(1, self.hrc.len)
        P = obs_to_class_probs(ma, Sa, self.hrc)
        # Probs sum to ≤ 1 since some binary codes are unused
        assert P.sum().item() <= 1.0 + 1e-5


# ──────────────────────────────────────────────────────────────────────────────
#  get_predicted_labels
# ──────────────────────────────────────────────────────────────────────────────


class TestGetPredictedLabels:
    def setup_method(self):
        self.hrc = class_to_obs(10)

    def test_output_shape(self):
        B = 16
        ma = torch.randn(B, self.hrc.len)
        Sa = torch.rand(B, self.hrc.len) * 0.1
        preds = get_predicted_labels(ma, Sa, self.hrc)
        assert preds.shape == (B,)

    def test_values_in_range(self):
        B = 32
        ma = torch.randn(B, self.hrc.len)
        Sa = torch.rand(B, self.hrc.len) * 0.1
        preds = get_predicted_labels(ma, Sa, self.hrc)
        assert torch.all(preds >= 0)
        assert torch.all(preds < 10)

    def test_matches_argmax_of_probs(self):
        torch.manual_seed(99)
        B = 8
        ma = torch.randn(B, self.hrc.len)
        Sa = torch.rand(B, self.hrc.len) * 0.1
        P = obs_to_class_probs(ma, Sa, self.hrc)
        preds = get_predicted_labels(ma, Sa, self.hrc)
        torch.testing.assert_close(preds, P.argmax(dim=1))


# ──────────────────────────────────────────────────────────────────────────────
#  compute_innovation_with_indices
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeInnovationWithIndices:
    def setup_method(self):
        torch.manual_seed(42)
        self.hrc = class_to_obs(10)
        self.B = 4

    def _make_inputs(self):
        ma = torch.randn(self.B, self.hrc.len)
        Sa = torch.rand(self.B, self.hrc.len) * 0.1 + 1e-4
        labels = torch.randint(0, 10, (self.B,))
        y_obs, y_idx = labels_to_hrc(labels, self.hrc)
        var_obs = torch.full_like(y_obs, 0.05**2)
        return ma, Sa, y_obs, var_obs, y_idx

    def test_output_shape(self):
        ma, Sa, y_obs, var_obs, y_idx = self._make_inputs()
        dm, dS = compute_innovation_with_indices(ma, Sa, y_obs, var_obs, y_idx)
        assert dm.shape == (self.B, self.hrc.len)
        assert dS.shape == (self.B, self.hrc.len)

    def test_delta_Sa_non_positive(self):
        """dS should be ≤ 0 everywhere (innovation always reduces variance)."""
        ma, Sa, y_obs, var_obs, y_idx = self._make_inputs()
        _, dS = compute_innovation_with_indices(ma, Sa, y_obs, var_obs, y_idx)
        assert torch.all(dS <= 0)

    def test_sparsity(self):
        """Only n_obs nodes per sample should be non-zero."""
        ma, Sa, y_obs, var_obs, y_idx = self._make_inputs()
        dm, _ = compute_innovation_with_indices(ma, Sa, y_obs, var_obs, y_idx)
        n_nonzero_per_row = (dm != 0).sum(dim=1)
        # Each row should have at most n_obs non-zero entries
        assert torch.all(n_nonzero_per_row <= self.hrc.n_obs)

    def test_zero_obs_gives_negative_dm_at_positive_mean(self):
        """If obs=-1 and ma>0 at a selected node, delta_mu should be negative."""
        ma = torch.ones(1, self.hrc.len) * 2.0
        Sa = torch.ones(1, self.hrc.len) * 0.1
        hrc = class_to_obs(10)
        # Class 1 has obs=[+1,+1,+1,-1] — the last bit is -1
        y_obs, y_idx = labels_to_hrc(torch.tensor([1]), hrc)
        var_obs = torch.full_like(y_obs, 0.01)
        dm, _ = compute_innovation_with_indices(ma, Sa, y_obs, var_obs, y_idx)
        # Node for last bit of class 1: obs=-1, ma=2.0 → dm = (-1-2)/(...) < 0
        last_node = y_idx[0, -1].item() - 1
        assert dm[0, last_node].item() < 0

    def test_all_zeros_for_zero_delta(self):
        """Targets matching predictions give near-zero mean innovation."""
        hrc = class_to_obs(10)
        # class 0 has all obs=+1; set ma=100 at all nodes → Phi(100/...) ≈ 1
        # The predicted mean for selected nodes ≈ 1 ≈ obs → dm ≈ 0
        ma = torch.zeros(1, hrc.len)
        Sa = torch.ones(1, hrc.len) * 0.1
        y_obs = torch.ones(1, hrc.n_obs)   # all +1
        y_idx = hrc.idx[:1]                 # class 0 indices
        var_obs = torch.full_like(y_obs, 0.01)
        dm, dS = compute_innovation_with_indices(ma, Sa, y_obs, var_obs, y_idx)
        # Only selected nodes should be non-zero
        assert (dS != 0).sum().item() == hrc.n_obs


class TestHierarchicalProbit:
    def setup_method(self):
        self.hrc = class_to_obs(10)

    def test_uniform_initial_probabilities_for_cifar10(self):
        ma = torch.zeros(1, self.hrc.len)
        Sa = torch.full_like(ma, 0.7)
        raw = obs_to_class_probs_probit(ma, Sa, self.hrc)
        torch.testing.assert_close(raw, torch.full_like(raw, 1.0 / 16.0))
        normalized = raw / raw.sum(dim=1, keepdim=True)
        torch.testing.assert_close(normalized, torch.full_like(raw, 0.1))

    def test_prediction_uses_fixed_unit_link_scale(self):
        ma = torch.full((1, self.hrc.len), 0.5)
        Sa = torch.zeros_like(ma)
        probabilities = obs_to_class_probs_probit(ma, Sa, self.hrc)
        q = torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(0.5))
        torch.testing.assert_close(probabilities[0, 0], q**self.hrc.n_obs)

        with pytest.raises(TypeError):
            obs_to_class_probs_probit(ma, Sa, self.hrc, sigma_v=0.3)

    def test_reconstructs_exact_skew_normal_output_moments(self):
        ma = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
        Sa = torch.tensor([[0.6, 0.3]], dtype=torch.float64)
        signs = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        var_obs = torch.tensor([[0.25, 1.0]], dtype=torch.float64)
        indices = torch.tensor([[1, 2]])

        dm, dS = compute_probit_innovation_with_indices(
            ma, Sa, signs, var_obs, indices
        )
        posterior_mean = ma + Sa * dm
        posterior_variance = Sa + Sa.square() * dS

        r2 = Sa + var_obs
        r = torch.sqrt(r2)
        gamma = signs * ma / r
        normal = torch.distributions.Normal(0.0, 1.0)
        mills = torch.exp(normal.log_prob(gamma)) / normal.cdf(gamma)
        expected_mean = ma + signs * Sa / r * mills
        expected_variance = Sa - Sa.square() / r2 * mills * (mills + gamma)

        torch.testing.assert_close(posterior_mean, expected_mean)
        torch.testing.assert_close(posterior_variance, expected_variance)

    def test_unvisited_nodes_are_unchanged(self):
        ma = torch.zeros(1, 3)
        Sa = torch.ones(1, 3)
        dm, dS = compute_probit_innovation_with_indices(
            ma,
            Sa,
            torch.tensor([[1.0, -1.0]]),
            torch.ones(1, 2),
            torch.tensor([[1, 3]]),
        )
        assert dm[0, 1] == 0
        assert dS[0, 1] == 0
        assert dm[0, 0] > 0 and dm[0, 2] < 0
        assert torch.all(dS <= 0)

    def test_inverse_mills_is_finite_in_extreme_left_tail(self):
        dm, dS = compute_probit_innovation_with_indices(
            torch.tensor([[-40.0]]),
            torch.tensor([[0.5]]),
            torch.ones(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.int64),
        )
        assert torch.isfinite(dm).all()
        assert torch.isfinite(dS).all()
        assert dm.item() > 0
        assert dS.item() < 0

    def test_invalid_sign_is_rejected(self):
        with pytest.raises(ValueError, match="signed"):
            compute_probit_innovation_with_indices(
                torch.zeros(1, 1),
                torch.ones(1, 1),
                torch.zeros(1, 1),
                torch.ones(1, 1),
                torch.ones(1, 1, dtype=torch.int64),
            )


class TestFullTreeBranchOffsets:
    """The full tree's branch offsets are a readout-only shift.

    ``class_to_obs_full`` stores ``offset[j] = Phi^-1(pi_j)`` so that a zero
    network output reproduces the branch prior, and ``hrc_probit.hrc_log_probs``
    adds ``tau * offset`` to the latent mean. ``network.step_hrc`` computes its
    innovation against the raw output and never reads ``hrc.offset``, so nothing
    the network learns depends on it: training fits the branch frequencies from
    data and the readout then applies the prior a second time.

    These tests pin that behaviour rather than endorse it. Making the offset a
    real prior means training against the shifted latent, which changes the
    model; until then the study runs the full tree with
    ``hrc_prior_offsets=False`` -- measured at gain 0.1 / sigma_v 0.3, 20
    epochs, zero init, the offsets cost 0.0233 NLL on CIFAR-100 and 0.0060 on
    CIFAR-10.
    """

    def test_offsets_change_the_tree_but_not_its_shape(self):
        from triton_tagi.hrc_softmax import class_to_obs_full

        with_offsets = class_to_obs_full(10)
        without = class_to_obs_full(10, use_prior_offsets=False)

        assert with_offsets.len == without.len == 9
        assert torch.equal(with_offsets.idx, without.idx)
        assert torch.equal(with_offsets.obs, without.obs)
        assert torch.equal(with_offsets.mask, without.mask)
        # only the unbalanced splits carry a nonzero shift
        assert bool((with_offsets.offset != 0).any())
        assert bool((without.offset == 0).all())

    def test_the_full_tree_is_not_the_padded_tree_with_padding_removed(self):
        """The two trees partition the classes differently, by construction.

        ``class_to_obs`` is a fixed-depth code, so for K = 10 its root splits
        eight classes against two; ``class_to_obs_full`` splits by recursive
        median, five against five. Neither is the other with dead nodes pruned,
        which is why an ``hrc:full`` row is not an ``hrc`` row minus padding.
        """

        from triton_tagi.hrc_softmax import class_to_obs_full

        padded, full = class_to_obs(10), class_to_obs_full(10)
        root_sign_padded = padded.obs[:, 0]
        root_sign_full = full.obs[:, 0]
        assert int((root_sign_padded > 0).sum()) == 8
        assert int((root_sign_full > 0).sum()) == 5

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the TAGI update kernels are Triton/CUDA"
    )
    def test_training_is_blind_to_the_offset(self):
        """Identical weights either way: no update ever depends on the offset."""

        from triton_tagi.classification import TAGILastLayerClassifier

        features = torch.randn(64, 8, generator=torch.Generator().manual_seed(0))
        labels = torch.arange(64) % 10

        def train(use_offsets: bool):
            torch.manual_seed(0)
            classifier = TAGILastLayerClassifier(
                8, 10, head="hrc", hrc_tree="full",
                hrc_prior_offsets=use_offsets, device="cuda",
                gain_w=0.1, gain_b=0.1, sigma_v=0.3, mean_init="zero",
            )
            classifier.fit(features, labels, epochs=2, batch_size=32, record_initial=False)
            return classifier

        on, off = train(True), train(False)
        assert bool((on.hrc.offset != 0).any())
        assert torch.allclose(on.linear.mw, off.linear.mw)
        assert torch.allclose(on.linear.Sw, off.linear.Sw)
        # ...and yet the readout differs, which is the whole defect
        probabilities_on = on.predict(features, sigma_v=on.sigma_v).probabilities
        on.hrc.offset = torch.zeros_like(on.hrc.offset)
        probabilities_off = on.predict(features, sigma_v=on.sigma_v).probabilities
        assert not torch.allclose(probabilities_on, probabilities_off)
