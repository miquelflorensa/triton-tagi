"""
Tests for inference-based initialization (triton_tagi/inference_init.py).

Checks that:
  1. _apply_constraints reduces |E[Z]| toward 0
  2. _apply_constraints moves E[Z^2] toward sigma_total_sq
  3. inference_init runs without error on a small FNN
  4. After inference_init, the forward-pass E[Z^2] is close to target
  5. Weight variances remain positive after update
  6. skip_last=True leaves the last Linear layer unchanged
  7. Large and tiny probe batches both succeed
"""

import torch

from triton_tagi import Sequential
from triton_tagi.inference_init import _apply_constraints, inference_init
from triton_tagi.kernels.common import triton_fused_var_forward
from triton_tagi.layers import Linear, ReLU, Remax

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

SIGMA_M = 1.0
SIGMA_Z = 1.0
SIGMA_M_SQ = SIGMA_M**2
SIGMA_Z_SQ = SIGMA_Z**2
S_TOT_SQ = SIGMA_M_SQ + SIGMA_Z_SQ  # 2.0


# ======================================================================
#  Helpers
# ======================================================================


def make_net(dims=(784, 256, 128, 10)):
    layers = []
    for i in range(len(dims) - 2):
        layers += [Linear(dims[i], dims[i + 1], device=DEVICE), ReLU()]
    layers += [Linear(dims[-2], dims[-1], device=DEVICE), Remax()]
    return Sequential(layers, device=DEVICE)


def get_layer_ez2(net, x, layer_idx):
    """E[Z^2] = mean(mz^2 + Sz) after forward through layer at layer_idx."""
    ma, Sa = x, torch.zeros_like(x)
    with torch.no_grad():
        for i, layer in enumerate(net.layers):
            if isinstance(layer, Linear):
                mz = torch.matmul(ma, layer.mw) + layer.mb
                Sz = triton_fused_var_forward(ma, Sa, layer.mw, layer.Sw, layer.Sb)
                if i == layer_idx:
                    return (mz**2 + Sz).mean().item()
                ma, Sa = mz, Sz
            else:
                ma, Sa = layer.forward(ma, Sa)
    return None


# ======================================================================
#  _apply_constraints unit tests
# ======================================================================


class TestApplyConstraints:
    def _random_hz(self, B=256, A=128):
        mz = torch.randn(B, A, device=DEVICE) * 2.0
        Sz = torch.ones(B, A, device=DEVICE) * 0.5
        return mz, Sz

    def test_sum_mean_goes_to_zero(self):
        mz, Sz = self._random_hz()
        A = mz.shape[1]
        mz_p, Sz_p = _apply_constraints(mz, Sz, A, SIGMA_M_SQ, SIGMA_Z_SQ, n_iter=50)
        assert mz_p.sum(dim=1).abs().max().item() < 0.05

    def test_sum_sq_goes_to_target(self):
        mz, Sz = self._random_hz()
        B, A = mz.shape
        mz_p, Sz_p = _apply_constraints(mz, Sz, A, SIGMA_M_SQ, SIGMA_Z_SQ, n_iter=50)
        Ez2_per_sample = (mz_p**2 + Sz_p).mean(dim=1)  # per-sample average
        err = (Ez2_per_sample - S_TOT_SQ).abs().max().item()
        assert err < 0.3, f"S2 constraint failed: max E[Z²] error = {err:.4f}"

    def test_variance_rescaling_exact(self):
        """After constraint, Σ Sz should be A * sigma_Z²."""
        mz, Sz = self._random_hz()
        A = mz.shape[1]
        _, Sz_p = _apply_constraints(mz, Sz, A, SIGMA_M_SQ, SIGMA_Z_SQ, n_iter=5)
        sig2_S = Sz_p.sum(dim=1)
        target = A * SIGMA_Z_SQ
        err = ((sig2_S - target) / target).abs().max().item()
        assert err < 0.01, f"Variance rescaling not exact: relative error {err:.4f}"

    def test_variances_stay_positive(self):
        mz, Sz = self._random_hz()
        _, Sz_p = _apply_constraints(mz, Sz, mz.shape[1], SIGMA_M_SQ, SIGMA_Z_SQ)
        assert (Sz_p > 0).all()

    def test_already_at_target_is_stable(self):
        """When mz²=σ_M² and Sz=σ_Z², all constraints are already met.
        The sum mean is 0 (alternating signs), Σ Sz = A·σ_Z²,
        and Σ(mz²+Sz) = A·(σ_M²+σ_Z²)."""
        B, A = 256, 64
        # Alternating signs so Σ mz = 0, each mz² = σ_M²
        signs = torch.ones(A, device=DEVICE)
        signs[1::2] = -1.0
        mz = signs.unsqueeze(0).expand(B, A) * (SIGMA_M_SQ**0.5)
        Sz = torch.full((B, A), SIGMA_Z_SQ, device=DEVICE)
        mz_p, Sz_p = _apply_constraints(mz, Sz, A, SIGMA_M_SQ, SIGMA_Z_SQ, n_iter=50)
        assert (mz_p - mz).abs().max().item() < 1e-2
        assert (Sz_p - Sz).abs().max().item() < 1e-2


# ======================================================================
#  inference_init integration tests
# ======================================================================


class TestInferenceInit:
    def _probe(self, B=1024, D=784):
        return torch.randn(B, D, device=DEVICE)

    def test_runs_without_error(self):
        net = make_net()
        inference_init(net, self._probe(), verbose=False)

    def test_weight_variances_positive(self):
        net = make_net()
        inference_init(net, self._probe(), verbose=False)
        for layer in net.layers:
            if isinstance(layer, Linear):
                assert (layer.Sw > 0).all(), "Sw has non-positive entries"
                assert (layer.Sb > 0).all(), "Sb has non-positive entries"

    def test_skip_last_leaves_last_layer_unchanged(self):
        net = make_net((784, 128, 10))
        x = self._probe()
        mw_before = net.layers[-2].mw.clone()
        Sw_before = net.layers[-2].Sw.clone()
        inference_init(net, x, skip_last=True, verbose=False)
        assert torch.allclose(net.layers[-2].mw, mw_before)
        assert torch.allclose(net.layers[-2].Sw, Sw_before)

    def test_skip_last_false_updates_last_layer(self):
        net = make_net((784, 128, 10))
        Sw_before = net.layers[-2].Sw.clone()
        inference_init(net, self._probe(), skip_last=False, verbose=False)
        assert not torch.allclose(net.layers[-2].Sw, Sw_before)

    def test_hidden_unit_variance_near_target(self):
        """After inference_init, E[Z^2] at the first Linear layer should be
        close to sigma_total_sq."""
        net = make_net((784, 256, 10))
        x = self._probe()
        inference_init(net, x, sigma_M=SIGMA_M, sigma_Z=SIGMA_Z, skip_last=True, verbose=False)
        ez2 = get_layer_ez2(net, x, layer_idx=0)
        # Should be within 20% of target
        err = abs(ez2 - S_TOT_SQ) / S_TOT_SQ
        assert err < 0.2, (
            f"E[Z²]={ez2:.4f} too far from target {S_TOT_SQ:.4f} (relative error {err:.2%})"
        )

    def test_small_probe_batch(self):
        net = make_net((784, 64, 10))
        inference_init(net, self._probe(B=64), verbose=False)

    def test_large_probe_batch(self):
        net = make_net((784, 128, 10))
        inference_init(net, self._probe(B=2048), verbose=False)

    def test_different_sigma_targets(self):
        for sigma_M, sigma_Z in [(0.5, 0.5), (2.0, 0.0), (0.1, 0.9)]:
            net = make_net((784, 64, 10))
            inference_init(net, self._probe(), sigma_M=sigma_M, sigma_Z=sigma_Z, verbose=False)
            for layer in net.layers:
                if isinstance(layer, Linear):
                    assert (layer.Sw > 0).all()


# ======================================================================
#  Run
# ======================================================================

if __name__ == "__main__":
    import time

    print(f"Device: {DEVICE}")
    print(f"Target σ_total² = {S_TOT_SQ}")
    print()

    for label, cls in [
        ("_apply_constraints", TestApplyConstraints),
        ("inference_init", TestInferenceInit),
    ]:
        print(f"─── {label} " + "─" * (55 - len(label)))
        obj = cls()
        for name in sorted(dir(obj)):
            if name.startswith("test_"):
                t0 = time.perf_counter()
                try:
                    getattr(obj, name)()
                    dt = time.perf_counter() - t0
                    print(f"  PASS  {name}  ({dt * 1000:.0f}ms)")
                except AssertionError as e:
                    print(f"  FAIL  {name}: {e}")
        print()
