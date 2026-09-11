"""EvenProbit activation layer — the CDF variance head for TAGI-V.

The interleaved ``2K`` TAGI-V output layout is shared with
:class:`~triton_tagi.layers.EvenExp` and
:class:`~triton_tagi.layers.EvenSoftplus`:

    - Even indices (0, 2, 4, ...): the prediction stream ``Z_i`` → identity
    - Odd  indices (1, 3, 5, ...): the variance head ``U_i`` → ``h(U_i)``

What changes is the activation on the odd stream. ``EvenExp`` sends
``U_i`` through ``exp``, which is positive but unbounded, so a variance head
that drifts either saturates at the floor or diverges. Here the activation is
the scaled Gaussian CDF of :mod:`triton_tagi.cdf_variance`,

    ``V2bar_i = h(U_i) := epsilon + kappa Phi(U_i)``,

which confines the learned aleatoric variance to ``(epsilon, epsilon + kappa)``
for every finite ``U_i``. Both constants carry units of squared logits and are
fixed before calibration.

The forward moments are *exact* for a Gaussian ``U_i ~ N(nu_i, r_i)`` — see
:func:`triton_tagi.cdf_variance.cdf_variance_moments` — so no delta-method
linearisation enters the layer.

Backward is an **identity passthrough**, for the same structural reason as in
``EvenExp`` but with a different mechanism: the training update
:func:`triton_tagi.update.observation.compute_cdf_tagiv_innovation` integrates
the head ``U_i`` explicitly by quadrature and returns normalized deltas that
are *already* in the pre-activation ``(Z, U)`` space. There is no activation
Jacobian left to apply. ``EvenProbit`` is therefore a forward-only moment layer
for the TAGI-V output head and must not be used as a generic hidden activation.

Because those deltas are stated against the head's own Gaussian ``(nu_i, r_i)``
and that Gaussian is not recoverable from the post-activation moments — the CDF
is a saturating many-to-one map in moment space — :meth:`EvenProbit.forward`
caches the odd-slot pre-activation moments on ``self``.
"""

from __future__ import annotations

from torch import Tensor

from ..base import Layer
from ..cdf_variance import cdf_variance_moments
from ..hsm_calibration import DEFAULT_OWEN_ORDER


class EvenProbit(Layer):
    """Scaled Gaussian-CDF variance-head activation, interleaved 2K layout.

    Applies ``h(u) = epsilon + kappa Phi(u)`` to odd-indexed positions through
    its exact Gaussian moments and passes even-indexed positions (the
    prediction stream) through unchanged. Output width must be
    ``2 * half_width``.

    Args:
        half_width: ``K``, the number of target dimensions (even/odd pairs).
        epsilon: Strictly positive variance floor of the activation.
        kappa: Strictly positive variance range of the activation, so the
            learned noise lies in ``(epsilon, epsilon + kappa)``.
        order: Gauss--Legendre order for the ``Var[Phi]`` integral behind
            :func:`~triton_tagi.cdf_variance.cdf_variance_moments`.

    Attributes:
        nu: Cached odd-slot pre-activation means ``nu_i``, float64.
        r: Cached odd-slot pre-activation variances ``r_i``, float64.
        cov_u_h: Cached ``Cov(U_i, V2bar_i)``, float64.
    """

    def __init__(
        self,
        half_width: int,
        *,
        epsilon: float,
        kappa: float,
        order: int = DEFAULT_OWEN_ORDER,
    ) -> None:
        if half_width < 1:
            raise ValueError(f"half_width must be positive, got {half_width}")
        if not epsilon > 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not kappa > 0.0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        self.half_width = int(half_width)
        self.epsilon = float(epsilon)
        self.kappa = float(kappa)
        self.order = int(order)
        self.nu: Tensor | None = None
        self.r: Tensor | None = None
        self.cov_u_h: Tensor | None = None

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """Map the variance head through the CDF activation's exact moments.

        Even slots are copied unchanged. For the odd-slot head prior
        ``(nu, r) = (mz[..., 1::2], Sz[..., 1::2])`` the outputs are

            ``ma[..., 1::2] = E[V2bar] = epsilon + kappa Phi(nu / sqrt(1 + r))``
            ``Sa[..., 1::2] = Var(V2bar) = kappa^2 Var[Phi(U)]``.

        ``Sa`` on the odd slots is ``Var(V2bar)``: uncertainty *about* a
        variance. It is **not** an additive noise variance and must never be
        added to the predictive ``e_i + h_i``. It belongs in fourth moments and
        in nonlinear predictive averages only.

        The odd-slot pre-activation moments ``(nu, r)`` and ``Cov(U, V2bar)``
        are cached on ``self`` for the training update, and the cache is
        overwritten on every call.

        Args:
            mz: Pre-activation means, shape (..., 2K).
            Sz: Pre-activation variances, shape (..., 2K), non-negative.

        Returns:
            ma: Post-activation means, shape (..., 2K).
            Sa: Post-activation variances, shape (..., 2K).
        """

        if mz.shape != Sz.shape:
            raise ValueError("EvenProbit moments must have matching shapes")
        if mz.dim() < 1 or mz.shape[-1] != 2 * self.half_width:
            raise ValueError(
                f"EvenProbit expects a width of {2 * self.half_width}, got {tuple(mz.shape)}"
            )

        odd = slice(1, None, 2)
        nu = mz[..., odd].double()
        r = Sz[..., odd].double().clamp_min(0.0)
        h_mean, h_variance, cov_u_h = cdf_variance_moments(
            nu, r, epsilon=self.epsilon, kappa=self.kappa, order=self.order
        )
        self.nu = nu
        self.r = r
        self.cov_u_h = cov_u_h

        ma = mz.clone()
        Sa = Sz.clone()
        ma[..., odd] = h_mean.to(ma.dtype)
        Sa[..., odd] = h_variance.clamp_min(0.0).to(Sa.dtype)
        return ma, Sa

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Pass the innovation deltas through unchanged.

        :func:`~triton_tagi.update.observation.compute_cdf_tagiv_innovation`
        integrates ``U_i`` under the observation likelihood directly and returns
        normalized deltas against the pre-activation pair ``(mu_i, e_i)`` and
        ``(nu_i, r_i)``. No activation Jacobian remains to be applied, so —
        exactly as in :class:`EvenExp`, where the smoother gain is instead
        folded into the updater — this backward is the identity.

        Args:
            delta_ma: Mean deltas, already in pre-activation space.
            delta_Sa: Variance deltas, already in pre-activation space.

        Returns:
            The same two tensors, unchanged.
        """

        return delta_ma, delta_Sa

    def __repr__(self) -> str:
        return (
            f"EvenProbit(half_width={self.half_width}, epsilon={self.epsilon}, kappa={self.kappa})"
        )
