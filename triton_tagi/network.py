"""
Network builder — a Sequential container for TAGI layers.

Supports both MLP and CNN architectures:

    # MLP
    net = Sequential([
        Linear(784, 256), ReLU(),
        Linear(256, 10),  Remax(),
    ])

    # CNN
    net = Sequential([
        Conv2D(1, 32, 5, padding=2), ReLU(), AvgPool2D(2),
        Conv2D(32, 64, 5, padding=2), ReLU(), AvgPool2D(2),
        Flatten(),
        Linear(3136, 256), ReLU(),
        Linear(256, 10),   Remax(),
    ])

The step() method follows cuTAGI's architecture:
    1. Forward pass — propagate moments
    2. Compute output innovation
    3. Backward pass — compute and store deltas on each layer (NO update)
    4. Update — apply capped deltas to all learnable layers
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Layer, LearnableLayer
from .layers.even_probit import EvenProbit
from .layers.multihead_attention import MultiheadAttentionV2
from .layers.resblock import ResBlock
from .logit_tagiv import (
    LOGIT_VARIANCE_FLOOR,
    compute_logit_mean_innovation,
    compute_logit_replicate_innovation,
    compute_logit_tagiv_innovation,
)
from .probitree import ProbiTree, probitree_label_update
from .update.observation import (
    compute_categorical_innovation,
    compute_cdf_tagiv_innovation,
    compute_hrc_tagiv_innovation,
    compute_innovation,
    compute_innovation_with_indices,
    compute_probit_innovation_with_indices,
)
from .update.parameters import get_cap_factor


class Sequential:
    """
    Sequential container for TAGI Bayesian neural networks.

    Parameters
    ----------
    layers : list of layer objects
    device : str or torch.device  (default "cuda")
    """

    def __init__(self, layers: list, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.layers = layers

        # Move learnable layers to the target device
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                # These blocks manage their own sub-layers
                layer.device = self.device
                for sub in layer._learnable:
                    self._move_layer_to_device(sub)
            elif isinstance(layer, LearnableLayer):
                self._move_layer_to_device(layer)
            # Move BatchNorm running stats
            if hasattr(layer, "running_mean"):
                layer.running_mean = layer.running_mean.to(self.device)
                layer.running_var = layer.running_var.to(self.device)

    def _move_layer_to_device(self, layer):
        """Move a single layer's parameters to self.device."""
        if isinstance(layer, MultiheadAttentionV2):
            layer.device = self.device
            for sub in (layer.q_proj, layer.k_proj, layer.v_proj):
                self._move_layer_to_device(sub)
            return
        if not hasattr(layer, "mw") or layer.mw is None:
            return
        layer.device = self.device
        layer.mw = layer.mw.to(self.device)
        if getattr(layer, "Sw", None) is not None:
            layer.Sw = layer.Sw.to(self.device)
        if getattr(layer, "mb", None) is not None:
            layer.mb = layer.mb.to(self.device)
            if getattr(layer, "Sb", None) is not None:
                layer.Sb = layer.Sb.to(self.device)
        if getattr(layer, "running_mean", None) is not None:
            layer.running_mean = layer.running_mean.to(self.device)
            layer.running_var = layer.running_var.to(self.device)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the entire network.

        Parameters
        ----------
        x : Tensor  input data (flat or spatial)

        Returns
        -------
        mu  : Tensor  predicted output means
        var : Tensor  predicted output variances
        """
        ma = x
        Sa = torch.zeros_like(x)

        for layer in self.layers:
            if isinstance(layer, Layer):
                ma, Sa = layer.forward(ma, Sa)
            else:
                raise TypeError(f"Unknown layer type: {type(layer)}")

        return ma, Sa

    # ------------------------------------------------------------------
    #  Single training step (cuTAGI-style: backward + capped update)
    # ------------------------------------------------------------------
    def step(self, x_batch: Tensor, y_batch: Tensor, sigma_v: float) -> tuple[Tensor, Tensor]:
        """
        Perform one forward + backward + capped-update TAGI step.

        Parameters
        ----------
        x_batch : Tensor  input mini-batch
        y_batch : Tensor  target mini-batch
        sigma_v : float   observation noise std

        Returns
        -------
        y_pred_mu  : Tensor  predicted means (before update)
        y_pred_var : Tensor  predicted variances (before update)
        """
        batch_size = x_batch.shape[0]

        # ── 1. Forward ──
        y_pred_mu, y_pred_var = self.forward(x_batch)

        # ── 2. Output innovation ──
        delta_mu, delta_var = compute_innovation(y_batch, y_pred_mu, y_pred_var, sigma_v)

        # ── 3. Backward (compute + store deltas, NO param update) ──
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # ── 4. Capped parameter update (cuTAGI-style) ──
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Hierarchical softmax training step
    # ------------------------------------------------------------------
    def step_hrc(
        self,
        x_batch: Tensor,
        labels: Tensor,
        hrc: "HierarchicalSoftmax",
        sigma_v: float,
    ) -> tuple[Tensor, Tensor]:
        """One forward + backward + capped-update step using hierarchical softmax.

        Uses the sparse output innovation from :func:`compute_innovation_with_indices`
        so that only the ``n_obs`` tree nodes on each class's binary path receive
        an update signal, matching cuTAGI's ``update_using_indices``.

        The output layer must have ``hrc.len`` output neurons::

            net = Sequential([..., Linear(hidden, hrc.len)])

        Args:
            x_batch: Input mini-batch, shape (B, in_features).
            labels:  Integer class labels, shape (B,).
            hrc:     HierarchicalSoftmax from :func:`triton_tagi.hrc_softmax.class_to_obs`.
            sigma_v: Observation noise standard deviation.

        Returns:
            y_pred_mu:  Predicted output means before update, shape (B, hrc.len).
            y_pred_var: Predicted output variances before update, shape (B, hrc.len).
        """
        from .hrc_softmax import labels_to_hrc, labels_to_hrc_mask

        batch_size = x_batch.shape[0]

        # 1. Forward pass. Sequence models output (B, S, hrc.len); flatten to
        #    (B*S, hrc.len) for innovation, then reshape the delta back.
        y_pred_mu, y_pred_var = self.forward(x_batch)
        pred_shape = y_pred_mu.shape
        if y_pred_mu.dim() == 3:
            ma_flat = y_pred_mu.reshape(-1, pred_shape[-1])
            Sa_flat = y_pred_var.reshape(-1, pred_shape[-1])
        else:
            ma_flat = y_pred_mu
            Sa_flat = y_pred_var

        # 2. Encode labels → (obs ±1, 1-indexed node positions)
        y_obs, y_idx = labels_to_hrc(labels, hrc)

        # var_obs: scalar sigma_v^2 broadcast to (N, n_obs)
        var_obs = torch.full_like(y_obs, sigma_v**2)

        # 3. Sparse output innovation
        delta_mu, delta_var = compute_innovation_with_indices(
            ma_flat, Sa_flat, y_obs, var_obs, y_idx, mask=labels_to_hrc_mask(labels, hrc)
        )

        if y_pred_mu.dim() == 3:
            delta_mu = delta_mu.reshape(pred_shape)
            delta_var = delta_var.reshape(pred_shape)

        # 4. Backward pass (identical to dense step)
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # 5. Capped parameter update
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)

        return y_pred_mu, y_pred_var

    def step_hrc_probit(
        self,
        x_batch: Tensor,
        labels: Tensor,
        hrc: "HierarchicalSoftmax",
    ) -> tuple[Tensor, Tensor]:
        """Update HRC path nodes with exact probit half-space moments.

        The observation model is the half-space event ``s * R_j > 0`` for
        ``R_j = Z_j + o_j + eps_j`` with ``eps_j ~ N(0, 1)``. With
        ``d = sqrt(S_j + 1)``, ``gamma = s (mu_j + o_j) / d`` and
        ``lambda = phi(gamma) / Phi(gamma)``, the exact moment projection gives
        ``delta_mu = s lambda / d`` and
        ``delta_S = -lambda (lambda + gamma) / d^2``.

        The likelihood this maximizes is exactly the one
        :func:`triton_tagi.hrc_probit.hrc_log_probs` reports at inference, and
        it has no free scale: the unit noise variance fixes the latent unit
        that ``(mu, S, tau) -> (a mu, a^2 S, a tau)`` would otherwise leave
        undetermined, and the offsets come from the class prior. There is no
        ``sigma_v`` and no ``tau`` to choose.

        Args:
            x_batch: Input mini-batch.
            labels:  Integer class labels, shape (B,).
            hrc:     Tree structure; variable-depth paths and branch priors are
                     honoured through its mask and offsets.
        """

        from .hrc_softmax import labels_to_hrc, labels_to_hrc_mask

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        pred_shape = y_pred_mu.shape
        if y_pred_mu.dim() == 3:
            ma_flat = y_pred_mu.reshape(-1, pred_shape[-1])
            Sa_flat = y_pred_var.reshape(-1, pred_shape[-1])
        elif y_pred_mu.dim() == 2:
            ma_flat = y_pred_mu
            Sa_flat = y_pred_var
        else:
            raise ValueError("HRC probit expects two- or three-dimensional output moments")

        y_obs, y_idx = labels_to_hrc(labels, hrc)
        mask = labels_to_hrc_mask(labels, hrc)
        var_obs = torch.ones_like(y_obs)
        latent_shift = None
        if hrc.offset is not None:
            latent_shift = hrc.node_offset(y_obs.device).to(y_obs.dtype)[y_idx.long() - 1]
        delta_mu, delta_var = compute_probit_innovation_with_indices(
            ma_flat,
            Sa_flat,
            y_obs,
            var_obs,
            y_idx,
            mask=mask,
            latent_shift=latent_shift,
        )
        if y_pred_mu.dim() == 3:
            delta_mu = delta_mu.reshape(pred_shape)
            delta_var = delta_var.reshape(pred_shape)

        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    def step_probitree(
        self,
        x_batch: Tensor,
        labels: Tensor,
        tree: ProbiTree,
        r: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one direct-probit ``ProbiTree`` observation update.

        All active gate messages are computed from the same forward-pass prior
        and sent through the layerwise TAGI backward recursion once.

        Minibatches use the same batch approximation as every other head here:
        each layer sums its parameter deltas over the batch dimension and the
        capped update is applied once, with the cap factor read from the batch
        size. Cross-sample covariance between the gate messages is neglected,
        exactly as in :meth:`step`, :meth:`step_hrc`, and
        :meth:`step_categorical`.

        Returns the pre-update output moments and the per-sample path log
        evidence, shaped ``(batch,)``.
        """

        batch_size = x_batch.shape[0]
        if labels.dim() != 1 or labels.numel() != batch_size:
            raise ValueError("ProbiTree labels must have shape (batch,) matching the inputs")
        y_pred_mu, y_pred_var = self.forward(x_batch)
        if y_pred_mu.dim() != 2 or y_pred_mu.shape[1] != tree.num_gates:
            raise ValueError("ProbiTree outputs must have shape (batch, K - 1)")

        update = probitree_label_update(tree, y_pred_mu, y_pred_var, labels, r)
        # Layer.backward consumes derivatives of log evidence with respect to
        # prior output means: exactly the stable g/h interface here.
        delta_mu = update.g.to(y_pred_mu.dtype)
        delta_var = update.h.to(y_pred_var.dtype)
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var, update.log_evidence

    # ------------------------------------------------------------------
    #  Fixed-noise-free categorical training steps
    # ------------------------------------------------------------------
    def step_categorical(
        self,
        x_batch: Tensor,
        labels: Tensor,
        num_classes: int,
    ) -> tuple[Tensor, Tensor]:
        """Update a dense categorical or interleaved categorical TAGI-V head."""

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        delta_mu, delta_var = compute_categorical_innovation(
            labels, y_pred_mu, y_pred_var, num_classes
        )
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    def step_logit_tagiv(
        self,
        x_batch: Tensor,
        targets: Tensor,
        variance_floor: float = LOGIT_VARIANCE_FLOOR,
        update_mean: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Regress continuous logit targets with a learned observation variance.

        The output layer is the interleaved ``2K`` TAGI-V head followed by
        :class:`~triton_tagi.layers.EvenExp`, so ``targets`` are teacher logits
        rather than class indices.

        Args:
            x_batch: Frozen features, shape (B, in_features).
            targets: Centered teacher logits, shape (B, K).
            variance_floor: The floor ``s_min2`` added to ``exp(G)``.
            update_mean: When false the latent stream is frozen and only the
                variance head learns.

        Returns:
            y_pred_mu: Output means before the update, shape (B, 2K).
            y_pred_var: Output variances before the update, shape (B, 2K).
        """

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        delta_mu, delta_var = compute_logit_tagiv_innovation(
            targets,
            y_pred_mu,
            y_pred_var,
            variance_floor=variance_floor,
            update_mean=update_mean,
        )
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    def step_logit_tagiv_mean(
        self,
        x_batch: Tensor,
        targets: Tensor,
        observation_variance: float,
    ) -> tuple[Tensor, Tensor]:
        """Warm up the latent stream of a logit TAGI-V head under fixed noise.

        The variance stream receives zero deltas, so it keeps its prior while
        the mean converges.

        Args:
            x_batch: Frozen features, shape (B, in_features).
            targets: Logit targets, shape (B, K).
            observation_variance: The fixed ``sigma_v**2``.

        Returns:
            y_pred_mu: Output means before the update, shape (B, 2K).
            y_pred_var: Output variances before the update, shape (B, 2K).
        """

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        delta_mu, delta_var = compute_logit_mean_innovation(
            targets, y_pred_mu, y_pred_var, observation_variance=observation_variance
        )
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    def step_logit_tagiv_replicates(
        self,
        x_batch: Tensor,
        residual_variance: Tensor,
        repeats: int,
        variance_floor: float = LOGIT_VARIANCE_FLOOR,
    ) -> tuple[Tensor, Tensor]:
        """Update the variance stream from a replicated sample variance.

        The latent stream receives zero deltas, so this is the second phase of a
        two-stage fit: the mean is already converged and the variance head reads
        an observation of ``S`` that never passes through it.

        Args:
            x_batch: Frozen features, shape (B, in_features).
            residual_variance: Unbiased sample variance over repeats, shape (B, K).
            repeats: The replicate count ``M``, at least two.
            variance_floor: The floor ``s_min2``.

        Returns:
            y_pred_mu: Output means before the update, shape (B, 2K).
            y_pred_var: Output variances before the update, shape (B, 2K).
        """

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        delta_mu, delta_var = compute_logit_replicate_innovation(
            residual_variance,
            y_pred_mu,
            y_pred_var,
            repeats=repeats,
            variance_floor=variance_floor,
        )
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    def step_cdf_tagiv(
        self,
        x_batch: Tensor,
        labels: Tensor,
        *,
        epsilon: float,
        kappa: float,
        hermite_order: int = 64,
    ) -> tuple[Tensor, Tensor]:
        """Train an interleaved TAGI-V head through the CDF Gaussian channel.

        The output layer is the interleaved ``2K`` head followed by
        :class:`~triton_tagi.layers.EvenProbit`. Integer class labels are
        expanded to the signed encoding ``y_i = 2 * 1{c == i} - 1`` over all
        ``K`` units, which is the encoding the channel's logit origin is fixed
        to; a ``0/1`` one-hot would move that origin. The head's own Gaussian
        prior ``(nu, r)`` is read from the ``EvenProbit`` forward cache, since
        the post-activation moments do not determine it.

        Args:
            x_batch: Input batch, shape (B, ...).
            labels: Integer class indices, shape (B,).
            epsilon: Strictly positive variance floor of the activation.
            kappa: Strictly positive variance range of the activation.
            hermite_order: Gauss--Hermite order for the head integral.

        Returns:
            y_pred_mu: Output means before the update, shape (B, 2K).
            y_pred_var: Output variances before the update, shape (B, 2K).
        """

        probit = next(
            (layer for layer in reversed(self.layers) if isinstance(layer, EvenProbit)),
            None,
        )
        if probit is None:
            raise ValueError("step_cdf_tagiv requires an EvenProbit layer in the network")

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        if y_pred_mu.dim() != 2 or y_pred_mu.shape[1] % 2:
            raise ValueError("CDF TAGI-V expects a two-dimensional output of width 2K")

        num_classes = y_pred_mu.shape[1] // 2
        flat_labels = labels.reshape(-1).long().to(y_pred_mu.device)
        if flat_labels.numel() != batch_size:
            raise ValueError("labels leading shape must match the prediction batch")
        if bool(((flat_labels < 0) | (flat_labels >= num_classes)).any()):
            raise ValueError("labels contain an invalid class index")
        targets = 2.0 * torch.nn.functional.one_hot(flat_labels, num_classes).double() - 1.0

        delta_mu, delta_var = compute_cdf_tagiv_innovation(
            targets,
            y_pred_mu[:, 0::2],
            y_pred_var[:, 0::2],
            probit.nu,
            probit.r,
            epsilon=epsilon,
            kappa=kappa,
            hermite_order=hermite_order,
        )
        delta_mu = delta_mu.to(y_pred_mu.dtype)
        delta_var = delta_var.to(y_pred_var.dtype)
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    def step_hrc_tagiv(
        self,
        x_batch: Tensor,
        labels: Tensor,
        hrc: "HierarchicalSoftmax",
    ) -> tuple[Tensor, Tensor]:
        """Update an interleaved TAGI-V head at sparse HRC path nodes."""

        from .hrc_softmax import labels_to_hrc

        batch_size = x_batch.shape[0]
        y_pred_mu, y_pred_var = self.forward(x_batch)
        if y_pred_mu.dim() != 2:
            raise ValueError("HRC TAGI-V currently expects two-dimensional last-layer output")
        y_obs, y_idx = labels_to_hrc(labels, hrc)
        delta_mu, delta_var = compute_hrc_tagiv_innovation(y_pred_mu, y_pred_var, y_obs, y_idx)
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Set all layers to training mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.train()

    def eval(self) -> None:
        """Set all layers to evaluation mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.eval()

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)

    def num_parameters(self) -> int:
        """Return total number of learnable scalars (means + variances)."""
        return sum(
            layer.num_parameters for layer in self.layers if isinstance(layer, LearnableLayer)
        )

    def get_attention_scores(self) -> dict[int, tuple[Tensor, Tensor]]:
        """Collect attention score moments (μ, var) from every attention layer.

        Returns an ordered dict keyed by the layer's position in ``self.layers``.
        Each value is ``(mu_score, var_score)`` of shape ``(B, H, S, S)`` from
        the most recent forward pass. Raises if no attention layer has run.
        """
        out: dict[int, tuple[Tensor, Tensor]] = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MultiheadAttentionV2):
                out[i] = layer.get_attention_scores()
        if not out:
            raise RuntimeError("No attention layers in this Sequential.")
        return out
