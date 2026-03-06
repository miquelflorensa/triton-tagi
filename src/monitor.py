"""
TAGI Network Monitor
====================

Tracks and visualises how activations, parameters, and signal flow evolve
through training.  Particularly useful for diagnosing gain selection:
a wrong gain causes variance explosion or vanishing at some layer, which
shows up immediately in the signal-flow plot.

Quick-start
-----------
    from src.monitor import TAGIMonitor

    monitor = TAGIMonitor(net, log_dir="run_logs")
    monitor.record(epoch=0, x_probe=x_train[:256])   # snapshot at init
    monitor.print_report()

    for epoch in range(1, n_epochs + 1):
        for xb, yb in batches:
            net.step(xb, yb, sigma_v)
        monitor.record(epoch, x_train[:256])
        monitor.print_report()

    monitor.plot("monitor.png")

Gain sweep
----------
    from src.monitor import sweep_gains

    sweep_gains(
        builder_fn = lambda gw: build_my_net(gain_w=gw),
        x_probe    = x_train[:256],
        gains      = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
        filename   = "gain_sweep.png",
    )
"""

import os
import math
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable

from .layers.linear     import Linear
from .layers.conv2d     import Conv2D
from .layers.batchnorm2d import BatchNorm2D
from .layers.resblock   import ResBlock

_LEARNABLE = (Linear, Conv2D, BatchNorm2D, ResBlock)


# ══════════════════════════════════════════════════════════════════════
#  Data records
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ActivationStats:
    """Per-layer activation statistics for one forward probe."""
    layer_idx:    int
    layer_name:   str
    # mean statistics
    mu_mean:      float   # E[μ]       — overall signal centering
    mu_std:       float   # std(μ)     — spread of activations
    mu_abs_mean:  float   # E[|μ|]     — unsigned magnitude
    mu_max:       float   # max(|μ|)   — worst outlier
    # variance statistics
    var_mean:     float   # E[σ²]      — average uncertainty / variance
    var_std:      float   # std(σ²)
    var_max:      float   # max(σ²)    — largest uncertainty
    # health indicators
    frac_dead:    float   # fraction of |μ| < 0.01 (dead activations)
    frac_explode: float   # fraction of |μ| > 100  (exploding activations)


@dataclass
class ParamStats:
    """Per-learnable-layer parameter statistics."""
    layer_idx:   int
    layer_name:  str
    # weight mean
    mw_mean:     float   # E[mw]
    mw_std:      float   # std(mw)
    mw_abs_mean: float   # E[|mw|]
    # weight variance
    Sw_mean:     float   # E[Sw]   — average weight uncertainty
    Sw_max:      float   # max(Sw)
    # bias mean / variance
    mb_std:      float   # std(mb)
    Sb_mean:     float   # E[Sb]
    # update ratio (filled by record(), not probe())
    update_ratio: float = 0.0   # |Δmw| / max(|mw|, 1e-8)


@dataclass
class EpochRecord:
    """Complete network snapshot at one epoch."""
    epoch:       int
    step:        int
    tag:         str
    activations: List[ActivationStats]
    params:      List[ParamStats]


# ══════════════════════════════════════════════════════════════════════
#  TAGIMonitor
# ══════════════════════════════════════════════════════════════════════

class TAGIMonitor:
    """
    Non-invasive training monitor for TAGI Sequential networks.

    Calling probe() reruns a forward pass on a small batch and records
    statistics at every layer.  It never modifies parameters and does not
    interfere with training because each training step overwrites all
    cached layer state (J, ma_in, …) at the start of its own forward pass.

    Parameters
    ----------
    net         : Sequential   the network being trained
    log_dir     : str          directory for logs and plots
    probe_size  : int          max batch size used for probing (default 256)
    dead_thresh : float        |μ| threshold for "dead" classification (0.01)
    """

    def __init__(self, net, log_dir="tagi_monitor",
                 probe_size: int = 256, dead_thresh: float = 0.01):
        self.net         = net
        self.log_dir     = log_dir
        self.probe_size  = probe_size
        self.dead_thresh = dead_thresh
        self.history: List[EpochRecord] = []
        self._step       = 0
        self._prev_mw: Dict[int, torch.Tensor] = {}   # for update_ratio
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _name(idx, layer):
        return f"[{idx:2d}] {type(layer).__name__}"

    def _act_stats(self, idx, layer, ma, Sa) -> ActivationStats:
        mu  = ma.detach().float().reshape(-1)
        var = Sa.detach().float().reshape(-1).clamp(min=0)
        return ActivationStats(
            layer_idx   = idx,
            layer_name  = self._name(idx, layer),
            mu_mean     = mu.mean().item(),
            mu_std      = mu.std().item(),
            mu_abs_mean = mu.abs().mean().item(),
            mu_max      = mu.abs().max().item(),
            var_mean    = var.mean().item(),
            var_std     = var.std().item(),
            var_max     = var.max().item(),
            frac_dead   = (mu.abs() < self.dead_thresh).float().mean().item(),
            frac_explode= (mu.abs() > 100).float().mean().item(),
        )

    def _param_stats(self, idx, layer) -> Optional[ParamStats]:
        if not isinstance(layer, _LEARNABLE):
            return None
        mw = layer.mw.detach().float().reshape(-1)
        Sw = layer.Sw.detach().float().reshape(-1).clamp(min=0)
        mb = layer.mb.detach().float().reshape(-1)
        Sb = layer.Sb.detach().float().reshape(-1).clamp(min=0)

        # Update ratio vs previous snapshot
        prev = self._prev_mw.get(idx)
        if prev is not None:
            delta = (mw - prev).abs().mean().item()
            denom = max(prev.abs().mean().item(), 1e-8)
            update_ratio = delta / denom
        else:
            update_ratio = 0.0
        self._prev_mw[idx] = mw.clone()

        return ParamStats(
            layer_idx    = idx,
            layer_name   = self._name(idx, layer),
            mw_mean      = mw.mean().item(),
            mw_std       = mw.std().item(),
            mw_abs_mean  = mw.abs().mean().item(),
            Sw_mean      = Sw.mean().item(),
            Sw_max       = Sw.max().item(),
            mb_std       = mb.std().item(),
            Sb_mean      = Sb.mean().item(),
            update_ratio = update_ratio,
        )

    # ------------------------------------------------------------------
    #  Core probe
    # ------------------------------------------------------------------

    def probe(self, x_batch: torch.Tensor) -> List[ActivationStats]:
        """
        Run a forward pass on ≤ probe_size samples and return per-layer stats.

        This is safe to call between training steps: any cached state written
        here (J, ma_in, …) is overwritten by the next step()'s forward pass.
        """
        x  = x_batch[:self.probe_size].detach()
        ma = x
        Sa = torch.zeros_like(x)
        stats = []
        for i, layer in enumerate(self.net.layers):
            ma, Sa = layer.forward(ma, Sa)
            stats.append(self._act_stats(i, layer, ma, Sa))
        return stats

    def snap_params(self) -> List[ParamStats]:
        """Read parameter statistics from all learnable layers."""
        out = []
        for i, layer in enumerate(self.net.layers):
            s = self._param_stats(i, layer)
            if s is not None:
                out.append(s)
        return out

    # ------------------------------------------------------------------
    #  Record
    # ------------------------------------------------------------------

    def count_step(self):
        """Call once per net.step() to maintain an accurate step counter."""
        self._step += 1

    def record(self, epoch: int, x_probe: torch.Tensor,
               tag: str = "") -> EpochRecord:
        """
        Snapshot the network state (activations + parameters).

        Call at the **end** of each epoch (or at init with epoch=0).

        Parameters
        ----------
        epoch   : int    current epoch number
        x_probe : Tensor sample batch used for probing
        tag     : str    optional label (e.g. "after-warmup")
        """
        acts   = self.probe(x_probe)
        params = self.snap_params()
        rec    = EpochRecord(epoch=epoch, step=self._step,
                             tag=tag, activations=acts, params=params)
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------
    #  Text report
    # ------------------------------------------------------------------

    @staticmethod
    def _health_flag(var_mean: float, frac_dead: float) -> str:
        if math.isnan(var_mean) or math.isinf(var_mean):
            return "  !! NaN/Inf !!"
        if var_mean > 1e4:
            return "  !! EXPLODE !!"
        if var_mean > 100:
            return "  /!\\ high σ²  "
        if var_mean < 1e-6:
            return "  !! VANISH  !!"
        if var_mean < 1e-3:
            return "  /!\\ low σ²  "
        if frac_dead > 0.9:
            return "  /!\\ DEAD    "
        return "  OK           "

    def print_report(self, record: Optional[EpochRecord] = None):
        """
        Print a formatted table for the given (or latest) record.
        Flags layers that are exploding, vanishing, or dead.
        """
        if record is None:
            if not self.history:
                print("  [Monitor] No records yet.")
                return
            record = self.history[-1]

        tag_str = f"  ({record.tag})" if record.tag else ""
        print(f"\n  {'─'*72}")
        print(f"  TAGI Monitor — epoch {record.epoch}  "
              f"step {record.step}{tag_str}")
        print(f"  {'─'*72}")

        # ── Activation table ──
        print(f"\n  {'Layer':<22}  {'E[μ]':>7}  {'std(μ)':>7}  "
              f"{'E[σ²]':>9}  {'max(σ²)':>9}  {'dead%':>6}  Status")
        print(f"  {'─'*72}")
        for a in record.activations:
            flag = self._health_flag(a.var_mean, a.frac_dead)
            print(f"  {a.layer_name:<22}  {a.mu_mean:>7.3f}  "
                  f"{a.mu_std:>7.3f}  "
                  f"{a.var_mean:>9.4f}  {a.var_max:>9.3f}  "
                  f"{a.frac_dead*100:>5.1f}%  {flag}")

        # ── Parameter table ──
        if record.params:
            print(f"\n  {'Layer':<22}  {'E[|mw|]':>9}  {'std(mw)':>9}  "
                  f"{'E[Sw]':>10}  {'max(Sw)':>10}  {'Δ/|mw|':>8}")
            print(f"  {'─'*72}")
            for p in record.params:
                sw_flag = ""
                if p.Sw_mean < 1e-10:   sw_flag = " ⚠Sw→0"
                elif p.Sw_mean > 1e3:   sw_flag = " ⚠Sw↑↑"
                print(f"  {p.layer_name:<22}  {p.mw_abs_mean:>9.5f}  "
                      f"{p.mw_std:>9.5f}  "
                      f"{p.Sw_mean:>10.3e}  {p.Sw_max:>10.3e}  "
                      f"{p.update_ratio*100:>7.2f}%{sw_flag}")

        print(f"  {'─'*72}")

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------

    def plot(self, filename: Optional[str] = None, show: bool = False):
        """
        Generate a 5-panel diagnostic figure and save to disk.

        Panels
        ------
        1. Heatmap: activation variance E[σ²] per (epoch, layer)
        2. Bar chart: signal flow at the latest epoch
        3. Line chart: E[σ²] trajectories for each layer over epochs
        4. Line chart: E[Sw] (weight variance) trajectories
        5. Line chart: dead-neuron % trajectories
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import matplotlib.colors as mcolors
        except ImportError:
            print("  [Monitor] matplotlib not found — skipping plot.")
            return

        if not self.history:
            print("  [Monitor] No history to plot.")
            return

        if filename is None:
            filename = os.path.join(self.log_dir, "monitor.png")

        epochs     = [r.epoch for r in self.history]
        act_names  = [a.layer_name for a in self.history[0].activations]
        n_layers   = len(act_names)
        n_epochs   = len(epochs)

        # Shape: (n_epochs, n_layers)
        var_means = np.array([
            [a.var_mean for a in r.activations] for r in self.history
        ], dtype=np.float64)
        frac_dead = np.array([
            [a.frac_dead for a in r.activations] for r in self.history
        ], dtype=np.float64)

        # Param history
        has_params = bool(self.history[0].params)
        if has_params:
            param_names = [p.layer_name for p in self.history[0].params]
            n_params    = len(param_names)
            sw_hist = np.array([
                [p.Sw_mean for p in r.params] for r in self.history
            ], dtype=np.float64)
            mw_hist = np.array([
                [p.mw_abs_mean for p in r.params] for r in self.history
            ], dtype=np.float64)
            upd_hist = np.array([
                [p.update_ratio * 100 for p in r.params] for r in self.history
            ], dtype=np.float64)

        # ── Layout ──
        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                hspace=0.48, wspace=0.32,
                                left=0.07, right=0.97,
                                top=0.93, bottom=0.06)

        cmap_health = mcolors.LinearSegmentedColormap.from_list(
            "health", ["#d32f2f", "#f57c00", "#388e3c"], N=256)

        # ── Panel 1: Heatmap ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        log_var = np.log10(np.clip(var_means.T, 1e-10, None))
        im = ax1.imshow(log_var, aspect="auto", cmap=cmap_health,
                        vmin=-4, vmax=2, origin="lower")
        ax1.set_xticks(range(n_epochs))
        ax1.set_xticklabels(epochs, rotation=45, ha="right", fontsize=7)
        ax1.set_yticks(range(n_layers))
        ax1.set_yticklabels(act_names, fontsize=7)
        ax1.set_xlabel("Epoch", fontsize=8)
        ax1.set_title(
            "Activation Variance E[σ²]  per layer per epoch  "
            "(log₁₀ · green ≈ 1 · red = bad)",
            fontsize=9, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
        cbar.set_label("log₁₀(E[σ²])", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        # annotate healthy band
        for v, lbl in [(-1, "0.1"), (0, "1.0"), (1, "10")]:
            ax1.axhline(-0.5, color="none")  # dummy
        ax1.axhline(-0.5, color="none")

        # ── Panel 2: Signal flow bar chart (latest epoch) ────────────
        ax2 = fig.add_subplot(gs[1, 0])
        latest_var = var_means[-1]
        log_lv = np.log10(np.clip(latest_var, 1e-10, None))
        colors = ["#d32f2f" if (v > 2 or v < -3) else
                  "#f57c00" if (v > 1 or v < -2) else "#388e3c"
                  for v in log_lv]
        bars = ax2.barh(range(n_layers), log_lv, color=colors, height=0.7)
        ax2.set_yticks(range(n_layers))
        ax2.set_yticklabels(act_names, fontsize=7)
        for threshold, color, style, lbl in [
            (-3, "#b71c1c", "--", "σ²=0.001"),
            (-1, "#f57c00", ":",  "σ²=0.1"),
            (0,  "#1b5e20", "-",  "σ²=1"),
            (2,  "#b71c1c", "--", "σ²=100"),
        ]:
            ax2.axvline(threshold, color=color, linestyle=style,
                        linewidth=1, label=lbl, alpha=0.8)
        ax2.set_xlabel("log₁₀(E[σ²])", fontsize=8)
        ax2.set_title(f"Signal Flow — epoch {epochs[-1]}  "
                      f"(green=healthy, red=bad)", fontsize=8)
        ax2.legend(fontsize=6, loc="lower right")
        ax2.set_xlim(-6, 5)

        # ── Panel 3: Variance trajectories ───────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        cmap_layers = plt.get_cmap("tab20", n_layers)
        for li in range(n_layers):
            ax3.semilogy(epochs, np.clip(var_means[:, li], 1e-10, None),
                         label=act_names[li], alpha=0.8,
                         color=cmap_layers(li), linewidth=1.2)
        ax3.axhline(1.0,  color="green",  linestyle="--",
                    linewidth=1, alpha=0.6, label="σ²=1 target")
        ax3.axhline(0.1,  color="orange", linestyle=":",
                    linewidth=1, alpha=0.5)
        ax3.axhline(100,  color="red",    linestyle="--",
                    linewidth=1, alpha=0.5)
        ax3.set_xlabel("Epoch", fontsize=8)
        ax3.set_ylabel("E[σ²]  (log scale)", fontsize=8)
        ax3.set_title("Activation Variance Trajectories", fontsize=8)
        ax3.tick_params(labelsize=7)
        if n_layers <= 12:
            ax3.legend(fontsize=5, loc="upper right",
                       ncol=2, framealpha=0.5)

        # ── Panel 4: Weight variance (Sw) trajectories ───────────────
        ax4 = fig.add_subplot(gs[2, 0])
        if has_params:
            cmap_params = plt.get_cmap("tab10", n_params)
            for pi in range(n_params):
                ax4.semilogy(epochs, np.clip(sw_hist[:, pi], 1e-15, None),
                             label=param_names[pi], alpha=0.85,
                             color=cmap_params(pi), linewidth=1.2)
            ax4.axhline(1e-10, color="red", linestyle="--",
                        linewidth=1, alpha=0.6, label="Sw→0 floor")
            ax4.set_xlabel("Epoch", fontsize=8)
            ax4.set_ylabel("E[Sw]  (log scale)", fontsize=8)
            ax4.set_title("Weight Variance E[Sw]  "
                          "(decreasing = model learning)", fontsize=8)
            ax4.tick_params(labelsize=7)
            if n_params <= 8:
                ax4.legend(fontsize=6, framealpha=0.5)
        else:
            ax4.text(0.5, 0.5, "No learnable layers",
                     ha="center", va="center", transform=ax4.transAxes)

        # ── Panel 5: Dead neurons ─────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 1])
        for li in range(n_layers):
            ax5.plot(epochs, frac_dead[:, li] * 100,
                     label=act_names[li], alpha=0.7,
                     color=cmap_layers(li), linewidth=1.2)
        ax5.axhline(50, color="red", linestyle="--",
                    linewidth=1, alpha=0.6, label="50% dead")
        ax5.axhline(90, color="darkred", linestyle="--",
                    linewidth=1, alpha=0.6, label="90% dead")
        ax5.set_ylim(0, 105)
        ax5.set_xlabel("Epoch", fontsize=8)
        ax5.set_ylabel("% of units with |μ| < 0.01", fontsize=8)
        ax5.set_title("Dead Activations %", fontsize=8)
        ax5.tick_params(labelsize=7)
        if n_layers <= 12:
            ax5.legend(fontsize=5, loc="upper right",
                       ncol=2, framealpha=0.5)

        fig.suptitle("TAGI Network Monitor", fontsize=13, fontweight="bold")

        plt.savefig(filename, dpi=130, bbox_inches="tight")
        print(f"  [Monitor] Plot saved → {filename}")
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    #  Convenience: save history as CSV
    # ------------------------------------------------------------------

    def save_csv(self, filename: Optional[str] = None):
        """
        Append the full history to a CSV file (one row per layer per epoch).
        Useful for later analysis or plotting outside Python.
        """
        if filename is None:
            filename = os.path.join(self.log_dir, "monitor.csv")

        import csv
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "epoch", "step", "tag",
                    "layer_idx", "layer_name",
                    "mu_mean", "mu_std", "mu_abs_mean", "mu_max",
                    "var_mean", "var_std", "var_max",
                    "frac_dead", "frac_explode",
                ])
            for rec in self.history:
                for a in rec.activations:
                    w.writerow([
                        rec.epoch, rec.step, rec.tag,
                        a.layer_idx, a.layer_name,
                        a.mu_mean, a.mu_std, a.mu_abs_mean, a.mu_max,
                        a.var_mean, a.var_std, a.var_max,
                        a.frac_dead, a.frac_explode,
                    ])
        print(f"  [Monitor] CSV saved → {filename}")


# ══════════════════════════════════════════════════════════════════════
#  Gain sweep — standalone utility
# ══════════════════════════════════════════════════════════════════════

def sweep_gains(
    builder_fn: Callable,
    x_probe: torch.Tensor,
    gains: List[float],
    filename: str = "gain_sweep.png",
    probe_size: int = 256,
    target_var: float = 1.0,
    show: bool = False,
):
    """
    Sweep gain values and visualise the resulting signal flow at initialisation.

    This directly addresses the gain bottleneck: for each candidate gain
    you see where the signal explodes or vanishes before any training begins.

    Parameters
    ----------
    builder_fn : callable
        Function that takes a single gain value and returns a fresh Sequential.
        Example: lambda g: build_simple_3cnn(gain_w=g, gain_b=g)
    x_probe    : Tensor
        A sample input batch (will be sliced to probe_size).
    gains      : list of float
        Gain values to try (e.g. [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]).
    filename   : str
        Output filename for the sweep figure.
    probe_size : int
        Max batch size for the forward pass.
    target_var : float
        Target activation variance (shown as a horizontal guide line).
    show       : bool
        If True, also display the figure interactively.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [sweep_gains] matplotlib not found — cannot plot.")
        return

    x = x_probe[:probe_size].detach()
    results: Dict[float, List[ActivationStats]] = {}
    layer_names = None

    for gain in gains:
        net = builder_fn(gain)
        ma, Sa = x, torch.zeros_like(x)
        stats = []
        for i, layer in enumerate(net.layers):
            ma, Sa = layer.forward(ma, Sa)
            mu_flat  = ma.detach().float().reshape(-1)
            var_flat = Sa.detach().float().reshape(-1).clamp(min=0)
            stats.append(ActivationStats(
                layer_idx   = i,
                layer_name  = f"[{i:2d}] {type(layer).__name__}",
                mu_mean     = mu_flat.mean().item(),
                mu_std      = mu_flat.std().item(),
                mu_abs_mean = mu_flat.abs().mean().item(),
                mu_max      = mu_flat.abs().max().item(),
                var_mean    = var_flat.mean().item(),
                var_std     = var_flat.std().item(),
                var_max     = var_flat.max().item(),
                frac_dead   = (mu_flat.abs() < 0.01).float().mean().item(),
                frac_explode= (mu_flat.abs() > 100).float().mean().item(),
            ))
        results[gain] = stats
        if layer_names is None:
            layer_names = [s.layer_name for s in stats]

    n_layers = len(layer_names)
    xs = list(range(n_layers))

    # ── Build figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap("plasma", len(gains))

    # Left: log10(var) per layer, one line per gain
    ax = axes[0]
    for gi, gain in enumerate(gains):
        var_vals = np.array([s.var_mean for s in results[gain]])
        log_var  = np.log10(np.clip(var_vals, 1e-10, None))
        has_nan  = any(math.isnan(v) or math.isinf(v) for v in var_vals)
        style    = "--" if has_nan else "-"
        ax.plot(xs, log_var, marker="o", markersize=4,
                linestyle=style, color=cmap(gi),
                label=f"gain={gain}", linewidth=1.6, alpha=0.9)

    target_log = math.log10(target_var)
    ax.axhline(target_log, color="black", linestyle="--",
               linewidth=1.5, label=f"target σ²={target_var}")
    ax.axhline(target_log + 2, color="red",    linestyle=":",
               linewidth=1, alpha=0.6, label="×100 from target")
    ax.axhline(target_log - 2, color="orange", linestyle=":",
               linewidth=1, alpha=0.6, label="×0.01 from target")
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("log₁₀(E[σ²])", fontsize=9)
    ax.set_title("Signal Variance per Layer at Init\n"
                 "(want: all lines near the black dashed target)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Right: heatmap (rows=gains, cols=layers)
    ax2 = axes[1]
    heat = np.array([
        [math.log10(max(s.var_mean, 1e-10)) for s in results[g]]
        for g in gains
    ])
    import matplotlib.colors as mcolors
    cmap_health = mcolors.LinearSegmentedColormap.from_list(
        "health", ["#d32f2f", "#f57c00", "#388e3c"], N=256)
    im = ax2.imshow(heat, aspect="auto", cmap=cmap_health,
                    vmin=target_log - 3, vmax=target_log + 3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(gains)))
    ax2.set_yticklabels([f"gain={g}" for g in gains], fontsize=8)
    ax2.set_title("Gain × Layer Heatmap  "
                  "(green = σ² near target, red = bad)", fontsize=9)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("log₁₀(E[σ²])", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    # Annotate each cell with the actual log value
    for gi in range(len(gains)):
        for li in range(n_layers):
            ax2.text(li, gi, f"{heat[gi, li]:.1f}",
                     ha="center", va="center", fontsize=5.5,
                     color="white" if abs(heat[gi, li] - target_log) > 1.5
                     else "black")

    fig.suptitle(
        f"Gain Sensitivity Analysis  (target σ²={target_var})",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    print(f"  [sweep_gains] Plot saved → {filename}")

    # ── Text summary ──────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Gain Sweep Summary  (target σ² = {target_var})")
    print(f"  {'─'*60}")
    print(f"  {'Gain':>8}  {'Min log σ²':>11}  {'Max log σ²':>11}"
          f"  {'Spread':>8}  Verdict")
    print(f"  {'─'*60}")
    for gain in gains:
        log_vars = [math.log10(max(s.var_mean, 1e-10)) for s in results[gain]]
        lo, hi   = min(log_vars), max(log_vars)
        spread   = hi - lo
        if any(math.isnan(v) or math.isinf(v)
               for s in results[gain] for v in [s.var_mean]):
            verdict = "CRASH / NaN"
        elif hi > target_log + 3:
            verdict = "EXPLODING"
        elif lo < target_log - 3:
            verdict = "VANISHING"
        elif spread > 3:
            verdict = "UNEVEN"
        elif abs((lo + hi) / 2 - target_log) < 1.5:
            verdict = "GOOD"
        else:
            verdict = "OK"
        print(f"  {gain:>8.4f}  {lo:>11.2f}  {hi:>11.2f}"
              f"  {spread:>8.2f}  {verdict}")
    print(f"  {'─'*60}")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  sigma_v sweep — observation noise sensitivity
# ══════════════════════════════════════════════════════════════════════

def sweep_sigma_v(
    net,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    sigma_vs: List[float],
    filename: str = "sigma_v_sweep.png",
    probe_size: int = 256,
    show: bool = False,
):
    """
    Sweep observation noise values and show how each one drives parameter
    updates through the cap system.

    The three coupled quantities visualised are:

    1. **Innovation magnitude** — |δμ| = |y − μ_pred| / (S_pred + σ_v²)
       This is what arrives at the output layer.  Larger σ_v → smaller signal.

    2. **Cap budget** per parameter — √Sw / cap_factor
       This is the maximum |Δmw| that the update rule will ever apply,
       regardless of how large the innovation is.

    3. **Fraction capped** — how many parameters are actually capped.
       If nearly all parameters are capped the innovation magnitude is
       irrelevant; the only thing that matters is Sw (gain).

    Call this BEFORE or AFTER training (never during a step).
    It does NOT modify parameters; any deltas written to layers are
    overwritten by the next net.step() call.

    Parameters
    ----------
    net       : Sequential
    x_probe   : Tensor (B, ...)  input sample
    y_probe   : Tensor (B, K)    target labels (one-hot)
    sigma_vs  : list of float    noise values to sweep
    filename  : str              output file
    probe_size: int              max batch size
    show      : bool             display figure interactively
    """
    from .update.observation import compute_innovation
    from .update.parameters  import get_cap_factor

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [sweep_sigma_v] matplotlib not found.")
        return

    x = x_probe[:probe_size].detach()
    y = y_probe[:probe_size].detach()
    B = x.shape[0]
    cap_factor = get_cap_factor(B)

    # ── Forward pass (run once, cached state reused for all sigma_vs) ──
    mu_pred, var_pred = net.forward(x)

    # ── Collect Sw budget per learnable layer (does not change) ──
    param_names, Sw_budgets, mw_magnitudes = [], [], []
    for i, layer in enumerate(net.layers):
        if isinstance(layer, _LEARNABLE):
            Sw = layer.Sw.detach().float().flatten()
            mw = layer.mw.detach().float().flatten()
            budget = (Sw.clamp(min=1e-10).sqrt() / cap_factor).mean().item()
            param_names.append(f"[{i:2d}] {type(layer).__name__}")
            Sw_budgets.append(budget)
            mw_magnitudes.append(mw.abs().mean().item())

    # ── Per-sigma_v: innovation + backward ──
    records = {}
    for sigma_v in sigma_vs:
        delta_mu, delta_var = compute_innovation(y, mu_pred, var_pred, sigma_v)

        innov_abs = delta_mu.detach().float().abs()
        innov_mean = innov_abs.mean().item()
        innov_max  = innov_abs.max().item()

        # Backward to get delta_mw for each learnable layer
        dm, dv = delta_mu.clone(), delta_var.clone()
        for layer in reversed(net.layers):
            dm, dv = layer.backward(dm, dv)

        # Per-layer update stats
        layer_stats = []
        for i, layer in enumerate(net.layers):
            if not isinstance(layer, _LEARNABLE):
                continue
            if layer.delta_mw is None:
                layer_stats.append(dict(
                    name="?", delta_mean=0., budget=0., frac_capped=0., ratio=0.))
                continue
            dmw = layer.delta_mw.detach().float().flatten()
            Sw  = layer.Sw.detach().float().flatten().clamp(min=1e-10)
            bar = Sw.sqrt() / cap_factor          # per-param cap budget
            dmw_abs   = dmw.abs()
            frac_cap  = (dmw_abs > bar).float().mean().item()
            applied   = torch.minimum(dmw_abs, bar).mean().item()
            layer_stats.append(dict(
                name       = f"[{i:2d}] {type(layer).__name__}",
                delta_mean = dmw_abs.mean().item(),
                budget     = bar.mean().item(),
                frac_capped= frac_cap,
                applied    = applied,
            ))

        records[sigma_v] = dict(
            innov_mean  = innov_mean,
            innov_max   = innov_max,
            layer_stats = layer_stats,
        )

    # ── Figure ──────────────────────────────────────────────────────
    n_sv     = len(sigma_vs)
    n_params = len(param_names)
    cmap     = plt.get_cmap("plasma", n_sv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: innovation magnitude per sigma_v
    ax = axes[0]
    innov_means = [records[sv]["innov_mean"] for sv in sigma_vs]
    innov_maxs  = [records[sv]["innov_max"]  for sv in sigma_vs]
    ax.loglog(sigma_vs, innov_means, "o-", color="steelblue",
              linewidth=2, label="E[|δμ|]")
    ax.loglog(sigma_vs, innov_maxs,  "s--", color="tomato",
              linewidth=1.5, label="max|δμ|")
    # Add the cap budget (flat horizontal line per layer) as reference
    for pi, (nm, bud) in enumerate(zip(param_names, Sw_budgets)):
        ax.axhline(bud, linestyle=":", linewidth=1.0, alpha=0.5,
                   label=f"cap {nm}")
    ax.set_xlabel("σ_v", fontsize=9)
    ax.set_ylabel("|δμ| at output", fontsize=9)
    ax.set_title("Innovation Magnitude vs σ_v\n"
                 "(dotted = cap budget per param layer)", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: fraction of parameters capped per layer, per sigma_v
    ax2 = axes[1]
    xs  = list(range(n_params))
    for si, sigma_v in enumerate(sigma_vs):
        frac_caps = [ls["frac_capped"]
                     for ls in records[sigma_v]["layer_stats"]]
        ax2.plot(xs, [f * 100 for f in frac_caps],
                 "o-", color=cmap(si), label=f"σ_v={sigma_v}",
                 linewidth=1.5, alpha=0.85)
    ax2.axhline(50, color="red",    linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(100, color="darkred", linestyle="--", linewidth=1, alpha=0.3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(param_names, rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("% parameters capped", fontsize=9)
    ax2.set_ylim(-5, 105)
    ax2.set_title("Fraction of Updates Hitting the Cap\n"
                  "(100% = σ_v irrelevant, only Sw/gain matters)", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: effective applied update magnitude (after cap)
    ax3 = axes[2]
    for si, sigma_v in enumerate(sigma_vs):
        applied = [ls["applied"]
                   for ls in records[sigma_v]["layer_stats"]]
        ax3.semilogy(xs, [max(a, 1e-15) for a in applied],
                     "o-", color=cmap(si), label=f"σ_v={sigma_v}",
                     linewidth=1.5, alpha=0.85)
    # Also plot Sw_budget as reference (dashed)
    ax3.semilogy(xs, [max(b, 1e-15) for b in Sw_budgets],
                 "k--", linewidth=2, label="cap budget √Sw/cap_f", alpha=0.7)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(param_names, rotation=40, ha="right", fontsize=7)
    ax3.set_ylabel("E[applied |Δmw|]  (log scale)", fontsize=9)
    ax3.set_title("Actually Applied Update Magnitude\n"
                  "(lines collapsing → dominated by cap, σ_v has no effect)",
                  fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"σ_v Sensitivity  (cap_factor={cap_factor}, batch={B})",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    print(f"  [sweep_sigma_v] Plot saved → {filename}")

    # ── Text summary ──────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  σ_v Sensitivity  (cap_factor={cap_factor:.1f}, B={B})")
    print(f"  {'─'*70}")
    print(f"  {'σ_v':>8}  {'E[|δμ|]':>10}  {'max|δμ|':>10}  "
          f"{'avg %capped':>12}  Verdict")
    print(f"  {'─'*70}")
    for sigma_v in sigma_vs:
        r  = records[sigma_v]
        fc = [ls["frac_capped"] for ls in r["layer_stats"]]
        avg_cap = sum(fc) / len(fc) * 100 if fc else 0.
        if avg_cap > 95:
            verdict = "FULLY CAPPED (σ_v irrelevant)"
        elif avg_cap > 50:
            verdict = "MOSTLY CAPPED"
        elif r["innov_mean"] < 1e-6:
            verdict = "SIGNAL TOO WEAK"
        else:
            verdict = "UNCAPPED — σ_v matters"
        print(f"  {sigma_v:>8.4f}  {r['innov_mean']:>10.3e}  "
              f"{r['innov_max']:>10.3e}  {avg_cap:>11.1f}%  {verdict}")
    print(f"  {'─'*70}")
    print(f"  NOTE: cap budget (avg √Sw/cap_f):")
    for nm, bud, mw in zip(param_names, Sw_budgets, mw_magnitudes):
        ratio = bud / max(mw, 1e-12) * 100
        print(f"    {nm:<24} budget={bud:.3e}  |mw|={mw:.3e}  "
              f"budget/|mw|={ratio:.2f}%")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  compare_heads — Remax vs Bernoulli side-by-side
# ══════════════════════════════════════════════════════════════════════

def compare_heads(
    net,
    x_probe: torch.Tensor,
    n_gh: int = 32,
    filename: str = "head_comparison.png",
    probe_size: int = 256,
    show: bool = False,
):
    """
    Run both Remax and Bernoulli on the same pre-head logits and compare
    their outputs and backward Jacobians side-by-side.

    The comparison shows:
    - **P distribution**: how confident each head is (max P, entropy)
    - **V distribution**: output variance going into the innovation
      — Bernoulli: V = P(1-P); Remax: V = P² · (exp(σ²_ln) - 1)
    - **J distribution**: backward Jacobian J = cov(z,a)/Sz
      — larger |J| = stronger gradient signal back through the network
    - **S_pred effect on σ_v sensitivity**: large V_pred means a given
      σ_v has less relative influence on the innovation

    Parameters
    ----------
    net       : Sequential   network that ends with Remax OR Bernoulli
    x_probe   : Tensor       sample input batch
    n_gh      : int          Gauss-Hermite order for Bernoulli (default 32)
    filename  : str          output file
    probe_size: int          max batch size for forward pass
    show      : bool         display interactively
    """
    from .layers.remax    import Remax
    from .layers.bernoulli import Bernoulli

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [compare_heads] matplotlib not found.")
        return

    x = x_probe[:probe_size].detach()

    # ── Forward through everything EXCEPT the last layer ──────────────
    ma, Sa = x, torch.zeros_like(x)
    for layer in net.layers[:-1]:
        ma, Sa = layer.forward(ma, Sa)

    mz = ma.detach().clone()   # logit means   (B, K)
    Sz = Sa.detach().clone()   # logit variances (B, K)

    # ── Run both heads ─────────────────────────────────────────────────
    remax_head = Remax()
    P_rm, V_rm = remax_head.forward(mz, Sz)
    J_rm = remax_head.J.detach().float()
    P_rm, V_rm = P_rm.detach().float(), V_rm.detach().float()

    bern_head = Bernoulli(n_gh=n_gh)
    P_bn, V_bn = bern_head.forward(mz, Sz)
    J_bn = bern_head.J.detach().float()
    P_bn, V_bn = P_bn.detach().float(), V_bn.detach().float()

    # ── Flatten for histograms ─────────────────────────────────────────
    p_rm_f = P_rm.reshape(-1).cpu().numpy()
    p_bn_f = P_bn.reshape(-1).cpu().numpy()
    v_rm_f = V_rm.reshape(-1).cpu().numpy()
    v_bn_f = V_bn.reshape(-1).cpu().numpy()
    j_rm_f = J_rm.reshape(-1).cpu().numpy()
    j_bn_f = J_bn.reshape(-1).cpu().numpy()

    # ── Summary scalars ───────────────────────────────────────────────
    max_P_rm = P_rm.max(dim=-1).values.mean().item()
    max_P_bn = P_bn.max(dim=-1).values.mean().item()

    # Shannon entropy per sample, then averaged
    eps = 1e-9
    H_rm = -(P_rm * (P_rm + eps).log()).sum(dim=-1).mean().item()
    H_bn = -(P_bn * (P_bn + eps).log()).sum(dim=-1).mean().item()

    # Effective S_pred (= mean output variance, what competes with sigma_v²)
    s_pred_rm = V_rm.mean().item()
    s_pred_bn = V_bn.mean().item()

    J_abs_rm = J_rm.abs().mean().item()
    J_abs_bn = J_bn.abs().mean().item()

    # ── Figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    kw = dict(alpha=0.6, bins=60, density=True)

    color_rm = "#1976D2"   # blue  — Remax
    color_bn = "#D32F2F"   # red   — Bernoulli

    # Row 0, col 0 — P distribution
    ax = axes[0, 0]
    ax.hist(p_rm_f, **kw, color=color_rm, label=f"Remax  (maxP={max_P_rm:.3f})")
    ax.hist(p_bn_f, **kw, color=color_bn, label=f"Bernoulli (maxP={max_P_bn:.3f})")
    ax.set_xlabel("P_i  (prob of class i)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title("Output Probability Distribution", fontsize=9)
    ax.legend(fontsize=7)

    # Row 0, col 1 — max P per sample (confidence)
    ax = axes[0, 1]
    mp_rm = P_rm.max(dim=-1).values.cpu().numpy()
    mp_bn = P_bn.max(dim=-1).values.cpu().numpy()
    ax.hist(mp_rm, bins=30, alpha=0.6, color=color_rm,
            label=f"Remax  μ={max_P_rm:.3f}  H={H_rm:.2f}")
    ax.hist(mp_bn, bins=30, alpha=0.6, color=color_bn,
            label=f"Bernoulli  μ={max_P_bn:.3f}  H={H_bn:.2f}")
    ax.set_xlabel("max_k P_k  (winner confidence)", fontsize=8)
    ax.set_title("Prediction Confidence (max P per sample)\nH = entropy",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Row 0, col 2 — V distribution (output variance → S_pred)
    ax = axes[0, 2]
    ax.hist(np.log10(np.clip(v_rm_f, 1e-10, None)), **kw, color=color_rm,
            label=f"Remax  E[V]={s_pred_rm:.4f}")
    ax.hist(np.log10(np.clip(v_bn_f, 1e-10, None)), **kw, color=color_bn,
            label=f"Bernoulli  E[V]={s_pred_bn:.4f}")
    ax.set_xlabel("log₁₀(V_i)  output variance", fontsize=8)
    ax.set_title("Output Variance V = Bernoulli/Remax variance\n"
                 "(large V → σ_v has less relative influence on δμ)", fontsize=9)
    ax.legend(fontsize=7)

    # Row 1, col 0 — J distribution
    ax = axes[1, 0]
    ax.hist(j_rm_f, **kw, color=color_rm,
            label=f"Remax  E[|J|]={J_abs_rm:.4f}")
    ax.hist(j_bn_f, **kw, color=color_bn,
            label=f"Bernoulli  E[|J|]={J_abs_bn:.4f}")
    ax.set_xlabel("J_k = cov(z_k, a_k) / Sz_k", fontsize=8)
    ax.set_title("Backward Jacobian J\n"
                 "(larger |J| = stronger gradient reaching upstream layers)",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Row 1, col 1 — |J| vs input variance Sz
    ax = axes[1, 1]
    sz_f = Sz.detach().float().reshape(-1).cpu().numpy()
    # subsample for scatter
    n_pts = min(2000, len(sz_f))
    idx   = np.random.choice(len(sz_f), n_pts, replace=False)
    ax.scatter(np.log10(np.clip(sz_f[idx], 1e-10, None)), j_rm_f[idx],
               s=3, alpha=0.4, color=color_rm, label="Remax")
    ax.scatter(np.log10(np.clip(sz_f[idx], 1e-10, None)), j_bn_f[idx],
               s=3, alpha=0.4, color=color_bn, label="Bernoulli")
    ax.set_xlabel("log₁₀(Sz_k)  input variance to head", fontsize=8)
    ax.set_ylabel("J_k", fontsize=8)
    ax.set_title("|J| vs Input Variance\n"
                 "(J→0 when Sz→0: gradient dies if pre-head Sz vanishes)", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)

    # Row 1, col 2 — σ_v break-even point
    ax = axes[1, 2]
    sv_range = np.logspace(-4, 1, 200)
    # Innovation = |y-mu| / (S_pred + sigma_v²)
    # Assuming |y-mu| ≈ 0.5 (half of 1-hot target distance)
    residual = 0.5
    innov_rm = residual / (s_pred_rm + sv_range**2)
    innov_bn = residual / (s_pred_bn + sv_range**2)
    ax.loglog(sv_range, innov_rm, color=color_rm, linewidth=2, label="Remax")
    ax.loglog(sv_range, innov_bn, color=color_bn, linewidth=2, label="Bernoulli")
    ax.set_xlabel("σ_v  (observation noise)", fontsize=8)
    ax.set_ylabel("|δμ|  (innovation magnitude)", fontsize=8)
    ax.set_title("Expected Innovation vs σ_v\n"
                 "(assuming |y−μ|≈0.5 — crossover = head dominates innovation)",
                 fontsize=9)
    # Vertical band for common sigma_v choices
    ax.axvspan(0.001, 0.1, alpha=0.12, color="gray", label="typical σ_v range")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Remax vs Bernoulli Head  (x_probe: B={x.shape[0]}, K={mz.shape[-1]})",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    print(f"  [compare_heads] Plot saved → {filename}")

    # ── Text summary ──────────────────────────────────────────────────
    print(f"\n  {'─'*62}")
    print(f"  Remax vs Bernoulli Head Comparison  (n_gh={n_gh}, B={x.shape[0]})")
    print(f"  {'─'*62}")
    print(f"  {'Metric':<30}  {'Remax':>10}  {'Bernoulli':>10}")
    print(f"  {'─'*62}")
    rows = [
        ("E[max_k P_k]  (confidence)", f"{max_P_rm:.4f}", f"{max_P_bn:.4f}"),
        ("Entropy H  (uncertainty)",   f"{H_rm:.4f}",     f"{H_bn:.4f}"),
        ("E[V]  (output variance)",    f"{s_pred_rm:.4e}", f"{s_pred_bn:.4e}"),
        ("E[|J|]  (Jacobian mag.)",    f"{J_abs_rm:.4f}", f"{J_abs_bn:.4f}"),
        ("J std  (spread)",            f"{J_rm.std().item():.4f}",
                                        f"{J_bn.std().item():.4f}"),
    ]
    for label, rm, bn in rows:
        print(f"  {label:<30}  {rm:>10}  {bn:>10}")
    print(f"  {'─'*62}")
    print(f"  σ_v crossover (E[V]=σ_v²):")
    print(f"    Remax    σ_v* ≈ {math.sqrt(s_pred_rm):.4f}")
    print(f"    Bernoulli σ_v* ≈ {math.sqrt(s_pred_bn):.4f}")
    print(f"  For σ_v >> σ_v*: head variance dominates → innovation ≈ 1/S_pred")
    print(f"  For σ_v << σ_v*: observation precision dominates → innov. ≈ 1/σ_v²")
    print(f"  {'─'*62}")

    if show:
        plt.show()
    plt.close(fig)
