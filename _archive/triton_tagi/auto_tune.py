"""
Auto-Tune — Automatic Gain and sigma_v Selection for TAGI
==========================================================

Two-phase search that finds the best (gain_w, gain_b, sigma_v) for a given
network architecture:

Phase 1 (gain):    Forward-only analysis of per-layer activation variance.
Phase 2 (sigma_v): Short training runs to evaluate learning dynamics.

Quick-start
-----------
    from src.auto_tune import auto_tune

    result = auto_tune(
        builder_fn = lambda gw, gb: build_my_net(gain_w=gw, gain_b=gb),
        x_probe    = x_train[:512],
        y_probe    = y_train_oh[:512],
        x_eval     = x_test[:1000],
        y_eval     = y_test_labels[:1000],
    )
    # result.gain_w, result.gain_b, result.sigma_v
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .layers.batchnorm2d import BatchNorm2D
from .layers.conv2d import Conv2D
from .layers.linear import Linear
from .layers.resblock import ResBlock
from .monitor import ActivationStats, TAGIMonitor
from .update.observation import compute_innovation
from .update.parameters import get_cap_factor

_LEARNABLE = (Linear, Conv2D, BatchNorm2D, ResBlock)


# ══════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════


@dataclass
class GainScore:
    """Score for a single (gain_w, gain_b) candidate."""

    gain_w: float
    gain_b: float
    total_score: float
    centering: float  # mean |log10(var) - target|
    spread: float  # max_log - min_log across layers
    dead_penalty: float
    explode_penalty: float
    nan_crash: bool
    verdict: str  # GOOD / OK / VANISHING / EXPLODING / UNEVEN / CRASH
    per_layer_log_var: list[float]


@dataclass
class GainResult:
    """Result of Phase 1: gain search."""

    best_gain_w: float
    best_gain_b: float
    best_score: GainScore
    all_scores: list[GainScore]
    target_var: float


@dataclass
class SigmaVScore:
    """Score for a single sigma_v candidate."""

    sigma_v: float
    total_score: float
    accuracy_after: float
    accuracy_delta: float
    avg_frac_capped: float
    avg_param_movement: float


@dataclass
class SigmaVResult:
    """Result of Phase 2: sigma_v search."""

    best_sigma_v: float
    best_score: SigmaVScore
    all_scores: list[SigmaVScore]
    crossover_sigma_v: float


@dataclass
class TuneResult:
    """Complete auto-tune result."""

    gain_w: float
    gain_b: float
    sigma_v: float
    gain_result: GainResult
    sigma_v_result: SigmaVResult


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: Gain scoring
# ══════════════════════════════════════════════════════════════════════


def _score_gain(
    stats: list[ActivationStats], target_var: float, gain_w: float, gain_b: float
) -> GainScore:
    """
    Score a single (gain_w, gain_b) candidate using per-layer variance health.

    Reuses the threshold logic from sweep_gains (monitor.py lines 685-701).
    """
    target_log = math.log10(max(target_var, 1e-10))

    # Check for NaN/Inf
    if any(math.isnan(s.var_mean) or math.isinf(s.var_mean) for s in stats):
        return GainScore(
            gain_w=gain_w,
            gain_b=gain_b,
            total_score=-1000.0,
            centering=-100.0,
            spread=100.0,
            dead_penalty=0.0,
            explode_penalty=0.0,
            nan_crash=True,
            verdict="CRASH",
            per_layer_log_var=[],
        )

    log_vars = [math.log10(max(s.var_mean, 1e-10)) for s in stats]
    lo, hi = min(log_vars), max(log_vars)
    spread = hi - lo

    # 1. Centering: mean distance from target (negative = worse)
    deviations = [abs(lv - target_log) for lv in log_vars]
    centering = -sum(deviations) / len(deviations)

    # 2. Spread penalty (want < 1 decade)
    spread_penalty = -max(0.0, spread - 1.0) * 2.0

    # 3. Dead activation penalty
    avg_dead = sum(s.frac_dead for s in stats) / len(stats)
    dead_penalty = -10.0 * max(0.0, avg_dead - 0.1)

    # 4. Exploding activation penalty
    avg_explode = sum(s.frac_explode for s in stats) / len(stats)
    explode_penalty = -20.0 * avg_explode

    # 5. Extreme variance penalty (VANISHING / EXPLODING)
    extreme_penalty = 0.0
    if lo < target_log - 3:
        extreme_penalty -= 5.0 * abs(lo - (target_log - 3))
    if hi > target_log + 3:
        extreme_penalty -= 5.0 * abs(hi - (target_log + 3))

    total = centering + spread_penalty + dead_penalty + explode_penalty + extreme_penalty

    # Verdict (matches sweep_gains logic)
    if hi > target_log + 3:
        verdict = "EXPLODING"
    elif lo < target_log - 3:
        verdict = "VANISHING"
    elif spread > 3:
        verdict = "UNEVEN"
    elif abs((lo + hi) / 2 - target_log) < 1.5:
        verdict = "GOOD"
    else:
        verdict = "OK"

    return GainScore(
        gain_w=gain_w,
        gain_b=gain_b,
        total_score=total,
        centering=centering,
        spread=spread,
        dead_penalty=dead_penalty,
        explode_penalty=explode_penalty,
        nan_crash=False,
        verdict=verdict,
        per_layer_log_var=log_vars,
    )


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: find_best_gain
# ══════════════════════════════════════════════════════════════════════


def find_best_gain(
    builder_fn: Callable,
    x_probe: torch.Tensor,
    gains_w: list[float] | None = None,
    bias_factors: list[float] | None = None,
    target_var: float = 1.0,
    refine: bool = True,
    n_refine: int = 10,
    probe_size: int = 256,
    verbose: bool = True,
) -> GainResult:
    """
    Find the best (gain_w, gain_b) by forward-only variance analysis.

    Parameters
    ----------
    builder_fn    : callable(gain_w, gain_b) → Sequential
    x_probe       : Tensor  sample input batch
    gains_w       : list of float  gain_w candidates (default: logspace -3..1, 15 pts)
    bias_factors  : list of float  gain_b = factor × gain_w (default: [0.5, 1.0, 2.0])
    target_var    : float  target per-layer activation variance (default 1.0)
    refine        : bool   refine around best gain_w (default True)
    n_refine      : int    refinement points (default 10)
    probe_size    : int    max batch for probing (default 256)
    verbose       : bool

    Returns
    -------
    GainResult with best (gain_w, gain_b) and all scores
    """
    if gains_w is None:
        gains_w = np.logspace(-3, 1, 15).tolist()
    if bias_factors is None:
        bias_factors = [0.5, 1.0, 2.0]

    x = x_probe[:probe_size].detach()
    all_scores: list[GainScore] = []

    if verbose:
        n_total = len(gains_w) * len(bias_factors)
        print(f"\n  Phase 1: Gain Search ({n_total} candidates, target_var={target_var})")
        print(f"  {'─' * 60}")

    for gw in gains_w:
        for bf in bias_factors:
            gb = gw * bf
            try:
                net = builder_fn(gw, gb)
                monitor = TAGIMonitor(net, probe_size=probe_size)
                stats = monitor.probe(x)
                score = _score_gain(stats, target_var, gw, gb)
            except Exception:
                score = GainScore(
                    gain_w=gw,
                    gain_b=gb,
                    total_score=-1000.0,
                    centering=-100.0,
                    spread=100.0,
                    dead_penalty=0.0,
                    explode_penalty=0.0,
                    nan_crash=True,
                    verdict="CRASH",
                    per_layer_log_var=[],
                )
            all_scores.append(score)
            del net
            torch.cuda.empty_cache()

    # Sort by score descending
    all_scores.sort(key=lambda s: s.total_score, reverse=True)
    best = all_scores[0]

    if verbose:
        _print_gain_table(all_scores[:10], target_var)

    # ── Refinement around best gain_w ──
    if refine and not best.nan_crash:
        best_gw = best.gain_w
        best_bf = best.gain_b / best.gain_w if best.gain_w > 0 else 1.0

        # Find neighbors in log-space
        log_best = math.log10(best_gw)
        log_lo = log_best - 0.5
        log_hi = log_best + 0.5
        refine_gws = np.logspace(log_lo, log_hi, n_refine).tolist()

        if verbose:
            print(f"\n  Refining around gain_w={best_gw:.4f} (±0.5 decades, {n_refine} points)")

        for gw in refine_gws:
            gb = gw * best_bf
            try:
                net = builder_fn(gw, gb)
                monitor = TAGIMonitor(net, probe_size=probe_size)
                stats = monitor.probe(x)
                score = _score_gain(stats, target_var, gw, gb)
            except Exception:
                score = GainScore(
                    gain_w=gw,
                    gain_b=gb,
                    total_score=-1000.0,
                    centering=-100.0,
                    spread=100.0,
                    dead_penalty=0.0,
                    explode_penalty=0.0,
                    nan_crash=True,
                    verdict="CRASH",
                    per_layer_log_var=[],
                )
            all_scores.append(score)
            del net
            torch.cuda.empty_cache()

        all_scores.sort(key=lambda s: s.total_score, reverse=True)
        best = all_scores[0]

        if verbose:
            print(
                f"  Refined best: gain_w={best.gain_w:.4f}, "
                f"gain_b={best.gain_b:.4f}  [{best.verdict}]"
            )

    if verbose:
        print(
            f"\n  BEST GAIN: gain_w={best.gain_w:.4f}, "
            f"gain_b={best.gain_b:.4f}, score={best.total_score:.3f} "
            f"[{best.verdict}]"
        )
        print(f"  {'─' * 60}")

    return GainResult(
        best_gain_w=best.gain_w,
        best_gain_b=best.gain_b,
        best_score=best,
        all_scores=all_scores,
        target_var=target_var,
    )


def _print_gain_table(scores: list[GainScore], target_var: float):
    """Print top gain candidates as a table."""
    print(f"\n  {'gain_w':>8}  {'gain_b':>8}  {'Score':>7}  {'Center':>7}  {'Spread':>7}  Verdict")
    print(f"  {'─' * 60}")
    for s in scores:
        print(
            f"  {s.gain_w:>8.4f}  {s.gain_b:>8.4f}  {s.total_score:>7.2f}  "
            f"{s.centering:>7.2f}  {s.spread:>7.2f}  {s.verdict}"
        )


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: sigma_v scoring
# ══════════════════════════════════════════════════════════════════════


def _score_sigma_v(
    accuracy_after, accuracy_delta, avg_frac_capped, avg_param_movement, sigma_v
) -> SigmaVScore:
    """
    Score a sigma_v candidate using combined metrics.

    Primary:    accuracy improvement
    Secondary:  fraction capped in sweet spot (10-50%)
    Tiebreaker: parameter movement (not zero, not excessive)
    """
    # 1. Accuracy improvement (primary, weighted heavily)
    acc_score = accuracy_delta * 100.0

    # 2. Fraction capped sweet spot
    fc = avg_frac_capped
    if fc > 0.95:
        cap_score = -5.0  # fully capped → sigma_v irrelevant
    elif fc < 0.01:
        cap_score = -3.0  # uncapped → potentially unstable
    else:
        # Prefer 10-50% capped; peak around 30%
        cap_score = -abs(fc - 0.3) * 5.0

    # 3. Parameter movement (tiebreaker)
    pm = avg_param_movement
    if pm < 1e-8:
        move_score = -5.0  # no learning at all
    elif pm > 0.5:
        move_score = -3.0  # too aggressive
    else:
        # Log-scale scoring, peak around 0.001-0.01
        move_score = (math.log10(max(pm, 1e-10)) + 3.0) * 0.5

    total = acc_score + cap_score + move_score

    return SigmaVScore(
        sigma_v=sigma_v,
        total_score=total,
        accuracy_after=accuracy_after,
        accuracy_delta=accuracy_delta,
        avg_frac_capped=avg_frac_capped,
        avg_param_movement=avg_param_movement,
    )


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: find_best_sigma_v
# ══════════════════════════════════════════════════════════════════════


def find_best_sigma_v(
    builder_fn: Callable,
    gain_w: float,
    gain_b: float,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    sigma_vs: list[float] | None = None,
    n_steps: int = 50,
    batch_size: int = 128,
    x_eval: torch.Tensor | None = None,
    y_eval: torch.Tensor | None = None,
    probe_size: int = 512,
    verbose: bool = True,
) -> SigmaVResult:
    """
    Find the best sigma_v by running short training trials.

    Parameters
    ----------
    builder_fn : callable(gain_w, gain_b) → Sequential
    gain_w     : float   from Phase 1
    gain_b     : float   from Phase 1
    x_train    : Tensor  training inputs (used for mini-batch sampling)
    y_train    : Tensor  training targets (one-hot)
    sigma_vs   : list of float  candidates (default: auto from crossover)
    n_steps    : int     training steps per candidate (default 50)
    batch_size : int     mini-batch size (default 128)
    x_eval     : Tensor  evaluation inputs (default: use x_train[:1000])
    y_eval     : Tensor  evaluation labels (integer class indices)
    probe_size : int     max samples from x_train for training (default 512)
    verbose    : bool

    Returns
    -------
    SigmaVResult with best sigma_v, crossover, and all scores
    """
    x_tr = x_train[:probe_size].detach()
    y_tr = y_train[:probe_size].detach()

    if x_eval is None:
        x_eval = x_train[:1000].detach()
    if y_eval is None:
        raise ValueError("y_eval (integer labels) is required for accuracy evaluation")

    x_ev = x_eval.detach()
    y_ev = y_eval.detach()

    # ── Compute crossover sigma_v* ──
    net_probe = builder_fn(gain_w, gain_b)
    net_probe.eval()
    mu_pred, var_pred = net_probe.forward(x_tr[:256])
    crossover_sv = math.sqrt(max(var_pred.mean().item(), 1e-10))
    del net_probe
    torch.cuda.empty_cache()

    if verbose:
        print(f"\n  Phase 2: sigma_v Search (n_steps={n_steps}, batch_size={batch_size})")
        print(f"  Crossover σ_v* = {crossover_sv:.4e}")
        print(f"  {'─' * 70}")

    # ── Default grid centered on crossover ──
    if sigma_vs is None:
        log_cross = math.log10(max(crossover_sv, 1e-6))
        sigma_vs = np.logspace(log_cross - 2, log_cross + 2, 8).tolist()

    if verbose:
        sv_str = ", ".join(f"{sv:.3e}" for sv in sigma_vs)
        print(f"  Candidates: [{sv_str}]")

    # ── Compute initial accuracy (before any training) ──
    net_init = builder_fn(gain_w, gain_b)
    net_init.eval()
    acc_init = _evaluate(net_init, x_ev, y_ev)
    del net_init
    torch.cuda.empty_cache()

    if verbose:
        print(f"  Initial accuracy: {acc_init * 100:.1f}%\n")

    all_scores: list[SigmaVScore] = []

    for sv in sigma_vs:
        # Build fresh net
        net = builder_fn(gain_w, gain_b)

        # Snapshot initial weights
        mw_before = {}
        for i, layer in enumerate(net.layers):
            if isinstance(layer, _LEARNABLE):
                mw_before[i] = layer.mw.detach().float().flatten().clone()

        # Train for n_steps
        net.train()
        n_train = len(x_tr)
        for step in range(n_steps):
            idx_start = (step * batch_size) % n_train
            idx_end = min(idx_start + batch_size, n_train)
            if idx_end - idx_start < 2:
                idx_start, idx_end = 0, min(batch_size, n_train)
            xb = x_tr[idx_start:idx_end]
            yb = y_tr[idx_start:idx_end]
            net.step(xb, yb, sv)

        # Evaluate accuracy
        net.eval()
        acc_after = _evaluate(net, x_ev, y_ev)
        acc_delta = acc_after - acc_init

        # Parameter movement
        movements = []
        for i, layer in enumerate(net.layers):
            if i not in mw_before:
                continue
            mw_now = layer.mw.detach().float().flatten()
            delta = (mw_now - mw_before[i]).abs().mean().item()
            denom = max(mw_before[i].abs().mean().item(), 1e-8)
            movements.append(delta / denom)
        avg_movement = sum(movements) / len(movements) if movements else 0.0

        # Fraction capped (one probe forward+backward)
        avg_frac = _measure_frac_capped(net, x_tr[:256], y_tr[:256], sv)

        score = _score_sigma_v(acc_after, acc_delta, avg_frac, avg_movement, sv)
        all_scores.append(score)

        if verbose:
            print(
                f"  σ_v={sv:.3e}  acc={acc_after * 100:5.1f}%  "
                f"Δacc={acc_delta * 100:+5.1f}%  "
                f"capped={avg_frac * 100:4.0f}%  "
                f"move={avg_movement:.3e}  "
                f"score={score.total_score:.2f}"
            )

        del net
        torch.cuda.empty_cache()

    all_scores.sort(key=lambda s: s.total_score, reverse=True)
    best = all_scores[0]

    if verbose:
        print(
            f"\n  BEST σ_v: {best.sigma_v:.4e}  "
            f"(acc={best.accuracy_after * 100:.1f}%, "
            f"Δacc={best.accuracy_delta * 100:+.1f}%, "
            f"capped={best.avg_frac_capped * 100:.0f}%)"
        )
        print(f"  {'─' * 70}")

    return SigmaVResult(
        best_sigma_v=best.sigma_v,
        best_score=best,
        all_scores=all_scores,
        crossover_sigma_v=crossover_sv,
    )


def _evaluate(net, x_eval, y_labels, batch_size=256):
    """Compute argmax accuracy."""
    correct = 0
    for i in range(0, len(x_eval), batch_size):
        xb = x_eval[i : i + batch_size]
        lb = y_labels[i : i + batch_size]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1) == lb).sum().item()
    return correct / len(x_eval)


def _measure_frac_capped(net, x_sample, y_sample, sigma_v):
    """
    Run one forward+backward pass and measure what fraction of parameter
    updates would exceed the cap budget.
    """
    net.eval()
    mu_pred, var_pred = net.forward(x_sample)
    delta_mu, delta_var = compute_innovation(y_sample, mu_pred, var_pred, sigma_v)

    dm, dv = delta_mu.clone(), delta_var.clone()
    for layer in reversed(net.layers):
        dm, dv = layer.backward(dm, dv)

    cap_factor = get_cap_factor(x_sample.shape[0])
    fracs = []
    for layer in net.layers:
        if not isinstance(layer, _LEARNABLE):
            continue
        if layer.delta_mw is None:
            continue
        dmw = layer.delta_mw.detach().float().flatten()
        Sw = layer.Sw.detach().float().flatten().clamp(min=1e-10)
        budget = Sw.sqrt() / cap_factor
        frac = (dmw.abs() > budget).float().mean().item()
        fracs.append(frac)

    return sum(fracs) / len(fracs) if fracs else 0.0


# ══════════════════════════════════════════════════════════════════════
#  Main entry point: auto_tune
# ══════════════════════════════════════════════════════════════════════


def auto_tune(
    builder_fn: Callable,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    # Gain search params
    gains_w: list[float] | None = None,
    bias_factors: list[float] | None = None,
    target_var: float = 1.0,
    refine_gain: bool = True,
    # Sigma_v search params
    sigma_vs: list[float] | None = None,
    n_steps: int = 50,
    batch_size: int = 128,
    # Evaluation
    x_eval: torch.Tensor | None = None,
    y_eval: torch.Tensor | None = None,
    # Output
    probe_size: int = 256,
    verbose: bool = True,
    plot_filename: str | None = "auto_tune.png",
) -> TuneResult:
    """
    Automatic hyperparameter tuning for TAGI networks.

    Phase 1: Forward-only gain search (fast).
    Phase 2: Short-training sigma_v search (uses best gain from Phase 1).

    Parameters
    ----------
    builder_fn    : callable(gain_w, gain_b) → Sequential
    x_probe       : Tensor  sample inputs for probing / training
    y_probe       : Tensor  sample targets (one-hot)
    gains_w       : list of float  gain_w candidates (default: auto)
    bias_factors  : list of float  gain_b = factor × gain_w (default: [0.5, 1.0, 2.0])
    target_var    : float  target activation variance (default 1.0)
    refine_gain   : bool   refine gain search (default True)
    sigma_vs      : list of float  sigma_v candidates (default: auto from crossover)
    n_steps       : int    training steps per sigma_v candidate (default 50)
    batch_size    : int    mini-batch size for sigma_v trials (default 128)
    x_eval        : Tensor  evaluation inputs (default: x_probe[:1000])
    y_eval        : Tensor  evaluation labels (integer class indices)
    probe_size    : int    max batch for probing (default 256)
    verbose       : bool
    plot_filename : str or None  save diagnostic plot (default "auto_tune.png")

    Returns
    -------
    TuneResult with recommended gain_w, gain_b, sigma_v
    """
    if verbose:
        print("=" * 62)
        print("  TAGI Auto-Tune")
        print("=" * 62)

    # ── Phase 1: Gain ──
    gain_result = find_best_gain(
        builder_fn=builder_fn,
        x_probe=x_probe,
        gains_w=gains_w,
        bias_factors=bias_factors,
        target_var=target_var,
        refine=refine_gain,
        probe_size=probe_size,
        verbose=verbose,
    )

    # ── Phase 2: sigma_v ──
    sigma_v_result = find_best_sigma_v(
        builder_fn=builder_fn,
        gain_w=gain_result.best_gain_w,
        gain_b=gain_result.best_gain_b,
        x_train=x_probe,
        y_train=y_probe,
        sigma_vs=sigma_vs,
        n_steps=n_steps,
        batch_size=batch_size,
        x_eval=x_eval if x_eval is not None else x_probe[:1000],
        y_eval=y_eval,
        probe_size=min(len(x_probe), 512),
        verbose=verbose,
    )

    result = TuneResult(
        gain_w=gain_result.best_gain_w,
        gain_b=gain_result.best_gain_b,
        sigma_v=sigma_v_result.best_sigma_v,
        gain_result=gain_result,
        sigma_v_result=sigma_v_result,
    )

    # ── Summary ──
    if verbose:
        _print_summary(result)

    # ── Diagnostic plot ──
    if plot_filename is not None:
        _plot_tune_results(result, plot_filename, verbose=verbose)

    return result


# ══════════════════════════════════════════════════════════════════════
#  Summary and plotting
# ══════════════════════════════════════════════════════════════════════


def _print_summary(result: TuneResult):
    """Print a compact text summary of the auto-tune results."""
    gr = result.gain_result
    sr = result.sigma_v_result
    bs = gr.best_score
    ss = sr.best_score

    print(f"\n  {'=' * 62}")
    print("  TAGI Auto-Tune Results")
    print(f"  {'=' * 62}")

    print(f"\n  Phase 1: Gain (forward-only, {len(gr.all_scores)} candidates)")
    print(f"    gain_w = {gr.best_gain_w:.4f},  gain_b = {gr.best_gain_b:.4f}")
    print(f"    Verdict: {bs.verdict}")
    print(
        f"    Score: {bs.total_score:.2f}  (centering={bs.centering:.2f}, spread={bs.spread:.2f})"
    )
    if bs.per_layer_log_var:
        lv_str = ", ".join(f"{v:.1f}" for v in bs.per_layer_log_var)
        print(f"    Per-layer log10(var): [{lv_str}]")

    print(
        f"\n  Phase 2: sigma_v ({len(sr.all_scores)} candidates, "
        f"crossover={sr.crossover_sigma_v:.3e})"
    )
    print(f"    sigma_v = {sr.best_sigma_v:.4e}")
    print(f"    Accuracy: {ss.accuracy_after * 100:.1f}%  (delta={ss.accuracy_delta * 100:+.1f}%)")
    print(f"    Fraction capped: {ss.avg_frac_capped * 100:.0f}%")
    print(f"    Param movement: {ss.avg_param_movement:.3e}")

    print("\n  RECOMMENDED:")
    print(f"    gain_w = {result.gain_w:.4f}")
    print(f"    gain_b = {result.gain_b:.4f}")
    print(f"    sigma_v = {result.sigma_v:.4e}")
    print(f"  {'=' * 62}")


def _plot_tune_results(result: TuneResult, filename: str, verbose: bool = True):
    """Generate a 4-panel diagnostic plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [auto_tune] matplotlib not found, skipping plot.")
        return

    gr = result.gain_result
    sr = result.sigma_v_result

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1 (top-left): Gain score landscape ──
    ax = axes[0, 0]
    # Group by gain_w, show best score across bias_factors
    gw_best = {}
    for s in gr.all_scores:
        if s.nan_crash:
            continue
        key = round(s.gain_w, 8)
        if key not in gw_best or s.total_score > gw_best[key].total_score:
            gw_best[key] = s

    if gw_best:
        gws = sorted(gw_best.keys())
        scores_y = [gw_best[g].total_score for g in gws]
        colors = []
        for g in gws:
            v = gw_best[g].verdict
            if v == "GOOD":
                colors.append("#388e3c")
            elif v == "OK":
                colors.append("#1976D2")
            elif v == "VANISHING":
                colors.append("#f57c00")
            elif v == "EXPLODING":
                colors.append("#d32f2f")
            else:
                colors.append("#757575")
        ax.scatter(
            [math.log10(g) for g in gws],
            scores_y,
            c=colors,
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.plot([math.log10(g) for g in gws], scores_y, "k-", alpha=0.3)
        ax.axvline(
            math.log10(gr.best_gain_w),
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="best",
        )

    ax.set_xlabel("log10(gain_w)", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(
        "Gain Score Landscape\n(green=GOOD, blue=OK, orange=VANISHING, red=EXPLODING)", fontsize=9
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2 (top-right): Per-layer variance at best gain ──
    ax2 = axes[0, 1]
    best_gs = gr.best_score
    if best_gs.per_layer_log_var:
        xs = range(len(best_gs.per_layer_log_var))
        target_log = math.log10(max(gr.target_var, 1e-10))
        colors2 = []
        for lv in best_gs.per_layer_log_var:
            dist = abs(lv - target_log)
            if dist < 1:
                colors2.append("#388e3c")
            elif dist < 2:
                colors2.append("#f57c00")
            else:
                colors2.append("#d32f2f")
        ax2.bar(xs, best_gs.per_layer_log_var, color=colors2, alpha=0.8)
        ax2.axhline(
            target_log,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"target={gr.target_var}",
        )
        ax2.axhline(target_log + 2, color="red", linestyle=":", linewidth=1, alpha=0.5)
        ax2.axhline(target_log - 2, color="orange", linestyle=":", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Layer index", fontsize=9)
    ax2.set_ylabel("log10(E[var])", fontsize=9)
    ax2.set_title(
        f"Per-Layer Variance at Best Gain\n"
        f"(gain_w={gr.best_gain_w:.4f}, gain_b={gr.best_gain_b:.4f})",
        fontsize=9,
    )
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3 (bottom-left): sigma_v score landscape ──
    ax3 = axes[1, 0]
    sv_vals = [s.sigma_v for s in sr.all_scores]
    sv_scores = [s.total_score for s in sr.all_scores]
    ax3.plot(
        [math.log10(sv) for sv in sv_vals],
        sv_scores,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
    )
    ax3.axvline(
        math.log10(sr.crossover_sigma_v),
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"crossover σ_v*={sr.crossover_sigma_v:.2e}",
    )
    ax3.axvline(
        math.log10(sr.best_sigma_v),
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"best σ_v={sr.best_sigma_v:.2e}",
    )
    ax3.set_xlabel("log10(σ_v)", fontsize=9)
    ax3.set_ylabel("Score", fontsize=9)
    ax3.set_title("sigma_v Score Landscape", fontsize=9)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # ── Panel 4 (bottom-right): Training dynamics vs sigma_v ──
    ax4 = axes[1, 1]
    svs_log = [math.log10(s.sigma_v) for s in sr.all_scores]

    ax4_acc = ax4
    ax4_acc.plot(
        svs_log,
        [s.accuracy_delta * 100 for s in sr.all_scores],
        "o-",
        color="#1976D2",
        linewidth=2,
        label="Δacc (%)",
    )
    ax4_acc.set_ylabel("Δ accuracy (%)", fontsize=9, color="#1976D2")
    ax4_acc.tick_params(axis="y", labelcolor="#1976D2")

    ax4b = ax4.twinx()
    ax4b.plot(
        svs_log,
        [s.avg_frac_capped * 100 for s in sr.all_scores],
        "s--",
        color="#d32f2f",
        linewidth=1.5,
        alpha=0.8,
        label="% capped",
    )
    ax4b.set_ylabel("% capped", fontsize=9, color="#d32f2f")
    ax4b.tick_params(axis="y", labelcolor="#d32f2f")
    ax4b.set_ylim(-5, 105)

    ax4.set_xlabel("log10(σ_v)", fontsize=9)
    ax4.set_title(
        "Training Dynamics vs sigma_v\n(blue=accuracy gain, red=fraction capped)", fontsize=9
    )
    ax4.grid(alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4_acc.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.suptitle(
        f"Auto-Tune Results  →  gain_w={result.gain_w:.4f}, "
        f"gain_b={result.gain_b:.4f}, σ_v={result.sigma_v:.3e}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=130, bbox_inches="tight")

    if verbose:
        print(f"  [auto_tune] Plot saved → {filename}")
    plt.close(fig)
