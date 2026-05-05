# triton-tagi: Engineering Plan

Single source of truth for the library's scope and roadmap. Update it when
scope changes; do not let it drift.

---

## 1. Vision

triton-tagi is a **minimal, Python/Triton reimplementation of cuTAGI**. It
mirrors the subset of cuTAGI needed to reproduce cuTAGI's headline examples,
with identical numerical behaviour, and it runs faster than cuTAGI on a single
GPU at modest-to-large batch sizes thanks to fused Triton kernels.

Goals, in priority order:

1. **Parity** — every kept example reproduces the cuTAGI result within the
   Phase-1 validation tolerance (and usually much tighter).
2. **Readable** — a new user can read the package top-to-bottom in an evening.
   No abstractions beyond what the kept layers need.
3. **Extensible** — adding a new layer means adding one file under
   `triton_tagi/layers/` and one test; no registry, no decorators.
4. **Fast** — fused Triton kernels; benchmarked against cuTAGI on the same
   hardware.

**Non-goals (for now):** LSTM, autograd-style computation graph,
Adam/Nadam/momentum optimizers, FRN/TLU/SharedVar variants,
ConvTranspose2D, posterior-tempering / cold-posterior studies,
multi-GPU/DDP. Code for the non-goals lives under `_archive/` and can be
restored if ever needed.

---

## 2. Library Surface

### Layers (`triton_tagi/layers/`)

| Layer | Used by |
|---|---|
| `Linear` | MLP, CNN heads, ResNet head |
| `Conv2D` | CNN, ResNet |
| `BatchNorm2D` | CNN, ResNet |
| `LayerNorm` | MLP variant |
| `AvgPool2D` | CNN, ResNet stem/head |
| `MaxPool2D` | CNN (alternate pool) |
| `ReLU` | all non-linear examples |
| `Flatten` | conv→FC boundary |
| `Remax` | classification head (cuTAGI-native output activation) |
| `ResBlock` + `Add` | ResNet-18 |
| `EvenSoftplus` | heteroscedastic regression noise head |
| `Embedding` | reverse_predictor |
| `PositionalEncoding` | reverse_predictor (sinusoidal, fixed) |
| `MultiheadAttentionV2` | reverse_predictor (V2, separate Q/K/V) |
| `RMSNorm` | reverse_predictor |

### Top-level (`triton_tagi/`)

- `base.py` — `Layer`, `LearnableLayer` ABCs
- `network.py` — `Sequential`
- `param_init.py` — He/Xavier/Gaussian init for Linear/Conv/Norm layers
- `hrc_softmax.py` — hierarchical softmax output for many-class classification
- `checkpoint.py` — `RunDir` (training-side I/O) and `load_model` (inference-side loader)
- `kernels/common.py` — fused Triton kernels for Linear / Conv2D / BN
  (variance forward, backward delta, weight gradient)
- `kernels/attention.py` — fused Triton kernels for `MultiheadAttentionV2`
  (`bmm_tagi_var` for full-Gaussian QKᵀ / Score@V, `bmm_shared_left/right`
  for the four backward reductions). Block sizes are picked by a
  `_pick_blocks(M,L,K)` heuristic — `@triton.autotune` was removed because
  its ~50-100µs/call dispatch overhead dominated at small attention shapes
  (see `feedback_triton_autotune_overhead.md` in memory)
- `update/observation.py`, `update/parameters.py` — innovation and update rules

That's it. Everything else is archived.

---

## 3. Parity Examples

The acceptance criterion for the library is: each of these examples produces
the same result as its cuTAGI counterpart, at the Phase-1 tolerance, on a
fixed seed. "Same result" means: final-epoch test accuracy / RMSE within the
documented tolerance, training curves qualitatively overlapping.

| Example | cuTAGI source | Status | Notes |
|---|---|---|---|
| `regression.py` | `regression.py` | ☑ parity | 1-D toy regression with epistemic bands |
| `regression_heteros.py` | `regression_heteros.py` | ☑ runs | Uses `EvenSoftplus`; cuTAGI uses `EvenExp` — qualitatively same, numerically differs |
| `mnist_mlp.py` | `classification.py` | ☑ parity | 95.3% @ ep5 (no head) / 96.9% @ ep3 (remax) |
| `mnist_cnn.py` | `classification.py` | ☑ parity | 97.75% @ ep3 |
| `cifar10_cnn.py` | `classification.py` | ☑ runs | 3-block BN-CNN, 80% @ ep100 w/ aug |
| `cifar10_resnet18.py` | `resnet18_cifar10.py` | ☑ runs | ResNet-18, gain=0.1, σ_v=0.05 |
| `cifar10_resnet18_hrc.py` | `softmax_cifar.py` | ☑ runs | ResNet-18 + hierarchical softmax head |
| `custom_layer.py` | (new) | ☑ works | ELU tutorial: Triton kernel → `Layer` subclass → MNIST |
| `reverse_predictor.py` | `reverse_predictor.py` (feat/attn-debug) | ☑ runs | Sinusoidal PE + MHA-V2 + RMSNorm + HRC head; sequence-to-sequence reversal |

### Example conventions

1. Use `RunDir` from `triton_tagi.checkpoint` for every artifact.
2. All hyperparameters live in a single `config: dict`, written to `config.json`.
3. Support `--help` via `argparse`; expose at minimum `n_epochs`, `batch_size`, `sigma_v`.
4. Print one line per epoch: epoch number, train metric, test metric, wall time.
5. Save a final figure as both `.pdf` and `.png` in `run.figures/`.

---

## 4. Testing

### Unit tests (`tests/unit/`)

One file per module. Minimum per layer:

| Test | What it checks |
|---|---|
| `test_<layer>_forward_shape` | Output shape for standard input |
| `test_<layer>_forward_zero_sa` | `Sa=0` reduces to deterministic forward |
| `test_<layer>_backward_shape` | Delta output shape matches delta input |
| `test_<layer>_backward_passthrough` | Zero deltas in → zero deltas out |
| `test_<layer>_update_decreases_sw` | Update with non-zero signal shrinks `Sw` |

### Validation tests (`tests/validation/`)

Every numerical claim in PLAN §3 is backed by a test. Each compares a layer's
forward / backward / update against a pytagi reference on a fixed batch,
asserting `torch.testing.assert_close(atol=1e-5, rtol=0)`.

Full training runs (`test_mnist_mlp.py`, `test_cifar10_cnn.py`,
`test_cifar10_resnet18.py`) are gated behind `@pytest.mark.slow` / `cuda`.

### Conventions

- Framework: `pytest`
- Tensor comparisons: `torch.testing.assert_close(atol=1e-5, rtol=0)` — never bare `assert a == b`
- GPU tests: `@pytest.mark.cuda`, auto-skipped without GPU
- Seed every test: `torch.manual_seed(42)`

---

## 5. Tensor / Naming Conventions

- Activations: `ma` (mean), `Sa` (variance). Capital `S` for all variance tensors.
- Weights: `mw`, `Sw`. Biases: `mb`, `Sb`.
- Deltas: `delta_ma`, `delta_Sa`, `delta_mz`, `delta_Sz`.
- Every layer inherits `Layer` (pure moment propagation) or `LearnableLayer`
  (+ `update`, `num_parameters`).
- Style: `ruff format`, line length 100, rules `E / F / I / UP / B`.

---

## 6. Benchmarks (`benchmarks/`)

- `bench_vs_cutagi.py` — Linear / Conv2D / BN networks at batch {1, 16, 32, 64, 256, 1024}; median of 50 runs.
- `bench_attention.py` — per-call `bmm_tagi_var` / `bmm_shared_*` vs torch
  baseline (two `matmul` + elementwise), plus end-to-end MHA forward+backward.
  Uses CUDA events for sub-100µs precision.
- `profile_bottlenecks.py` — `torch.profiler` trace with hot-kernel breakdown.
- `results.md` — current numbers (batch-1024: Linear 70×, Conv2D 9.7×, BN 8.7× vs cuTAGI).
  No attention row yet; `bench_attention.py` reports inline.

Benchmark code only exercises the minimal surface. No archived layers.

---

## 7. Archive (`_archive/`)

Nothing is deleted. Moved-out code lives under:

- `_archive/triton_tagi/` — archived layers (Bernoulli, ConvTranspose2D,
  FRN/TLU/FRNResBlock, LeakyReLU, SiLU, SharedVar\*), optimizers (AdamTAGI,
  NadamTAGI, StateSpaceMomentum), research utilities (`auto_tune.py`,
  `monitor.py`, `inference_init.py`, `init.py`),
  `update/shared_var_parameters.py`.
- `_archive/tests/` — tests for the archived code.
- `_archive/diagnostics/` — one-off `_diag_*.py` investigation scripts
  moved out of `tests/validation/` on 2026-04-23 (Remax/MMCDF/ResNet-18
  parity debugging; the bugs they chased are all fixed).
- `_archive/workspace/` — old training scripts, `run_logs_*` directories,
  top-level figures/JSONs, `tagi_monitor/`, `figures/`.

To recover something: copy the file(s) back into place and re-add the export
in `__init__.py`.

---

## 8. Commit Conventions

Lightweight, Conventional-Commits-inspired. The point is filterable history,
not ceremony — when in doubt prefer a clear subject over the perfect prefix.

**Subject line** — `<type>: <imperative summary>`, ≤72 chars, sentence case.

| Type | Use for |
|---|---|
| `feat` | New layer, kernel, example, public API |
| `fix` | Bug fix that doesn't change scope or interface |
| `perf` | Speed/memory change with no semantic difference |
| `parity` | cuTAGI parity work (tolerance fix, reference alignment, MixtureReLU-style maths). Project-specific because parity is load-bearing here |
| `refactor` | Rename / restructure without behaviour change |
| `docs` | PLAN.md, README, docstrings, comments |
| `test` | Test-only change (add/remove/repair). Test-supporting fixtures count |
| `chore` | Version bump, dep bump, archive moves, formatter runs |

**Body** (optional but encouraged when the change isn't self-explanatory):

- Blank line after subject, wrap at 72.
- Explain **why**, not what — the diff already shows what.
- Note any non-obvious choices (e.g. "kept allow_tf32=False because TF32
  breaks the 1e-4 cuTAGI variance tolerance").
- For `parity` commits, say which cuTAGI source file/branch the fix targets.

**Footer**:

- Reference issues with `Refs: #N` or `Fixes: #N` if any exist.
- No `Co-Authored-By:` lines for AI-assisted commits — assume AI assistance
  is the default and skip the noise.

**Versioning**:

- Bump `pyproject.toml` version on user-visible API changes (new public
  layer/export, removed export, changed signature). Pure-internal perf or
  refactor doesn't need a bump.
- Bump commit gets its own `chore: bump version 0.X.Y → 0.X.Y+1` and is
  tagged `git tag v0.X.Y && git push --tags`.
- Update both `pyproject.toml` and `triton_tagi/__init__.py` `__version__`
  in the same commit; they must agree.

**Examples**:

```
feat: add MultiheadAttentionV2 with separate Q/K/V projections

Mirrors cuTAGI's attention.cpp on feat/attn-debug. Bayesian Q·Kᵀ uses
the fused bmm_tagi_var kernel; backward fuses the four δQ/δK/δV/δscore
reductions through bmm_shared_{left,right}.
```

```
parity: fix Remax to use MixtureReLU + log-normal cov path

cuTAGI computes Remax via MixtureReLU moments and log-normal identities,
not the Softplus + Taylor approximation we had. Variance from the old
path was off by ~5e-4 which fails test_remax_parity at atol=1e-4.

Refs: cuTAGI src/output_layer_cpu.cpp::Remax
```

```
perf: drop @triton.autotune from kernels/attention.py

Autotune adds ~50-100µs of Python dispatch per call even on cache hits,
which dominated reverse_predictor's small attention shapes (S=8, D=32).
Replaced with a shape-adaptive _pick_blocks heuristic. Per-call 17-24%
faster; MHA forward 5.6% faster end-to-end. All 18 attention validation
tests pass.
```

---

## 9. Scope Changes

Last touched **2026-04-23**: documented `kernels/attention.py` and
`bench_attention.py`; archived `_diag_*.py` investigation scripts to
`_archive/diagnostics/`; bumped version to 0.2.0; recorded the
`@triton.autotune` dispatch-overhead finding.

The plan was last *rewritten* on **2026-04-22** to add self-attention
(Embedding, PositionalEncoding, MultiheadAttentionV2, RMSNorm) and the
`reverse_predictor` example. The previous rewrite was on **2026-04-19** to
pare the library back to a minimal cuTAGI-parity core. Prior plans (pre-2026-04-19)
included Phase 5+ autograd, Phase 5.5 cold-posterior / BatchNorm audit, and
ImageNet / time-series / autoencoder examples. Those sections were removed.
If any of that work resumes, restore from git history and re-add here
explicitly.
