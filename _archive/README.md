# _archive/

Code, tests, and workspace artifacts that were removed from the main library
when triton-tagi was pared back to a minimal cuTAGI-parity core (2026-04-19).

**Nothing here is deleted.** To restore something, copy the file back to its
original location and re-add its export in the corresponding `__init__.py`.

## Layout

```
_archive/
├── triton_tagi/
│   ├── layers/
│   │   ├── bernoulli.py
│   │   ├── convtranspose2d.py
│   │   ├── frn.py
│   │   ├── frn_resblock.py
│   │   ├── leaky_relu.py
│   │   ├── silu.py
│   │   ├── tlu.py
│   │   ├── shared_var_batchnorm2d.py
│   │   ├── shared_var_conv2d.py
│   │   ├── shared_var_linear.py
│   │   └── shared_var_resblock.py
│   ├── update/
│   │   └── shared_var_parameters.py
│   ├── auto_tune.py         # gain / σ_v grid search
│   ├── inference_init.py    # inference-aware weight init research
│   ├── init.py              # reinit_net, init_residual_aware
│   ├── momentum.py          # StateSpaceMomentum
│   ├── monitor.py           # TAGIMonitor diagnostics + sweep_* helpers
│   ├── nadam_optimizer.py   # NadamTAGI
│   └── optimizer.py         # AdamTAGI
├── tests/
│   ├── unit/                # tests for the archived modules
│   └── validation/          # validation tests for archived layers
└── workspace/
    ├── scripts/             # research / paper-driving scripts
    ├── run_logs_*           # historical training logs
    ├── figures/             # inference-init figures
    └── tagi_monitor/        # (empty)
```

## Why this exists

The previous version of the library had drifted toward a research monorepo:
multiple optimizers, several normalization variants, shared-variance sibling
layers for every learnable layer, auto-tuning utilities, and 40+ ad-hoc scripts
at the repo root. The current minimal surface (see `../PLAN.md`) is 11 layers
plus `Sequential`, `RunDir`, and HRC softmax.

If you are restoring something, also restore:

- its export in `triton_tagi/__init__.py` (and `layers/__init__.py` if it is a layer)
- its tests (move back from `_archive/tests/`)
- any `Sequential`-side plumbing (e.g. `FRNResBlock` needed special handling in `network.py`)
