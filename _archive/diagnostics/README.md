# Archived diagnostic scripts

One-off investigation scripts moved here on 2026-04-23 from
`tests/validation/`. None of these are pytest tests — they are
ad-hoc reproductions used while resolving specific parity bugs.
The bugs they were chasing have all been fixed; the scripts are
kept for reference only.

| Script | Investigation |
|---|---|
| `_diag_conv_*.py` | Conv2D layout / im2col probe |
| `_diag_cpu_vs_gpu.py` | Conv2D CPU vs GPU divergence |
| `_diag_init_from_cutagi.py` | Init parity via cuTAGI state-dict export |
| `_diag_peel.py` | Layer-by-layer activation diff |
| `_diag_pre_remax.py`, `_diag_remax_parity.py` | Remax parity (resolved: MixtureReLU + log-normal cov path) |
| `_diag_resnet18_*.py` | ResNet-18 stuck-at-87% debugging |
| `_diag_sd_probe.py`, `_diag_verify.py` | State-dict transposition checks |

To re-run, copy back into `tests/validation/` and rename without the
underscore prefix to make pytest collect it (or run directly with
`python <script>.py`).
