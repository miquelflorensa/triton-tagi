# Benchmark Results: triton-tagi vs cuTAGI

**GPU:** NVIDIA GeForce RTX 4070 Ti SUPER  
**Metric:** median wall-clock time over 50 runs (ms), 10 warmup iterations  
**Step:** forward + backward + update  

---

### Linear

_Standalone layer: Linear(512, 512). Triton-tagi times the layer directly; cuTAGI wraps it in a 1-layer Sequential._

| Batch | triton-tagi (ms) | cuTAGI (ms) | Speedup |
|------:|----------------:|------------:|--------:|
|     1 |           0.590 |       0.258 |   0.44× |
|    16 |           0.625 |       0.739 |   1.18× |
|    32 |           0.610 |       1.218 |   2.00× |
|    64 |           0.610 |       2.274 |   3.73× |
|   256 |           0.608 |       8.053 |  13.25× |
|  1024 |           0.643 |      45.019 |  70.01× |

### Conv2D network

_Full network: Conv2D(32,32,3,pad=1,16×16) → ReLU → Flatten → Linear(8192,64). Conv2D standalone backward crashes in pytagi, so both sides time the full network._

| Batch | triton-tagi (ms) | cuTAGI (ms) | Speedup |
|------:|----------------:|------------:|--------:|
|     1 |           1.805 |       1.092 |   0.61× |
|    16 |           1.809 |       2.198 |   1.22× |
|    32 |           1.805 |       3.198 |   1.77× |
|    64 |           1.818 |       5.162 |   2.84× |
|   256 |           2.942 |      19.155 |   6.51× |
|  1024 |          10.986 |     106.405 |   9.69× |

### BatchNorm2D network

_Full network: Conv2D(32,32,3,pad=1,16×16) → BN(32) → ReLU → Flatten → Linear(8192,64). pytagi BatchNorm2d requires a preceding Conv2d, so both sides time the full network._

| Batch | triton-tagi (ms) | cuTAGI (ms) | Speedup |
|------:|----------------:|------------:|--------:|
|     1 |           2.499 |       1.214 |   0.49× |
|    16 |           2.520 |       2.462 |   0.98× |
|    32 |           2.511 |       3.743 |   1.49× |
|    64 |           2.526 |       5.799 |   2.30× |
|   256 |           3.402 |      20.535 |   6.04× |
|  1024 |          12.500 |     108.505 |   8.68× |

---

## Phase 3 Analysis

### Profiling (batch=256, `torch.profiler`, 20 steps after 10 warmup)

**Linear(512→512) — top bottlenecks**

| Op | CUDA ms | CUDA% |
|---|---:|---:|
| `_fused_backward_delta_kernel` | 1.01 | 25.6% |
| `_fused_var_forward_kernel` | 0.66 | 16.8% |
| `aten::mm` (cuBLAS matmul) | 0.64 | 16.3% |

**Conv2D network — top bottlenecks**

| Op | CUDA ms | CUDA% |
|---|---:|---:|
| `_col2im_kernel` | 13.8 | 15.9% |
| `_fused_var_forward_kernel` | 13.4 | 15.5% |
| `_fused_backward_delta_kernel` | 13.1 | 15.2% |
| `aten::mm` (cuBLAS) | 11.1 | 12.8% |
| `_im2col_kernel` | 6.7 | 7.7% |
| `aten::pow` (squaring for grad) | 3.9 | 4.5% |

### Optimization attempts

**1. Autotuning `_fused_var_forward_kernel` and `_fused_backward_delta_kernel` — APPLIED**

Added `@triton.autotune` with 11 tile configs to both kernels (BLOCK_M ∈ {16…128}, BLOCK_N ∈ {16…128}, BLOCK_K ∈ {16…128}). Autotune caches the best config per `(M, N, K)` shape after the first encounter.

Result: ~11–13% wall-clock improvement for Conv2D/BN networks at batch ≥ 256. Linear shows minimal gain because it is already dispatch-overhead bound at this size. No numerical regression (158 tests pass).

**2. Fused weight gradient kernel (`tl.trans` in `tl.dot`) — ATTEMPTED, REVERTED**

The `aten::mm` calls for `patches_ma.T @ dmz` and `(patches_ma²).T @ dSz` appear in the top-4.  A custom Triton kernel computing the transposed matmul was implemented, but measured 8× slower than cuBLAS SGEMM for the relevant `(K, NL)^T × (NL, C_out)` shapes (Triton cannot match cuBLAS for A^T@B patterns). Reverted.

**3. `F.fold` as col2im replacement — ATTEMPTED, REVERTED**

`_col2im_kernel` accounts for 15.9% of Conv2D CUDA time. PyTorch's `F.fold` (backed by cuDNN) was evaluated as a drop-in replacement. Despite avoiding the explicit scatter loop, `F.fold` internally forces a `aten::copy_` even on contiguous inputs (2ms overhead at batch=256), making net performance worse. Reverted to the custom Triton scatter kernel.

### Summary

The primary lever available in pure Python/Triton is kernel tile autotuning; more significant gains for the conv path would require restructuring the im2col/col2im memory access pattern (gather vs. scatter) or using cuDNN convolution directly, neither of which is consistent with the library's Triton-native goal.
