"""Low-level fused Triton kernels for TAGI operations."""

from .common import (
    triton_fused_var_forward,
    triton_fused_backward_delta,
)
