"""Validated single-token fused affine gate/up Metal kernel.

This is an inference prototype, not a claimed improvement over MLX. The caller
must benchmark the stock path before selecting this backend for a shape.
"""

from functools import lru_cache
from importlib.resources import files

import mlx.core as mx
import mlx.nn as nn

SUPPORTED_BITS = (3, 4, 6)
SUPPORTED_GROUPS = (32, 64, 128)


@lru_cache(maxsize=2)
def _kernel(variant):
    if variant not in ("scalar", "packed"):
        raise ValueError("variant must be scalar or packed")
    filename = "gate_up_packed.metal" if variant == "packed" else "gate_up.metal"
    source = files("paretoquant").joinpath(f"kernels/{filename}").read_text()
    return mx.fast.metal_kernel(
        name=f"paretoquant_gate_up_{variant}",
        input_names=["x", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"],
        output_names=["out"],
        source=source,
    )


def _validate(x, gate, up, bits, group_size, rows_per_group):
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if group_size not in SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUPS}")
    if rows_per_group not in (1, 2, 4, 8):
        raise ValueError("rows_per_group must be 1, 2, 4, or 8")
    if x.ndim < 1 or x.size != x.shape[-1] or x.size == 0:
        raise ValueError("fused decode supports a single token only")
    if x.dtype not in (mx.float16, mx.float32):
        raise ValueError("input dtype must be float16 or float32")
    if len(gate) != 3 or len(up) != 3:
        raise ValueError("affine packed weights require weight, scales, and biases")
    k = x.shape[-1]
    if k % group_size:
        raise ValueError("input shape must be divisible by group_size")
    if gate[0].ndim != 2 or up[0].ndim != 2:
        raise ValueError("packed weights must have a two-dimensional shape")
    n = gate[0].shape[0]
    if n == 0:
        raise ValueError("output shape cannot be empty")
    for q in (gate, up):
        if q[0].shape != (n, k * bits // 32):
            raise ValueError("packed weight shape does not match input, precision, or gate/up")
        if q[0].dtype != mx.uint32:
            raise ValueError("packed weights must have uint32 dtype")
        for arr in q[1:]:
            if arr.shape != (n, k // group_size) or arr.dtype != x.dtype:
                raise ValueError("scale/bias shape and dtype must match input and packed weights")
    return n, k


def fused_gate_up(x, gate, up, *, bits, group_size=64, rows_per_group=4, variant="packed"):
    """Compute SiLU(gate @ x) * (up @ x) without expanding packed weights."""
    n, k = _validate(x, gate, up, bits, group_size, rows_per_group)
    output = _kernel(variant)(
        inputs=[x.reshape(-1), *gate, *up],
        template=[("T", x.dtype), ("BITS", bits), ("K", k), ("N", n), ("GROUP", group_size)],
        grid=(32, n, 1),
        threadgroup=(32, rows_per_group, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]
    return output.reshape((*x.shape[:-1], n))


def stock_gate_up(x, gate, up, *, bits, group_size=64):
    g = mx.quantized_matmul(x, *gate, transpose=True, group_size=group_size, bits=bits)
    u = mx.quantized_matmul(x, *up, transpose=True, group_size=group_size, bits=bits)
    return nn.silu(g) * u


# Explicit array arguments keep weights dynamic across models and saved plans.
# Compile both paths so isolated timings do not grant fusion an unfair dispatch advantage.
compiled_fused_gate_up = mx.compile(fused_gate_up)
compiled_stock_gate_up = mx.compile(stock_gate_up)
