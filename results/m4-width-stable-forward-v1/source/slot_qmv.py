"""Isolated, unverified slot-shared affine QMV experiment."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class ArrayMeta:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ShapePlan:
    slots: int
    k: int
    n: int
    output_shape: tuple[int, ...]
    output_dtype: str
    fast: bool
    packs_per_thread: int
    block_size: int
    final_values: int
    safe_final: bool
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]


def shape_plan(
    x: ArrayMeta,
    packed_weights: ArrayMeta,
    scales: ArrayMeta,
    affine_bias: ArrayMeta,
    *,
    bits: int,
    group_size: int = 64,
) -> ShapePlan:
    """Pure metadata admission: no runtime imports, evaluation, or allocations."""
    if any(type(meta) is not ArrayMeta for meta in (x, packed_weights, scales, affine_bias)):
        raise ValueError("shape_plan requires ArrayMeta inputs")
    if type(bits) is not int or bits not in (3, 4):
        raise ValueError("bits must be a built-in int in {3, 4}")
    if type(group_size) is not int or group_size not in (32, 64, 128):
        raise ValueError("group_size must be a built-in int in {32, 64, 128}")
    if (
        type(x.shape) is not tuple
        or len(x.shape) not in (2, 3)
        or any(type(d) is not int or d <= 0 for d in x.shape)
    ):
        raise ValueError("x must have positive built-in integer dimensions (T, K)")
    if len(x.shape) == 3 and x.shape[0] != 1:
        raise ValueError("only a singleton leading batch is admitted")
    slots, k = x.shape[-2:]
    if not 1 <= slots <= 9 or k % group_size:
        raise ValueError("require 1 <= T <= 9 and K divisible by group_size")
    for name, meta in (
        ("packed_weights", packed_weights),
        ("scales", scales),
        ("affine_bias", affine_bias),
    ):
        if (
            type(meta.shape) is not tuple
            or len(meta.shape) != 2
            or any(type(d) is not int or d <= 0 for d in meta.shape)
        ):
            raise ValueError(f"{name} must have two positive built-in integer dimensions")
    n = packed_weights.shape[0]
    if n < 8 or n % 8:
        raise ValueError("require N >= 8 and N divisible by 8")
    if packed_weights.shape != (n, k * bits // 32):
        raise ValueError("packed_weights must have shape (N, K * bits / 32)")
    if scales.shape != (n, k // group_size) or affine_bias.shape != scales.shape:
        raise ValueError("scales and affine_bias must have shape (N, K / group_size)")
    if type(x.dtype) is not str or x.dtype not in ("float16", "float32"):
        raise ValueError("x dtype must be float16 or float32")
    if type(packed_weights.dtype) is not str or packed_weights.dtype != "uint32":
        raise ValueError("packed_weights dtype must be uint32")
    if (
        type(scales.dtype) is not str
        or type(affine_bias.dtype) is not str
        or scales.dtype != x.dtype
        or affine_bias.dtype != x.dtype
    ):
        raise ValueError("scales and affine_bias dtype must exactly match x")
    if max(k * bits, n * k * bits // 8, n * (k // group_size), slots * k, slots * n) > 2**31 - 1:
        raise ValueError("signed 32-bit QMV indexing limit exceeded")
    fast = k % 512 == 0
    packs_per_thread = 2 if fast else 1
    block_size = 8 * packs_per_thread * 32
    return ShapePlan(
        slots,
        k,
        n,
        x.shape[:-1] + (n,),
        x.dtype,
        fast,
        packs_per_thread,
        block_size,
        (k - 1) % block_size + 1,
        not fast,
        (32, n // 4, 1),
        (32, 2, 1),
    )


def _array_meta(array):
    try:
        dtype = str(array.dtype)
        if dtype.startswith("mlx.core."):
            dtype = dtype[len("mlx.core.") :]
        return ArrayMeta(array.shape, dtype)
    except AttributeError as exc:
        raise ValueError("all inputs must expose shape and dtype") from exc


@lru_cache(maxsize=1)
def _kernel():
    """Only called after host admission; importing this module is CPU-safe."""
    mx = import_module("mlx.core")
    here = Path(__file__).resolve().parent
    kernel = mx.fast.metal_kernel(
        name="slot_shared_affine_qmv",
        input_names=["x", "packed_weights", "scales", "affine_bias"],
        output_names=["y"],
        source=(here / "slot_qmv_body.metal").read_text(),
        header=(here / "slot_qmv_header.metal").read_text(),
        ensure_row_contiguous=True,
    )
    return mx, kernel


def slot_qmv(x, packed_weights, scales, affine_bias, *, bits, group_size=64):
    """Unverified affine matmul only; no output-bias or activation fusion."""
    plan = shape_plan(
        *(_array_meta(a) for a in (x, packed_weights, scales, affine_bias)),
        bits=bits,
        group_size=group_size,
    )
    mx, kernel = _kernel()
    if any(not isinstance(a, mx.array) for a in (x, packed_weights, scales, affine_bias)):
        raise ValueError("slot_qmv requires MLX arrays")
    return kernel(
        inputs=[x, packed_weights, scales, affine_bias],
        template=[
            ("T", x.dtype),
            ("BITS", bits),
            ("GROUP_SIZE", group_size),
            ("SLOTS", plan.slots),
            ("K", plan.k),
            ("N", plan.n),
            ("FAST", plan.fast),
        ],
        grid=plan.grid,
        threadgroup=plan.threadgroup,
        output_shapes=[plan.output_shape],
        output_dtypes=[x.dtype],
    )[0]
