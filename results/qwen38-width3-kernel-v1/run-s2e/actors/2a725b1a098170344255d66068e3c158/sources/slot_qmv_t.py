"""031 tunable slot-shared QMV: same pinned per-slot arithmetic as slot_qmv (030), with launch geometry
(ROWS output rows per simdgroup, SIMDS simdgroups per threadgroup) as template parameters. Geometry
changes scheduling only; per-output K partition and summation order are those of stock qmv.
Admission is per (shape, slots, geometry) by a tuner requiring bitwise equality to stock singleton."""
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from slot_qmv import shape_plan, _array_meta

GEOMETRIES_RS = ((4, 2), (1, 2), (2, 2), (2, 4), (1, 4), (1, 8), (2, 1), (4, 1), (4, 4), (8, 1), (8, 2), (1, 16), (2, 8))


@lru_cache(maxsize=1)
def _kernel():
    mx = import_module("mlx.core")
    here = Path(__file__).resolve().parent
    return mx, mx.fast.metal_kernel(
        name="slot_shared_affine_qmv_t",
        input_names=["x", "packed_weights", "scales", "affine_bias"],
        output_names=["y"],
        source=(here / "slot_qmv_body_t.metal").read_text(),
        header=(here / "slot_qmv_header_t.metal").read_text(),
        ensure_row_contiguous=True,
    )


def slot_qmv_t(x, packed_weights, scales, affine_bias, *, bits, group_size=64, rows=4, simds=2, mode=0):
    plan = shape_plan(*(_array_meta(a) for a in (x, packed_weights, scales, affine_bias)), bits=bits, group_size=group_size)
    if (rows, simds) not in GEOMETRIES_RS or plan.n % (rows * simds):
        raise ValueError("unadmitted geometry")
    if mode not in (0, 1, 2):
        raise ValueError("unadmitted mode")
    mx, kernel = _kernel()
    return kernel(
        inputs=[x, packed_weights, scales, affine_bias],
        template=[("T", x.dtype), ("BITS", bits), ("GROUP_SIZE", group_size), ("SLOTS", plan.slots),
                  ("K", plan.k), ("N", plan.n), ("FAST", plan.fast), ("ROWS", rows), ("SIMDS", simds), ("MODE", int(mode))],
        grid=(32, plan.n // rows, 1),
        threadgroup=(32, simds, 1),
        output_shapes=[plan.output_shape],
        output_dtypes=[x.dtype],
    )[0]

GEOMETRIES = tuple((r, s, m) for m in (0, 1, 2) for r, s in GEOMETRIES_RS)
