"""CPU-only admission and source-contract tests; never import MLX/NumPy."""

import ast
import builtins
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("isolated_slot_qmv", HERE / "slot_qmv.py")
assert spec is not None and spec.loader is not None
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


def test_valid_rank_two_metadata_plan():
    assert callable(getattr(candidate, "shape_plan", None))
    meta = candidate.ArrayMeta
    plan = candidate.shape_plan(
        meta((2, 512), "float16"),
        meta((8, 64), "uint32"),
        meta((8, 8), "float16"),
        meta((8, 8), "float16"),
        bits=4,
    )
    assert (plan.slots, plan.k, plan.n) == (2, 512, 8)
    assert plan.output_shape == (2, 8)
    assert plan.output_dtype == "float16"


class IntSubclass(int):
    pass


def valid_args(k=512, n=8, slots=2, bits=4, group_size=64, dtype="float16"):
    meta = candidate.ArrayMeta
    return (
        meta((slots, k), dtype),
        meta((n, k * bits // 32), "uint32"),
        meta((n, k // group_size), dtype),
        meta((n, k // group_size), dtype),
    )


@pytest.mark.parametrize(
    "field,value",
    [("bits", v) for v in (True, False, 3.0, 4.0, "4", None, 2, 5, 6, 8, IntSubclass(4))]
    + [
        ("group_size", v) for v in (True, False, 64.0, "64", None, 0, -64, 16, 256, IntSubclass(64))
    ],
)
def test_rejects_noncanonical_quantization_configuration(field, value):
    with pytest.raises(ValueError):
        candidate.shape_plan(
            *valid_args(), **{field: value, **({"bits": 4} if field != "bits" else {})}
        )


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (512,),
        (0, 512),
        (10, 512),
        (-1, 512),
        (2, 0),
        (2, -512),
        (2, 511),
        (True, 512),
        (2, True),
        (2.0, 512),
        (2, 512.0),
        (IntSubclass(2), 512),
        (2, IntSubclass(512)),
        (2, 2, 512),
        (1, 1, 2, 512),
        [2, 512],
        None,
    ],
)
def test_rejects_unadmitted_input_shape(shape):
    args = list(valid_args())
    args[0] = candidate.ArrayMeta(shape, "float16")
    with pytest.raises(ValueError):
        candidate.shape_plan(*args, bits=4)


@pytest.mark.parametrize("slots", range(1, 10))
def test_rank_three_preserves_singleton_batch_output(slots):
    args = list(valid_args(slots=slots))
    args[0] = candidate.ArrayMeta((1, slots, 512), "float16")
    plan = candidate.shape_plan(*args, bits=4)
    assert plan.slots == slots
    assert plan.output_shape == (1, slots, 8)


@pytest.mark.parametrize(
    "index,shape",
    [
        (1, (7, 64)),
        (1, (9, 64)),
        (1, (0, 64)),
        (1, (4, 64)),
        (1, (8, 63)),
        (1, (8, 65)),
        (1, (1, 8, 64)),
        (1, (64,)),
        (1, (8.0, 64)),
        (1, (8, True)),
        (1, [8, 64]),
        (1, None),
        (2, (8, 7)),
        (2, (1, 8, 8)),
        (2, (7, 8)),
        (2, (8, 8.0)),
        (3, (8, 9)),
        (3, (1, 8, 8)),
        (3, (8,)),
        (3, (True, 8)),
    ],
)
def test_rejects_packed_or_affine_shape_mismatch(index, shape):
    args = list(valid_args())
    args[index] = candidate.ArrayMeta(shape, args[index].dtype)
    with pytest.raises(ValueError):
        candidate.shape_plan(*args, bits=4)


@pytest.mark.parametrize(
    "index,dtype",
    [(0, d) for d in ("bfloat16", "float64", "int32", "uint32", "half", None)]
    + [(1, d) for d in ("int32", "uint8", "uint16", "float16", None)]
    + [(i, d) for i in (2, 3) for d in ("float32", "bfloat16", "float64", "int32", None)],
)
def test_rejects_unsupported_or_mismatched_dtype(index, dtype):
    args = list(valid_args())
    args[index] = candidate.ArrayMeta(args[index].shape, dtype)
    with pytest.raises(ValueError):
        candidate.shape_plan(*args, bits=4)


@pytest.mark.parametrize("bits", (3, 4))
@pytest.mark.parametrize("group_size", (32, 64, 128))
@pytest.mark.parametrize(
    "k,fast,block,final",
    [
        (128, False, 256, 128),
        (256, False, 256, 256),
        (384, False, 256, 128),
        (512, True, 512, 512),
        (768, False, 256, 256),
        (896, False, 256, 128),
        (4864, False, 256, 256),
        (5120, True, 512, 512),
        (27648, True, 512, 512),
    ],
)
def test_plan_matches_single_token_dispatch_and_safe_final(k, fast, block, final, bits, group_size):
    plan = candidate.shape_plan(
        *valid_args(k=k, bits=bits, group_size=group_size, dtype="float32"),
        bits=bits,
        group_size=group_size,
    )
    assert hasattr(plan, "fast")
    assert plan.fast is fast
    assert plan.block_size == block
    assert plan.packs_per_thread == (2 if fast else 1)
    assert plan.final_values == final
    assert plan.safe_final is (not fast)
    assert plan.grid == (32, 2, 1)
    assert plan.threadgroup == (32, 2, 1)
    assert plan.output_dtype == "float32"


@pytest.mark.parametrize(
    "k,n",
    [
        (1 << 31, 8),
        (512, 1 << 31),
        (1 << 20, 8192),
        (512, 1 << 26),
    ],
)
def test_rejects_signed_integer_address_overflow(k, n):
    with pytest.raises(ValueError):
        candidate.shape_plan(*valid_args(k=k, n=n), bits=4)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", [None, object(), ((2, 512), "float16")])
def test_metadata_admission_rejects_untyped_values(index, value):
    args = list(valid_args())
    args[index] = value
    with pytest.raises(ValueError):
        candidate.shape_plan(*args, bits=4)


class FakeArray:
    """Host-only stand-in: tests dispatch, never arithmetic or GPU parity."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


@pytest.mark.parametrize(
    "change",
    [
        {"bits": True},
        {"bits": 4.0},
        {"bits": 8},
        {"group_size": True},
        {"group_size": 64.0},
        {"group_size": 16},
        {"xshape": (2, 2, 512)},
        {"xshape": (10, 512)},
        {"xshape": (True, 512)},
        {"xdtype": "bfloat16"},
        {"wshape": (7, 64)},
        {"wdtype": "int32"},
        {"sdtype": "float32"},
        {"bshape": (8, 9)},
    ],
)
def test_wrapper_rejects_before_runtime_getter(change, monkeypatch):
    assert callable(getattr(candidate, "slot_qmv", None))

    def forbidden():
        pytest.fail("runtime getter reached for an inadmissible input")

    monkeypatch.setattr(candidate, "_kernel", forbidden)
    x = FakeArray(change.get("xshape", (2, 512)), change.get("xdtype", "float16"))
    w = FakeArray(change.get("wshape", (8, 64)), change.get("wdtype", "uint32"))
    s = FakeArray((8, 8), change.get("sdtype", "float16"))
    b = FakeArray(change.get("bshape", (8, 8)), "float16")
    with pytest.raises(ValueError):
        candidate.slot_qmv(
            x, w, s, b, bits=change.get("bits", 4), group_size=change.get("group_size", 64)
        )


@pytest.mark.parametrize("rank", (2, 3))
@pytest.mark.parametrize("slots", range(1, 10))
@pytest.mark.parametrize("bits,k", ((3, 896), (4, 512)))
def test_wrapper_launch_contract_without_real_runtime(rank, slots, bits, k, monkeypatch):
    args = [FakeArray(a.shape, a.dtype) for a in valid_args(k=k, slots=slots, bits=bits)]
    if rank == 3:
        args[0].shape = (1, slots, k)
    calls = []
    output = FakeArray(args[0].shape[:-1] + (8,), "float16")

    def launch(**kwargs):
        calls.append(kwargs)
        return [output]

    runtime = SimpleNamespace(
        array=FakeArray, float16="float16", float32="float32", uint32="uint32"
    )
    monkeypatch.setattr(candidate, "_kernel", lambda: (runtime, launch))
    result = candidate.slot_qmv(*args, bits=bits)
    assert result is output
    assert len(calls) == 1
    call = calls[0]
    assert call["inputs"] == args
    assert call["output_shapes"] == [output.shape]
    assert call["output_dtypes"] == ["float16"]
    assert call["grid"] == (32, 2, 1)
    assert call["threadgroup"] == (32, 2, 1)
    assert dict(call["template"]) == {
        "T": "float16",
        "BITS": bits,
        "GROUP_SIZE": 64,
        "SLOTS": slots,
        "K": k,
        "N": 8,
        "FAST": k % 512 == 0,
    }


@pytest.mark.parametrize("index", range(4))
def test_wrapper_rejects_nonarray_after_metadata_before_launch(index, monkeypatch):
    args = [FakeArray(a.shape, a.dtype) for a in valid_args()]
    args[index] = candidate.ArrayMeta(args[index].shape, args[index].dtype)

    def forbidden(**kwargs):
        pytest.fail("nonarray reached launch")

    runtime = SimpleNamespace(
        array=FakeArray, float16="float16", float32="float32", uint32="uint32"
    )
    monkeypatch.setattr(candidate, "_kernel", lambda: (runtime, forbidden))
    with pytest.raises(ValueError):
        candidate.slot_qmv(*args, bits=4)


def test_metal_helpers_are_exact_pinned_address_space_adaptation():
    path = HERE / "slot_qmv_header.metal"
    assert path.is_file(), "derived Metal helpers are missing"
    header = path.read_text()
    helpers = header.split("// BEGIN PINNED HELPERS\n", 1)[1].split("// END PINNED HELPERS", 1)[0]
    assert (
        hashlib.sha256(helpers.encode()).hexdigest()
        == "66cfc54321acdc2a2da5e5ba4f99d430ae25430e4fdee0d2356f7899edc8bb6b"
    )
    assert "sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];" in helpers
    assert "return scale * accum + sum * bias;" in helpers
    assert "const thread uint16_t* ws = (const thread uint16_t*)w;" in helpers


def test_fast_source_stages_weight_bytes_outside_slot_dot_loop():
    header = (HERE / "slot_qmv_header.metal").read_text()
    assert "inline void slot_qmv_block(" in header, "slot-shared block is missing"
    block = header.split("inline void slot_qmv_block(", 1)[1].split("// END SLOT BLOCK", 1)[0]
    assert "thread uint32_t weight_storage[(weight_bytes + 3) / 4];" in block
    assert "thread uint8_t* staged = (thread uint8_t*)weight_storage;" in block
    assert "staged[byte] = wl[byte];" in block
    stage = block.index("staged[byte] = wl[byte];")
    dot_loop = block.index("for (int slot = 0; slot < SLOTS; slot++)", stage)
    dot = block.index("slot_qdot<float, values_per_thread, BITS>", dot_loop)
    assert stage < dot_loop < dot
    assert "slot_load_vector<T, float, values_per_thread, BITS>" in block
    assert "result[slot][row] +=" in block
    kernel = header.split("inline void slot_qmv_impl(", 1)[1]
    assert "thread float result[SLOTS][4] = {0};" in kernel
    assert "k < K; k += block_size" in kernel
    assert "result[slot][row] = simd_sum(result[slot][row]);" in kernel
    assert "y[slot * N + out_row + row] = static_cast<T>(result[slot][row]);" in kernel
    assert "const int out_row = int(group.y) * 8 + int(simd_gid) * 4;" in kernel


def test_slow_source_preserves_safe_last_step_even_for_full_final_block():
    header = (HERE / "slot_qmv_header.metal").read_text()
    kernel = header.split("inline void slot_qmv_impl(", 1)[1]
    assert "for (; k < K - block_size; k += block_size)" in kernel, (
        "stock slow last-step schedule missing"
    )
    assert "const int remaining = clamp(K - k - int(simd_lid) * values_per_thread," in kernel
    assert "if (remaining > 0)" in kernel
    assert "packs_per_thread, true>" in kernel
    assert "slot_load_vector_safe<T, float, values_per_thread, BITS>" in header
    assert "slot_qdot_safe<float, values_per_thread, BITS>" in header
    assert "const int loaded_bytes = SAFE ? remaining * BITS / 8 : weight_bytes;" in header
    assert "byte < loaded_bytes" in header
    assert '"slow source pending"' not in kernel


def test_metal_entry_maps_lane_and_simd_to_one_eight_row_group():
    path = HERE / "slot_qmv_body.metal"
    assert path.is_file(), "Metal entry body missing"
    body = path.read_text()
    assert "slot_qmv_impl<T, GROUP_SIZE, BITS, SLOTS, K, N, FAST>" in body
    assert "packed_weights, scales, affine_bias, x, y" in body
    assert "threadgroup_position_in_grid" in body
    assert "thread_position_in_threadgroup.y" in body
    assert "thread_position_in_threadgroup.x" in body
    assert "simdgroup_index_in_threadgroup" not in body


def test_runtime_getter_is_lazy_cached_and_builds_contiguous_kernel(monkeypatch):
    assert callable(getattr(candidate, "import_module", None)), "lazy runtime loader missing"
    imports = []
    builds = []
    sentinel = object()

    def build(**kwargs):
        builds.append(kwargs)
        return sentinel

    fake_mx = SimpleNamespace(fast=SimpleNamespace(metal_kernel=build))

    def load(name):
        imports.append(name)
        assert name == "mlx.core"
        return fake_mx

    monkeypatch.setattr(candidate, "import_module", load)
    candidate._kernel.cache_clear()
    try:
        assert not imports
        assert candidate._kernel() == (fake_mx, sentinel)
        assert candidate._kernel() == (fake_mx, sentinel)
        assert imports == ["mlx.core"]
        assert len(builds) == 1
        build_args = builds[0]
        assert build_args["name"] == "slot_shared_affine_qmv"
        assert build_args["input_names"] == ["x", "packed_weights", "scales", "affine_bias"]
        assert build_args["output_names"] == ["y"]
        assert build_args["ensure_row_contiguous"] is True
        assert build_args["header"] == (HERE / "slot_qmv_header.metal").read_text()
        assert build_args["source"] == (HERE / "slot_qmv_body.metal").read_text()
    finally:
        candidate._kernel.cache_clear()


def test_module_import_does_not_load_array_runtimes(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "numpy", "mlx_lm"}:
            pytest.fail("array runtime imported during CPU-only module import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    other_spec = importlib.util.spec_from_file_location("cpu_only_slot_qmv", HERE / "slot_qmv.py")
    assert other_spec is not None and other_spec.loader is not None
    other = importlib.util.module_from_spec(other_spec)
    monkeypatch.setitem(sys.modules, other_spec.name, other)
    other_spec.loader.exec_module(other)
    assert other._kernel.cache_info().currsize == 0
    tree = ast.parse((HERE / "slot_qmv.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] not in {"mlx", "numpy", "mlx_lm"} for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] not in {"mlx", "numpy", "mlx_lm"}


@pytest.mark.parametrize(
    "kwargs",
    [{"transpose": False}, {"mode": "affine"}, {"output_bias": None}, {"activation": "silu"}],
)
def test_no_hidden_transpose_mode_output_bias_or_activation_options(kwargs):
    with pytest.raises(TypeError):
        candidate.slot_qmv(*valid_args(), bits=4, **kwargs)


@pytest.mark.parametrize("bits", (3, 4))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize(
    "k,n",
    [
        (5120, 5120),
        (5120, 27648),
        (27648, 5120),
        (5120, 152064),
        (896, 896),
        (896, 4864),
        (4864, 896),
        (896, 151936),
    ],
)
def test_checkpoint_projection_and_head_metadata(bits, dtype, k, n):
    plan = candidate.shape_plan(*valid_args(k=k, n=n, slots=9, bits=bits, dtype=dtype), bits=bits)
    assert plan.output_shape == (9, n)
    assert plan.grid == (32, n // 4, 1)
    assert plan.fast is (k % 512 == 0)


@pytest.mark.parametrize("bits", (3, 4))
@pytest.mark.parametrize("group_size", (32, 64, 128))
@pytest.mark.parametrize("k", (128, 256, 384, 512, 768, 896, 4864, 5120, 27648))
def test_lane_schedule_covers_bytes_and_affine_groups_exactly(bits, group_size, k):
    plan = candidate.shape_plan(
        *valid_args(k=k, bits=bits, group_size=group_size), bits=bits, group_size=group_size
    )
    values_per_lane = plan.packs_per_thread * 8
    seen = []
    staged = []
    for start in range(0, k, plan.block_size):
        for lane in range(32):
            count = min(max(k - start - lane * values_per_lane, 0), values_per_lane)
            if not count:
                continue
            assert count % 8 == 0
            lo = start + lane * values_per_lane
            seen.extend(range(lo, lo + count))
            staged.extend(range(lo * bits // 8, (lo + count) * bits // 8))
            group_index = start // group_size + lane // (group_size // values_per_lane)
            assert group_index == lo // group_size
            assert group_index == (lo + count - 1) // group_size
    assert seen == list(range(k))
    assert staged == list(range(k * bits // 8))


def test_metal_template_guards_match_admitted_plan():
    header = (HERE / "slot_qmv_header.metal").read_text()
    kernel = header.split("inline void slot_qmv_impl(", 1)[1]
    assert "GROUP_SIZE == 32 || GROUP_SIZE == 64 || GROUP_SIZE == 128" in kernel
    assert "K > 0 && K % GROUP_SIZE == 0" in kernel
    assert "FAST == (K % 512 == 0)" in kernel


def test_upstream_license_is_exact_pinned_copy():
    assert hashlib.sha256((HERE / "LICENSE").read_bytes()).hexdigest() == (
        "ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01"
    )
