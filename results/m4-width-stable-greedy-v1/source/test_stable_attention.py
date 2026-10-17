"""CPU-only metadata and structural tests, not GPU/math/parity proof."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def block_runtime_imports(monkeypatch):
    forbidden = ("mlx", "mlx_lm", "numpy")
    assert not any(name.split(".")[0] in forbidden for name in sys.modules)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError(f"Forbidden CPU-test import: {name}")
        return original(name, *args, **kwargs)

    class BlockRuntimeFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise AssertionError(f"Forbidden CPU-test import: {fullname}")
            return None

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(sys, "meta_path", [BlockRuntimeFinder(), *sys.meta_path])
    yield
    assert not any(name.split(".")[0] in forbidden for name in sys.modules)


def load_component():
    path = Path(__file__).with_name("stable_attention.py")
    assert path.is_file(), "isolated attention component has not been implemented"
    spec = importlib.util.spec_from_file_location("isolated_stable_attention", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(**overrides):
    data = dict(
        query_shape=(1, 14, 5, 64),
        key_shape=(1, 2, 12, 64),
        value_shape=(1, 2, 12, 64),
        query_dtype="float16",
        key_dtype="float16",
        value_dtype="float16",
        cache=SimpleNamespace(offset=12),
        scale=0.125,
        mask="causal",
    )
    data.update(overrides)
    return data


def test_plan_uses_position_specific_causal_prefixes():
    component = load_component()
    assert component.plan_causal_prefixes(**metadata()) == (8, 9, 10, 11, 12)


class StructuralArray:
    """Slicing-only double: no array values or numerical operations exist."""

    def __init__(self, shape, dtype="float16", source=None, span=None):
        self.shape = shape
        self.dtype = dtype
        self.source = source
        self.span = span

    def __getitem__(self, index):
        assert index[0] is Ellipsis
        assert index[2] == slice(None)
        start, stop, step = index[1].indices(self.shape[-2])
        assert step == 1
        shape = self.shape[:-2] + (stop - start, self.shape[-1])
        return StructuralArray(shape, self.dtype, self, (start, stop))


class StructuralRuntime:
    """Unavoidable native-call double: structural proof only."""

    def __init__(self):
        self.fast = SimpleNamespace(scaled_dot_product_attention=self.sdpa)
        self.calls = []
        self.contiguous_inputs = []
        self.concatenations = []

    def contiguous(self, array):
        self.contiguous_inputs.append(array)
        return array

    def sdpa(self, q, k, v, **options):
        result = StructuralArray(q.shape, q.dtype)
        self.calls.append((q, k, v, options, result))
        return result

    def concatenate(self, arrays, axis):
        self.concatenations.append((arrays, axis))
        assert axis == -2
        shape = arrays[0].shape[:-2] + (sum(a.shape[-2] for a in arrays), arrays[0].shape[-1])
        return StructuralArray(shape, arrays[0].dtype)


class ReadOnlyCache:
    def __init__(self, offset):
        object.__setattr__(self, "offset", offset)

    def __setattr__(self, name, value):
        raise AssertionError("cache must not be mutated")

    def __getattr__(self, name):
        if name == "bits":
            raise AttributeError(name)
        raise AssertionError(f"cache storage/operations must not be accessed: {name}")


@pytest.mark.parametrize(
    "width,length,heads,kv_heads,dim",
    [
        (2, 2, 14, 2, 64),
        (5, 12, 14, 2, 64),
        (9, 19, 20, 4, 128),
    ],
)
def test_structural_single_query_calls_preserve_causal_prefix_and_cache(
    monkeypatch,
    width,
    length,
    heads,
    kv_heads,
    dim,
):
    component = load_component()
    assert callable(getattr(component, "stable_causal_attention", None)), "missing runtime wrapper"
    runtime = StructuralRuntime()
    monkeypatch.setattr(component, "_load_mx", lambda: runtime)
    q = StructuralArray((1, heads, width, dim))
    k = StructuralArray((1, kv_heads, length, dim))
    v = StructuralArray(k.shape)
    cache = ReadOnlyCache(length)
    result = component.stable_causal_attention(q, k, v, cache=cache, scale=0.125, mask="causal")
    assert result.shape == q.shape
    assert cache.offset == length
    assert len(runtime.calls) == width
    assert len(runtime.contiguous_inputs) == width
    for position, (sq, sk, sv, options, _) in enumerate(runtime.calls):
        end = length - width + position + 1
        assert sq.shape == (1, heads, 1, dim)
        assert sq.source is q and sq.span == (position, position + 1)
        assert sk.source is k and sk.span == (0, end)
        assert sv.source is v and sv.span == (0, end)
        assert sk.shape == sv.shape == (1, kv_heads, end, dim)
        assert runtime.contiguous_inputs[position] is sq
        assert options == dict(scale=0.125, mask=None, sinks=None)
    outputs, axis = runtime.concatenations[0]
    assert axis == -2
    assert outputs == [call[4] for call in runtime.calls]


@pytest.mark.parametrize("mask", [None, "causal"])
def test_width_one_is_original_native_call_without_transforms(monkeypatch, mask):
    component = load_component()
    runtime = StructuralRuntime()
    monkeypatch.setattr(component, "_load_mx", lambda: runtime)
    q = StructuralArray((1, 14, 1, 64))
    k = StructuralArray((1, 2, 13, 64))
    v = StructuralArray(k.shape)
    cache = ReadOnlyCache(13)
    result = component.stable_causal_attention(q, k, v, cache=cache, scale=1, mask=mask)
    assert len(runtime.calls) == 1
    sq, sk, sv, options, output = runtime.calls[0]
    assert sq is q and sk is k and sv is v and result is output
    assert options == dict(scale=1, mask=mask, sinks=None)
    assert runtime.contiguous_inputs == []
    assert runtime.concatenations == []
    assert cache.offset == 13


def assert_invalid_before_runtime(monkeypatch, overrides, message):
    component = load_component()
    data = metadata(**overrides)

    def forbidden_loader():
        pytest.fail("invalid metadata reached lazy MLX import")

    monkeypatch.setattr(component, "_load_mx", forbidden_loader)
    with pytest.raises(ValueError, match=message):
        component.plan_causal_prefixes(**data)
    q = StructuralArray(data.pop("query_shape"), data.pop("query_dtype"))
    k = StructuralArray(data.pop("key_shape"), data.pop("key_dtype"))
    v = StructuralArray(data.pop("value_shape"), data.pop("value_dtype"))
    with pytest.raises(ValueError, match=message):
        component.stable_causal_attention(q, k, v, **data)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(query_shape=(14, 5, 64)),
        dict(query_shape=(1, 14, 5, 64, 1)),
        dict(query_shape=None),
        dict(key_shape="bad"),
        dict(value_shape=(1, 2, 12)),
        dict(query_shape=(True, 14, 5, 64)),
        dict(key_shape=(1, True, 12, 64)),
        dict(value_shape=(1, 2, 12, False)),
        dict(query_shape=(1, 14.0, 5, 64)),
        dict(key_shape=(1, 2, 12.0, 64)),
        dict(query_shape=(1, 0, 5, 64)),
        dict(query_shape=(1, -1, 5, 64)),
        dict(key_shape=(1, 0, 12, 64)),
        dict(key_shape=(1, 2, 0, 64)),
        dict(query_shape=(1, 14, 5, 0)),
        dict(query_shape=(2, 14, 5, 64)),
        dict(key_shape=(2, 2, 12, 64)),
        dict(value_shape=(2, 2, 12, 64)),
        dict(query_shape=(1, 13, 5, 64)),
        dict(query_shape=(1, 14, 0, 64)),
        dict(query_shape=(1, 14, 10, 64)),
        dict(key_shape=(1, 2, 4, 64), value_shape=(1, 2, 4, 64)),
        dict(key_shape=(1, 2, 12, 128), value_shape=(1, 2, 12, 128)),
        dict(value_shape=(1, 2, 11, 64)),
        dict(value_shape=(1, 3, 12, 64)),
        dict(value_shape=(1, 2, 12, 128)),
    ],
)
def test_invalid_shapes_fail_before_lazy_import(monkeypatch, overrides):
    assert_invalid_before_runtime(monkeypatch, overrides, "shape")


@pytest.mark.parametrize(
    "overrides",
    [
        dict(query_dtype="float64"),
        dict(key_dtype="bfloat16"),
        dict(value_dtype="int16"),
        dict(query_dtype="Float16"),
        dict(query_dtype="prefix.float16"),
        dict(query_dtype="mlx.core.float16.extra"),
        dict(query_dtype="float16 "),
        dict(query_dtype=None),
        dict(key_dtype=True),
        dict(query_dtype="float32"),
        dict(value_dtype="mlx.core.float32"),
    ],
)
def test_invalid_dtypes_fail_before_lazy_import(monkeypatch, overrides):
    assert_invalid_before_runtime(monkeypatch, overrides, "dtype")


@pytest.mark.parametrize("dtype", ["float16", "float32", "mlx.core.float16", "mlx.core.float32"])
def test_explicit_supported_dtype_names(dtype):
    component = load_component()
    assert component.plan_causal_prefixes(
        **metadata(
            query_dtype=dtype,
            key_dtype=dtype,
            value_dtype=dtype,
        )
    ) == (8, 9, 10, 11, 12)


@pytest.mark.parametrize(
    "cache",
    [
        None,
        object(),
        SimpleNamespace(offset=True),
        SimpleNamespace(offset=12.0),
        SimpleNamespace(offset=11),
        SimpleNamespace(offset=13),
        SimpleNamespace(offset=-1),
        SimpleNamespace(offset=12, bits=4),
        SimpleNamespace(offset=12, bits=None),
        SimpleNamespace(offset=100, max_size=12),
    ],
)
def test_incompatible_cache_fails_before_lazy_import(monkeypatch, cache):
    assert_invalid_before_runtime(monkeypatch, dict(cache=cache), "cache")


class ArrayLikeMask:
    def __eq__(self, other):
        raise AssertionError("must not compare array-like mask values")


class CausalSubclass(str):
    pass


@pytest.mark.parametrize(
    "mask", [None, "CAUSAL", "causal ", "", False, [], ArrayLikeMask(), CausalSubclass("causal")]
)
def test_multi_query_requires_exact_causal_string(monkeypatch, mask):
    assert_invalid_before_runtime(monkeypatch, dict(mask=mask), "mask")


@pytest.mark.parametrize("mask", ["none", False, [], ArrayLikeMask(), CausalSubclass("causal")])
def test_width_one_rejects_noncausal_mask(monkeypatch, mask):
    assert_invalid_before_runtime(monkeypatch, dict(query_shape=(1, 14, 1, 64), mask=mask), "mask")


@pytest.mark.parametrize("sinks", [False, 0, [], object()])
def test_attention_sinks_are_rejected(monkeypatch, sinks):
    assert_invalid_before_runtime(monkeypatch, dict(sinks=sinks), "sinks")


@pytest.mark.parametrize(
    "scale",
    [
        True,
        False,
        None,
        "0.125",
        0,
        -1,
        -0.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        1j,
        10**400,
    ],
)
def test_scale_must_be_positive_finite_real(monkeypatch, scale):
    assert_invalid_before_runtime(monkeypatch, dict(scale=scale), "scale")


@pytest.mark.parametrize("scale", [1, 0.125, 1e-300])
def test_positive_finite_real_scale_is_accepted(scale):
    component = load_component()
    assert component.plan_causal_prefixes(**metadata(scale=scale)) == (8, 9, 10, 11, 12)


@pytest.mark.parametrize(
    "which,attribute",
    [
        ("q", "shape"),
        ("k", "shape"),
        ("v", "shape"),
        ("q", "dtype"),
        ("k", "dtype"),
        ("v", "dtype"),
    ],
)
def test_missing_array_metadata_is_validation_error(monkeypatch, which, attribute):
    component = load_component()

    def forbidden_loader():
        pytest.fail("missing metadata reached runtime")

    monkeypatch.setattr(component, "_load_mx", forbidden_loader)
    arrays = dict(
        q=StructuralArray((1, 14, 5, 64)),
        k=StructuralArray((1, 2, 12, 64)),
        v=StructuralArray((1, 2, 12, 64)),
    )
    delattr(arrays[which], attribute)
    with pytest.raises(ValueError, match=attribute):
        component.stable_causal_attention(
            arrays["q"],
            arrays["k"],
            arrays["v"],
            cache=ReadOnlyCache(12),
            scale=0.125,
            mask="causal",
        )


class NativeDtypeRepresentation:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"mlx.core.{self.name}"


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_structural_wrapper_accepts_native_dtype_representation(monkeypatch, dtype):
    component = load_component()
    runtime = StructuralRuntime()
    monkeypatch.setattr(component, "_load_mx", lambda: runtime)
    native_dtype = NativeDtypeRepresentation(dtype)
    q = StructuralArray((1, 20, 5, 128), native_dtype)
    k = StructuralArray((1, 4, 5, 128), native_dtype)
    v = StructuralArray(k.shape, native_dtype)
    result = component.stable_causal_attention(
        q, k, v, cache=ReadOnlyCache(5), scale=0.125, mask="causal"
    )
    assert result.shape == q.shape
    assert [call[1].shape[-2] for call in runtime.calls] == [1, 2, 3, 4, 5]


def test_import_guard_blocks_actual_lazy_loader():
    component = load_component()
    with pytest.raises(AssertionError, match="Forbidden CPU-test import: mlx"):
        component._load_mx()


def test_import_guard_blocks_importlib_runtime_loading():
    import importlib

    with pytest.raises(AssertionError, match="Forbidden CPU-test import: numpy"):
        importlib.import_module("numpy")
