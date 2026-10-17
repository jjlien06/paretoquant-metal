"""Opt-in BF16 storage-to-F16 runtime transform; NOT equivalent arithmetic.

No checkpoint writes or repacking. Original payload hashes belong to the loader,
not to the converted runtime arrays. F32 (including A_log), F16 and U32 stay put.
"""

import hashlib
import json
import math
import re


def layout_map(rows, names):
    """Strict JSON dtype/shape/byte inventory, independent of MLX imports."""
    widths = {
        "mlx.core.uint32": 4,
        "mlx.core.float32": 4,
        "mlx.core.bfloat16": 2,
        "mlx.core.float16": 2,
    }
    if not isinstance(rows, list) or len(rows) != len(names):
        raise ValueError("Incomplete compute parameter inventory")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "shape", "dtype", "bytes"}:
            raise ValueError("Invalid compute parameter inventory row")
        name, shape, dtype, size = (row[k] for k in ("name", "shape", "dtype", "bytes"))
        if (
            not isinstance(name, str)
            or name not in names
            or name in result
            or not isinstance(shape, list)
            or any(type(s) is not int or s <= 0 for s in shape)
            or not isinstance(dtype, str)
            or dtype not in widths
            or type(size) is not int
            or size != math.prod(shape) * widths[dtype]
        ):
            raise ValueError("Invalid compute parameter shape/dtype/bytes")
        result[name] = row
    if set(result) != set(names):
        raise ValueError("Compute parameter names differ from admitted inventory")
    return result


def verify_compute_receipt(rank, mode, names):
    """Bind trusted compute admission to independently constrained runtime evidence."""
    validate_compute_dtype(mode)
    compute = rank.get("compute")
    loading = rank.get("weight_loading") or {}
    if (
        rank.get("compute_dtype") != mode
        or not isinstance(compute, dict)
        or compute.get("requested_compute_dtype") != mode
        or loading.get("tensor_hash_scope") != "serialized_source_payloads_before_runtime_cast"
        or compute.get("hash_scope") != "converted_runtime_arrays_only"
    ):
        raise ValueError("Compute mode/hash scope differs from admitted execution")
    source = layout_map(compute.get("source_parameter_layout"), names)
    runtime = layout_map(compute.get("runtime_parameter_layout"), names)
    if source != layout_map(loading.get("source_parameter_layout"), names) or runtime != layout_map(
        rank.get("parameter_layout"), names
    ):
        raise ValueError("Compute source/runtime inventory contradicts rank/loading")
    changed = []
    for name, original in source.items():
        expected = dict(original)
        if name.endswith(".scales"):
            weight = source.get(name.removesuffix(".scales") + ".weight", {})
            if weight.get("dtype") != "mlx.core.uint32":
                raise ValueError("Quantized packed weights must remain uint32")
        if mode != "native" and original["dtype"] == "mlx.core.bfloat16":
            if name.split(".")[-1] == "A_log":
                raise ValueError("Cannot cast protected BF16 source")
            expected["dtype"] = "mlx.core.float16"
            changed.append(name)
        if runtime[name] != expected:
            raise ValueError("Runtime parameter changed outside eligible BF16 cast")
    changed.sort()
    if mode == "float16-rounded":
        if compute.get("cast_policy") != "finite-fp16-rounding-v1":
            raise ValueError("Missing admitted rounded cast policy")
        diagnostics = compute.get("conversion_diagnostics")
        if not isinstance(diagnostics, dict) or set(diagnostics) != set(changed):
            raise ValueError("Missing per-tensor rounding diagnostics")
        for name in changed:
            d = diagnostics[name]
            fields = {
                "source_sha256",
                "runtime_sha256",
                "element_count",
                "numerically_changed_count",
                "roundtrip_exact_count",
                "max_abs_error",
                "max_rel_error",
            }
            if not isinstance(d, dict) or set(d) != fields:
                raise ValueError("Invalid rounding diagnostic fields")
            count = math.prod(source[name]["shape"])
            if (
                any(
                    type(d[k]) is not int
                    for k in ("element_count", "numerically_changed_count", "roundtrip_exact_count")
                )
                or d["element_count"] != count
                or not 0 <= d["numerically_changed_count"] <= count
                or d["roundtrip_exact_count"] != count - d["numerically_changed_count"]
                or any(
                    type(d[k]) not in (int, float) or not math.isfinite(d[k]) or d[k] < 0
                    for k in ("max_abs_error", "max_rel_error")
                )
                or (d["max_abs_error"] == 0) != (d["numerically_changed_count"] == 0)
                or (d["max_rel_error"] == 0) != (d["numerically_changed_count"] == 0)
                or d["source_sha256"] != loading.get("tensor_sha256", {}).get(name)
                or d["runtime_sha256"] != compute.get("converted_tensor_sha256", {}).get(name)
            ):
                raise ValueError("Rounding diagnostics contradict source/runtime evidence")
        expected_exact = all(
            d["roundtrip_exact_count"] == d["element_count"] for d in diagnostics.values()
        )
    else:
        expected_exact = True if mode == "float16" else None
    digests = compute.get("converted_tensor_sha256")
    if (
        compute.get("converted_tensor_names") != changed
        or not isinstance(digests, dict)
        or set(digests) != set(changed)
        or any(
            not isinstance(d, str) or re.fullmatch(r"[0-9a-f]{64}", d) is None
            for d in digests.values()
        )
        or compute.get("arithmetic_changed") is not bool(changed)
        or compute.get("roundtrip_exact") is not expected_exact
        or compute.get("precision_equivalent") is not (False if mode != "native" else None)
    ):
        raise ValueError("Missing or forged runtime cast evidence")
    prefix = "language_model.model.embed_tokens"
    anchor = prefix + (".scales" if prefix + ".scales" in runtime else ".weight")
    actual = compute.get("actual_compute_dtype")
    if (
        anchor not in runtime
        or runtime[anchor]["dtype"] != actual
        or (mode != "native" and actual != "mlx.core.float16")
    ):
        raise ValueError("Actual compute dtype contradicts embedding/runtime inventory")
    codes = compute_codes(mode, actual)
    peer = compute.get("peer_compute_codes")
    if (
        not isinstance(peer, list)
        or len(peer) != 2
        or any(
            not isinstance(p, list)
            or len(p) != 2
            or any(type(v) is not int for v in p)
            or p != codes
            for p in peer
        )
    ):
        raise ValueError("Missing actual both-rank compute agreement")
    return actual


def receipt_sha256(receipt):
    data = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def compute_codes(mode, actual):
    validate_compute_dtype(mode)
    codes = {"mlx.core.float16": 16, "mlx.core.bfloat16": 17, "mlx.core.float32": 32}
    if actual not in codes:
        raise ValueError("Unsupported actual compute dtype")
    return [{"native": 0, "float16": 1, "float16-rounded": 2}[mode], codes[actual]]


def agree_compute(receipt, group):
    """Check mode AND actual handoff dtype before any model communication."""
    import mlx.core as mx

    codes = compute_codes(receipt["requested_compute_dtype"], receipt["actual_compute_dtype"])
    if group is None:
        return [codes]
    if group.size() != 2:
        raise ValueError("Require two ranks for compute agreement")
    gathered = mx.distributed.all_gather(mx.array(codes, mx.int32), group=group, stream=mx.cpu)
    mx.eval(gathered)
    actual = gathered.tolist()
    if actual != codes * 2:
        raise ValueError("Pipeline ranks disagree on compute mode/dtype")
    return [actual[:2], actual[2:]]


def validate_compute_dtype(mode):
    if mode not in ("native", "float16", "float16-rounded"):
        raise ValueError("Invalid compute dtype; require native, float16 or float16-rounded")


def parameter_layout(model):
    from mlx.utils import tree_flatten

    return [
        {"name": name, "shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes}
        for name, a in sorted(tree_flatten(model.parameters()))
    ]


def embedding_compute_dtype(model):
    embedding = model.language_model.model.embed_tokens
    return str(embedding.scales.dtype if hasattr(embedding, "scales") else embedding.weight.dtype)


def apply_compute_dtype(model, mode="native"):
    """Fail closed before generation; a failed transform is never a runnable model.

    Validate and install one array at a time. On failure the process must discard
    the partially transformed model. Original parameters are never serialized.
    """
    validate_compute_dtype(mode)
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    source = parameter_layout(model)
    changed, hashes, diagnostics = [], {}, {}
    rounded = mode == "float16-rounded"
    predicate = getattr(model, "cast_predicate", lambda path: True)
    if mode in ("float16", "float16-rounded"):
        # Hold names, not the full original parameter tree, across the transform.
        names = [row["name"] for row in source if row["dtype"] == str(mx.bfloat16)]
        for name in names:
            if name.split(".")[-1] == "A_log" or not predicate(name):
                raise ValueError(f"Cannot cast protected BF16 parameter: {name}")
            original = dict(tree_flatten(model.parameters()))[name]
            cast = original.astype(mx.float16)
            restored = cast.astype(mx.bfloat16)
            finite = mx.all(mx.isfinite(original)) & mx.all(mx.isfinite(cast))
            exact = mx.array_equal(original.view(mx.uint16), restored.view(mx.uint16))
            mx.eval(cast, finite, exact)
            if not finite.item():
                raise ValueError(f"Nonfinite/overflow runtime cast: {name}")
            if not rounded and not exact.item():
                raise ValueError(f"BF16-to-F16 roundtrip is not exact: {name}")
            hashes[name] = hashlib.sha256(np.asarray(cast).tobytes()).hexdigest()
            if rounded:
                # F64 CPU diagnostics avoid BF16/F32 relative-error underflow.
                before = np.asarray(original.astype(mx.float32)).astype(np.float64)
                after = np.asarray(cast).astype(np.float64)
                error = np.abs(after - before)
                relative = np.divide(
                    error, np.abs(before), out=np.zeros_like(error), where=before != 0
                )
                original_bits = np.asarray(original.view(mx.uint16))
                restored_bits = np.asarray(restored.view(mx.uint16))
                diagnostics[name] = {
                    "source_sha256": hashlib.sha256(original_bits.tobytes()).hexdigest(),
                    "runtime_sha256": hashes[name],
                    "element_count": int(before.size),
                    "numerically_changed_count": int(np.count_nonzero(before != after)),
                    "roundtrip_exact_count": int(np.count_nonzero(original_bits == restored_bits)),
                    "max_abs_error": float(error.max(initial=0)),
                    "max_rel_error": float(relative.max(initial=0)),
                }
            model.load_weights([(name, cast)], strict=False)
            changed.append(name)
            del original, cast, restored, finite, exact
        if embedding_compute_dtype(model) != str(mx.float16):
            raise ValueError("float16 compute requires a float16 embedding after casting")
    result = {
        "requested_compute_dtype": mode,
        "actual_compute_dtype": embedding_compute_dtype(model),
        "arithmetic_changed": bool(changed),
        "precision_equivalent": False if mode != "native" else None,
        "roundtrip_exact": (
            all(d["roundtrip_exact_count"] == d["element_count"] for d in diagnostics.values())
            if rounded
            else True
            if mode == "float16"
            else None
        ),
        "hash_scope": "converted_runtime_arrays_only",
        "converted_tensor_names": sorted(changed),
        "converted_tensor_sha256": hashes,
        "source_parameter_layout": source,
        "runtime_parameter_layout": parameter_layout(model),
        "arithmetic_note": (
            "BF16 arrays rounded to finite F16; values may change including underflow to zero; "
            "practical numerics tradeoff, not same arithmetic, not precision equivalence; "
            "no quality guarantee"
            if rounded
            else "BF16 arrays cast to F16 exactly as values; dequantization, kernels, "
            "activations and accumulation arithmetic may differ; not precision equivalence"
            if mode == "float16"
            else "native runtime arrays unchanged"
        ),
    }
    if rounded:
        result["cast_policy"] = "finite-fp16-rounding-v1"
        result["conversion_diagnostics"] = diagnostics
    return result
