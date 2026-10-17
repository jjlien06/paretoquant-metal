"""Isolated, unadmitted slot-stable causal attention candidate.

Only Python metadata is inspected before the optional MLX runtime import.
CPU tests establish validation and call structure, not GPU/numerical parity.
No decoder patching, cache updates, tensor evaluation, or performance claim.
"""

import math


def plan_causal_prefixes(
    query_shape,
    key_shape,
    value_shape,
    query_dtype,
    key_dtype,
    value_dtype,
    *,
    cache,
    scale,
    mask,
    sinks=None,
):
    """Validate metadata and return each query's exclusive causal KV end.

    Shapes are (1, Hq, L, D) and identical (1, Hkv, K, D) for K/V.
    Require positive Python-int dimensions, Hq divisible by Hkv, 1 <= L <= 9,
    and K >= L. Dtypes are matching float16/float32 names, either plain or
    exact MLX string representations. The already-updated floating cache
    must lack ``bits`` and have a Python-int offset exactly equal to K.

    No cache tensors, update/trim methods, offsets beyond this read, or
    runtime modules are accessed. Invalid metadata raises ValueError.
    This checks layout metadata, not native-cache identity or tensor values.
    """
    for shape in (query_shape, key_shape, value_shape):
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ValueError("shape must be rank four")
        if any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive Python integers")
        if shape[0] != 1:
            raise ValueError("shape batch must be one")
    if tuple(key_shape) != tuple(value_shape):
        raise ValueError("key/value shape must match exactly")
    if query_shape[-1] != key_shape[-1]:
        raise ValueError("query/key/value shape head dimensions must match")
    if query_shape[1] % key_shape[1] != 0:
        raise ValueError("shape query heads must be divisible by KV heads")
    if not 1 <= query_shape[-2] <= 9 or key_shape[-2] < query_shape[-2]:
        raise ValueError("shape requires 1 <= query width <= 9 and KV length >= width")
    names = {
        "float16": "float16",
        "mlx.core.float16": "float16",
        "float32": "float32",
        "mlx.core.float32": "float32",
    }
    dtypes = (query_dtype, key_dtype, value_dtype)
    if any(type(dtype) is not str or dtype not in names for dtype in dtypes):
        raise ValueError("dtype must be exactly float16 or float32")
    if len({names[dtype] for dtype in dtypes}) != 1:
        raise ValueError("query/key/value dtype must match")
    if cache is None or hasattr(cache, "bits"):
        raise ValueError("cache must be an already-updated floating KV cache")
    offset = getattr(cache, "offset", None)
    if type(offset) is not int or offset != key_shape[-2]:
        raise ValueError("cache offset must be a Python int equal to KV length")
    if sinks is not None:
        raise ValueError("sinks are not supported")
    causal = type(mask) is str and mask == "causal"
    if not causal and not (query_shape[-2] == 1 and mask is None):
        raise ValueError("mask must be the exact causal string (or None for width one)")
    if type(scale) not in (int, float) or scale <= 0:
        raise ValueError("scale must be a positive finite Python int or float")
    try:
        finite = math.isfinite(scale)
    except OverflowError as exc:
        raise ValueError("scale exceeds finite native real range") from exc
    if not finite:
        raise ValueError("scale must be finite")
    width = query_shape[-2]
    length = key_shape[-2]
    return tuple(length - width + i + 1 for i in range(width))


def _load_mx():
    import mlx.core as mx

    return mx


def stable_causal_attention(queries, keys, values, *, cache, scale, mask, sinks=None):
    """Use already-updated KV tensors without changing their native cache.

    L == 1 delegates the original arrays and supplied mask to one native
    fast SDPA call. For L >= 2 the exact "causal" mask is required; each
    contiguous one-query slice attends only to its visible K/V prefix,
    with mask=None and sinks=None. Outputs concatenate on query axis -2.
    Call mx.fast directly, never a potentially scoped-patched model helper.
    """
    prefixes = plan_causal_prefixes(
        getattr(queries, "shape", None),
        getattr(keys, "shape", None),
        getattr(values, "shape", None),
        str(getattr(queries, "dtype", None)),
        str(getattr(keys, "dtype", None)),
        str(getattr(values, "dtype", None)),
        cache=cache,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )
    mx = _load_mx()
    if len(prefixes) == 1:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
    outputs = []
    for position, end in enumerate(prefixes):
        query = mx.contiguous(queries[..., position : position + 1, :])
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                query,
                keys[..., :end, :],
                values[..., :end, :],
                scale=scale,
                mask=None,
                sinks=None,
            )
        )
    return mx.concatenate(outputs, axis=-2)
