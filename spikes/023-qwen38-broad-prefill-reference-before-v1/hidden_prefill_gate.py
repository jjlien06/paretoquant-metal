"""Mandatory native-arithmetic hidden-prefill equivalence, never a bypass flag.

Three fresh native target cache trees: original MTP (all-vocabulary projection),
optimized (all normalized hidden, no projection), and native cache-only prefill.
Only rank0 owns a head; both physical owners attest all their target cache states.
"""

import weakref
from copy import deepcopy

from verification import ATOL, RTOL, cache_metrics, metric

SCHEMA = "hidden-only-prefill-v1"
# Retain only receipts/metadata, never model weights or native cache tensors.
# JSON receipts cannot mint a same-process live-model capability.
_LIVE_LAYOUTS = {}


def trusted_layout(evidence, *, model=None):
    record = _LIVE_LAYOUTS.get(id(evidence))
    if record is None or record[0] is not evidence:
        raise ValueError("Missing live hidden prefill cache layout")
    if model is not None and record[1]() is not model:
        raise ValueError("Hidden prefill cache layout belongs to another live model")
    return deepcopy(record[2])


def architecture_layout(args, cache_types, *, dtype, head_dtype):
    """Required parts/shapes, independent of the submitted metric inventory."""
    if dtype not in ("float16", "bfloat16", "float32") or head_dtype not in (
        "float16",
        "bfloat16",
        "float32",
    ):
        raise ValueError("Unsupported hidden prefill cache dtype")
    target = []

    def kv(layer, dt):
        return [
            {
                "layer": layer,
                "part": part,
                "shape": [1, args.num_key_value_heads, None, args.head_dim],
                "dtype": dt,
            }
            for part in ("keys", "values")
        ] + [{"layer": layer, "part": "offset"}]

    for layer, kind in enumerate(cache_types):
        if kind == "KVCache":
            target.extend(kv(layer, dtype))
        elif kind == "ArraysCache":
            target.extend(
                [
                    {
                        "layer": layer,
                        "part": "state.0.0",
                        "shape": [
                            1,
                            args.linear_conv_kernel_dim - 1,
                            2 * args.linear_num_key_heads * args.linear_key_head_dim
                            + args.linear_num_value_heads * args.linear_value_head_dim,
                        ],
                        "dtype": dtype,
                    },
                    {
                        "layer": layer,
                        "part": "state.0.1",
                        "shape": [
                            1,
                            args.linear_num_value_heads,
                            args.linear_value_head_dim,
                            args.linear_key_head_dim,
                        ],
                        "dtype": "float32",
                    },
                ]
            )
        else:
            raise ValueError("Unknown native hidden prefill cache class")
    return {
        "target": target,
        "head": kv(0, head_dtype),
        "dtype": dtype,
        "head_dtype": head_dtype,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
    }


def metadata_layout(options, *, scope, owned_layers, rank):
    """Detached controller boundary: independently hash-bound checkpoint config.

    This never infers cache inventory/dimensions/dtype from submitted metrics.
    It follows the pinned native qwen3_5 make_cache class schedule, not optional
    checkpoint layer_types (which that implementation does not use).
    """
    import json
    from pathlib import Path

    from mlx_lm.models.qwen3_5 import TextModelArgs
    from verification import bind_metadata

    if scope == "small":
        path = options.small_model_local
        config_sha, index_sha = options.small_config_sha256, options.small_index_sha256
    else:
        path = options.local_metadata
        config_sha, index_sha = options.target_config_sha256, options.target_index_sha256
    bind_metadata(path, config_sha, index_sha)
    config = json.loads((Path(path) / "config.json").read_text())
    text = config.get("text_config", config)
    args = TextModelArgs.from_dict(text)
    lo, hi = owned_layers
    if (
        type(lo) is not int
        or type(hi) is not int
        or not 0 <= lo < hi <= args.num_hidden_layers
        or (scope == "small" and [lo, hi] != [0, args.num_hidden_layers])
    ):
        raise ValueError("Hidden prefill architecture ownership mismatch")
    dtype = text.get(
        "dtype", text.get("torch_dtype", config.get("dtype", config.get("torch_dtype")))
    )
    classes = [
        "KVCache" if (i + 1) % args.full_attention_interval == 0 else "ArraysCache"
        for i in range(lo, hi)
    ]
    layout = architecture_layout(
        args, classes, dtype=dtype, head_dtype="float32" if scope == "small" else dtype
    )
    layout.update(owned_layers=[lo, hi], rank=0 if scope == "small" else rank)
    return layout


def _native_layout(model, cache, head_cache, *, dtype, head_dtype):
    from mlx_lm.models.cache import ArraysCache, KVCache

    fresh = model.make_cache()
    if len(fresh) != len(cache) or not fresh:
        raise ValueError("Hidden prefill native cache coverage")
    for expected, actual in zip(fresh, cache):
        if type(expected) not in (ArraysCache, KVCache) or type(actual) is not type(expected):
            raise ValueError("Hidden prefill native cache class")
        if type(expected) is ArraysCache and (
            len(expected.cache) != 2
            or len(actual.cache) != 2
            or actual.left_padding is not None
            or actual.lengths is not None
        ):
            raise ValueError("Hidden prefill native ArraysCache layout")
    if head_cache is not None and type(head_cache) is not KVCache:
        raise ValueError("Hidden prefill head cache class")
    layout = architecture_layout(
        model.language_model.args,
        [type(c).__name__ for c in fresh],
        dtype=dtype,
        head_dtype=head_dtype,
    )
    layout.update(
        owned_layers=[
            model.model.start_idx,
            model.model.end_idx or model.language_model.args.num_hidden_layers,
        ]
    )
    return layout


def verify_prefill_evidence(evidence, *, scope, chunk_size, layer_count, expected_layout=None):
    """Validate complete physical receipts, not just a passed/greedy summary."""
    import math

    def require(condition):
        if not condition:
            raise ValueError("Incomplete strict hidden prefill evidence")

    e = evidence or {}
    layout = trusted_layout(e) if expected_layout is None else expected_layout
    require(len({p["layer"] for p in layout["target"]}) == layer_count)
    lengths = gate_lengths(scope, chunk_size)
    require(
        e.get("schema") == SCHEMA
        and e.get("passed") is True
        and e.get("same_process") is True
        and e.get("target_forward_mode") == "native"
        and e.get("atol") == ATOL
        and e.get("rtol") == RTOL
        and e.get("chunk_size") == chunk_size
        and e.get("prompt_lengths") == lengths
        and type(e.get("hidden_size")) is int
        and e["hidden_size"] > 0
        and type(e.get("vocab_size")) is int
        and e["vocab_size"] > 1
        and len(e.get("cases", [])) == len(lengths)
    )
    hidden_size, vocab = e["hidden_size"], e["vocab_size"]
    require(hidden_size == layout["hidden_size"] and vocab == layout["vocab_size"])
    require(e.get("owned_layers") == layout["owned_layers"])
    require(e.get("rank") == layout["rank"])
    if scope == "full" and chunk_size < 256:
        representative = e.get("representative_256")
        verify_prefill_evidence(
            representative,
            scope="representative",
            chunk_size=256,
            layer_count=layer_count,
            expected_layout=layout,
        )
        require(
            all(
                representative.get(k) == e.get(k)
                for k in ("owned_layers", "rank", "hidden_size", "vocab_size")
            )
        )

    def check_metric(m, shape, dtype=None):
        require(
            isinstance(m, dict)
            and m.get("strict_pass") is True
            and m.get("shape") == shape
            and all(type(d) is int and d > 0 for d in m["shape"])
            and m.get("dtype") == (layout["dtype"] if dtype is None else dtype)
            and m.get("finite") is True
            and type(m.get("exact")) is bool
            and type(m.get("max_abs")) in (int, float)
            and math.isfinite(m["max_abs"])
            and m["max_abs"] >= 0
            and type(m.get("max_tolerance")) in (int, float)
            and math.isfinite(m["max_tolerance"])
            and m["max_tolerance"] > 0
            and m["max_abs"] <= m["max_tolerance"]
            and type(m.get("max_tolerance_ratio")) in (int, float)
            and math.isfinite(m["max_tolerance_ratio"])
            and 0 <= m["max_tolerance_ratio"] <= 1
            and (not m["exact"] or (m["max_abs"] == 0 and m["max_tolerance_ratio"] == 0))
        )

    def check_cache(parts, offset, head=False):
        expected = layout["head" if head else "target"]
        require(isinstance(parts, list) and len(parts) == len(expected))
        require(
            all(
                isinstance(p, dict)
                and type(p.get("layer")) is int
                and isinstance(p.get("part"), str)
                for p in parts
            )
        )
        required = {(p["layer"], p["part"]): p for p in expected}
        inventory = [(p["layer"], p["part"]) for p in parts]
        require(len(set(inventory)) == len(inventory) and set(inventory) == set(required))
        for p in parts:
            require(p.get("strict_pass") is True)
            if p["part"] == "offset":
                require(
                    type(p.get("value")) is int
                    and p["value"] == offset
                    and p.get("exact") is True
                    and type(p.get("max_abs")) in (int, float)
                    and p["max_abs"] == 0
                )
            else:
                shape = [
                    offset if d is None else d for d in required[p["layer"], p["part"]]["shape"]
                ]
                check_metric(p, shape, required[p["layer"], p["part"]]["dtype"])

    def check_row(row, offset, *, head_offset=None, output=False):
        require(row.get("passed") is True and row.get("cache_pass") is True)
        for field in ("caches", "native_caches"):
            check_cache(row.get(field), offset)
        if head_offset is not None:
            require(row.get("head_offset") == head_offset)
            check_cache(row.get("head_caches"), head_offset, head=True)
        if output:
            require(row.get("argmax_equal") is True and row.get("native_argmax_equal") is True)
            for field in ("logits", "native_logits"):
                check_metric(row.get(field), [1, 1, vocab])
            for field in ("hidden", "native_hidden"):
                check_metric(row.get(field), [1, 1, hidden_size])

    for case, length in zip(e["cases"], lengths):
        require(case.get("passed") is True and case.get("prompt_tokens") == length)
        intervals = [(s, min(s + chunk_size, length - 1)) for s in range(0, length - 1, chunk_size)]
        require([(r.get("start"), r.get("end")) for r in case.get("chunks", [])] == intervals)
        prompt = [1 + i % min(100, vocab - 1) for i in range(length)]
        for row, (start, end) in zip(case["chunks"], intervals):
            require(row.get("shifted_tokens") == prompt[start + 1 : end + 1])
            check_metric(row.get("hidden"), [1, end - start, hidden_size])
            check_row(row, end, head_offset=end)
        final = case.get("final", {})
        require(final.get("token") == prompt[-1])
        check_row(final, length, output=True)
        continuation = case.get("continuation", [])
        require(len(continuation) == 2)
        for index, (row, token) in enumerate(zip(continuation, (6 % vocab, 7 % vocab)), 1):
            require(row.get("token") == token)
            check_row(row, length + index, head_offset=length - 1 + index, output=True)
            check_metric(row.get("head_logits"), [1, 1, vocab], layout["head_dtype"])


def gate_lengths(scope, chunk_size):
    if type(chunk_size) is not int or not 1 <= chunk_size <= 256:
        raise ValueError("Hidden gate chunk size must be in 1..256")
    if scope == "small":
        return [256, 257, 511, 512, 513]
    if scope == "full":
        return [2 * chunk_size + 1]
    if scope == "representative" and chunk_size == 256:
        return [257]
    raise ValueError("Unknown hidden gate scope")


def full_gate(model, head, *, chunk_size=256, **kwargs):
    """Two configured chunks plus mandatory native256 coverage if reduced."""
    result = prefill_gate(
        model,
        head,
        prompt_lengths=gate_lengths("full", chunk_size),
        chunk_size=chunk_size,
        **kwargs,
    )
    if chunk_size < 256:
        representative = prefill_gate(model, head, prompt_lengths=[257], chunk_size=256, **kwargs)
        result["representative_256"] = representative
        result["passed"] = result["passed"] and representative["passed"]
    return result


def prefill_gate(
    model, head, *, prompt_lengths, chunk_size=256, group=None, eval_group_size=1, report=None
):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from phased_decode import ownership
    from verifier import forward

    gate_lengths("full", chunk_size)
    if (
        not isinstance(prompt_lengths, (list, tuple))
        or not prompt_lengths
        or any(type(n) is not int or not 2 <= n <= 513 for n in prompt_lengths)
    ):
        raise ValueError("Require bounded nontrivial hidden gate prompts")
    rank, size = ownership(model, group)
    if (rank == 0) != (head is not None):
        raise ValueError("Hidden gate requires live head only on final rank0")
    emit = report or (lambda e: None)
    channel = None
    if size == 2:
        from distributed_control import Channel

        channel = Channel(group)

    def exchange(values, local_pass):
        # Publish rank0 all-position/head metrics AND combine independent cache
        # attestations before either rank can branch or advance to another case.
        if channel is not None:
            if rank == 0:
                channel.send(values)
                channel.send({"passed": local_pass})
                local_pass = channel.receive()["passed"] and local_pass
            else:
                values = channel.receive()
                peer = channel.receive()["passed"]
                channel.send({"passed": local_pass})
                local_pass = peer and local_pass
        return values, local_pass

    def run(cache, tokens, *, logits=True, hidden=True):
        return forward(
            model,
            mx.array([tokens], mx.int32),
            cache,
            group=group,
            eval_group_size=eval_group_size,
            all_logits=True,
            return_hidden=hidden,
            logits=logits,
            verifier_mode="native",
        )

    def compare(caches, outputs=None, head_caches=None, head_scores=None, **binding):
        left, right, native = caches
        parts, native_parts = cache_metrics(left, right), cache_metrics(left, native)
        values = {}
        if rank == 0:
            if outputs is not None:
                a, b, c = outputs
                values.update(
                    logits=metric(a[0], b[0]),
                    hidden=metric(a[1], b[1]),
                    native_logits=metric(a[0], c[0]),
                    native_hidden=metric(a[1], c[1]),
                    argmax_equal=mx.argmax(a[0], -1).tolist() == mx.argmax(b[0], -1).tolist(),
                    native_argmax_equal=mx.argmax(a[0], -1).tolist()
                    == mx.argmax(c[0], -1).tolist(),
                )
            if head_caches is not None:
                values["head_caches"] = cache_metrics(*head_caches)
                values["head_offset"] = head_caches[0][0].offset
            if head_scores is not None:
                values["head_logits"] = metric(*head_scores)
        local_pass = all(p["strict_pass"] for p in parts + native_parts)
        values, local_pass = exchange(values, local_pass)
        strict = all(v["strict_pass"] for v in values.values() if isinstance(v, dict))
        strict = strict and all(p["strict_pass"] for p in values.get("head_caches", []))
        strict = (
            strict and values.get("argmax_equal", True) and values.get("native_argmax_equal", True)
        )
        return {
            **binding,
            **values,
            "caches": parts,
            "native_caches": native_parts,
            "cache_pass": local_pass,
            "passed": local_pass and strict,
        }

    cases = []
    layout = None
    for length in prompt_lengths:
        vocab = model.language_model.args.vocab_size
        prompt = [1 + i % min(100, vocab - 1) for i in range(length)]
        caches = [model.make_cache() for _ in range(3)]
        left, right, native = caches
        heads = (KVCache(), KVCache()) if rank == 0 else None
        chunks = []
        for start in range(0, length - 1, chunk_size):
            end = min(start + chunk_size, length - 1)
            tokens = prompt[start:end]
            a = run(left, tokens)
            b = run(right, tokens, logits=False)
            run(native, tokens, logits=False, hidden=False)
            values = {}
            if rank == 0:
                if b[0] is not None:
                    raise ValueError("Hidden prefill unexpectedly projected vocabulary")
                values["hidden"] = metric(a[1], b[1])
                shifted = mx.array([prompt[start + 1 : end + 1]], mx.int32)
                for out, cache in zip((a, b), heads):
                    head.forward(out[1], shifted, cache, position=start, logits=False)
            if layout is None:
                # Native make_cache decides classes even while arrays are empty;
                # warm reference state supplies activation dtype, never inventory.
                dtypes = (
                    {
                        "dtype": str(a[1].dtype).removeprefix("mlx.core."),
                        "head_dtype": str(heads[0].keys.dtype).removeprefix("mlx.core."),
                    }
                    if rank == 0
                    else {}
                )
                dtypes, _ = exchange(dtypes, True)
                layout = _native_layout(model, left, heads[0] if rank == 0 else None, **dtypes)
            row = compare(
                caches,
                head_caches=([heads[0]], [heads[1]]) if rank == 0 else None,
                start=start,
                end=end,
                shifted_tokens=prompt[start + 1 : end + 1],
            )
            values, _ = exchange(values, True)
            row.update(values)
            row["passed"] = row["passed"] and values["hidden"]["strict_pass"]
            chunks.append(row)
            emit({"phase": "hidden_prefill_gate_chunk", "prompt_tokens": length, **row})
            # Do not retain unused all-vocab arrays across chunks or cases.
            del a, b
        outputs = [run(c, [prompt[-1]]) for c in caches]
        final = compare(caches, outputs, token=prompt[-1])
        continuation = []
        for token in (6 % vocab, 7 % vocab):
            scores = None
            if rank == 0:
                scores = [
                    head.forward(out[1], mx.array([[token]], mx.int32), hc, position=hc.offset)
                    for out, hc in zip(outputs[:2], heads)
                ]
            outputs = [run(c, [token]) for c in caches]
            row = compare(
                caches,
                outputs,
                head_caches=([heads[0]], [heads[1]]) if rank == 0 else None,
                head_scores=scores,
                token=token,
            )
            continuation.append(row)
        case = {
            "prompt_tokens": length,
            "chunks": chunks,
            "final": final,
            "continuation": continuation,
            "passed": all(r["passed"] for r in chunks + [final] + continuation),
        }
        cases.append(case)
        emit(
            {"phase": "hidden_prefill_gate_case", "prompt_tokens": length, "passed": case["passed"]}
        )
    result = {
        "schema": SCHEMA,
        "passed": all(c["passed"] for c in cases),
        "same_process": True,
        "target_forward_mode": "native",
        "rank": rank,
        "atol": ATOL,
        "rtol": RTOL,
        "chunk_size": chunk_size,
        "prompt_lengths": list(prompt_lengths),
        "cases": cases,
        "hidden_size": model.language_model.args.hidden_size,
        "vocab_size": model.language_model.args.vocab_size,
        "owned_layers": [
            model.model.start_idx,
            model.model.end_idx or model.language_model.args.num_hidden_layers,
        ],
    }
    layout["rank"] = rank
    _LIVE_LAYOUTS[id(result)] = (result, weakref.ref(model), layout)
    return result
