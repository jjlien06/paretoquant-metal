"""Independent expected inventory, derived from pinned JSON, never receipt rows."""

from . import harness as h

FIELDS = (
    "num_hidden_layers hidden_size vocab_size num_key_value_heads head_dim "
    "full_attention_interval linear_conv_kernel_dim linear_num_key_heads "
    "linear_key_head_dim linear_num_value_heads linear_value_head_dim"
).split()


def architecture(binding):
    from pathlib import Path

    root = Path(binding["path"])
    config = h.load_bound(root / "config.json", binding["config_sha256"], limit=1024**2)
    index = h.load_bound(
        root / "model.safetensors.index.json", binding["index_sha256"], limit=8 * 1024**2
    )
    text = config.get("text_config", config)
    h.require(
        all(type(text.get(k)) is int and text[k] > 0 for k in FIELDS),
        "Missing architecture dimensions",
    )
    h.require(text["linear_conv_kernel_dim"] > 1, "Invalid convolution dimensions")
    dtype = text.get(
        "dtype", text.get("torch_dtype", config.get("dtype", config.get("torch_dtype")))
    )
    h.require(dtype in ("float16", "bfloat16", "float32"), "Unknown architecture dtype")
    weights = index.get("weight_map")
    h.require(
        isinstance(weights, dict)
        and weights
        and all(isinstance(k, str) and isinstance(v, str) for k, v in weights.items()),
        "Missing checkpoint index",
    )
    # The index, not submitted metric inventory, must cover every original layer.
    for layer in range(text["num_hidden_layers"]):
        h.require(
            any(
                k.startswith((f"language_model.model.layers.{layer}.", f"model.layers.{layer}."))
                for k in weights
            ),
            "Checkpoint index missing original layer",
        )
    return {**{k: text[k] for k in FIELDS}, "dtype": dtype}


def layout(a, lo, hi, rank, *, small=False):
    """Native qwen3_5 modulo cache schedule, local cache-list layer numbering."""
    dtype, head_dtype = a["dtype"], "float32" if small else a["dtype"]

    def kv(layer, dt):
        return [
            {
                "layer": layer,
                "part": p,
                "shape": [1, a["num_key_value_heads"], None, a["head_dim"]],
                "dtype": dt,
            }
            for p in ("keys", "values")
        ] + [{"layer": layer, "part": "offset"}]

    target = []
    for local, original in enumerate(range(lo, hi)):
        if (original + 1) % a["full_attention_interval"] == 0:
            target.extend(kv(local, dtype))
        else:
            target.extend(
                [
                    {
                        "layer": local,
                        "part": "state.0.0",
                        "shape": [
                            1,
                            a["linear_conv_kernel_dim"] - 1,
                            2 * a["linear_num_key_heads"] * a["linear_key_head_dim"]
                            + a["linear_num_value_heads"] * a["linear_value_head_dim"],
                        ],
                        "dtype": dtype,
                    },
                    {
                        "layer": local,
                        "part": "state.0.1",
                        "shape": [
                            1,
                            a["linear_num_value_heads"],
                            a["linear_value_head_dim"],
                            a["linear_key_head_dim"],
                        ],
                        "dtype": "float32",
                    },
                ]
            )
    return {
        "target": target,
        "head": kv(0, head_dtype),
        "dtype": dtype,
        "head_dtype": head_dtype,
        "hidden_size": a["hidden_size"],
        "vocab_size": a["vocab_size"],
        "rank": rank,
        "owned_layers": [lo, hi],
    }


def expected_plan(target, small):
    """Versioned complete numerical expectations, with no pass/admission bits.

    Full width gates deliberately use explicit tokens:[1,2,3,4,5], independent
    of prompt tokenization. Actual benchmark request prompts remain unchanged.
    """
    result = {
        "schema": "qwen38-overlap-numerical-plan-v2",
        "metadata": {"full": target, "small": small},
        "atol": 1e-6,
        "rtol": 1e-5,
        "chunk_size": 256,
        "width_prompt_ids": [1, 2, 3, 4, 5],
        "forced_ids": [6, 7, 8, 9, 10, 11],
    }
    for scope, binding in (("small", small), ("full", target)):
        a = architecture(binding)
        total = a["num_hidden_layers"]
        h.require(total >= 2 and (scope != "full" or total == 64), "Checkpoint ownership differs")
        first = total - total // 2 if scope == "small" else 36
        result[scope] = {}
        for rank in (0, 1):
            lo, hi = (first, total) if rank == 0 else (0, first)
            overlap = layout(a, lo, hi, rank, small=scope == "small")
            native = layout(a, 0, total, 0, small=True) if scope == "small" else overlap
            lengths = [1, 2, 255, 256, 257, 511, 512, 513] if scope == "small" else [513]
            hidden_lengths = [256, 257, 511, 512, 513] if scope == "small" else [513]

            def cases(lengths):
                return [
                    {
                        "prompt_tokens": n,
                        "intervals": [[s, min(s + 256, n - 1)] for s in range(0, n - 1, 256)],
                        "final_target_offset": n,
                        "final_head_offset": n - 1,
                        "continuation_offsets": [[n + j, n - 1 + j] for j in (1, 2)],
                    }
                    for n in lengths
                ]

            result[scope][str(rank)] = {
                "overlap_layout": overlap,
                "native_layout": native,
                "overlap_comparison": (
                    "overlap vs serial; ordinary final singleton outside compound"
                ),
                "overlap_cases": cases(lengths),
                "hidden_comparison": (
                    "hidden-only vs original projected MTP and native cache-only prefill"
                ),
                "hidden_cases": cases(hidden_lengths),
                "width_windows": [
                    {
                        "inputs": [6 + j * 2, 7 + j * 2],
                        "target_offset": 7 + j * 2,
                        "output_width": 2,
                    }
                    for j in range(3)
                ],
                "prefix_boundaries": [
                    {"name": "same_width", "target_offset": 7, "output_width": 2},
                    *[
                        {
                            "name": f"prefix{keep}",
                            "target_offset": 5 + keep,
                            "output_width": int(keep > 0),
                            "continuation_target_offset": 6 + keep,
                            "continuation_width": 1,
                        }
                        for keep in (0, 1, 2)
                    ],
                ],
            }
    return result


def check_metric(value, shape, dtype):
    import math

    h.require(
        type(value) is dict
        and value.get("shape") == shape
        and all(type(v) is int and v > 0 for v in value["shape"])
        and value.get("dtype") == dtype
        and value.get("finite") is True
        and value.get("strict_pass") is True
        and type(value.get("exact")) is bool,
        "Numerical metric shape/dtype/finite differs",
    )
    for key in ("max_abs", "max_tolerance", "max_tolerance_ratio"):
        h.require(
            type(value.get(key)) in (int, float) and math.isfinite(value[key]),
            "Nonfinite numerical metric",
        )
    error, tolerance, ratio = (
        value[k] for k in ("max_abs", "max_tolerance", "max_tolerance_ratio")
    )
    h.require(
        0 <= error <= tolerance
        and tolerance > 0
        and 0 <= ratio <= 1
        and (error == 0 or ratio > 0)
        and ratio + max(1e-12, error / tolerance * 1e-6) >= error / tolerance
        and (not value["exact"] or error == ratio == 0),
        "Numerical tolerance contradiction",
    )


def check_parts(values, specs, offset):
    h.require(
        isinstance(values, list) and len(values) == len(specs), "Exact cache-part count differs"
    )
    expected = {(v["layer"], v["part"]): v for v in specs}
    h.require(
        all(type(v.get("layer")) is int and type(v.get("part")) is str for v in values),
        "Cache-part identity differs",
    )
    keys = [(v["layer"], v["part"]) for v in values]
    h.require(
        len(set(keys)) == len(keys) and set(keys) == set(expected),
        "Exact cache-part inventory differs",
    )
    for value in values:
        spec = expected[value["layer"], value["part"]]
        if spec["part"] == "offset":
            h.require(
                type(value.get("value")) is int
                and value["value"] == offset
                and value.get("strict_pass") is True
                and value.get("exact") is True
                and type(value.get("max_abs")) in (int, float)
                and value["max_abs"] == 0,
                "Cache offset differs",
            )
        else:
            check_metric(value, [offset if d is None else d for d in spec["shape"]], spec["dtype"])


def audit_legacy(evidence, plan, api):
    """Strengthen actual025 old width/prefix checks with every cache part/shape."""
    layout = plan["native_layout"]
    h.require(
        evidence.get("atol") == 1e-6
        and evidence.get("rtol") == 1e-5
        and evidence.get("owned_layers") == layout["owned_layers"]
        and evidence.get("passed") is True,
        "Legacy gate ownership/tolerances differ",
    )
    windows = evidence.get("windows")
    h.require(isinstance(windows, list) and len(windows) == 3, "Missing width windows")

    def row(value, offset, width, *, hidden=False):
        h.require(value.get("cache_pass") is True, "Legacy cache comparison failed")
        check_parts(value.get("caches"), layout["target"], offset)
        if width:
            h.require(value.get("argmax_equal") is True, "Legacy output argmax differs")
            check_metric(value.get("logits"), [1, width, layout["vocab_size"]], layout["dtype"])
            if hidden:
                check_metric(
                    value.get("hidden"), [1, width, layout["hidden_size"]], layout["dtype"]
                )

    for value, spec in zip(windows, plan["width_windows"], strict=True):
        h.require(value.get("inputs") == spec["inputs"], "Width input ids differ")
        row(value, spec["target_offset"], spec["output_width"])
    prefix = evidence.get("prefix_rollback")
    api("supervise").verify_prefix_evidence(
        prefix, layout["owned_layers"][1] - layout["owned_layers"][0]
    )
    row(prefix["same_width"], 7, 2, hidden=True)
    for value, keep in zip(prefix["prefixes"], (0, 1, 2), strict=True):
        row(value, 5 + keep, int(keep > 0), hidden=True)
        row(value["continuation"], 6 + keep, 1, hidden=True)
