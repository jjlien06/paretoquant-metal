"""Same-process strict width admission. Receipts are evidence, not bypass flags."""

import hashlib
import re
from pathlib import Path

ATOL, RTOL = 1e-6, 1e-5


def bind_metadata(path, config_sha256, index_sha256):
    result = {}
    for name, expected in [
        ("config.json", config_sha256),
        ("model.safetensors.index.json", index_sha256),
    ]:
        if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
            raise ValueError("Explicit trusted metadata SHA256 required")
        actual = hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Trusted metadata digest mismatch")
        result[name] = actual
    return result


def metric(a, b):
    import mlx.core as mx

    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Numerical metadata mismatch")
    x, y = a.astype(mx.float32), b.astype(mx.float32)
    finite = bool((mx.all(mx.isfinite(x)) & mx.all(mx.isfinite(y))).item())
    error = mx.abs(x - y)
    tolerance = ATOL + RTOL * mx.abs(y)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype).removeprefix("mlx.core."),
        "finite": finite,
        "max_abs": float(mx.max(error).item()) if finite else None,
        "max_tolerance": float(mx.max(tolerance).item()) if finite else None,
        "max_tolerance_ratio": float(mx.max(error / tolerance).item()) if finite else None,
        "exact": bool(mx.array_equal(a, b).item()),
        "strict_pass": finite and bool(mx.allclose(x, y, atol=ATOL, rtol=RTOL).item()),
    }


def cache_metrics(left, right):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    if len(left) != len(right):
        raise ValueError("Cache coverage")
    rows = []

    def compare(a, b, layer, name):
        if isinstance(a, mx.array):
            if not isinstance(b, mx.array):
                raise ValueError("Cache type")
            rows.append({"layer": layer, "part": name, **metric(a, b)})
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("Cache tree")
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, layer, f"{name}.{i}")
        elif type(a) is not type(b) or a != b:
            raise ValueError("Cache scalar metadata")

    for i, (a, b) in enumerate(zip(left, right)):
        if type(a) is not type(b):
            raise ValueError("Cache class")
        if isinstance(a, KVCache):
            if a.offset != b.offset:
                raise ValueError("Cache offset")
            compare(a.keys[..., : a.offset, :], b.keys[..., : b.offset, :], i, "keys")
            compare(a.values[..., : a.offset, :], b.values[..., : b.offset, :], i, "values")
            rows.append(
                {
                    "layer": i,
                    "part": "offset",
                    "value": a.offset,
                    "strict_pass": True,
                    "exact": True,
                    "max_abs": 0,
                }
            )
        else:
            compare(a.state, b.state, i, "state")
    return rows


def width_gate(
    model,
    prompt,
    forced,
    *,
    mode,
    group=None,
    eval_group_size=1,
    report=None,
    prefix_rollback=False,
):
    import mlx.core as mx
    from verifier import forward

    if not prompt or len(forced) != 6:
        raise ValueError("Three two-token windows required")
    rank = group.rank() if group is not None else 0
    left, right = model.make_cache(), model.make_cache()
    emit = report or (lambda event: None)
    for start in range(0, len(prompt) - 1, 256):
        chunk = prompt[start : min(start + 256, len(prompt) - 1)]
        for cache in (left, right):
            forward(
                model,
                mx.array([chunk], mx.int32),
                cache,
                group=group,
                logits=False,
                eval_group_size=eval_group_size,
            )
    for cache in (left, right):
        forward(
            model,
            mx.array([[prompt[-1]]], mx.int32),
            cache,
            group=group,
            logits=False,
            eval_group_size=eval_group_size,
        )
    windows = []
    for start in range(0, 6, 2):
        pair = forced[start : start + 2]
        serial = [
            forward(
                model,
                mx.array([[t]], mx.int32),
                left,
                group=group,
                all_logits=True,
                eval_group_size=eval_group_size,
            )
            for t in pair
        ]
        batched = forward(
            model,
            mx.array([pair], mx.int32),
            right,
            group=group,
            all_logits=True,
            verifier_mode=mode,
            eval_group_size=eval_group_size,
        )
        parts = cache_metrics(left, right)
        row = {"inputs": pair, "caches": parts, "cache_pass": all(p["strict_pass"] for p in parts)}
        if rank == 0:
            expected = mx.concatenate(serial, axis=1)
            row.update(
                logits=metric(expected, batched),
                sequential_argmax=mx.argmax(expected, -1).tolist(),
                batch_argmax=mx.argmax(batched, -1).tolist(),
            )
            row["argmax_equal"] = row["sequential_argmax"] == row["batch_argmax"]
        else:
            row.update(logits=None, argmax_equal=None)
        if group is not None:
            from distributed_control import Channel

            channel = Channel(group)
            if rank == 0:
                channel.send(
                    {
                        key: row[key]
                        for key in ("logits", "argmax_equal", "sequential_argmax", "batch_argmax")
                    }
                )
            else:
                row.update(channel.receive())
            row["logits_owner_rank"] = 0
        windows.append(row)
        emit({"phase": "width_gate_window", "window": start // 2, "verifier": mode, **row})
    passed = all(
        w["cache_pass"] and w["logits"]["strict_pass"] and w["argmax_equal"] for w in windows
    )
    prefix = (
        capture_gate(
            model, prompt, forced[:2], mode=mode, group=group, eval_group_size=eval_group_size
        )
        if prefix_rollback
        else None
    )
    return {
        "prefix_rollback": prefix,
        "passed": passed and (prefix is None or prefix["passed"]),
        "rank": rank,
        "mode": mode,
        "atol": ATOL,
        "rtol": RTOL,
        "windows": windows,
        "owned_layers": [
            model.model.start_idx,
            model.model.end_idx or model.language_model.args.num_hidden_layers,
        ],
    }


def capture_gate(model, prompt, pair, *, mode, group=None, eval_group_size=1):
    """Same-width observer gate FIRST, then every native singleton prefix.

    Each prefix also compares normalized hidden, full logits and continuation
    after commit; rank0 metrics are explicitly exchanged to both shard owners.
    """
    import mlx.core as mx
    from cache_state import restore, snapshot
    from prefix_rollback import Capture
    from verifier import forward

    rank = 0 if group is None else group.rank()

    def run(cache, tokens, verify=False):
        return forward(
            model,
            mx.array([tokens], mx.int32),
            cache,
            group=group,
            eval_group_size=eval_group_size,
            all_logits=True,
            return_hidden=True,
            verifier_mode=mode if verify else "native",
        )

    def compare(left, right, a, b):
        parts = cache_metrics(left, right)
        result = {"caches": parts, "cache_pass": all(p["strict_pass"] for p in parts)}
        values = {}
        if rank == 0 and a is not None:
            values = {
                "logits": metric(a[0], b[0]),
                "hidden": metric(a[1], b[1]),
                "argmax_equal": mx.argmax(a[0], -1).tolist() == mx.argmax(b[0], -1).tolist(),
            }
        if group is not None:
            from distributed_control import Channel

            channel = Channel(group)
            if rank == 0:
                channel.send(values)
            else:
                values = channel.receive()
        result.update(values)
        # Both physical owners must agree before any branch can advance.
        cache_pass = result["cache_pass"]
        if group is not None:
            if rank == 0:
                channel.send({"cache_pass": cache_pass})
                cache_pass = channel.receive()["cache_pass"] and cache_pass
            else:
                peer_pass = channel.receive()["cache_pass"]
                channel.send({"cache_pass": cache_pass})
                cache_pass = cache_pass and peer_pass
        result["passed"] = (
            cache_pass
            and all(result[k]["strict_pass"] for k in ("logits", "hidden") if k in result)
            and result.get("argmax_equal", True)
        )
        return result

    left, right = model.make_cache(), model.make_cache()
    for start in range(0, len(prompt), 256):
        for cache in (left, right):
            run(cache, prompt[start : start + 256])
    saved = snapshot(left)
    baseline = run(left, pair, True)
    cap = Capture(model, right)
    with cap:
        captured = run(right, pair, True)
    same = compare(left, right, baseline, captured)
    # Fail before using a prefix if mere instrumentation changed arithmetic.
    if not same["passed"]:
        return {"passed": False, "same_width": same, "prefixes": [], "atol": ATOL, "rtol": RTOL}
    prefixes = []
    for keep in (0, 1, 2):
        restore(left, saved)
        restore(right, saved)
        expected = None
        for token in pair[:keep]:
            expected = run(left, [token])
        cap = Capture(model, right)
        with cap:
            actual = run(right, pair, True)
        cap.commit(keep)
        selected = tuple(x[:, keep - 1 : keep] for x in actual) if rank == 0 and keep else None
        row = compare(left, right, expected, selected)
        # Prefix0 has no emitted logits; compare its next native singleton too.
        a = run(left, [pair[0]])
        b = run(right, [pair[0]])
        continuation = compare(left, right, a, b)
        row.update(keep=keep, continuation=continuation)
        row["passed"] = row["passed"] and continuation["passed"]
        prefixes.append(row)
    return {
        "passed": same["passed"] and all(p["passed"] for p in prefixes),
        "same_width": same,
        "prefixes": prefixes,
        "atol": ATOL,
        "rtol": RTOL,
    }


class Admission:
    """A live model-bound capability, minted only after physical small+target gates."""

    def __init__(
        self,
        model,
        mode,
        small,
        full,
        peer,
        *,
        prefix_rollback=False,
        hidden_only_prefill=False,
        hidden_gate_chunk_size=256,
    ):
        if (
            not small["passed"]
            or not full["passed"]
            or not peer["small"]["passed"]
            or not peer["full"]["passed"]
        ):
            raise ValueError("Width gate failed; speculation disabled")
        if small["mode"] != mode or full["mode"] != mode:
            raise ValueError("Verifier admission mismatch")
        if prefix_rollback and not all(
            (g.get("prefix_rollback") or {}).get("passed")
            for g in (small, full, peer["small"], peer["full"])
        ):
            raise ValueError("Missing same-process prefix rollback admission")
        if type(hidden_only_prefill) is not bool:
            raise ValueError("Hidden prefill opt-in must be a bool")
        if hidden_only_prefill:
            from hidden_prefill_gate import (
                SCHEMA,
                gate_lengths,
                trusted_layout,
                verify_prefill_evidence,
            )

            for scope, local in (("small", small), ("full", full)):
                width = 256 if scope == "small" else hidden_gate_chunk_size
                lengths = gate_lengths(scope, width)
                lo, hi = local["owned_layers"]
                layout = trusted_layout(
                    local.get("hidden_only_prefill"), model=model if scope == "full" else None
                )
                if layout["owned_layers"] != [lo, hi]:
                    raise ValueError("Hidden prefill cache layout ownership mismatch")
                if (
                    scope == "full"
                    and width < 256
                    and trusted_layout(
                        local["hidden_only_prefill"].get("representative_256"), model=model
                    )
                    != layout
                ):
                    raise ValueError("Hidden prefill native256 cache layout binding mismatch")
                verify_prefill_evidence(
                    local.get("hidden_only_prefill"),
                    scope=scope,
                    chunk_size=width,
                    layer_count=hi - lo,
                    expected_layout=layout,
                )
                for receipt in (local, peer[scope]):
                    hidden = receipt.get("hidden_only_prefill") or {}
                    if (
                        hidden.get("passed") is not True
                        or hidden.get("same_process") is not True
                        or hidden.get("schema") != SCHEMA
                        or hidden.get("chunk_size") != width
                        or hidden.get("prompt_lengths") != lengths
                    ):
                        raise ValueError("Missing same-process small/full/peer hidden prefill gate")
        self.hidden_only_prefill = hidden_only_prefill
        self.prefix_rollback = prefix_rollback
        self.model, self.mode, self.small, self.full, self.peer = model, mode, small, full, peer

    def require(self, model, mode, *, prefix_rollback=False, hidden_only_prefill=False):
        if self.hidden_only_prefill != hidden_only_prefill:
            raise ValueError("Hidden prefill admission belongs to another mode")
        if self.model is not model or self.mode != mode or self.prefix_rollback != prefix_rollback:
            raise ValueError("Admission belongs to another live model/mode")
