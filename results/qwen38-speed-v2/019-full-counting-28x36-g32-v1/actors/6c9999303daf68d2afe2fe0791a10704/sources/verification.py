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
    return {
        "shape": list(a.shape),
        "max_abs": float(mx.max(mx.abs(x - y)).item()),
        "exact": bool(mx.array_equal(a, b).item()),
        "strict_pass": bool(mx.allclose(x, y, atol=ATOL, rtol=RTOL).item()),
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


def width_gate(model, prompt, forced, *, mode, group=None, eval_group_size=1, report=None):
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
    return {
        "passed": passed,
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


class Admission:
    """A live model-bound capability, minted only after physical small+target gates."""

    def __init__(self, model, mode, small, full, peer):
        if (
            not small["passed"]
            or not full["passed"]
            or not peer["small"]["passed"]
            or not peer["full"]["passed"]
        ):
            raise ValueError("Width gate failed; speculation disabled")
        if small["mode"] != mode or full["mode"] != mode:
            raise ValueError("Verifier admission mismatch")
        self.model, self.mode, self.small, self.full, self.peer = model, mode, small, full, peer

    def require(self, model, mode):
        if self.model is not model or self.mode != mode:
            raise ValueError("Admission belongs to another live model/mode")
