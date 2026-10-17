"""Fresh physical two-rank serial/compound gate and detached strict verifier.

No import-time MLX. JSON dictionaries cannot grant live scheduling capability.
GPU execution is owner-only; CPU validators never call pipeline_gate.
"""

import math
import weakref
from copy import deepcopy

from verification import ATOL, RTOL

SCHEMA = "qwen38-prefill-overlap-gate-v1"
_LIVE = {}


def lengths(scope):
    if scope == "small":
        return [1, 2, 255, 256, 257, 511, 512, 513]
    if scope == "full":
        return [513]
    raise ValueError("Unknown pipeline gate scope")


def require_live_overlap(receipt, model, eval_group_size):
    record = _LIVE.get(id(receipt))
    if (
        record is None
        or record[0] is not receipt
        or (model is not None and record[1]() is not model)
        or record[2] != eval_group_size
        or receipt.get("passed") is not True
    ):
        raise ValueError("Missing fresh live physical overlap admission")
    return deepcopy(record[3])


def verify_overlap_gate(evidence, *, expected_layout, scope, eval_group_size):
    from overlap_evidence import require

    e = evidence or {}
    layout = expected_layout
    require(
        e.get("schema") == SCHEMA
        and e.get("passed") is True
        and e.get("same_process") is True
        and e.get("physical_gpu") is True
        and e.get("scope") == scope
        and e.get("modes") == ["serial", "overlap"]
        and e.get("chunk_size") == 256
        and e.get("prompt_lengths") == lengths(scope)
        and e.get("atol") == ATOL
        and e.get("rtol") == RTOL
        and type(e.get("eval_group_size")) is int
        and e["eval_group_size"] == eval_group_size
        and type(e.get("rank")) is int
        and e["rank"] == layout["rank"]
        and e.get("owned_layers") == layout["owned_layers"],
        "Incomplete strict overlap gate",
    )

    def metric(m, shape, dtype):
        require(
            type(m) is dict
            and m.get("strict_pass") is True
            and m.get("finite") is True
            and m.get("shape") == shape
            and type(m.get("shape")) is list
            and all(type(d) is int and d > 0 for d in m["shape"])
            and m.get("dtype") == dtype
            and type(m.get("exact")) is bool
        )
        for k in ("max_abs", "max_tolerance", "max_tolerance_ratio"):
            require(type(m.get(k)) in (int, float) and math.isfinite(m[k]))
        require(
            0 <= m["max_abs"] <= m["max_tolerance"]
            and m["max_tolerance"] > 0
            and 0 <= m["max_tolerance_ratio"] <= 1
        )
        require(
            (m["max_abs"] == 0 or m["max_tolerance_ratio"] > 0)
            and m["max_tolerance_ratio"] + max(1e-12, m["max_abs"] / m["max_tolerance"] * 1e-6)
            >= m["max_abs"] / m["max_tolerance"]
        )
        require(not m["exact"] or m["max_abs"] == m["max_tolerance_ratio"] == 0)

    def parts(values, offset, head=False):
        expected = layout["head" if head else "target"]
        require(type(values) is list and len(values) == len(expected))
        require(
            all(
                type(p) is dict and type(p.get("layer")) is int and type(p.get("part")) is str
                for p in values
            )
        )
        keys = [(p["layer"], p["part"]) for p in values]
        inventory = {(p["layer"], p["part"]): p for p in expected}
        require(len(set(keys)) == len(keys) and set(keys) == set(inventory))
        for p in values:
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
                spec = inventory[p["layer"], p["part"]]
                metric(p, [offset if d is None else d for d in spec["shape"]], spec["dtype"])

    require(type(e.get("cases")) is list and len(e["cases"]) == len(lengths(scope)))
    for case, length in zip(e["cases"], lengths(scope)):
        require(case.get("prompt_tokens") == length and type(case.get("prompt_tokens")) is int)
        prompt = [1 + i % min(100, layout["vocab_size"] - 1) for i in range(length)]
        intervals = [(s, min(s + 256, length - 1)) for s in range(0, length - 1, 256)]
        require([(c.get("start"), c.get("end")) for c in case.get("chunks", [])] == intervals)

        def row(r, offset, head_offset, hidden_width, output=False):
            require(r.get("passed") is True)
            parts(r.get("caches"), offset)
            # Empty one-token head prefix has no initialized arrays until continuation.
            if head_offset:
                parts(r.get("head_caches"), head_offset, True)
            else:
                require(r.get("head_caches") == [])
            metric(r.get("hidden"), [1, hidden_width, layout["hidden_size"]], layout["dtype"])
            if output:
                require(r.get("argmax_equal") is True)
                metric(r.get("logits"), [1, 1, layout["vocab_size"]], layout["dtype"])

        for r, (start, end) in zip(case["chunks"], intervals):
            require(type(r.get("start")) is int and type(r.get("end")) is int)
            require(
                type(r.get("shifted_tokens")) is list
                and all(type(t) is int for t in r["shifted_tokens"])
                and r["shifted_tokens"] == prompt[start + 1 : end + 1]
            )
            row(r, end, end, end - start)
        row(case.get("final", {}), length, length - 1, 1, True)
        require(len(case.get("continuation", [])) == 2)
        for i, r in enumerate(case["continuation"], 1):
            row(r, length + i, length - 1 + i, 1, True)
            metric(r.get("head_logits"), [1, 1, layout["vocab_size"]], layout["head_dtype"])
    return True


def metadata_layout(options, *, scope, rank, native_small=False):
    """Derive full cache-part inventory from separately hash-bound JSON only."""
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from hidden_prefill_gate import architecture_layout
    from verification import bind_metadata

    path = options.small_model_local if scope == "small" else options.local_metadata
    config_sha = options.small_config_sha256 if scope == "small" else options.target_config_sha256
    index_sha = options.small_index_sha256 if scope == "small" else options.target_index_sha256
    bind_metadata(path, config_sha, index_sha)
    config = json.loads((Path(path) / "config.json").read_text())
    text = deepcopy(config.get("text_config", config))
    fields = (
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "num_key_value_heads",
        "head_dim",
        "full_attention_interval",
        "linear_conv_kernel_dim",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
    )
    if any(type(text.get(key)) is not int or text[key] < 1 for key in fields):
        raise ValueError("Missing trusted native overlap cache dimensions")
    total = text["num_hidden_layers"]
    if scope == "small":
        first = total - total // 2
    elif scope == "full" and total == sum(options.split):
        first = options.split[1]
    else:
        raise ValueError("Trusted overlap ownership mismatch")
    lo, hi = (first, total) if rank == 0 else (0, first)
    if native_small and scope == "small":
        lo, hi, rank = 0, total, 0
    dtype = text.get(
        "dtype", text.get("torch_dtype", config.get("dtype", config.get("torch_dtype")))
    )
    classes = [
        "KVCache" if (i + 1) % text["full_attention_interval"] == 0 else "ArraysCache"
        for i in range(lo, hi)
    ]
    layout = architecture_layout(
        SimpleNamespace(**text),
        classes,
        dtype=dtype,
        head_dtype="float32" if scope == "small" else dtype,
    )
    layout.update(rank=rank, owned_layers=[lo, hi])
    return layout


def pipeline_gate(model, head, *, group, scope, eval_group_size=32, report=None):
    """Exercise real MTPBackend.fresh/Worker/Channel on both physical owners.

    Retains only bounded short-gate snapshots, never request-time activation queues.
    Existing width/prefix/hidden/native256 gates remain independently mandatory.
    """
    import mlx.core as mx
    from cache_state import restore, snapshot
    from distributed_control import Channel, serve_request
    from hidden_prefill_gate import _native_layout
    from mlx_backend import MTPBackend, Rank1Target
    from mlx_lm.models.cache import KVCache
    from phased_decode import ownership
    from verification import Admission, cache_metrics, metric

    if mx.default_device() != mx.gpu or group is None or group.size() != 2:
        raise ValueError("Pipeline capability requires physical GPU and two-rank group")
    rank, size = ownership(model, group)
    if (rank == 0) != (head is not None):
        raise ValueError("Head must be local to rank0")
    emit = report or (lambda event: None)
    channel = Channel(group)

    class GateAdmission(Admission):
        def require(self, live, mode, **kwargs):
            if live is not model:
                raise ValueError("Gate live model mismatch")

        def require_overlap(self, live, width):
            if live is not model or width != eval_group_size:
                raise ValueError("Gate model/group mismatch")

    # Private, short-lived permit is never returned or minted from JSON/CPU fixtures.
    permit = object.__new__(GateAdmission)
    cases = []
    layout = None
    for length in lengths(scope):
        vocab = model.language_model.args.vocab_size
        prompt = [1 + i % min(100, vocab - 1) for i in range(length)]
        baseline = []
        rows = []
        backend = None
        target = None

        def record(output=None, head_cache=None, head_scores=None):
            local = backend.cache if rank == 0 else target.cache
            saved = snapshot(local)
            hs = snapshot([head_cache]) if head_cache is not None and head_cache.offset else None
            if mode == "serial":
                baseline.append((saved, hs, output, head_scores))
                return
            reference, reference_head, old_output, old_head_scores = baseline.pop(0)
            scratch = model.make_cache()
            restore(scratch, reference)
            values = {"caches": cache_metrics(scratch, local)}
            if rank == 0:
                values["hidden"] = metric(old_output[1], output[1])
                if output[0] is not None:
                    values.update(
                        logits=metric(old_output[0], output[0]),
                        argmax_equal=mx.argmax(old_output[0], -1).tolist()
                        == mx.argmax(output[0], -1).tolist(),
                    )
                values["head_caches"] = []
                if reference_head is not None:
                    hc = KVCache()
                    restore([hc], reference_head)
                    values["head_caches"] = cache_metrics([hc], [head_cache])
                if head_scores is not None:
                    values["head_logits"] = metric(old_head_scores, head_scores)
            values["passed"] = all(p["strict_pass"] for p in values["caches"])
            if rank == 0:
                values["passed"] &= all(p["strict_pass"] for p in values["head_caches"])
                values["passed"] &= all(
                    v["strict_pass"] for v in values.values() if type(v) is dict
                )
                values["passed"] &= values.get("argmax_equal", True)
            rows.append(values)

        class ObservedHead:
            def forward(self, hidden, tokens, cache, *, position, logits=True):
                out = head.forward(hidden, tokens, cache, position=position, logits=logits)
                if not logits and tokens.shape[1] >= 1:
                    record((None, hidden), cache)
                return out

        for mode in ("serial", "overlap"):
            if rank == 0:
                backend = MTPBackend(
                    model,
                    ObservedHead(),
                    group=group,
                    peer=channel,
                    eval_group_size=eval_group_size,
                    admission=permit,
                    hidden_only_prefill=True,
                    prefill_mode=mode,
                )
                backend._gate_observe = True
                # fresh() includes ordinary final singleton AND its ACK, outside compound.
                _, hidden = backend.fresh(prompt)
                # Capture final full logits using the actual forward adapter, without re-running it.
                # _command observer retains only the final evaluated scores for this gate.
                final_output = backend._gate_last_output
                record(final_output, backend.mtp_cache)
                for token in (6 % vocab, 7 % vocab):
                    hs = head.forward(
                        hidden,
                        mx.array([[token]], mx.int32),
                        backend.mtp_cache,
                        position=backend.mtp_cache.offset,
                    )
                    output = backend._command("forward", [token])
                    hidden = output[1]
                    record(output, backend.mtp_cache, hs)
                backend.finish()
            else:
                target = Rank1Target(
                    model,
                    group,
                    eval_group_size=eval_group_size,
                    admission=permit,
                    hidden_only_prefill=True,
                    prefill_mode=mode,
                )

                def progress(event):
                    if (
                        event.get("phase") == "pipeline_send_evaluated"
                        or event.get("phase") == "rank_action"
                        and event.get("op") in ("prefill_hidden", "forward")
                    ):
                        record()

                target.progress = progress
                serve_request(channel, target, progress)
        if baseline:
            raise ValueError("Serial/compound gate boundary count differs")
        count = len(range(0, length - 1, 256))
        for r, (s, e) in zip(
            rows[:count], [(s, min(s + 256, length - 1)) for s in range(0, length - 1, 256)]
        ):
            r.update(start=s, end=e, shifted_tokens=prompt[s + 1 : e + 1])
        case = {
            "prompt_tokens": length,
            "chunks": rows[:count],
            "final": rows[count],
            "continuation": rows[count + 1 :],
        }
        local_pass = all(r["passed"] for r in rows)
        # Transfer only compact rank0 output/head metrics, not owned-target cache trees.
        if rank == 0:
            shared = deepcopy(case)
            for r in shared["chunks"] + [shared["final"]] + shared["continuation"]:
                r.pop("caches")
            channel.send({"case": shared, "passed": local_pass})
            local_pass = channel.receive()["passed"] and local_pass
            dtypes = {
                "dtype": str(hidden.dtype).removeprefix("mlx.core."),
                "head_dtype": str(backend.mtp_cache.keys.dtype).removeprefix("mlx.core."),
            }
            channel.send(dtypes)
        else:
            shared = channel.receive()
            channel.send({"passed": local_pass})
            local_pass &= shared["passed"]
            for local, global_row in zip(
                rows,
                shared["case"]["chunks"]
                + [shared["case"]["final"]]
                + shared["case"]["continuation"],
            ):
                local.update({k: v for k, v in global_row.items() if k != "passed"})
                local["passed"] &= global_row["passed"]
            dtypes = channel.receive()
        layout = _native_layout(
            model,
            backend.cache if rank == 0 else target.cache,
            backend.mtp_cache if rank == 0 else None,
            **dtypes,
        )
        layout["rank"] = rank
        case["passed"] = local_pass
        cases.append(case)
        emit(
            {
                "phase": "overlap_gate_case",
                "scope": scope,
                "prompt_tokens": length,
                "passed": local_pass,
                "modes": ["serial", "overlap"],
                "chunk_size": 256,
            }
        )
    result = {
        "schema": SCHEMA,
        "passed": all(c["passed"] for c in cases),
        "same_process": True,
        "physical_gpu": True,
        "scope": scope,
        "modes": ["serial", "overlap"],
        "chunk_size": 256,
        "prompt_lengths": lengths(scope),
        "rank": rank,
        "owned_layers": layout["owned_layers"],
        "atol": ATOL,
        "rtol": RTOL,
        "eval_group_size": eval_group_size,
        "cases": cases,
    }
    verify_overlap_gate(
        result, expected_layout=layout, scope=scope, eval_group_size=eval_group_size
    )
    _LIVE[id(result)] = (result, weakref.ref(model), eval_group_size, deepcopy(layout))
    return result
