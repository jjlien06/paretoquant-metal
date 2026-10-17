"""Explicit TEST-only actor documents from real025 pure receipt helpers.

No processes, pressure probes, GPU arrays or physical receipts are produced.
"""

import json

from benchmarks.qwen38_overlap.cli import encoded


def actor_documents(api, path, sources, *, role="rank0", output=None, events=None, nonce="a" * 32):
    path.mkdir()
    actor_pid = {"rank0": 100, "rank1": 200, "server": 300}[role]
    child_pid = actor_pid + 1
    host = "local" if role == "rank0" else "remote"
    binding = dict(
        host_role=host, actor_role=role, nonce=nonce, actor_pid=actor_pid, child_pid=child_pid
    )
    clock = [100.0]
    guard = api("pressure_guard").PressureGuard(True, sampler=lambda: 1, clock=lambda: clock[0])
    clock[0] = 100.1
    guard.sample("pre_spawn", force=True)
    clock[0] = 100.5
    guard.sample("owned_child", force=True)
    clock[0] = 100.9
    guard.sample("post_child", force=True)
    pressure = guard.receipt()
    pressure.update(binding=binding, actor_started_monotonic=100.0)
    execution = dict(child_start_seconds=0.2, gate_release_seconds=0.3, child_end_seconds=0.8)
    startup = dict(
        actor_pid=actor_pid,
        nonce=nonce,
        deadline=900,
        require_normal_pressure=True,
        host_role=host,
        actor_role=role,
        actor_started_monotonic=100.0,
    )
    pids = dict(binding, pgid=child_pid, actor_started_monotonic=100.0, child_start_seconds=0.2)
    result = dict(
        binding,
        returncode=0,
        reason="completed",
        log_evidence_complete=True,
        log_cap_exceeded=False,
        require_normal_pressure=True,
        pressure=pressure,
        actor_started_monotonic=100.0,
        execution=execution,
        elapsed=1.0,
        cleanup=dict(child_reaped=True, group_gone=True, pending_os_gpu_termination=False),
    )
    for name, value in dict(
        startup=startup, pids=pids, result=result, source_sha256=sources
    ).items():
        (path / (name + ".json")).write_bytes(encoded(value))
    (path / "stdout").write_text(json.dumps(output or {"rank": 0, "requests": []}) + "\n")
    (path / "stderr").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (events or [dict(phase="process_start", rank=0, pid=child_pid)])
        )
    )
    raw = api("remote_actor").job_status(path.parent, path.name, records=True)
    context = dict(binding, actor_started_monotonic=100.0, execution=execution, elapsed=1.0)
    return raw, context


def symbolic_gate(api, layout, *, scope, overlap):
    """TEST grammar values; real025 fixture metric/parts + gate length helpers.

    Producer-required physical_gpu is merely schema input in this TEST namespace;
    this helper never creates live gate capability or measured numeric evidence.
    """
    from copy import deepcopy

    helper = api("test_overlap_gate")

    def metric(shape, dtype):
        return {**helper.metric(shape), "dtype": dtype}

    def parts(specs, offset):
        values = helper.parts(specs, offset)
        for value, spec in zip(values, specs, strict=True):
            if spec["part"] != "offset":
                value["dtype"] = spec["dtype"]
        return values

    def row(offset, head_offset, width=1, output=False):
        result = dict(
            passed=True,
            cache_pass=True,
            caches=parts(layout["target"], offset),
            native_caches=parts(layout["target"], offset),
            head_caches=parts(layout["head"], head_offset) if head_offset else [],
            head_offset=head_offset,
            hidden=metric([1, width, layout["hidden_size"]], layout["dtype"]),
        )
        if output:
            result.update(
                argmax_equal=True,
                native_argmax_equal=True,
                logits=metric([1, 1, layout["vocab_size"]], layout["dtype"]),
            )
            result.update(
                native_hidden=deepcopy(result["hidden"]), native_logits=deepcopy(result["logits"])
            )
        return result

    lengths = (
        api("overlap_gate").lengths(scope)
        if overlap
        else api("hidden_prefill_gate").gate_lengths(scope, 256)
    )
    cases = []
    for n in lengths:
        prompt = [1 + j % min(100, layout["vocab_size"] - 1) for j in range(n)]
        chunks = [
            dict(row(e, e, e - s), start=s, end=e, shifted_tokens=prompt[s + 1 : e + 1])
            for s, e in [(s, min(s + 256, n - 1)) for s in range(0, n - 1, 256)]
        ]
        final = dict(row(n, n - 1, output=True), token=prompt[-1])
        continuation = [
            dict(
                row(n + j, n - 1 + j, output=True),
                token=(5 + j) % layout["vocab_size"],
                head_logits=metric([1, 1, layout["vocab_size"]], layout["head_dtype"]),
            )
            for j in (1, 2)
        ]
        cases.append(
            dict(
                passed=True, prompt_tokens=n, chunks=chunks, final=final, continuation=continuation
            )
        )
    return dict(
        schema=api("overlap_gate").SCHEMA if overlap else api("hidden_prefill_gate").SCHEMA,
        passed=True,
        same_process=True,
        physical_gpu=True,
        scope=scope,
        modes=["serial", "overlap"],
        target_forward_mode="native",
        chunk_size=256,
        prompt_lengths=lengths,
        rank=layout["rank"],
        owned_layers=layout["owned_layers"],
        hidden_size=layout["hidden_size"],
        vocab_size=layout["vocab_size"],
        atol=1e-6,
        rtol=1e-5,
        eval_group_size=32,
        cases=cases,
    )


def symbolic_legacy(api, plan, mode, binding):
    """Width/prefix grammar, not a numeric measurement."""
    layout = plan["native_layout"]
    helper = api("test_overlap_gate")

    def row(offset, width):
        caches = helper.parts(layout["target"], offset)
        for value, spec in zip(caches, layout["target"], strict=True):
            if spec["part"] != "offset":
                value["dtype"] = spec["dtype"]
        return dict(
            passed=True,
            cache_pass=True,
            caches=caches,
            argmax_equal=True,
            logits={
                **helper.metric([1, width or 1, layout["vocab_size"]]),
                "dtype": layout["dtype"],
            },
            hidden={
                **helper.metric([1, width or 1, layout["hidden_size"]]),
                "dtype": layout["dtype"],
            },
            sequential_argmax=[[1] * width],
            batch_argmax=[[1] * width],
        )

    prefix = dict(
        passed=True,
        atol=1e-6,
        rtol=1e-5,
        same_width=row(7, 2),
        prefixes=[
            dict(row(5 + keep, int(keep > 0)), keep=keep, continuation=row(6 + keep, 1))
            for keep in (0, 1, 2)
        ],
    )
    return dict(
        passed=True,
        mode=mode,
        owned_layers=layout["owned_layers"],
        atol=1e-6,
        rtol=1e-5,
        metadata_sha256=binding,
        prefix_rollback=prefix,
        windows=[
            dict(row(spec["target_offset"], 2), inputs=spec["inputs"])
            for spec in plan["width_windows"]
        ],
    )


def symbolic_request(api, rank, spec, prompt, layout, ordinal):
    """Use actual025 Worker/Command and pipeline/credit codec on symbolic seams."""
    from types import SimpleNamespace

    nonce = f"{ordinal + 1000:032x}"
    events = []
    command_type = api("rank_control").Command
    pipeline = api("overlap_prefill")

    def forward(tokens, *, progress, receive_stage=None, logits=False):
        if rank == 0:
            progress(dict(phase="handoff_receive_before"))
            progress(
                dict(
                    phase="handoff_receive_done",
                    shape=[1, len(tokens), layout["hidden_size"]],
                    dtype=layout["dtype"],
                )
            )
            if receive_stage:
                receive_stage()
        else:
            progress(dict(phase="embedding_before"))
            progress(dict(phase="embedding_done"))
        lo, hi = layout["owned_layers"]
        for start in range(lo, hi, 32):
            end = min(start + 32, hi)
            progress(dict(phase="block_before", start_idx=start, end_idx=end))
            progress(dict(phase="block_done", start_idx=start, end_idx=end))
            for layer in range(start, end):
                progress(dict(phase="layer_done", layer_index=layer))
        if rank == 0:
            progress(dict(phase="head_done" if logits else "hidden_norm_before"))
            if not logits:
                progress(dict(phase="hidden_norm_done"))
            return None, "TEST_SYMBOLIC_HIDDEN"
        progress(dict(phase="handoff_send_before"))
        progress(
            dict(
                phase="handoff_send_done",
                shape=[1, len(tokens), layout["hidden_size"]],
                dtype=layout["dtype"],
            )
        )

    class Target:
        def dispatch(self, op, tokens):
            if op in ("forward", "prefill_hidden"):
                forward(tokens, progress=events.append, logits=op == "forward")

        def dispatch_pipeline(self, command):
            if rank == 0:
                pipeline.run_rank0_pipeline(
                    command,
                    SimpleNamespace(send=lambda _: None),
                    forward,
                    lambda *args: None,
                    events.append,
                )
                events.append(
                    dict(
                        phase="pipeline_drained",
                        request_nonce=nonce,
                        command_sequence=command.sequence,
                        prefill_mode="overlap",
                        pending_handoffs=0,
                        rank=rank,
                    )
                )
            else:
                credits = [pipeline.credit(command, *chunk) for chunk in pipeline.chunks(prompt)]
                pipeline.run_rank1_pipeline(
                    command, SimpleNamespace(receive=lambda: credits.pop(0)), forward, events.append
                )

    worker = api("rank_control").Worker(
        Target().dispatch, overlapped_prefill=spec["prefill_mode"] == "overlap"
    )
    commands = [("reset", [])]
    if spec["prefill_mode"] == "overlap" and len(prompt) > 1:
        commands.append(("prefill_pipeline", prompt))
    else:
        commands.extend(
            ("prefill_hidden", prompt[s : min(s + 256, len(prompt) - 1)])
            for s in range(0, len(prompt) - 1, 256)
        )
    commands.extend([("forward", [prompt[-1]]), ("finish", [])])
    for sequence, (op, tokens) in enumerate(commands):
        command = command_type(nonce, sequence, op, tokens)
        ack = worker.handle(command)
        events.append(
            dict(
                phase="rank_action",
                request=command.request,
                sequence=sequence,
                op=op,
                tokens=tokens,
                ok=ack["ok"],
            )
        )
    common = {
        k: spec[k]
        for k in (
            "id",
            "case_id",
            "method",
            "warmup",
            "max_tokens",
            "prefill_mode",
            "source_request_id",
        )
    }
    common.update(
        repeat=ordinal,
        sample_repeat=spec["repeat"],
        eval_group_size=32,
        prompt_sha256=__import__("hashlib").sha256(spec["prompt"].encode()).hexdigest(),
        rank=rank,
    )
    result = dict(
        common,
        local_rank=rank,
        prompt_tokens=len(prompt),
        prompt_token_ids=prompt,
        request_nonce=nonce,
        cache_epoch=nonce,
        token_ids=[248044],
        finish_reason="stop",
        text="",
        verifier_mode=("native", "slot-qmv")[rank],
        first_token_seconds=1.0,
        wall_seconds=2.0,
        wall_tokens_per_second=0.5,
        post_first_tokens_per_second=None,
        local_rank_wall_seconds=2.0 + rank,
        local_peak_mlx_bytes=100 + rank,
        accepted_drafts=0,
        verification_rounds=0,
        teacher_forwards=0,
        sidecar_resident_on_rank0=True,
    )
    result.pop("rank")
    for event in events:
        event.update(common)
    return result, [
        dict(common, phase="request_before", prompt_token_ids=prompt),
        dict(common, phase="request_start"),
        *events,
        dict(common, phase="request_done", request_result=result),
    ]
