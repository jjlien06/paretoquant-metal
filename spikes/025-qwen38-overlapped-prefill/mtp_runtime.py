"""Bounded real loaded-shard MTP orchestration; GPU selection is caller-owned."""

import gc
import time

from pressure_guard import policy_name
from verification import Admission, bind_metadata, width_gate


def small_gate(args, eval_groups, report, *, group=None):
    import mlx.core as mx
    from mlx_lm import load

    binding = bind_metadata(args.small_model, args.small_config_sha256, args.small_index_sha256)
    model, _ = load(
        args.small_model,
        trust_remote_code=False,
        tokenizer_config={"trust_remote_code": False, "local_files_only": True},
    )
    small_head = None
    if getattr(args, "hidden_only_prefill", False):
        import mlx.nn as nn
        from mtp_head import BoundHead, MTPHead

        # Small checkpoint has no trusted target-sized sidecar. Exercise the
        # exact native head recurrence using a deterministic quantized fixture;
        # full admission below always uses the actual trusted loaded sidecar.
        mx.random.seed(2401)
        module = MTPHead(model.language_model.args)
        nn.quantize(module, group_size=64, bits=4)
        module.eval()
        mx.eval(module.parameters())
        output = (
            model.model.embed_tokens.as_linear
            if model.language_model.args.tie_word_embeddings
            else model.language_model.lm_head
        )
        small_head = BoundHead(module, model.model.embed_tokens, output)
    receipts = {}
    for size in sorted(set(eval_groups)):
        receipts[size] = width_gate(
            model,
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            mode=args.verifier_mode,
            eval_group_size=size,
            report=report,
            prefix_rollback=getattr(args, "prefix_rollback", False),
        )
        if getattr(args, "hidden_only_prefill", False):
            from hidden_prefill_gate import gate_lengths, prefill_gate

            hidden = prefill_gate(
                model,
                small_head,
                prompt_lengths=gate_lengths("small", 256),
                chunk_size=256,
                eval_group_size=size,
                report=report,
            )
            receipts[size]["hidden_only_prefill"] = hidden
            receipts[size]["passed"] = receipts[size]["passed"] and hidden["passed"]
        receipts[size]["metadata_sha256"] = binding
        receipts[size]["device"] = mx.device_info()
        if not receipts[size]["passed"]:
            raise ValueError("Physical small checkpoint width admission failed")
    if getattr(args, "overlapped_prefill", False):
        from overlap_gate import pipeline_gate

        if group is None or group.size() != 2:
            raise ValueError("Overlap small gate requires physical two-rank group")
        count = model.language_model.args.num_hidden_layers
        if count < 2:
            raise ValueError("Small pipeline requires two nonempty shards")
        model.model.pipeline(group, split=[count // 2, count - count // 2])
        for size in sorted(set(eval_groups)):
            overlap_small = pipeline_gate(
                model,
                small_head if group.rank() == 0 else None,
                group=group,
                scope="small",
                eval_group_size=size,
                report=report,
            )
            receipts[size]["overlapped_prefill"] = overlap_small
            receipts[size]["passed"] &= overlap_small["passed"]
    del model
    gc.collect()
    mx.clear_cache()
    return receipts


def run_loaded(
    model,
    tokenizer,
    ids,
    args,
    *,
    group,
    small,
    eval_groups,
    report,
    head=None,
    config=None,
    request_specs=None,
):
    from distributed_control import Channel, serve_request
    from mlx_backend import MTPBackend, Rank1Target
    from mtp_head import load_trusted_head
    from speculative import generate

    rank = 0 if group is None else group.rank()
    channel = None if group is None else Channel(group)
    hidden_only = getattr(args, "hidden_only_prefill", False)
    overlap_enabled = getattr(args, "overlapped_prefill", False)
    hidden_width = getattr(args, "hidden_gate_chunk_size", 256)
    if hidden_only and rank == 0 and head is None:
        head = load_trusted_head(
            args.sidecar, model, target_config=config, target_revision=args.target_revision
        )
    gates = {}
    admissions = {}
    for size in sorted(set(eval_groups)):
        full = width_gate(
            model,
            ids,
            [6, 7, 8, 9, 10, 11],
            mode=args.verifier_mode,
            group=group,
            eval_group_size=size,
            report=report,
            prefix_rollback=getattr(args, "prefix_rollback", False),
        )
        if hidden_only:
            from hidden_prefill_gate import full_gate

            full["hidden_only_prefill"] = full_gate(
                model,
                head,
                chunk_size=hidden_width,
                group=group,
                eval_group_size=size,
                report=report,
            )
            full["passed"] = full["passed"] and full["hidden_only_prefill"]["passed"]
        if overlap_enabled:
            from overlap_gate import pipeline_gate

            full["overlapped_prefill"] = pipeline_gate(
                model,
                head,
                group=group,
                scope="full",
                eval_group_size=size,
                report=report,
            )
            full["passed"] &= full["overlapped_prefill"]["passed"]
        local = {
            "small": {
                "passed": small[size]["passed"],
                "mode": args.verifier_mode,
                "prefix_rollback": {
                    "passed": (small[size].get("prefix_rollback") or {}).get("passed", False)
                },
            },
            "full": {
                "passed": full["passed"],
                "mode": args.verifier_mode,
                "prefix_rollback": {
                    "passed": (full.get("prefix_rollback") or {}).get("passed", False)
                },
            },
        }
        if hidden_only:
            for scope, receipt in (("small", small[size]), ("full", full)):
                evidence = receipt.get("hidden_only_prefill") or {}
                local[scope]["hidden_only_prefill"] = {
                    k: evidence.get(k)
                    for k in ("passed", "same_process", "schema", "chunk_size", "prompt_lengths")
                }
        if overlap_enabled:
            for scope, receipt in (("small", small[size]), ("full", full)):
                local[scope]["overlapped_prefill"] = {
                    k: receipt["overlapped_prefill"][k]
                    for k in (
                        "schema",
                        "passed",
                        "same_process",
                        "physical_gpu",
                        "scope",
                        "modes",
                        "chunk_size",
                        "prompt_lengths",
                        "eval_group_size",
                    )
                }
        if channel is None:
            peer = local
        elif rank == 0:
            channel.send(local)
            peer = channel.receive()
        else:
            peer = channel.receive()
            channel.send(local)
        admissions[size] = Admission(
            model,
            args.verifier_mode,
            small[size],
            full,
            peer,
            hidden_only_prefill=hidden_only,
            hidden_gate_chunk_size=hidden_width,
            overlapped_prefill=overlap_enabled,
            eval_group_size=size,
            prefix_rollback=getattr(args, "prefix_rollback", False),
        )
        gates[size] = {"small": small[size], "full": full, "peer": peer}
    request_modes = [
        {k: row[k] for k in ("id", "prefill_mode", "source_request_id")}
        for row in (request_specs or [])
        if "prefill_mode" in row
    ]
    verification = {
        "passed": True,
        "pressure_policy": policy_name(args),
        "require_normal_pressure": getattr(args, "require_normal_pressure", False),
        "mode": args.verifier_mode,
        "gate_only": args.verification_gate_only,
        "groups": gates,
        "same_loaded_shards": True,
        "request_prefill_modes": request_modes,
        "overlapped_prefill": overlap_enabled,
        "prefill_modes": ["serial", "overlap"] if overlap_enabled else ["serial"],
        "hidden_only_prefill": hidden_only,
        "hidden_gate_chunk_size": hidden_width,
        "prefix_rollback": getattr(args, "prefix_rollback", False),
        "target_revision": args.target_revision,
    }
    report(
        {
            "phase": "numerical_admission_done",
            "pressure_policy": policy_name(args),
            "require_normal_pressure": getattr(args, "require_normal_pressure", False),
            "request_prefill_modes": request_modes,
            "hidden_only_prefill": hidden_only,
            "overlapped_prefill": overlap_enabled,
            "prefill_modes": ["serial", "overlap"] if overlap_enabled else ["serial"],
            "hidden_gate_chunk_size": hidden_width,
            "prefix_rollback": getattr(args, "prefix_rollback", False),
            "verifier": args.verifier_mode,
            "gate_only": args.verification_gate_only,
        }
    )
    if args.verification_gate_only:
        return {"requests": [], "verification": verification}
    if rank == 0 and head is None:
        head = load_trusted_head(
            args.sidecar, model, target_config=config, target_revision=args.target_revision
        )
    if request_specs is not None:
        from workload_runtime import run_requests

        return {
            "requests": run_requests(
                model,
                tokenizer,
                request_specs,
                args,
                group=group,
                channel=channel,
                head=head,
                admissions=admissions,
                eval_groups=eval_groups,
                report=report,
            ),
            "verification": verification,
            "sidecar_resident_on_rank0": True,
        }
    requests = []
    for repeat, size in enumerate(eval_groups):

        def progress(event):
            report({**event, "repeat": repeat, "eval_group_size": size})

        progress({"phase": "request_before"})
        if rank == 0:
            backend = MTPBackend(
                model,
                head,
                group=group,
                peer=channel,
                eval_group_size=size,
                admission=admissions[size],
                verifier_mode=args.verifier_mode,
                prefix_rollback=getattr(args, "prefix_rollback", False),
                hidden_only_prefill=getattr(args, "hidden_only_prefill", False),
                progress=progress,
            )
            began = time.perf_counter()
            result = generate(
                backend, ids, max_tokens=args.max_tokens, eos_ids=tokenizer.eos_token_ids
            )
            backend.finish()
            elapsed = time.perf_counter() - began
            first = backend.first_token_at - began
            result.update(
                repeat=repeat,
                eval_group_size=size,
                prompt_tokens=len(ids),
                wall_seconds=elapsed,
                wall_tokens_per_second=len(result["token_ids"]) / elapsed,
                first_token_seconds=first,
                post_first_tokens_per_second=(len(result["token_ids"]) - 1) / (elapsed - first)
                if len(result["token_ids"]) > 1
                else None,
                timing_convention=(
                    "EOS included; fresh-cache request, setup/gates/head load excluded; "
                    "finish ACK included"
                ),
                text=tokenizer.decode(result["token_ids"], skip_special_tokens=True),
                numerical_admission="same-process-small-and-full-strict-passed",
                verifier_mode=args.verifier_mode,
            )
            if channel is not None:
                channel.send({"request_result": result})
        else:
            target = Rank1Target(
                model,
                group,
                eval_group_size=size,
                admission=admissions[size],
                verifier_mode=args.verifier_mode,
                prefix_rollback=getattr(args, "prefix_rollback", False),
                hidden_only_prefill=getattr(args, "hidden_only_prefill", False),
                progress=progress,
            )
            serve_request(channel, target, progress)
            result = channel.receive()["request_result"]
        requests.append(result)
        progress({"phase": "request_done"})
    return {"requests": requests, "verification": verification}
