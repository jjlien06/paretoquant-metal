"""Bounded real loaded-shard MTP orchestration; GPU selection is caller-owned."""

import gc
import time

from verification import Admission, bind_metadata, width_gate


def small_gate(args, eval_groups, report):
    import mlx.core as mx
    from mlx_lm import load

    binding = bind_metadata(args.small_model, args.small_config_sha256, args.small_index_sha256)
    model, _ = load(
        args.small_model,
        trust_remote_code=False,
        tokenizer_config={"trust_remote_code": False, "local_files_only": True},
    )
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
        receipts[size]["metadata_sha256"] = binding
        receipts[size]["device"] = mx.device_info()
        if not receipts[size]["passed"]:
            raise ValueError("Physical small checkpoint width admission failed")
    del model
    gc.collect()
    mx.clear_cache()
    return receipts


def run_loaded(
    model, tokenizer, ids, args, *, group, small, eval_groups, report, head=None, config=None
):
    from distributed_control import Channel, serve_request
    from mlx_backend import MTPBackend, Rank1Target
    from mtp_head import load_trusted_head
    from speculative import generate

    rank = 0 if group is None else group.rank()
    channel = None if group is None else Channel(group)
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
            prefix_rollback=getattr(args, "prefix_rollback", False),
        )
        gates[size] = {"small": small[size], "full": full, "peer": peer}
    verification = {
        "passed": True,
        "mode": args.verifier_mode,
        "gate_only": args.verification_gate_only,
        "groups": gates,
        "same_loaded_shards": True,
        "prefix_rollback": getattr(args, "prefix_rollback", False),
        "target_revision": args.target_revision,
    }
    report(
        {
            "phase": "numerical_admission_done",
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
                progress=progress,
            )
            serve_request(channel, target, progress)
            result = channel.receive()["request_result"]
        requests.append(result)
        progress({"phase": "request_done"})
    return {"requests": requests, "verification": verification}
