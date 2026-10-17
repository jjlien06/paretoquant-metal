"""Mixed fresh-cache requests, only after unchanged same-loaded-shard gates."""

import gc
import hashlib
import time

from workload_schedule import OVERLAP_SCHEMA, SCHEMA, validate_schedule


def tokenize_requests(tokenizer, rows):
    from phased_decode import validate_options
    from probe import prepare_prompt

    validate_schedule(
        {
            "schema": OVERLAP_SCHEMA if any("prefill_mode" in r for r in rows) else SCHEMA,
            "requests": rows,
        }
    )
    prepared = []
    for row in rows:
        prompt = prepare_prompt(tokenizer, row["prompt"])
        ids = (
            tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str)
            else list(prompt)
        )
        validate_options(ids, row["max_tokens"], 256, 4096)
        if "prefill_mode" in row and len(ids) > 2048:
            raise ValueError("Overlap-study schedule prompt exceeds2048 IDs")
        prepared.append(ids)
    return prepared


def run_requests(
    model, tokenizer, rows, args, *, group, channel, head, admissions, eval_groups, report
):
    from overlap_prefill import validate_modes

    validate_modes(args, rows)
    import mlx.core as mx
    from distributed_control import serve_request
    from mlx_backend import MTPBackend, Rank1Target
    from phased_decode import phased_generate
    from speculative import generate

    prepared = tokenize_requests(tokenizer, rows)
    if len(eval_groups) != len(rows):
        raise ValueError("Workload/evaluation schedule length differs")
    rank = 0 if group is None else group.rank()
    requests = []
    for ordinal, (spec, ids, size) in enumerate(zip(rows, prepared, eval_groups)):
        binding = {k: spec[k] for k in ("id", "case_id", "method", "warmup", "max_tokens")}
        if "prefill_mode" in spec:
            binding.update(
                prefill_mode=spec["prefill_mode"], source_request_id=spec["source_request_id"]
            )
        binding.update(
            repeat=ordinal,
            sample_repeat=spec["repeat"],
            eval_group_size=size,
            prompt_sha256=hashlib.sha256(spec["prompt"].encode("utf-8")).hexdigest(),
        )

        def progress(event):
            report({**event, **binding})

        mx.synchronize()
        mx.reset_peak_memory()
        mx.random.seed(0)
        progress({"phase": "request_before", "prompt_token_ids": ids})
        if "prefill_mode" in spec:
            progress({"phase": "request_start", "prompt_token_ids": ids})
        local_began = time.perf_counter()
        if spec["method"] == "native":
            # Both ranks enter the collective width-one native path; no Channel or MTP backend.
            result = phased_generate(
                model,
                tokenizer,
                ids,
                max_tokens=spec["max_tokens"],
                group=group,
                progress=progress,
                prefill_step_size=256,
                context_limit=4096,
                eval_group_size=size,
            )
            result.update(accepted_drafts=0, verification_rounds=0, teacher_forwards=0)
        elif rank == 0:
            backend = MTPBackend(
                model,
                head,
                group=group,
                peer=channel,
                eval_group_size=size,
                admission=admissions[size],
                verifier_mode=args.verifier_mode,
                prefix_rollback=args.prefix_rollback,
                hidden_only_prefill=getattr(args, "hidden_only_prefill", False),
                prefill_mode=spec.get("prefill_mode", "serial"),
                progress=progress,
            )
            began = time.perf_counter()
            result = generate(
                backend, ids, max_tokens=spec["max_tokens"], eos_ids=tokenizer.eos_token_ids
            )
            backend.finish()
            mx.synchronize()
            elapsed = time.perf_counter() - began
            result.update(
                wall_seconds=elapsed,
                first_token_seconds=backend.first_token_at - began,
                prompt_tokens=len(ids),
                request_nonce=backend.request,
                cache_epoch=backend.request,
                text=tokenizer.decode(result["token_ids"], skip_special_tokens=True),
            )
            if channel is not None:
                channel.send({"request_result": result})
            del backend
        else:
            target = Rank1Target(
                model,
                group,
                eval_group_size=size,
                admission=admissions[size],
                verifier_mode=args.verifier_mode,
                prefix_rollback=args.prefix_rollback,
                hidden_only_prefill=getattr(args, "hidden_only_prefill", False),
                prefill_mode=spec.get("prefill_mode", "serial"),
                progress=progress,
            )
            serve_request(channel, target, progress)
            result = channel.receive()["request_result"]
            del target
        mx.synchronize()
        local_elapsed = time.perf_counter() - local_began
        # Rank1 retains its own independent wall/peak receipt, never the transferred rank0 peak.
        result = dict(
            result,
            **binding,
            prompt_token_ids=ids,
            local_rank=rank,
            local_rank_wall_seconds=local_elapsed,
            local_peak_mlx_bytes=mx.get_peak_memory(),
            sidecar_resident_on_rank0=True,
            numerical_admission="same-process-small-and-full-strict-passed",
            verifier_mode="native" if spec["method"] == "native" else args.verifier_mode,
        )
        n = len(result["token_ids"])
        wall, first = result["wall_seconds"], result["first_token_seconds"]
        result.update(
            wall_tokens_per_second=n / wall,
            post_first_tokens_per_second=(n - 1) / (wall - first) if n > 1 else None,
            timing_convention=(
                "EOS included; fresh cache; gates/head load excluded; MTP finish ACK included; "
                "independent local_rank_wall includes final synchronization"
            ),
        )
        requests.append(result)
        # Flushed JSON phase record is durable partial evidence even if final stdout never arrives.
        progress({"phase": "request_done", "request_result": result})
        # Backend/target refs were deleted above. Collect equally after BOTH methods,
        # outside all reported request times, before the next request peak reset.
        mx.synchronize()
        gc.collect()
    return requests
