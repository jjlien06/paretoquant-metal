"""Standalone strict two-rank MLX ring communication smoke probe."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from phased_decode import validate_eval_schedule
from shard_plan import select_pipeline_rank, text_config
from stream_weights import (
    checkpoint_inventory,
    load_from_tensors,
    receive_json,
    receive_records,
    tensor_records,
)

_PROCESS_STARTED = time.monotonic()


def phased_options(args):
    size = getattr(args, "eval_group_size", 1)
    validate_eval_schedule(
        size, getattr(args, "eval_group_schedule", None), getattr(args, "repeats", 1), args.phased
    )
    return {"eval_group_size": size}


def phase_reporter(*, rank, output=None, memory=None):
    """MLX allocation is allocator accounting, not system residency."""
    output = sys.stderr if output is None else output
    began = _PROCESS_STARTED

    def report(event):
        values = memory() if memory is not None else {}
        record = {
            **event,
            "active_mlx_bytes": None,
            "peak_mlx_bytes": None,
            "cache_mlx_bytes": None,
            **values,
            "pid": os.getpid(),
            "rank": rank,
            "timestamp": time.time(),
            "elapsed_seconds": time.monotonic() - began,
            "process_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "process_maxrss_units": "bytes (macOS)" if sys.platform == "darwin" else "KiB",
            "memory_accounting": "MLX allocator bytes; not system residency",
        }
        print(json.dumps(record, allow_nan=False), file=output, flush=True)

    return report


def collective():
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    group = mx.distributed.init(strict=True, backend="ring")
    rank, size = group.rank(), group.size()
    if size != 2 or rank not in (0, 1):
        raise ValueError("This smoke probe requires exactly two ranks")
    total = mx.distributed.all_sum(mx.array(rank + 1), group=group)
    gathered = mx.distributed.all_gather(mx.array([rank]), group=group)
    mx.eval(total, gathered)
    if total.item() != 3 or gathered.tolist() != [0, 1]:
        raise RuntimeError("Collective numerical check failed")
    if rank == 0:
        mx.eval(mx.distributed.send(mx.array([11.0, 12.0]), 1, group=group))
        received = mx.distributed.recv((2,), mx.float32, 1, group=group)
    else:
        received = mx.distributed.recv((2,), mx.float32, 0, group=group)
        mx.eval(received)
        mx.eval(mx.distributed.send(mx.array([21.0, 22.0]), 0, group=group))
    mx.eval(received)
    return {
        "mode": "collective",
        "rank": rank,
        "world_size": size,
        "all_sum": total.item(),
        "all_gather": gathered.tolist(),
        "received": received.tolist(),
        "mlx_version": mx.__version__,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "device": mx.device_info(),
        "peak_mlx_bytes": mx.get_peak_memory(),
    }


def prepare_prompt(tokenizer, prompt):
    if prompt.startswith("tokens:"):
        ids = json.loads(prompt.removeprefix("tokens:"))
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("Invalid raw token prompt")
        return ids
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generation(args):
    request_specs = None
    if getattr(args, "benchmark_schedule", False):
        from workload_schedule import NAME, load_schedule

        request_specs = load_schedule(Path(__file__).parent / NAME, args.benchmark_schedule_sha256)
        if (
            args.repeats != len(request_specs)
            or args.max_tokens != max(r["max_tokens"] for r in request_specs)
            or args.split != [28, 36]
            or args.eval_group_size != 32
            or args.eval_group_schedule
            or not args.prefix_rollback
        ):
            raise ValueError("023 request schedule args mismatch")
        if (getattr(args, "adaptive_tau", None) is not None) != any(
            r["method"] == "adaptive" for r in request_specs
        ) or not 0.0 <= (getattr(args, "adaptive_tau", None) or 0.0) <= 1.0:
            raise ValueError("030 adaptive tau must be set iff schedule has adaptive rows")
    elif getattr(args, "adaptive_tau", None) is not None:
        raise ValueError("Adaptive depth requires the explicit request schedule")
    elif getattr(args, "benchmark_schedule_sha256", None):
        raise ValueError("Schedule hash requires explicit schedule mode")
    eval_groups = validate_eval_schedule(
        args.eval_group_size, getattr(args, "eval_group_schedule", None), args.repeats, args.phased
    )
    import mlx.core as mx
    from cpu_communication import synchronous_cpu_pipeline
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_tokenizer
    from phased_decode import phased_generate

    if args.max_tokens <= 0 or args.max_tokens > 512 or args.repeats <= 0:
        raise ValueError("Require 1..512 tokens and positive repeats")
    model_path = Path(args.model).resolve(strict=True)
    config = json.loads((model_path / "config.json").read_text())
    text = text_config(config)
    from mtp_head import TARGET_REVISION
    from verification import bind_metadata

    if (
        args.target_revision != TARGET_REVISION
        or args.single_host
        or not args.phased
        or (
            request_specs is None and (not 1 <= args.max_tokens <= 96 or not 1 <= args.repeats <= 3)
        )
    ):
        raise ValueError("019 requires bounded pinned distributed phased target")
    target_binding = bind_metadata(model_path, args.target_config_sha256, args.target_index_sha256)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    mx.reset_peak_memory()
    early = phase_reporter(
        rank=int(os.environ.get("MLX_RANK", "0")),
        memory=lambda: {
            "active_mlx_bytes": mx.get_active_memory(),
            "peak_mlx_bytes": mx.get_peak_memory(),
            "cache_mlx_bytes": mx.get_cache_memory(),
        },
    )
    if not args.single_host and not all(os.environ.get(k) for k in ("MLX_HOSTFILE", "MLX_RANK")):
        raise ValueError("Distributed generation requires explicit MLX_HOSTFILE and MLX_RANK")
    early({"phase": "distributed_init_before"})
    group = None if args.single_host else mx.distributed.init(strict=True, backend="ring")
    early({"phase": "distributed_init_done"})
    rank, size = (0, 1) if group is None else (group.rank(), group.size())
    if not args.single_host and size != 2:
        raise ValueError("The distributed probe requires exactly two ranks")
    report = phase_reporter(
        rank=rank,
        memory=lambda: {
            "active_mlx_bytes": mx.get_active_memory(),
            "peak_mlx_bytes": mx.get_peak_memory(),
            "cache_mlx_bytes": mx.get_cache_memory(),
        },
    )
    report({"phase": "loading_before", "execution": "phased" if args.phased else "upstream"})
    from mtp_runtime import run_loaded, small_gate

    small = small_gate(args, eval_groups, report)
    mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
    budget = args.parameter_budget_bytes or int(
        mx.device_info()["max_recommended_working_set_size"] * 0.90
    )
    weight_loading = None
    if args.stream_local or args.weight_server:

        class SingleGroup:
            def rank(self):
                return 0

            def size(self):
                return 1

        selected_split = args.split or [text["num_hidden_layers"] // size] * size
        if sum(selected_split) != text["num_hidden_layers"]:
            raise ValueError("An explicit valid split is required")
        selected_group = group or SingleGroup()
        if args.stream_local:
            tensors, provenance = checkpoint_inventory(model_path)
            plan = select_pipeline_rank(
                tensors,
                model_type=config["model_type"],
                num_hidden_layers=text["num_hidden_layers"],
                split=selected_split,
                rank=rank,
                tie_word_embeddings=text.get("tie_word_embeddings", False),
            )
            selected = {k: tensors[k] for k in plan["selected_keys"]}
            model, weight_loading = load_from_tensors(
                config,
                selected_group,
                selected_split,
                selected,
                tensor_records(selected, provenance),
                budget_bytes=budget,
                progress=report,
            )
        else:
            address, port = args.weight_server.rsplit(":", 1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.settimeout(45)
                connection.bind((args.weight_client_bind, 0))
                connection.connect((address, int(port)))
                token = os.environ["PARETOQUANT_WEIGHT_TOKEN"].encode("ascii")
                if len(token) != 64:
                    raise ValueError("Invalid stream session token")
                connection.sendall(token)
                metadata = receive_json(connection)
                config_bytes = (model_path / "config.json").read_bytes()
                if (
                    metadata["config_text"].encode() != config_bytes
                    or metadata["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
                    or metadata["rank"] != rank
                    or metadata["split"] != selected_split
                ):
                    raise ValueError("Server metadata differs from admitted checkpoint/rank/split")
                selected = metadata["tensors"]
                model, weight_loading = load_from_tensors(
                    config,
                    selected_group,
                    selected_split,
                    selected,
                    receive_records(connection, selected),
                    budget_bytes=budget,
                    progress=report,
                )
                connection.sendall(b"K")
        tokenizer = load_tokenizer(
            model_path,
            {"trust_remote_code": False, "local_files_only": True},
            eos_token_ids=config.get("eos_token_id"),
        )
    else:
        model, tokenizer = load(
            str(model_path),
            lazy=True,
            trust_remote_code=False,
            tokenizer_config={"trust_remote_code": False, "local_files_only": True},
        )
        if group is not None:
            model.model.pipeline(group, split=args.split)
        elif args.split:
            raise ValueError("A pipeline split cannot be applied to a single-host control")
    parameters = tree_flatten(model.parameters())
    parameter_bytes = sum(v.nbytes for _, v in parameters)
    if parameter_bytes > budget:
        raise ValueError(
            f"Local stored parameter bytes {parameter_bytes} exceed preflight budget {budget}"
        )
    mx.eval(model.parameters())
    if group is not None:
        mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
    report({"phase": "loading_done", "parameter_bytes": parameter_bytes})
    prompt = prepare_prompt(tokenizer, args.prompt)
    if isinstance(prompt, str):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        ids = list(prompt)
    from phased_decode import validate_options

    validate_options(ids, args.max_tokens, 256, 4096)
    runtime = run_loaded(
        model,
        tokenizer,
        ids,
        args,
        group=group,
        small=small,
        eval_groups=eval_groups,
        report=report,
        config=config,
        request_specs=request_specs,
    )
    return {
        "mode": "generate",
        "execution": "phased",
        "rank": rank,
        "world_size": size,
        "eval_group_size": args.eval_group_size,
        "eval_group_schedule": list(args.eval_group_schedule or []),
        "model_path": str(model_path),
        "prompt": args.prompt,
        "formatted_prompt": prompt if isinstance(prompt, str) else json.dumps(prompt),
        "model_type": config["model_type"],
        "trust_remote_code": False,
        "layer_start": model.model.start_idx,
        "layer_end": model.model.end_idx or text["num_hidden_layers"],
        "parameter_bytes": parameter_bytes,
        "parameter_budget_bytes": budget,
        "parameter_layout": [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.nbytes}
            for k, v in parameters
        ],
        "weight_loading": weight_loading,
        "device": mx.device_info(),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "target_binding": target_binding,
        "target_revision": args.target_revision,
        "verifier_mode": args.verifier_mode,
        "prefix_rollback": args.prefix_rollback,
        "adaptive_tau": getattr(args, "adaptive_tau", None),
        "benchmark_schedule_sha256": getattr(args, "benchmark_schedule_sha256", None),
        **runtime,
    }
    requests = []
    communication_context = (
        synchronous_cpu_pipeline(group)
        if args.cpu_communication and group is not None
        else nullcontext({"send": 0, "recv_like": 0, "all_gather": 0})
    )
    with communication_context as communication_calls:
        for repeat in range(args.repeats):
            mx.random.seed(0)
            actual_group = eval_groups[repeat]
            report({"phase": "request_before", "repeat": repeat, "eval_group_size": actual_group})
            if args.phased:
                request = phased_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    group=group,
                    progress=lambda event: report(
                        {**event, "repeat": repeat, "eval_group_size": actual_group}
                    ),
                    eval_group_size=actual_group,
                )
                requests.append({"repeat": repeat, **request, "eval_group_size": actual_group})
                report({"phase": "request_done", "repeat": repeat, "eval_group_size": actual_group})
                continue
            started = time.perf_counter()
            first_token = None
            tokens, text_parts = [], []
            last = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=make_sampler(0.0),
                prefill_step_size=2048,
            ):
                if first_token is None:
                    first_token = time.perf_counter() - started
                tokens.append(int(response.token))
                report(
                    {
                        "phase": "token_done",
                        "repeat": repeat,
                        "eval_group_size": actual_group,
                        "token_index": len(tokens) - 1,
                        "token_id": int(response.token),
                    }
                )
                text_parts.append(response.text)
                last = response
            mx.synchronize()
            elapsed = time.perf_counter() - started
            if last is None:
                raise RuntimeError("No genuine token was generated")
            requests.append(
                {
                    "repeat": repeat,
                    "eval_group_size": actual_group,
                    "token_ids": tokens,
                    "text": "".join(text_parts),
                    "wall_seconds": elapsed,
                    "first_token_seconds": first_token,
                    "wall_tokens_per_second": len(tokens) / elapsed,
                    "native_generation_tps": last.generation_tps,
                    "prompt_tokens": last.prompt_tokens,
                    "finish_reason": last.finish_reason,
                }
            )
            report({"phase": "request_done", "repeat": repeat, "eval_group_size": actual_group})
    if any(r["token_ids"] != requests[0]["token_ids"] for r in requests):
        raise RuntimeError("Greedy repeated requests changed tokens")
    return {
        "mode": "generate",
        "execution": "phased" if args.phased else "upstream",
        "eval_group_size": phased_options(args)["eval_group_size"],
        "eval_group_schedule": list(getattr(args, "eval_group_schedule", None) or []),
        "rank": rank,
        "world_size": size,
        "model_path": str(model_path),
        "prompt": args.prompt,
        "formatted_prompt": prompt if isinstance(prompt, str) else json.dumps(prompt),
        "model_type": config["model_type"],
        "trust_remote_code": False,
        "layer_start": model.model.start_idx,
        "layer_end": model.model.end_idx or text["num_hidden_layers"],
        "parameter_bytes": parameter_bytes,
        "parameter_budget_bytes": budget,
        "parameter_layout": [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.nbytes}
            for k, v in parameters
        ],
        "active_mlx_bytes": mx.get_active_memory(),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "cache_mlx_bytes": mx.get_cache_memory(),
        "device": mx.device_info(),
        "mlx_version": mx.__version__,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "requests": requests,
        "weight_loading": weight_loading,
        "communication": {
            "mode": (
                "phased_synchronized_layers_cpu_handoffs"
                if args.phased
                else "synchronous_cpu_boundaries"
                if args.cpu_communication and group is not None
                else "upstream"
                if group is not None
                else "none"
            ),
            "calls": communication_calls,
        },
        "implementation": (
            (
                "explicit synchronized Qwen3.5 layers/CPU handoffs; "
                if args.phased
                else "upstream mlx-lm Qwen3.5 pipeline; "
            )
            + "stock quantized operators; "
            "no custom fusion or compilation"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["collective", "generate"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--verification-gate-only", action="store_true")
    parser.add_argument("--benchmark-schedule", action="store_true")
    parser.add_argument("--benchmark-schedule-sha256")
    parser.add_argument("--prefix-rollback", action="store_true")
    parser.add_argument("--adaptive-tau", type=float, default=None)
    parser.add_argument(
        "--verifier-mode", choices=("native", "singleton-qmm", "slot-qmv", "slot-qmv-tuned-m2", "slot-qmv-tuned-m4"), default="native"
    )
    parser.add_argument("--target-revision")
    parser.add_argument("--target-config-sha256")
    parser.add_argument("--target-index-sha256")
    parser.add_argument("--small-model")
    parser.add_argument("--small-config-sha256")
    parser.add_argument("--small-index-sha256")
    parser.add_argument("--sidecar")
    parser.add_argument("--phased", action="store_true")
    parser.add_argument("--eval-group-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=1)
    parser.add_argument("--eval-group-schedule", type=int, nargs="*", choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument(
        "--cpu-communication",
        action="store_true",
        help="Diagnostic synchronous CPU-stream pipeline boundaries",
    )
    streaming = parser.add_mutually_exclusive_group()
    streaming.add_argument("--stream-local", action="store_true")
    streaming.add_argument("--weight-server")
    parser.add_argument("--weight-client-bind", default="169.254.248.125")
    parser.add_argument("--split", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parameter-budget-bytes", type=int, default=0)
    parser.add_argument("--prompt", default="What is 2 plus 2? Explain briefly.")
    args = parser.parse_args()
    try:
        phased_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    from remote_actor import source_hashes

    phase_reporter(rank=int(os.environ.get("MLX_RANK", "0")))(
        {
            "phase": "process_start",
            "source_sha256": source_hashes(Path(__file__).parent),
            "execution": (
                "explicit synchronized layers/CPU handoffs; stock quantized ops; "
                "no custom fusion; no compile"
            )
            if args.phased
            else "upstream",
        }
    )
    if args.phased and args.cpu_communication:
        parser.error("--phased and --cpu-communication are exclusive")
    if args.mode == "generate" and not args.model:
        parser.error("generate requires a local checkpoint path")
    if args.mode == "collective":
        result = collective()
    else:
        # Apply the upstream recommended process-local limit before allocating
        # checkpoint buffers, not only inside stream_generate after loading.
        import mlx.core as mx

        recommended = mx.device_info()["max_recommended_working_set_size"]
        previous = mx.set_wired_limit(recommended)
        try:
            result = generation(args)
            result["wired_before_load_bytes"] = recommended
        finally:
            mx.synchronize()
            mx.set_wired_limit(previous)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
