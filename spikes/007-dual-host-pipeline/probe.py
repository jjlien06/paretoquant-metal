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

from shard_plan import select_pipeline_rank
from stream_weights import (
    checkpoint_inventory,
    load_from_tensors,
    receive_json,
    receive_records,
    tensor_records,
)

_PROCESS_STARTED = time.monotonic()


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


def generation(args):
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
    if config.get("model_type") != "qwen2":
        raise ValueError("Only the verified Qwen2 pipeline architecture is admitted")
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

        selected_split = args.split or [config["num_hidden_layers"] // size] * size
        if sum(selected_split) != config["num_hidden_layers"]:
            raise ValueError("An explicit valid split is required")
        selected_group = group or SingleGroup()
        if args.stream_local:
            tensors, provenance = checkpoint_inventory(model_path)
            plan = select_pipeline_rank(
                tensors,
                model_type=config["model_type"],
                num_hidden_layers=config["num_hidden_layers"],
                split=selected_split,
                rank=rank,
                tie_word_embeddings=config.get("tie_word_embeddings", False),
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
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    requests = []
    communication_context = (
        synchronous_cpu_pipeline(group)
        if args.cpu_communication and group is not None
        else nullcontext({"send": 0, "recv_like": 0, "all_gather": 0})
    )
    with communication_context as communication_calls:
        for repeat in range(args.repeats):
            mx.random.seed(0)
            report({"phase": "request_before", "repeat": repeat})
            if args.phased:
                request = phased_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    group=group,
                    progress=lambda event: report({**event, "repeat": repeat}),
                )
                requests.append({"repeat": repeat, **request})
                report({"phase": "request_done", "repeat": repeat})
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
    if any(r["token_ids"] != requests[0]["token_ids"] for r in requests):
        raise RuntimeError("Greedy repeated requests changed tokens")
    return {
        "mode": "generate",
        "execution": "phased" if args.phased else "upstream",
        "rank": rank,
        "world_size": size,
        "model_path": str(model_path),
        "prompt": args.prompt,
        "formatted_prompt": prompt,
        "trust_remote_code": False,
        "layer_start": model.model.start_idx,
        "layer_end": model.model.end_idx or config["num_hidden_layers"],
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
                "explicit synchronized Qwen2 layers/CPU handoffs; "
                if args.phased
                else "upstream mlx-lm Qwen2 pipeline; "
            )
            + "stock quantized operators; "
            "no custom fusion or compilation"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["collective", "generate"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--phased", action="store_true")
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
