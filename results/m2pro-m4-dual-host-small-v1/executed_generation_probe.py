"""Standalone strict two-rank MLX ring communication smoke probe."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def collective():
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
        "mode": "collective", "rank": rank, "world_size": size,
        "all_sum": total.item(), "all_gather": gathered.tolist(),
        "received": received.tolist(), "mlx_version": mx.__version__,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "device": mx.device_info(), "peak_mlx_bytes": mx.get_peak_memory(),
    }


def generation(args):
    if args.max_tokens <= 0 or args.max_tokens > 512 or args.repeats <= 0:
        raise ValueError("Require 1..512 tokens and positive repeats")
    model_path = Path(args.model).resolve(strict=True)
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise ValueError("Only the verified Qwen2 pipeline architecture is admitted")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    mx.reset_peak_memory()
    group = None if args.single_host else mx.distributed.init(strict=True, backend="ring")
    rank, size = (0, 1) if group is None else (group.rank(), group.size())
    if not args.single_host and size != 2:
        raise ValueError("The distributed probe requires exactly two ranks")
    model, tokenizer = load(
        str(model_path), lazy=True, trust_remote_code=False,
        tokenizer_config={"trust_remote_code": False, "local_files_only": True},
    )
    if group is not None:
        model.model.pipeline(group, split=args.split)
    elif args.split:
        raise ValueError("A pipeline split cannot be applied to a single-host control")
    parameters = tree_flatten(model.parameters())
    parameter_bytes = sum(v.nbytes for _, v in parameters)
    budget = args.parameter_budget_bytes or int(
        mx.device_info()["max_recommended_working_set_size"] * 0.90
    )
    if parameter_bytes > budget:
        raise ValueError(
            f"Local stored parameter bytes {parameter_bytes} exceed preflight budget {budget}"
        )
    mx.eval(model.parameters())
    if group is not None:
        mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True,
    )
    requests = []
    for repeat in range(args.repeats):
        mx.random.seed(0)
        started = time.perf_counter()
        first_token = None
        tokens, text_parts = [], []
        last = None
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
            sampler=make_sampler(0.0), prefill_step_size=2048,
        ):
            if first_token is None:
                first_token = time.perf_counter() - started
            tokens.append(int(response.token))
            text_parts.append(response.text)
            last = response
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if last is None:
            raise RuntimeError("No genuine token was generated")
        requests.append({
            "repeat": repeat, "token_ids": tokens, "text": "".join(text_parts),
            "wall_seconds": elapsed, "first_token_seconds": first_token,
            "wall_tokens_per_second": len(tokens) / elapsed,
            "native_generation_tps": last.generation_tps,
            "prompt_tokens": last.prompt_tokens, "finish_reason": last.finish_reason,
        })
    if any(r["token_ids"] != requests[0]["token_ids"] for r in requests):
        raise RuntimeError("Greedy repeated requests changed tokens")
    return {
        "mode": "generate", "rank": rank, "world_size": size,
        "model_path": str(model_path), "prompt": args.prompt,
        "formatted_prompt": prompt, "trust_remote_code": False,
        "layer_start": model.model.start_idx,
        "layer_end": model.model.end_idx or config["num_hidden_layers"],
        "parameter_bytes": parameter_bytes, "parameter_budget_bytes": budget,
        "parameter_layout": [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype),
                              "bytes": v.nbytes} for k, v in parameters],
        "active_mlx_bytes": mx.get_active_memory(), "peak_mlx_bytes": mx.get_peak_memory(),
        "cache_mlx_bytes": mx.get_cache_memory(), "device": mx.device_info(),
        "mlx_version": mx.__version__, "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "requests": requests,
        "implementation": (
            "upstream mlx-lm Qwen2 pipeline; stock quantized operators; "
            "no custom fusion or compilation"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["collective", "generate"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--split", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parameter-budget-bytes", type=int, default=0)
    parser.add_argument("--prompt", default="What is 2 plus 2? Explain briefly.")
    args = parser.parse_args()
    if args.mode == "generate" and not args.model:
        parser.error("generate requires a local checkpoint path")
    result = collective() if args.mode == "collective" else generation(args)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
