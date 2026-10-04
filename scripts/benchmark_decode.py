#!/usr/bin/env python3
"""Opt-in full-model compiled-decode experiment, never implicit in generation."""

import argparse
import gc
import hashlib
import json
from pathlib import Path


def publish_result(output, result):
    """Serialize before exclusively opening; a competing result is never truncated."""
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=("short", "medium", "long"), default=None)
    args = parser.parse_args(argv)
    if args.decode_steps < 1 or args.repeats < 2 or args.warmup < 1:
        raise ValueError("positive decode steps, >=2 repeats and >=1 warmup required")
    from paretoquant.benchmark import environment
    from paretoquant.cli import _chat_prompt, _local_model, validate_model_dispatch
    from paretoquant.manifest import load_manifest, validate_manifest

    source = _local_model(args.model)
    output = args.output.resolve()
    if output.is_relative_to(source):
        raise ValueError("evidence output must be outside model directory")
    if output.exists():
        raise FileExistsError(output)
    manifest = load_manifest(source / "execution_manifest.json")
    before = environment()
    dispatch = validate_manifest(source, manifest, before)

    import mlx.core as mx
    from mlx_lm import load

    from paretoquant.decode import (
        FixedDecoder,
        _DynamicDecoder,
        benchmark_decoders,
        greedy_token_ids,
    )
    from paretoquant.evaluation import reference_schedule
    from paretoquant.pipeline import model_bytes
    from paretoquant.runtime import install_fusion

    stock, tokenizer = load(str(source))
    fused, _ = load(str(source))
    validate_model_dispatch(stock, dispatch)
    validate_model_dispatch(fused, dispatch)
    installed = install_fusion(fused, dispatch)
    if not installed:
        raise ValueError("manifest selected no fusion; cannot measure fusion contribution")
    mx.eval(stock.parameters(), fused.parameters())
    parameter_bytes = model_bytes(stock)
    if parameter_bytes != model_bytes(fused):
        raise ValueError("stock/fused parameter storage differs")
    calibration_path = Path(__file__).resolve().parents[1] / "src/paretoquant/data/calibration.json"
    calibration = json.loads(calibration_path.read_text())
    prompts = {
        "short": "Explain binary search briefly.",
        "medium": calibration[0] + "\nExplain the boundary cases of binary search.",
        "long": "\n".join([calibration[1]] * 6)
        + "\nExplain why memory access patterns matter for GPU kernels.",
    }
    cases = {}
    for name in dict.fromkeys(args.cases or prompts):
        formatted = _chat_prompt(tokenizer, prompts[name])
        ids = tokenizer.encode(formatted)
        print(f"Checking and measuring {name}: {len(ids)} prompt tokens", flush=True)
        schedule = reference_schedule(stock, ids, steps=args.decode_steps)
        case = benchmark_decoders(
            {"stock": stock, "fused": fused},
            ids,
            schedule,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        generation = {}
        for backend, model in (("stock", stock), ("fused", fused)):
            count = min(32, args.decode_steps)
            eos = tokenizer.eos_token_ids
            native = greedy_token_ids(_DynamicDecoder(model), ids, max_tokens=count, eos_tokens=eos)
            compiled = greedy_token_ids(
                FixedDecoder(model, capacity=len(ids) + count),
                ids,
                max_tokens=count,
                eos_tokens=eos,
            )
            if native != compiled:
                raise ValueError(f"compiled autoregressive tokens differ from native {backend}")
            generation[backend] = {
                "native_token_ids": native,
                "compiled_token_ids": compiled,
                "tokens_identical": True,
                "generated_token_count": len(compiled),
                "max_tokens": count,
                "stop_reason": "eos" if compiled[-1] in eos else "token_limit",
                "response": tokenizer.decode(compiled),
            }
        case["autoregressive_generation_check"] = generation
        case.update(
            prompt_text=prompts[name],
            formatted_prompt=formatted,
            prompt_token_ids=ids,
            schedule_token_ids=schedule,
        )
        cases[name] = case
        for comparison in ("combined_vs_stock_eager", "compiled_fusion", "fused_compile_and_cache"):
            ratio = case["ratios"][comparison]
            print(
                f"  {comparison}: {ratio['estimate']:.4f}x "
                f"[{ratio['ci_low']:.4f}, {ratio['ci_high']:.4f}]",
                flush=True,
            )
        gc.collect()
        mx.clear_cache()
    root = Path(__file__).resolve().parents[1]
    code = [
        Path(__file__),
        *sorted((root / "src/paretoquant").rglob("*.py")),
        *sorted((root / "src/paretoquant/kernels").glob("*.metal")),
    ]
    result = {
        "schema_version": 1,
        "scope": "single_sequence_fixed_capacity_Qwen2_decode",
        "environment_before": before,
        "environment_after": environment(),
        "source_model_sha256": manifest["model_sha256"],
        "manifest_sha256": hashlib.sha256(
            (source / "execution_manifest.json").read_bytes()
        ).hexdigest(),
        "source_code_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code
        },
        "source": str(source),
        "parameter_bytes_each_model": parameter_bytes,
        "installed_fused_pairs": installed,
        "peak_mlx_allocated_bytes": mx.get_peak_memory(),
        "cases": cases,
        "limitations": [
            "Teacher-forced cached decode, not sampled user-facing request throughput.",
            "Compilation plus fixed-capacity cache is separated from compilation-only controls.",
            "Full-capacity masked attention may add work compared with dynamic KV slicing.",
            "Native eager prefill and fixed-cache padding are excluded from decode timing.",
            "First compilation and full numerical admission are excluded from warmed timing.",
            "Trace counters are Python graph tracing observations, not runtime kernel counts.",
            "Intervals cover sampled paired trials, not other devices/models/workloads.",
            "Memory pressure and swap may affect results; no background apps were terminated.",
            "Parameter bytes exclude KV/activations; peak MLX allocation is not process RSS.",
        ],
    }
    publish_result(output, result)
    print(f"Published {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
