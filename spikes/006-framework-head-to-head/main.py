"""Real sequential, warm, single-request framework benchmarks on local weights."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import urllib.request
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]


def load_verified_dispatch(source):
    from paretoquant.benchmark import environment
    from paretoquant.manifest import load_manifest, validate_manifest

    return validate_manifest(
        source, load_manifest(source / "execution_manifest.json"), environment()
    )


def validate_server_model(models, source, context_limit):
    """Check API-observed identity; this is not loaded-weight attestation."""
    data = models.get("data") if isinstance(models, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("server model metadata is missing")
    matches = [
        item for item in data if isinstance(item, dict) and item.get("id") == "paretoquant-mixed"
    ]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one paretoquant-mixed server model")
    item = matches[0]
    root = item.get("root")
    if (
        not isinstance(root, str)
        or not root
        or not Path(root).is_absolute()
        or Path(root).resolve() != source.resolve()
    ):
        raise RuntimeError("server root differs or is unobservable")
    if type(item.get("max_model_len")) is not int or item["max_model_len"] != context_limit:
        raise RuntimeError("server context limit differs or is unobservable")
    return {
        "server_root_matches_local_model": True,
        "server_reported_root": root,
        "same_saved_mixed_weights": None,
        "server_loaded_weight_sha256": None,
        "prefix_caching": None,
        "max_num_seqs": None,
        "max_model_len": item["max_model_len"],
        "unverified_settings": ["prefix_caching", "max_num_seqs", "server_loaded_weight_sha256"],
        "requested_settings": {
            "prefix_caching": False,
            "max_num_seqs": 1,
            "max_model_len": context_limit,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("paretoquant_mlx", "stock_mlx", "vllm_metal"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "artifacts/m2pro-saved-retune-v1/model"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if args.repeats < 2 or args.warmup < 1 or args.max_tokens < 1:
        parser.error("positive tokens/warmup and at least two repeats required")
    if args.output.exists():
        parser.error("refusing to overwrite existing measurements")
    fixture_path = ROOT / "results/ollama-head-to-head-v1/comparison.json"
    fixture = json.loads(fixture_path.read_text())
    raw_prompts = fixture["raw_prompts"]
    if len(raw_prompts) != 3:
        raise RuntimeError("expected exactly three shared benchmark cases")
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    from paretoquant.cli import _local_model

    source = _local_model(args.model)
    dispatch = load_verified_dispatch(source)
    metadata["weight_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.glob("*.safetensors"))
    }
    metadata["model_path"] = str(source)
    metadata["precision"] = "saved affine precision map"
    metadata["gate_up_bits"] = sorted({row["bits"] for row in dispatch.values()})
    metadata["execution_manifest_sha256"] = hashlib.sha256(
        (source / "execution_manifest.json").read_bytes()
    ).hexdigest()
    metadata["local_manifest_validated"] = True

    if args.engine == "vllm_metal":
        with urllib.request.urlopen("http://127.0.0.1:11436/v1/models", timeout=10) as response:
            models = json.load(response)
        metadata["server_models"] = models
        metadata.update(validate_server_model(models, source, 1024))
        metadata["versions"] = json.loads(
            subprocess.check_output(
                [
                    str(ROOT / "artifacts/vllm-metal-venv/bin/python"),
                    "-c",
                    "import json; from importlib.metadata import version; "
                    "print(json.dumps({n:version(n) for n in ['vllm','vllm-metal','mlx','mlx-lm','torch']}))",
                ],
                text=True,
            )
        )
        metadata["version_observation"] = (
            "Local launcher environment, not attested running-server versions"
        )
        metadata["mode"] = "vLLM Metal/MLX GPU serving, not CUDA; local HTTP included"
    else:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        from paretoquant.cli import _runtime_counters
        from paretoquant.manifest import validate_model_dispatch
        from paretoquant.runtime import install_fusion

        model, tokenizer = load(str(source))
        mx.eval(model.parameters())
        validate_model_dispatch(model, dispatch)
        installed = install_fusion(model, dispatch) if args.engine == "paretoquant_mlx" else []
        metadata.update(
            {
                "versions": {name: version(name) for name in ("mlx", "mlx-lm")},
                "device": mx.device_info(),
                "fused_pair_count": len(installed),
            }
        )
    trials = []
    warmups = []
    for trial in range(-args.warmup, args.repeats):
        for case, raw in enumerate(raw_prompts):
            if args.engine == "vllm_metal":
                payload = {
                    "model": "paretoquant-mixed",
                    "prompt": raw,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "seed": 0,
                    "ignore_eos": True,
                    "stream": False,
                }
                request = urllib.request.Request(
                    "http://127.0.0.1:11436/v1/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                started = perf_counter()
                with urllib.request.urlopen(request, timeout=120) as response:
                    actual = json.load(response)
                elapsed = perf_counter() - started
                response_text = actual["choices"][0]["text"]
                count = actual["usage"]["completion_tokens"]
                prompt_count = actual["usage"]["prompt_tokens"]
                token_ids = actual["choices"][0].get("token_ids")
                metadata["actual_system_fingerprint"] = actual.get("system_fingerprint")
            else:
                mx.synchronize()
                started = perf_counter()
                text = []
                token_ids = []
                final = None
                for chunk in stream_generate(
                    model,
                    tokenizer,
                    raw,
                    max_tokens=args.max_tokens,
                    sampler=make_sampler(temp=0.0),
                ):
                    text.append(chunk.text)
                    token_ids.append(chunk.token)
                    final = chunk
                mx.synchronize()
                elapsed = perf_counter() - started
                if final is None:
                    raise RuntimeError("no actual MLX generation")
                response_text = "".join(text)
                count = final.generation_tokens
                prompt_count = final.prompt_tokens
            if count != args.max_tokens or not response_text:
                raise RuntimeError("actual generation did not meet fixed-token workload")
            if token_ids is not None and len(token_ids) != count:
                raise RuntimeError("token-ID count disagrees with generation metadata")
            record = {
                "engine": args.engine,
                "trial": trial,
                "case": case,
                "prompt_tokens": prompt_count,
                "generated_tokens": count,
                "token_ids": token_ids,
                "response": response_text,
                "wall_seconds": elapsed,
                "wall_tokens_per_second": count / elapsed,
            }
            (trials if trial >= 0 else warmups).append(record)
            print(
                f"{args.engine} trial={trial} case={case} tokens={count} "
                f"wall_tps={count / elapsed:.2f}",
                flush=True,
            )
    counts = Counter(row["case"] for row in trials)
    if dict(counts) != {case: args.repeats for case in range(len(raw_prompts))}:
        raise RuntimeError("collected trial counts do not match the request")
    summaries = []
    for case in range(len(raw_prompts)):
        rows = [row for row in trials if row["case"] == case]
        summaries.append(
            {
                "case": case,
                "prompt_tokens": rows[0]["prompt_tokens"],
                "repeats": len(rows),
                "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
                "median_wall_tokens_per_second": statistics.median(
                    row["wall_tokens_per_second"] for row in rows
                ),
            }
        )
    if args.engine != "vllm_metal":
        metadata["runtime_counters"] = _runtime_counters(model)
    result = {
        "schema_version": 1,
        "engine": args.engine,
        "metadata": metadata,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "max_generated_tokens": args.max_tokens,
        "warmup_per_case": args.warmup,
        "raw_prompts": raw_prompts,
        "trials": trials,
        "warmups": warmups,
        "summaries": summaries,
        "machine_swap_after": subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True
        ).strip(),
        "limitations": [
            "Sequential backend blocks, not cross-backend paired trials; machine state may drift.",
            "Wall time includes tokenization, prefill, greedy generation and text detokenization.",
            "vLLM measurements include localhost HTTP and serving/scheduling overhead; MLX uses its Python API.",
            "vLLM Metal uses MLX, not CUDA; its MLX version differs from the main environment.",
            "Three short-context small-model cases do not establish broad task quality or larger-model speed.",
            "Server root/context are API-observed; loaded weight hashes, cache and scheduler settings are not attested.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Saved actual measurements: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
