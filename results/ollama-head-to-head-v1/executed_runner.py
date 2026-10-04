"""Live practical comparison, not an identical-weight kernel ablation.

Run from the repository with its .venv/bin/python. Requires a localhost Ollama
server and qwen2.5:0.5b already downloaded. Does not install or start anything.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter
import urllib.request

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from paretoquant.benchmark import environment
from paretoquant.cli import _chat_prompt
from paretoquant.pipeline import model_bytes
from paretoquant.runtime import install_fusion

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=7)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--max-tokens", type=int, default=64)
args = parser.parse_args()
if args.repeats < 2 or args.warmup < 1 or args.max_tokens < 1:
    parser.error("at least two repeats, one warmup and one generated token required")
if args.output.exists():
    parser.error("refusing to overwrite existing evidence")

base = "http://127.0.0.1:11434"
ollama_metadata = {}
for endpoint, body in (("version", None), ("show", {"model": "qwen2.5:0.5b"})):
    request = urllib.request.Request(
        f"{base}/api/{endpoint}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        ollama_metadata[endpoint] = json.load(response)
if ollama_metadata["show"]["model_info"].get("general.finetune") != "Instruct":
    raise RuntimeError("Ollama model is not the expected instruct checkpoint family")

source = ROOT / "artifacts/m2pro-mixed-v2/model"
manifest = json.loads((source / "execution_manifest.json").read_text())
before = environment()
if (manifest["mlx"] != before["mlx"] or manifest["mlx_lm"] != before["mlx_lm"]
        or manifest["profile_device"]["device_name"] != before["device"]["device_name"]):
    raise RuntimeError("saved dispatch is not profiled for this hardware/runtime")
model, tokenizer = load(str(source))
mx.eval(model.parameters())
installed = install_fusion(model, manifest["dispatch"])
if not installed:
    raise RuntimeError("no custom fused execution was selected")

context = (
    "A database transaction groups multiple reads and writes into one logical operation. "
    "Atomicity requires either all changes or none. Consistency preserves invariants. "
    "Isolation controls interactions between concurrent transactions. Durability preserves "
    "committed changes after a crash. Write-ahead logging records changes before pages are written. "
)
questions = [
    "Explain binary search with its invariant, pseudocode, and a worked example on eight elements.",
    context + "Explain each ACID property with a concrete banking example and common failure modes.",
    context * 3 + "Compare locking and multiversion concurrency control, with examples and tradeoffs.",
]
raw_prompts = [_chat_prompt(tokenizer, question) for question in questions]
prompt_counts = [len(tokenizer.encode(prompt)) for prompt in raw_prompts]
if any(count + args.max_tokens > 1024 for count in prompt_counts):
    raise RuntimeError("requested workload exceeds the configured Ollama context")
trials = []
warmups = []
for trial in range(-args.warmup, args.repeats):
    for case, raw in enumerate(raw_prompts):
        order = ("paretoquant_mlx", "ollama") if trial % 2 == 0 else ("ollama", "paretoquant_mlx")
        for position, engine in enumerate(order):
            if engine == "paretoquant_mlx":
                mx.synchronize()
                started = perf_counter()
                text = []
                final = None
                for chunk in stream_generate(
                    model, tokenizer, prompt=raw, max_tokens=args.max_tokens,
                    sampler=make_sampler(temp=0.0),
                ):
                    text.append(chunk.text)
                    final = chunk
                mx.synchronize()
                elapsed = perf_counter() - started
                if final is None:
                    raise RuntimeError("MLX returned no generation metadata")
                count = final.generation_tokens
                record = {
                    "response": "".join(text), "generated_tokens": count,
                    "finish_reason": final.finish_reason,
                    "engine_reported_generation_tps": final.generation_tps,
                    "engine_reported_prompt_tps": final.prompt_tps,
                    "engine_reported_prompt_tokens": final.prompt_tokens,
                }
            else:
                payload = {
                    "model": "qwen2.5:0.5b", "prompt": raw, "raw": True,
                    "stream": False, "keep_alive": "10m",
                    "options": {"temperature": 0.0, "seed": 0,
                                "num_predict": args.max_tokens, "num_ctx": 1024},
                }
                request = urllib.request.Request(
                    f"{base}/api/generate", data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                started = perf_counter()
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                elapsed = perf_counter() - started
                count = result["eval_count"]
                if not result.get("done") or count < 1 or result["eval_duration"] <= 0:
                    raise RuntimeError(f"invalid Ollama generation metadata: {result}")
                record = {
                    "response": result["response"], "generated_tokens": count,
                    "finish_reason": result.get("done_reason"),
                    "engine_reported_generation_tps": count * 1e9 / result["eval_duration"],
                    "engine_reported_prompt_tokens": result.get("prompt_eval_count"),
                    "api_timings_ns": {key: result.get(key) for key in (
                        "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
                }
            if count < 1 or elapsed <= 0:
                raise RuntimeError("empty generation or invalid elapsed time")
            record.update({
                "engine": engine, "trial": trial, "case": case,
                "order_position": position, "wall_seconds": elapsed,
                "wall_tokens_per_second": count / elapsed,
                "full_prompt_token_count_mlx_tokenizer": prompt_counts[case],
            })
            (trials if trial >= 0 else warmups).append(record)
            print(f"trial={trial} case={case} engine={engine} tokens={count} "
                  f"wall_tps={count / elapsed:.2f}", flush=True)

summaries = []
for case in range(len(raw_prompts)):
    summary = {"case": case, "prompt_tokens": prompt_counts[case]}
    for engine in ("paretoquant_mlx", "ollama"):
        records = [row for row in trials if row["case"] == case and row["engine"] == engine]
        if len(records) != args.repeats:
            raise RuntimeError("collected count does not match requested repeats")
        summary[engine] = {
            "repeats": len(records),
            "median_wall_tokens_per_second": statistics.median(
                row["wall_tokens_per_second"] for row in records),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in records),
            "generated_token_counts": [row["generated_tokens"] for row in records],
            "median_engine_reported_generation_tps": statistics.median(
                row["engine_reported_generation_tps"] for row in records),
        }
    summary["ratio_mlx_over_ollama_wall_tps"] = (
        summary["paretoquant_mlx"]["median_wall_tokens_per_second"]
        / summary["ollama"]["median_wall_tokens_per_second"]
    )
    summaries.append(summary)
with urllib.request.urlopen(f"{base}/api/ps", timeout=30) as response:
    ollama_ps = json.load(response)
if not ollama_ps.get("models") or not ollama_ps["models"][0].get("size_vram"):
    raise RuntimeError("Ollama GPU residency was not verified")
fused_calls = sum(module.stats["fused_calls"] for _, module in model.named_modules()
                  if hasattr(module, "stats"))
if fused_calls < 1:
    raise RuntimeError("custom kernel never ran")
result = {
    "schema_version": 1, "environment_before": before, "environment_after": environment(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "mlx_model": str(source), "mlx_parameter_bytes": model_bytes(model),
    "mlx_weight_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob("*.safetensors"))
    },
    "execution_manifest_sha256": hashlib.sha256((source / "execution_manifest.json").read_bytes()).hexdigest(),
    "installed_fused_pair_count": len(installed), "runtime_fused_calls": fused_calls,
    "ollama_version": ollama_metadata["version"],
    "ollama_details": ollama_metadata["show"]["details"],
    "ollama_model_info": ollama_metadata["show"]["model_info"],
    "ollama_ps": ollama_ps, "quantization_matched": False,
    "max_generated_tokens": args.max_tokens, "repeats": args.repeats,
    "warmup_per_case": args.warmup, "context_limit_ollama": 1024,
    "raw_prompts": raw_prompts, "summaries": summaries, "trials": trials, "warmups": warmups,
    "limitations": [
        "Same Qwen2.5-0.5B-Instruct family, different weight quantizations: affine 3/4-bit versus GGUF Q4_K_M.",
        "Not an identical-weight isolation of kernel contribution; output quality is not assessed.",
        "Wall TPS includes prompt processing, greedy generation and text detokenization; Ollama includes local HTTP overhead.",
        "Native generation TPS definitions differ; do not treat them as a fair isolated decode comparison.",
        "Ollama may reuse prompt-prefix KV cache; MLX generation starts with fresh KV cache.",
        "Output token sequences and EOS behavior can differ; actual token counts are retained.",
        "Both models remain resident, run sequentially, and share unified memory with macOS/background processes.",
        "GPU allocation/residency metrics from MLX and Ollama are not equivalent total-memory measures.",
        "A small-model practical test does not establish larger-model capability or broad task quality.",
    ],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(summaries, indent=2), flush=True)
print(f"Saved: {args.output.resolve()}", flush=True)
