#!/usr/bin/env python3
"""Bounded-residency profiling for real local checkpoints; no downloads."""
import argparse
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path


def profile(args):
    import mlx.core as mx
    from mlx_lm import load

    from paretoquant.allocator import Option
    from paretoquant.benchmark import environment
    from paretoquant.cli import _dump, _local_model, _prepare_output, _source_hashes, _texts
    from paretoquant.evaluation import text_nll
    from paretoquant.manifest import kernel_hashes
    from paretoquant.pipeline import apply_plan, calibrate_inputs, model_bytes, profile_units

    source = _local_model(args.model)
    output = _prepare_output(args.output, source)
    before = environment()
    model, tokenizer = load(str(source))
    if any(hasattr(module, "bits") for _, module in model.named_modules()):
        raise ValueError("profiling requires an unquantized local reference")
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    reference_bytes = model_bytes(model)
    evaluation = _texts(None, "data/evaluation.json")
    reference_quality = text_nll(model, tokenizer, evaluation)
    calibration = _texts(None, "data/calibration.json")
    inputs = calibrate_inputs(model, tokenizer, calibration)
    measured = profile_units(model, inputs, repeats=args.profile_repeats)
    choices = {u["name"]: next(Option(**o) for o in u["options"]
                               if o["bits"] == 4 and o["backend"] == "stock")
               for u in measured["units"]}
    del inputs
    config = json.loads((source / "config.json").read_text())
    model, _, _ = apply_plan(model, config, choices)
    uniform_bytes = model_bytes(model)
    measured.update({
        "source_weights_sha256": _source_hashes(source),
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "profile_kernel_sha256": kernel_hashes(),
        "source_model_type": config.get("model_type"),
        "source_model_name": source.name,
        "reference_parameter_bytes": reference_bytes,
        "reference_smoke_quality": reference_quality,
        "fixed_parameter_bytes": uniform_bytes - sum(o.memory_bytes for o in choices.values()),
        "uniform4_parameter_bytes": uniform_bytes,
        "calibration_texts_sha256": hashlib.sha256(json.dumps(calibration).encode()).hexdigest(),
        "evaluation_texts_sha256": hashlib.sha256(json.dumps(evaluation).encode()).hexdigest(),
        "environment_before": before,
        "environment_after": environment(),
        "peak_gpu_allocated_bytes": mx.get_peak_memory(),
        "residency_note": "One full-precision reference, then one uniform4 model; no variant bank.",
    })
    _dump(output / "profile.json", measured)
    del model
    gc.collect()
    mx.clear_cache()
    print(json.dumps({"profile": str(output / "profile.json"),
                      "reference_mib": reference_bytes / 1024**2,
                      "uniform4_mib": uniform_bytes / 1024**2,
                      "units": len(measured["units"])}, indent=2))
    return 0


def evaluate(args):
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.utils import save_config, save_model

    from paretoquant.allocator import allocate
    from paretoquant.benchmark import environment
    from paretoquant.cli import (
        _chat_prompt,
        _dump,
        _local_model,
        _prepare_output,
        _runtime_counters,
        _source_hashes,
        _texts,
    )
    from paretoquant.evaluation import cached_decode_benchmark, reference_schedule, text_nll
    from paretoquant.manifest import create_manifest, kernel_hashes
    from paretoquant.pipeline import apply_plan, model_bytes, units_from_profile

    source = _local_model(args.model)
    measured = json.loads(Path(args.profile).read_text())
    if measured["source_weights_sha256"] != _source_hashes(source):
        raise ValueError("profile source checkpoint hashes differ")
    if measured.get("source_config_sha256") != hashlib.sha256(
        (source / "config.json").read_bytes()
    ).hexdigest():
        raise ValueError("profile source config differs or is unbound; regenerate profile")
    if measured.get("profile_kernel_sha256") != kernel_hashes():
        raise ValueError("profile Metal kernel hashes differ or are unbound; regenerate profile")
    current = environment()
    previous = measured["environment"]
    if (previous["device"]["device_name"] != current["device"]["device_name"]
            or previous["mlx"] != current["mlx"] or previous["mlx_lm"] != current["mlx_lm"]):
        raise ValueError("profile hardware/runtime differs; regenerate the profile")
    stock_units = units_from_profile(measured, stock_only=True)
    baseline = {u.name: next(o for o in u.options if o.bits == 4) for u in stock_units}
    fixed_bytes = measured["fixed_parameter_bytes"]
    budget_bytes = int(measured["uniform4_parameter_bytes"] * args.memory_fraction)
    if budget_bytes < fixed_bytes:
        raise ValueError("budget is smaller than fixed parameters")
    latency_budget = sum(o.latency_ms for o in baseline.values()) * args.latency_factor
    plan = allocate(units_from_profile(measured), budget_bytes - fixed_bytes,
                    latency_budget, args.max_states)
    output = _prepare_output(args.output, source)
    config = json.loads((source / "config.json").read_text())
    evaluation = _texts(None, "data/evaluation.json")
    prompt = "Explain why a mutex prevents a data race."
    variants = {}
    schedule = None
    for name, selection in (("uniform4", baseline), ("mixed", plan.choices)):
        print(f"Sequentially loading/evaluating {name}...", flush=True)
        model, tokenizer = load(str(source))
        model.set_dtype(mx.float16)
        model, cfg, dispatch = apply_plan(model, config, selection)
        actual_bytes = model_bytes(model)
        expected_bytes = fixed_bytes + sum(o.memory_bytes for o in selection.values())
        if actual_bytes != expected_bytes:
            raise ValueError("profile parameter byte model differs from constructed model")
        if name == "mixed" and actual_bytes > budget_bytes:
            raise ValueError("constructed model exceeds parameter budget")
        ids = tokenizer.encode(_chat_prompt(tokenizer, prompt))
        if schedule is None:
            schedule = reference_schedule(model, ids, steps=args.decode_steps)
        quality = text_nll(model, tokenizer, evaluation)
        timing = cached_decode_benchmark({name: model}, ids, schedule,
                                        repeats=args.repeats, warmup=2)[name]
        text = generate(model, tokenizer, prompt=_chat_prompt(tokenizer, prompt),
                        max_tokens=args.decode_steps, verbose=False)
        if not text.strip():
            raise ValueError("real generation returned empty text")
        destination = output / name / "model"
        save_model(destination, model)
        save_config(cfg, destination / "config.json")
        tokenizer.save_pretrained(destination)
        for filename in ("generation_config.json", "LICENSE"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, destination / filename)
        _dump(destination / "execution_manifest.json",
              create_manifest(destination, dispatch, current))
        variants[name] = {
            "parameter_bytes": actual_bytes,
            "quality": quality,
            "timing": timing,
            "generation": text,
            "precision_histogram": {str(bit): sum(o.bits == bit for o in selection.values())
                                    for bit in (3, 4, 6)},
            "fused_pair_count": sum(o.backend == "fused" for o in selection.values()),
            "runtime_counters": _runtime_counters(model),
            "saved_weights_sha256": _source_hashes(destination),
        }
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    result = {
        "schema_version": 1,
        "profile_sha256": hashlib.sha256(Path(args.profile).read_bytes()).hexdigest(),
        "source_weights_sha256": measured["source_weights_sha256"],
        "source_model": source.name,
        "environment_before": current,
        "environment_after": environment(),
        "memory_fraction": args.memory_fraction,
        "latency_factor": args.latency_factor,
        "budget_bytes": budget_bytes,
        "predicted_gate_up_latency_budget_ms": latency_budget,
        "plan": plan.to_dict(),
        "variants": variants,
        "reference_smoke_quality": measured["reference_smoke_quality"],
        "schedule_token_ids": schedule,
        "generation_prompt": prompt,
        "peak_gpu_allocated_bytes": mx.get_peak_memory(),
        "limitations": [
            "Variants are evaluated in sequential blocks, not paired timing trials.",
            "Same uniform4 teacher-forced cached token schedule; excludes sampling/rendering.",
            "Smoke quality is not a standard task-quality benchmark.",
            "Parameter bytes exclude KV cache, activations, and other process memory.",
            "Summed isolated gate/up latency is a surrogate, not full-model latency.",
            "Zero fused pairs is a valid measured stock fallback, not a fusion speedup.",
        ],
    }
    _dump(output / "results.json", result)
    print(json.dumps({"results": str(output / "results.json"), "variants": {
        n: {"mib": v["parameter_bytes"] / 1024**2,
            "tok_per_second": v["timing"]["median_decode_tokens_per_second"],
            "fused_pairs": v["fused_pair_count"]} for n, v in variants.items()}}, indent=2))
    return 0


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("profile")
    command.add_argument("--model", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--profile-repeats", type=positive_int, default=20)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--profile", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--memory-fraction", type=positive_float, default=0.94)
    evaluation.add_argument("--latency-factor", type=positive_float, default=1.15)
    evaluation.add_argument("--decode-steps", type=positive_int, default=64)
    evaluation.add_argument("--repeats", type=positive_int, default=15)
    evaluation.add_argument("--max-states", type=positive_int, default=20000)
    args = parser.parse_args(argv)
    try:
        return profile(args) if args.command == "profile" else evaluate(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
