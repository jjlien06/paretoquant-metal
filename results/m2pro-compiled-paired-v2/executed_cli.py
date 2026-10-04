"""Runnable local-model experiment; no automatic model download or remote code."""

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from .allocator import allocate
from .manifest import (
    create_manifest,
    load_manifest,
    validate_manifest,
    validate_model_dispatch,
    validate_schema,
    verify_manifest,
)


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_execution_manifest(saved, dispatch, profile):
    _dump(Path(saved) / "execution_manifest.json", create_manifest(saved, dispatch, profile))


def _texts(path, default):
    text = Path(path).read_text() if path else files("paretoquant").joinpath(default).read_text()
    data = json.loads(text)
    if (
        not isinstance(data, list)
        or not data
        or not all(isinstance(t, str) and t.strip() for t in data)
    ):
        raise ValueError("text corpus must be a nonempty JSON list of nonempty strings")
    return data


def _local_model(path):
    source = Path(path).expanduser().resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError(f"A local model directory with config.json is required: {source}")
    return source


def _prepare_output(output, source):
    output = Path(output).expanduser().resolve()
    if (output / "model").resolve() == Path(source).resolve():
        raise ValueError("output model destination would overwrite the source model")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory; preserve prior evidence")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _chat_prompt(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def _source_hashes(source):
    paths = sorted(source.glob("model*.safetensors"))
    if not paths:
        raise ValueError("local model directory contains no model*.safetensors weights")
    result = {}
    for path in paths:
        with path.open("rb") as handle:
            result[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def _precision_summary(choices):
    return {
        str(bit): count for bit, count in sorted(Counter(o.bits for o in choices.values()).items())
    }


def _runtime_counters(model):
    from .runtime import FusedMLP

    return {
        name: dict(module.stats)
        for name, module in model.named_modules()
        if type(module) is FusedMLP
    }


def run_experiment(args):
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.utils import save_config, save_model

    from .benchmark import environment
    from .evaluation import cached_decode_benchmark, reference_schedule, text_nll
    from .pipeline import (
        apply_plan,
        calibrate_inputs,
        model_bytes,
        profile_units,
        units_from_profile,
    )

    source = _local_model(args.model)
    output = _prepare_output(args.output, source)
    calibration = _texts(args.calibration, "data/calibration.json")
    evaluation = _texts(args.evaluation, "data/evaluation.json")
    if set(calibration) & set(evaluation):
        raise ValueError("calibration and evaluation texts must be disjoint")
    print(
        "Loading local unquantized reference and casting floating weights to float16...", flush=True
    )
    model, tokenizer = load(str(source))
    if any(hasattr(module, "bits") for _, module in model.named_modules()):
        raise ValueError(
            "reference must be unquantized; quantized source is not a full-precision teacher"
        )
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    config = json.loads((source / "config.json").read_text())
    initial_environment = environment()
    source_bytes = model_bytes(model)
    source_hashes = _source_hashes(source)
    print("Evaluating reference on held-out authored smoke texts...", flush=True)
    reference_quality = text_nll(model, tokenizer, evaluation)
    prompt = "Explain binary search briefly."
    prompt_ids = tokenizer.encode(_chat_prompt(tokenizer, prompt))
    schedule = reference_schedule(model, prompt_ids, steps=args.decode_steps)
    inputs = calibrate_inputs(model, tokenizer, calibration)
    profile = profile_units(model, inputs, repeats=args.profile_repeats, verbose=True)
    units = units_from_profile(profile)
    stock_units = units_from_profile(profile, stock_only=True)
    uniform4 = {
        unit.name: next(o for o in unit.options if o.bits == 4 and o.backend == "stock")
        for unit in stock_units
    }
    model, base_config, _ = apply_plan(model, config, uniform4)
    baseline_bytes = model_bytes(model)
    fixed_bytes = baseline_bytes - sum(o.memory_bytes for o in uniform4.values())
    total_budget = (
        int(args.memory_budget_mib * 1024**2) if args.memory_budget_mib else baseline_bytes
    )
    gate_budget = total_budget - fixed_bytes
    if gate_budget < 0:
        raise ValueError("memory budget is smaller than the fixed non-gate/up parameters")
    latency_budget = sum((uniform4[name].latency_ms for name in sorted(uniform4)), 0.0)
    latency_budget *= args.latency_factor
    profile.update(
        {
            "source_weights_sha256": source_hashes,
            "source_model_type": config.get("model_type"),
            "fixed_parameter_bytes": fixed_bytes,
            "uniform4_parameter_bytes": baseline_bytes,
            "total_parameter_budget_bytes": total_budget,
            "gate_up_latency_budget_ms": latency_budget,
            "candidate_scope_note": (
                "Only coupled gate/up pairs vary; all other eligible weights are 4-bit"
            ),
        }
    )
    _dump(output / "profile.json", profile)
    print("Solving exact memory/latency constrained precision plans...", flush=True)
    plans = {
        "sensitivity_only": allocate(stock_units, gate_budget, max_states=args.max_states),
        "hardware_stock": allocate(stock_units, gate_budget, latency_budget, args.max_states),
        "hardware_fused": allocate(units, gate_budget, latency_budget, args.max_states),
    }
    choices = {"uniform4": uniform4, **{name: plan.choices for name, plan in plans.items()}}
    choices["hardware_fused_stock_replay"] = {
        name: replace(option, label=f"q{option.bits}:stock", backend="stock")
        for name, option in plans["hardware_fused"].choices.items()
    }
    forced = {}
    for unit in profile["units"]:
        timing = unit["measurements"]["4"]["validation"]
        fused = [name for name in timing if name != "stock"]
        original = uniform4[unit["name"]]
        if fused:
            forced[unit["name"]] = replace(
                original,
                label=f"q4:{fused[0]}",
                backend="fused",
                latency_ms=timing[fused[0]]["median_ms"],
            )
        else:
            forced[unit["name"]] = original
    choices["uniform4_forced_fused"] = forced
    _dump(output / "plans.json", {name: plan.to_dict() for name, plan in plans.items()})
    models = {"uniform4": model}
    configs = {"uniform4": base_config}
    dispatches = {}
    for name, selection in choices.items():
        if name == "uniform4":
            continue
        print(
            f"Building {name}: gate/up precision histogram {_precision_summary(selection)}",
            flush=True,
        )
        loaded, _ = load(str(source))
        loaded.set_dtype(mx.float16)
        loaded, cfg, dispatch = apply_plan(loaded, config, selection)
        models[name], configs[name], dispatches[name] = loaded, cfg, dispatch
    variants = {}
    for name, candidate in models.items():
        print(f"Evaluating {name} on held-out smoke texts...", flush=True)
        variants[name] = {
            "parameter_bytes": model_bytes(candidate),
            "gate_up_precision_histogram": _precision_summary(choices[name]),
            "fused_pair_count": sum(o.backend == "fused" for o in choices[name].values()),
            "quality": text_nll(candidate, tokenizer, evaluation),
        }
    print("Benchmarking identical teacher-forced cached decode schedules...", flush=True)
    timing = cached_decode_benchmark(
        models, prompt_ids, schedule, repeats=args.decode_repeats, warmup=1
    )
    for name, measured in timing.items():
        variants[name]["timing"] = measured
    selected = models["hardware_fused"]
    sample = generate(
        selected,
        tokenizer,
        prompt=_chat_prompt(tokenizer, prompt),
        max_tokens=args.decode_steps,
        verbose=False,
    )
    saved = output / "model"
    save_model(saved, selected)
    save_config(configs["hardware_fused"], saved / "config.json")
    tokenizer.save_pretrained(saved)
    for filename in ("generation_config.json", "LICENSE"):
        if (source / filename).is_file():
            shutil.copy2(source / filename, saved / filename)
    _save_execution_manifest(saved, dispatches["hardware_fused"], initial_environment)
    result = {
        "schema_version": 1,
        "environment_before": initial_environment,
        "environment_after": environment(),
        "source_model": source.name,
        "source_weights_sha256": source_hashes,
        "reference_parameter_bytes": source_bytes,
        "reference_quality": reference_quality,
        "calibration_texts_sha256": hashlib.sha256(json.dumps(calibration).encode()).hexdigest(),
        "evaluation_texts_sha256": hashlib.sha256(json.dumps(evaluation).encode()).hexdigest(),
        "calibration_text_count": len(calibration),
        "evaluation_text_count": len(evaluation),
        "total_parameter_budget_bytes": total_budget,
        "gate_up_latency_budget_ms": latency_budget,
        "fixed_parameter_bytes": fixed_bytes,
        "variants": variants,
        "reference_decode_token_ids": schedule,
        "generation_prompt": prompt,
        "hardware_fused_generation": sample,
        "saved_model": "model",
        "runtime_counters": _runtime_counters(selected),
        "peak_gpu_allocated_all_variants_bytes": mx.get_peak_memory(),
        "limitations": [
            "Authored smoke-text NLL is not a standard reasoning/coding accuracy benchmark.",
            "Only gate/up pairs vary; other eligible weights use 4-bit affine quantization.",
            "Parameter budget excludes KV cache, activations, allocator workspace, and macOS.",
            "Summed isolated gate/up latency is a proxy, not a full-model latency guarantee.",
            "Teacher-forced decode timing excludes token sampling and text rendering.",
            "Forced-fusion ablation bypasses automatic performance fallback on purpose.",
            "Reference must fit during calibration; layer-sharded loading is not implemented.",
            "Prior swap and background workloads can influence measurements.",
        ],
    }
    _dump(output / "results.json", result)
    _write_report(output / "report.md", result)
    print(f"Saved actual results: {output / 'results.json'}", flush=True)
    print(f"Saved runnable mixed-precision model: {saved}", flush=True)
    print(f"Actual generation: {sample}", flush=True)
    return 0


def _write_report(path, result):
    baseline = result["variants"]["uniform4"]["timing"]["median_decode_tokens_per_second"]
    lines = [
        "# ParetoQuant-Metal measured smoke experiment",
        "",
        f"Device: {result['environment_before']['device']['device_name']}",
        f"Source: {result['source_model']}",
        "",
        "These are actual measurements, not projected larger-model results.",
        "",
        "| Variant | Parameter MiB | Smoke NLL | Decode tok/s | Ratio vs uniform4 | Fused pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, variant in result["variants"].items():
        speed = variant["timing"]["median_decode_tokens_per_second"]
        lines.append(
            f"| {name} | {variant['parameter_bytes'] / 1024**2:.2f} | "
            f"{variant['quality']['mean_nll']:.4f} | {speed:.2f} | "
            f"{speed / baseline:.3f}x | {variant['fused_pair_count']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {s}" for s in result["limitations"]],
            "",
            "## Actual mixed-precision model output",
            "",
            result["hardware_fused_generation"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def generate_local(args):
    from mlx_lm import generate, load

    from .benchmark import environment
    from .runtime import install_fusion

    source = _local_model(args.model)
    model, tokenizer = load(str(source))
    manifest = source / "execution_manifest.json"
    if not args.stock:
        try:
            if not manifest.is_file():
                raise ValueError("no execution manifest; fused dispatch is unverified")
            dispatch = verify_manifest(source, load_manifest(manifest), environment())
            validate_model_dispatch(model, dispatch)
            install_fusion(model, dispatch)
        except (ValueError, OSError, RuntimeError) as error:
            # install_fusion validates all replacements before mutating the model.
            print(f"Using stock kernels: {error}", file=sys.stderr)
    if getattr(args, "compiled", False):
        from .decode import FixedDecoder, greedy_token_ids

        ids = tokenizer.encode(_chat_prompt(tokenizer, args.prompt))
        decoder = FixedDecoder(model, capacity=len(ids) + args.max_tokens)
        tokens = greedy_token_ids(
            decoder, ids, max_tokens=args.max_tokens, eos_tokens=tokenizer.eos_token_ids
        )
        if tokens and tokens[-1] in tokenizer.eos_token_ids:
            tokens = tokens[:-1]
        print(tokenizer.decode(tokens))
        return 0
    print(
        generate(
            model,
            tokenizer,
            prompt=_chat_prompt(tokenizer, args.prompt),
            max_tokens=args.max_tokens,
            verbose=False,
        )
    )
    return 0


def replay_experiment(args):
    if args.repeats < 2:
        raise ValueError("paired replay requires at least two repeats")
    from .benchmark import environment

    source = _local_model(args.model)
    manifest_path = source / "execution_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("replay requires a saved model with execution_manifest.json")
    manifest = load_manifest(manifest_path)
    current = environment()
    dispatch = verify_manifest(source, manifest, current)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from .evaluation import cached_decode_benchmark, reference_schedule
    from .pipeline import model_bytes
    from .runtime import install_fusion
    from .statistics import paired_latency_ratio

    output = _prepare_output(args.output, source)
    stock, tokenizer = load(str(source))
    fused, _ = load(str(source))
    validate_model_dispatch(fused, dispatch)
    installed = install_fusion(fused, dispatch)
    if not installed:
        raise ValueError("manifest selected no fused pairs; there is no fusion effect to replay")
    if model_bytes(stock) != model_bytes(fused):
        raise ValueError("replay models do not have identical parameter storage")
    calibration = _texts(None, "data/calibration.json")
    contexts = {
        "short": "Explain binary search briefly.",
        "medium": calibration[0] + "\nExplain the boundary cases of binary search.",
        "long": "\n".join([calibration[1]] * 6)
        + "\nExplain why memory access patterns matter for GPU kernels.",
    }
    cases = {}
    for name, prompt in contexts.items():
        print(f"Paired replay: {name} context...", flush=True)
        ids = tokenizer.encode(_chat_prompt(tokenizer, prompt))
        schedule = reference_schedule(stock, ids, steps=args.decode_steps)
        caches = [make_prompt_cache(stock), make_prompt_cache(fused)]
        for candidate, cache in zip((stock, fused), caches):
            mx.eval(candidate(mx.array([ids]), cache=cache))
        stock_logits = stock(mx.array([[schedule[0]]]), cache=caches[0]).astype(mx.float32)
        fused_logits = fused(mx.array([[schedule[0]]]), cache=caches[1]).astype(mx.float32)
        delta = stock_logits - fused_logits
        numerical = {
            "all_logits_finite": bool(
                mx.all(mx.isfinite(stock_logits)).item()
                and mx.all(mx.isfinite(fused_logits)).item()
            ),
            "max_absolute_logit_error": mx.max(mx.abs(delta)).item(),
            "logit_rmse": mx.sqrt(mx.mean(delta**2)).item(),
            "next_token_argmax_matches": bool(
                mx.argmax(stock_logits[0, -1]).item() == mx.argmax(fused_logits[0, -1]).item()
            ),
        }
        if not numerical["all_logits_finite"]:
            raise ValueError("nonfinite logits in replay numerical check")
        timing = cached_decode_benchmark(
            {"stock": stock, "fused": fused}, ids, schedule, repeats=args.repeats, warmup=2
        )
        interval = paired_latency_ratio(
            timing["stock"]["decode_samples_ms"], timing["fused"]["decode_samples_ms"]
        )
        cases[name] = {
            "prompt_token_count": len(ids),
            "schedule_token_ids": schedule,
            "timing": timing,
            "fusion_latency_ratio": interval,
            "numerical_check": numerical,
        }
        print(
            f"  observed ratio {interval['estimate']:.3f}x; bootstrap interval "
            f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]",
            flush=True,
        )
    root = Path(__file__).parent
    code_paths = sorted([*root.rglob("*.py"), *root.rglob("*.metal")])
    result = {
        "schema_version": 1,
        "environment_before": current,
        "environment_after": environment(),
        "source_weights_sha256": _source_hashes(source),
        "source_code_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_paths
        },
        "parameter_bytes_each_model": model_bytes(stock),
        "fused_pair_count": len(installed),
        "cases": cases,
        "limitations": [
            "Same saved precision map and weights; only gate/up execution backend differs.",
            "Teacher-forced wall-clock timing excludes token sampling and rendering.",
            "Bootstrap intervals cover sampled trials, not other models or devices.",
            "Background load, cache residency, and prior swap can influence latency.",
            "Three authored contexts are not a standard task-accuracy benchmark.",
        ],
    }
    _dump(output / "replay.json", result)
    lines = [
        "# Same-weight fused/stock replay",
        "",
        "| Context | Prompt tokens | Stock tok/s | Fused tok/s | Ratio | Bootstrap 95% interval |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, case in cases.items():
        timing, interval = case["timing"], case["fusion_latency_ratio"]
        lines.append(
            f"| {name} | {case['prompt_token_count']} | "
            f"{timing['stock']['median_decode_tokens_per_second']:.2f} | "
            f"{timing['fused']['median_decode_tokens_per_second']:.2f} | "
            f"{interval['estimate']:.3f}x | "
            f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}] |"
        )
    lines += ["", "## Limits", "", *[f"- {s}" for s in result["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Saved paired replay: {output / 'replay.json'}", flush=True)
    return 0


def seal_model(args):
    """Explicitly adopt current artifact/kernel bytes, without replacing profile evidence."""
    source = _local_model(args.model)
    output = Path(args.output).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("seal output must be outside and unrelated to the source directory")
    if output.exists():
        raise ValueError("seal output must be a new directory; preserve prior evidence")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("seal requires regular files, not symlinks, to preserve evidence")
    original_path = source / "execution_manifest.json"
    original = original_path.read_bytes()
    old = load_manifest(original_path, allow_legacy=True)
    if old["schema_version"] == 2:
        validate_manifest(source, old, None, strict_runtime=False)
    profile = {"device": old["profile_device"], "mlx": old["mlx"], "mlx_lm": old["mlx_lm"]}
    sealed = create_manifest(source, old["dispatch"], profile)
    digest = hashlib.sha256(original).hexdigest()
    sealed["seal"] = {"source_manifest_sha256": digest, "profiling_performed": False}
    sealed = validate_schema(sealed)
    # All validation precedes copying; never modify or overwrite the original model.
    shutil.copytree(source, output)
    archive = output / f"execution_manifest.original-{digest}.json"
    if archive.exists() and archive.read_bytes() != original:
        raise ValueError("source contains conflicting original-manifest evidence")
    archive.write_bytes(original)
    # Detect copying races before authorizing the copied model's dispatch.
    validate_manifest(output, sealed, None, strict_runtime=False)
    _dump(output / "execution_manifest.json", sealed)
    validate_manifest(
        output, load_manifest(output / "execution_manifest.json"), None, strict_runtime=False
    )
    print(f"Saved sealed model: {output}")
    print("Sealing did not perform profiling; original hardware/runtime metadata is preserved.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mixed-precision allocation + fused Metal decode experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print actual local MLX/Metal environment")
    seal = sub.add_parser(
        "seal", help="Copy an existing saved model and bind its dispatch (no profiling)"
    )
    seal.add_argument("--model", required=True)
    seal.add_argument("--output", required=True, help="New model directory outside the source")
    replay = sub.add_parser("replay", help="Compare saved fused/stock execution with paired trials")
    replay.add_argument("--model", required=True)
    replay.add_argument("--output", default="artifacts/replay")
    replay.add_argument("--decode-steps", type=_positive_int, default=64)
    replay.add_argument("--repeats", type=_positive_int, default=15)
    run = sub.add_parser(
        "run", help="Calibrate, profile, allocate, evaluate, benchmark and save a local model"
    )
    run.add_argument(
        "--model", required=True, help="Local unquantized MLX-compatible model directory"
    )
    run.add_argument("--output", default="artifacts/run")
    run.add_argument("--calibration", help="JSON list of calibration texts")
    run.add_argument("--evaluation", help="Disjoint JSON list of held-out texts")
    run.add_argument("--memory-budget-mib", type=_positive)
    run.add_argument("--latency-factor", type=_positive, default=1.0)
    run.add_argument("--profile-repeats", type=_positive_int, default=20)
    run.add_argument("--decode-steps", type=_positive_int, default=32)
    run.add_argument("--decode-repeats", type=_positive_int, default=5)
    run.add_argument("--max-states", type=_positive_int, default=10000)
    gen = sub.add_parser(
        "generate", help="Reload saved precision map and compatible fused dispatch"
    )
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=_positive_int, default=64)
    gen.add_argument(
        "--compiled",
        action="store_true",
        help="Opt-in compiled Qwen2 one-token decode with fixed-capacity KV state (eager prefill)",
    )
    gen.add_argument(
        "--stock", action="store_true", help="Ignore fused manifest for a stock-kernel replay"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .benchmark import environment

            print(json.dumps(environment(), indent=2))
            return 0
        _local_model(args.model)
        if args.command == "run":
            return run_experiment(args)
        if args.command == "replay":
            return replay_experiment(args)
        if args.command == "seal":
            return seal_model(args)
        return generate_local(args)
    except (ValueError, FileNotFoundError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
