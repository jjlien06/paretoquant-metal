"""Retune execution on saved weights, never allocate or change their precision.

CPU policy and filesystem checks intentionally do not import MLX.
"""

import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path


def _no_symlink(path):
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlinks are not allowed: {part}")


def snapshot_hashes(root):
    """Hash every regular file, including tokenizer and original provenance."""
    root = Path(root).absolute()
    _no_symlink(root)
    if not root.is_dir():
        raise ValueError(f"local saved model directory required: {root}")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"regular non-symlink files/directories required: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                hashes[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    return hashes


def check_paths(source, output):
    """Read-only preflight; an output must be explicitly new and disjoint."""
    source, output = Path(source).expanduser().absolute(), Path(output).expanduser().absolute()
    _no_symlink(source)
    _no_symlink(output)
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must not overlap")
    if output.exists():
        raise ValueError("output must be a new nonexistent directory (never overwrite)")
    snapshot_hashes(source)
    if not (source / "config.json").is_file() or not list(source.glob("model*.safetensors")):
        raise ValueError("local saved config.json and model*.safetensors are required")
    return source, output


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)


def read_saved_config(source):
    """Reject unsupported saved quantization before loading or writing anything."""
    config = _read_json(Path(source) / "config.json")
    if not isinstance(config, dict) or config.get("model_type") != "qwen2":
        raise ValueError("retuning supports only saved Qwen2 models")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("retuning requires exact Qwen2 SiLU MLP semantics")
    quantization = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quantization, dict) or not quantization:
        raise ValueError("saved affine quantization metadata required")
    if "quantization" in config and "quantization_config" in config:
        if config["quantization"] != config["quantization_config"]:
            raise ValueError("conflicting saved quantization metadata")
    specifications = [quantization]
    for key, value in quantization.items():
        if key not in ("bits", "group_size", "mode"):
            if not isinstance(value, dict):
                raise ValueError(f"unsupported saved quantization override: {key}")
            specifications.append(value)
    for spec in specifications:
        if (
            spec.get("mode", "affine") != "affine"
            or type(spec.get("bits")) is not int
            or spec["bits"] not in (3, 4, 6)
            or type(spec.get("group_size")) is not int
            or spec["group_size"] not in (32, 64, 128)
        ):
            raise ValueError("unsupported saved quantization: affine bits 3/4/6, groups 32/64/128")
    return config


def select_dispatch(validation, best, *, bits, min_improvement=0.05):
    """Use fresh medians and a strict margin; this is not a significance test."""
    if type(bits) is not int or bits not in (3, 4, 6):
        raise ValueError("bits must be 3, 4, or 6")
    if (
        isinstance(min_improvement, bool)
        or not math.isfinite(min_improvement)
        or not (0 <= min_improvement < 1)
    ):
        raise ValueError("min_improvement must be finite and in [0, 1)")
    stock = {"backend": "stock", "rows_per_group": 4, "bits": bits}
    if best is None:
        return stock
    if best not in {f"fused:rpg{rpg}" for rpg in (1, 2, 4, 8)}:
        raise ValueError("unknown fused candidate")
    medians = {}
    for name in ("stock", best):
        values = validation.get(name, {}).get("samples_ms", [])
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            return stock
        medians[name] = statistics.median(values)
    if len(validation["stock"]["samples_ms"]) != len(validation[best]["samples_ms"]):
        return stock
    if medians[best] < medians["stock"] * (1 - min_improvement):
        return {"backend": "fused", "rows_per_group": int(best[-1]), "bits": bits}
    return stock


def numerical_errors(actual, expected, *, rtol=0.01, atol=0.02):
    """JSON-safe finite error diagnostics against the stock affine operation."""
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    result = {
        "valid": False,
        "rtol": rtol,
        "atol": atol,
        "elements": int(expected.size),
        "max_absolute_error": None,
        "max_relative_error": None,
        "relative_mse": None,
    }
    if actual.shape != expected.shape or not expected.size:
        return {**result, "reason": "shape_mismatch_or_empty"}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {**result, "reason": "nonfinite"}
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    difference = np.abs(a - e)
    valid = bool(np.allclose(a, e, rtol=rtol, atol=atol))
    result.update(
        valid=valid,
        max_absolute_error=float(difference.max()),
        max_relative_error=float((difference / np.maximum(np.abs(e), 1e-12)).max()),
        relative_mse=float(np.mean(difference**2) / max(float(np.mean(e**2)), 1e-12)),
        reason="allclose" if valid else "tolerance_exceeded",
    )
    return result


def profile_pair(
    inputs,
    gate,
    up,
    *,
    bits,
    group_size,
    repeats=20,
    warmup=3,
    min_improvement=0.05,
    stock_function=None,
    fused_function=None,
    benchmark=None,
    evaluate=None,
):
    """Validate all captured rows, tune four launches, then remeasure stock vs best.

    Dependency injection allows CPU policy tests without executing/importing Metal.
    Production defaults use compiled functions with explicit dynamic weight inputs.
    """
    import numpy as np

    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative integers")
    select_dispatch({}, None, bits=bits, min_improvement=min_improvement)
    if inputs.ndim != 2 or inputs.shape[0] < 1:
        raise ValueError("nonempty two-dimensional captured inputs required")
    if stock_function is None or fused_function is None:
        from .metal import compiled_fused_gate_up, compiled_stock_gate_up

        stock_function = stock_function or compiled_stock_gate_up
        fused_function = fused_function or compiled_fused_gate_up
    if benchmark is None:
        from .benchmark import benchmark_functions

        benchmark = benchmark_functions
    if evaluate is None:
        import mlx.core as mx

        def evaluate(value):
            mx.eval(value)
            return np.array(value)

    def stock(x=inputs[-1:]):
        return stock_function(x, gate, up, bits=bits, group_size=group_size)

    expected = np.concatenate([evaluate(stock(inputs[i : i + 1])) for i in range(inputs.shape[0])])
    candidates, errors = {"stock": stock}, {}
    for rpg in (1, 2, 4, 8):

        def candidate(x=inputs[-1:], rpg=rpg):
            return fused_function(x, gate, up, bits=bits, group_size=group_size, rows_per_group=rpg)

        try:
            actual = np.concatenate(
                [evaluate(candidate(inputs[i : i + 1])) for i in range(inputs.shape[0])]
            )
            diagnostic = numerical_errors(actual, expected)
        except (ValueError, RuntimeError) as error:
            diagnostic = {"valid": False, "reason": "kernel_error", "error": str(error)}
        errors[str(rpg)] = diagnostic
        if diagnostic["valid"]:
            candidates[f"fused:rpg{rpg}"] = candidate
    tuning = benchmark(candidates, warmup=warmup, repeats=repeats)
    valid_names = [name for name in candidates if name != "stock"]
    best = min(valid_names, key=lambda name: tuning[name]["median_ms"]) if valid_names else None
    functions = {"stock": stock}
    if best is not None:
        functions[best] = candidates[best]
    validation = benchmark(functions, warmup=warmup, repeats=repeats)
    dispatch = select_dispatch(validation, best, bits=bits, min_improvement=min_improvement)
    return {
        "bits": bits,
        "group_size": group_size,
        "calibration_input_count": int(inputs.shape[0]),
        "input_dtype": str(inputs.dtype),
        "timing_input": "last_captured_token",
        "numerical_validation": errors,
        "tuning": tuning,
        "best_tuning_candidate": best,
        "validation": validation,
        "dispatch": dispatch,
    }


def read_calibration(path=None):
    """Use local JSON only; bind the exact bytes and capture settings."""
    path = Path(path) if path is not None else Path(__file__).parent / "data/calibration.json"
    _no_symlink(path.absolute())
    raw = path.read_bytes()
    texts = _read_json(path)
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(text, str) and text.strip() for text in texts)
    ):
        raise ValueError("calibration JSON must be a nonempty list of nonempty strings")
    settings = {"texts": texts, "max_tokens": 128, "samples_per_text": 16}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return texts, {
        **settings,
        "path": str(path.resolve()),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_sha256": digest,
    }


def source_binding(source):
    """Check an existing v2 model seal only; never reuse its dispatch or runtime."""
    from .manifest import model_hashes

    path = Path(source) / "execution_manifest.json"
    if not path.exists():
        return {"schema_version": None, "verified": False, "model_sha256": model_hashes(source)}
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("source manifest must be a JSON object")
    version = data.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported source manifest version")
    hashes = model_hashes(source)
    if version == 2 and data.get("model_sha256") != hashes:
        raise ValueError("source model binding SHA-256 mismatch; retuning cannot repair corruption")
    return {"schema_version": version, "verified": version == 2, "model_sha256": hashes}


def publish_retune(source, output, profile):
    """Copy untouched files; archive only the manifest replaced by fresh dispatch."""
    from .manifest import create_manifest, load_manifest, validate_manifest

    source, output = check_paths(source, output)
    original = snapshot_hashes(source)
    if original != profile["source_file_sha256"]:
        raise ValueError("source changed since profiling began")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "model"
    shutil.copytree(source, target, symlinks=True)
    copied = snapshot_hashes(target)
    if copied != original:
        raise ValueError("copied source snapshot SHA-256 mismatch")
    old_manifest = target / "execution_manifest.json"
    if old_manifest.exists():
        archive = output / "provenance" / "source_execution_manifest.json"
        archive.parent.mkdir()
        shutil.copy2(old_manifest, archive)
        if snapshot_hashes(archive.parent)[archive.name] != original[old_manifest.name]:
            raise ValueError("original manifest archive SHA-256 mismatch")
    manifest = create_manifest(target, profile["dispatch"], profile["environment"])
    old_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    new_files = snapshot_hashes(target)
    source_payload = {k: v for k, v in original.items() if k != "execution_manifest.json"}
    new_payload = {k: v for k, v in new_files.items() if k != "execution_manifest.json"}
    if source_payload != new_payload or snapshot_hashes(source) != original:
        raise ValueError("retuning changed source or saved payload SHA-256")
    validated = load_manifest(old_manifest)
    if validate_manifest(target, validated, profile["environment"]) != profile["dispatch"]:
        raise ValueError("published manifest dispatch mismatch")
    report = {
        **profile,
        "source": str(source),
        "output_model": str(target),
        "copied_source_file_sha256": copied,
        "new_file_sha256": new_files,
        "source_payload_sha256": source_payload,
        "new_payload_sha256": new_payload,
        "source_execution_manifest_sha256": original.get("execution_manifest.json"),
        "source_execution_manifest_archive": (
            "provenance/source_execution_manifest.json"
            if "execution_manifest.json" in original
            else None
        ),
        "original_dispatch_trusted": False,
        "kernel_sha256": validated["kernel_sha256"],
        "model_sha256": validated["model_sha256"],
    }
    (output / "retune_profile.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    # Read back the exact publication targets rather than trusting write success.
    if _read_json(output / "retune_profile.json") != report:
        raise ValueError("published profile readback mismatch")
    return report


def _source_code_hashes():
    root = Path(__file__).parent
    paths = [
        root / name
        for name in (
            "retune.py",
            "metal.py",
            "benchmark.py",
            "pipeline.py",
            "adapters.py",
            "runtime.py",
            "manifest.py",
        )
    ]
    script = root.parents[1] / "scripts/retune_model.py"
    if script.is_file():
        paths.append(script)
    return {
        (
            f"scripts/{path.name}" if path == script else f"src/paretoquant/{path.name}"
        ): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def retune_saved_model(
    model_dir,
    output,
    *,
    calibration=None,
    repeats=20,
    warmup=3,
    min_improvement=0.05,
    verbose=True,
):
    """Fresh local pair profiling, with unchanged loaded/saved precision and weights.

    No legacy dispatch is installed and no allocation, conversion, custom down
    projection, attention optimization, or whole-model decode benchmark is run.
    """
    source, output = check_paths(model_dir, output)
    config = read_saved_config(source)
    binding = source_binding(source)
    texts, calibration_metadata = read_calibration(calibration)
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative integers")
    select_dispatch({}, None, bits=4, min_improvement=min_improvement)
    source_files, code_hashes = snapshot_hashes(source), _source_code_hashes()

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model, load_tokenizer

    from .benchmark import environment
    from .manifest import create_manifest, validate_model_dispatch
    from .pipeline import calibrate_inputs, mlp_modules, model_bytes

    if not mx.metal.is_available():
        raise ValueError("saved-model retuning requires an available Apple Silicon Metal GPU")
    observed = environment()
    # Direct local loaders do not activate execution manifests or trust remote code.
    model, _ = load_model(source, strict=True, trust_remote_code=False)
    tokenizer = load_tokenizer(source, eos_token_ids=config.get("eos_token_id"))
    modules = mlp_modules(model)
    if not modules:
        raise ValueError("no saved quantized Qwen2 gate/up pairs found")
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and module.mode != "affine":
            raise ValueError("unsupported loaded quantization mode; affine required")
    saved_dispatch = {}
    for name, module in modules.items():
        gate, up = module.gate_proj, module.up_proj
        if (
            type(gate) is not nn.QuantizedLinear
            or type(up) is not nn.QuantizedLinear
            or gate.bits != up.bits
            or gate.bits not in (3, 4, 6)
            or gate.group_size != up.group_size
            or gate.group_size not in (32, 64, 128)
            or gate.mode != "affine"
            or up.mode != "affine"
            or "bias" in gate
            or "bias" in up
        ):
            raise ValueError(f"unsupported loaded affine quantization pair: {name}")
        saved_dispatch[name] = {"backend": "stock", "bits": gate.bits, "rows_per_group": 4}
    # This is a NEW stock dispatch from loaded metadata, never the original manifest.
    create_manifest(source, saved_dispatch, observed)
    validate_model_dispatch(model, saved_dispatch)
    parameters = {name: value for name, value in tree_flatten(model.parameters())}
    resident_bytes = model_bytes(model)
    inputs = calibrate_inputs(
        model,
        tokenizer,
        texts,
        max_tokens=calibration_metadata["max_tokens"],
        samples_per_text=calibration_metadata["samples_per_text"],
    )
    units, dispatch, captured_hashes = [], {}, {}
    for name, module in modules.items():
        x = inputs[name]
        host = np.array(x)
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                {"shape": list(host.shape), "dtype": str(host.dtype)}, sort_keys=True
            ).encode()
        )
        hasher.update(host.tobytes())
        captured_hashes[name] = hasher.hexdigest()
        gate, up = module.gate_proj, module.up_proj
        unit = profile_pair(
            x,
            (gate.weight, gate.scales, gate.biases),
            (up.weight, up.scales, up.biases),
            bits=gate.bits,
            group_size=gate.group_size,
            repeats=repeats,
            warmup=warmup,
            min_improvement=min_improvement,
        )
        unit.update(
            name=name,
            packed_shape=list(gate.weight.shape),
            captured_input_sha256=captured_hashes[name],
        )
        units.append(unit)
        dispatch[name] = unit["dispatch"]
        if verbose:
            best = unit["best_tuning_candidate"]
            stock_ms = unit["validation"]["stock"]["median_ms"]
            candidate_ms = unit["validation"][best]["median_ms"] if best else None
            print(
                f"Retuned {name} q{gate.bits}: {unit['dispatch']['backend']}; "
                f"stock={stock_ms:.4f} ms best={candidate_ms} ms",
                flush=True,
            )
    final_parameters = dict(tree_flatten(model.parameters()))
    if (
        set(parameters) != set(final_parameters)
        or any(final_parameters[name] is not value for name, value in parameters.items())
        or model_bytes(model) != resident_bytes
    ):
        raise ValueError("profiling altered loaded model parameters")
    current = environment()
    if any(current[key] != observed[key] for key in ("device", "mlx", "mlx_lm")):
        raise ValueError("hardware/runtime changed during retuning")
    if _source_code_hashes() != code_hashes:
        raise ValueError("profiling source code changed during retuning")
    bits_counts = {
        str(bits): sum(unit["bits"] == bits for unit in units)
        for bits in (3, 4, 6)
        if any(unit["bits"] == bits for unit in units)
    }
    fused_count = sum(selection["backend"] == "fused" for selection in dispatch.values())
    report = {
        "schema_version": 1,
        "scope": "saved_affine_qwen2_gate_up_pairs_single_token",
        "whole_model_speedup_claim": False,
        "custom_down_or_attention_profiled": False,
        "original_dispatch_trusted": False,
        "environment": observed,
        "environment_end": current,
        "repeats": repeats,
        "warmup": warmup,
        "min_improvement": min_improvement,
        "selection_policy": "fresh_median_strict_margin_not_statistical_significance",
        "calibration": {**calibration_metadata, "captured_input_sha256": captured_hashes},
        "calibration_sha256": calibration_metadata["calibration_sha256"],
        "source_binding": binding,
        "source_file_sha256": source_files,
        "source_code_sha256": code_hashes,
        "resident_parameter_bytes": resident_bytes,
        "dispatch": dispatch,
        "units": units,
        "counts": {
            "pairs": len(units),
            "fused_pairs": fused_count,
            "stock_pairs": len(units) - fused_count,
            "bits": bits_counts,
            "numerically_valid_candidates": sum(
                error["valid"] for unit in units for error in unit["numerical_validation"].values()
            ),
        },
    }
    return publish_retune(source, output, report)
