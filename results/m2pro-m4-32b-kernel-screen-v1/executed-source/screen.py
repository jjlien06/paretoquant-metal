"""Actual packed-weight GPU diagnostic; import-safe on a CPU-only machine."""

import argparse
import hashlib
import json
import math
import random
import re
import stat
import statistics
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bounded_file(path, limit):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError(f"requires bounded regular file, no symlink: {path}")
    return info.st_size


EXPECTED = {
    f"model.layers.0.mlp.{projection}.{kind}": (
        [n, k // (8 if kind == "weight" else 64)],
        "<u4" if kind == "weight" else "<f2",
    )
    for projection, n, k in (
        ("gate_proj", 27648, 5120),
        ("up_proj", 27648, 5120),
        ("down_proj", 5120, 27648),
    )
    for kind in ("weight", "scales", "biases")
}


def validate_assets(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("assets must be a real directory")
    bounded_file(directory / "manifest.json", 1024 * 1024)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["configuration"] != {"bits": 4, "group_size": 64}:
        raise ValueError("requires affine 4-bit group_size=64")
    if not re.fullmatch("[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("source revision must be a pinned lowercase commit")
    if not re.fullmatch("[0-9a-f]{64}", manifest["source_config_sha256"]):
        raise ValueError("invalid source config hash")
    if set(manifest["tensors"]) != set(EXPECTED):
        raise ValueError("requires exactly nine original tensor names")
    total, paths = 0, set()
    for name, (shape, dtype) in EXPECTED.items():
        info = manifest["tensors"][name]
        if (info["shape"], info["dtype"], info["original_name"]) != (shape, dtype, name):
            raise ValueError(f"incorrect metadata: {name}")
        filename = info["file"]
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.npy", filename):
            raise ValueError(f"only basename .npy paths allowed: {filename}")
        if filename in paths:
            raise ValueError("duplicate tensor file")
        paths.add(filename)
        total += bounded_file(directory / filename, 256 * 1024 * 1024)
        for key in ("npy_sha256", "raw_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", info[key]):
                raise ValueError(f"invalid {key}: {name}")
    if total > 300 * 1024 * 1024:
        raise ValueError("total tensor files exceed 300 MiB")
    for name, info in manifest["tensors"].items():
        if digest(directory / info["file"]) != info["npy_sha256"]:
            raise ValueError(f"file hash mismatch: {name}")
    return manifest


def load_arrays(directory, manifest, np):
    arrays = {}
    for name, info in manifest["tensors"].items():
        path = Path(directory) / info["file"]
        bounded_file(path, 256 * 1024 * 1024)
        if digest(path) != info["npy_sha256"]:
            raise ValueError(f"file hash changed: {name}")
        array = np.load(path, allow_pickle=False)
        shape, dtype = EXPECTED[name]
        if tuple(array.shape) != tuple(shape) or array.dtype.str != dtype:
            raise ValueError(f"actual array shape/dtype mismatch: {name}")
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != info["raw_sha256"]:
            raise ValueError(f"raw array hash mismatch: {name}")
        arrays[name] = array
    return arrays


RTOL, ATOL = 0.01, 0.02


def synthetic_rows(seed):
    rows = []
    for offset in range(3):
        rng = random.Random(seed + offset)
        row = [rng.gauss(0, 1) for _ in range(5120)]
        rms = math.sqrt(sum(v * v for v in row) / len(row))
        rows.append([v / rms for v in row])
    return rows


def parity(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError("parity requires equal nonempty output shapes")
    finite = all(math.isfinite(v) for v in (*reference, *candidate))
    if not finite:
        return dict(passed=False, finite=False, relative_mse=None, max_abs_error=None)
    errors = [abs(a - b) for a, b in zip(reference, candidate)]
    return dict(
        passed=all(e <= ATOL + RTOL * abs(a) for a, e in zip(reference, errors)),
        finite=True,
        max_abs_error=max(errors),
        relative_mse=sum(e * e for e in errors) / max(sum(a * a for a in reference), 1e-30),
    )


def screen_functions(functions, inputs, read, benchmark, *, repeats, warmup):
    checks = {name: [] for name in functions}
    for x in inputs:
        reference = read(functions["native_stock"](x))
        compiled = read(functions["compiled_stock"](x))
        for name, fn in functions.items():
            output = read(fn(x))
            native_check, compiled_check = parity(reference, output), parity(compiled, output)
            checks[name].append(
                dict(
                    native_check,
                    vs_compiled_stock=compiled_check,
                    passed=native_check["passed"] and compiled_check["passed"],
                )
            )
    valid = [name for name in functions if all(v["passed"] for v in checks[name])]
    result = dict(status="screened", parity=checks, timings={}, summary={}, winners=[])
    if "native_stock" not in valid or "compiled_stock" not in valid:
        result["status"] = "control_parity_failed"
        return result
    for phase_index, phase in enumerate(("tuning", "validation")):
        rounds = []
        for index, x in enumerate(inputs):
            shift = (index + phase_index) % len(valid)
            order = valid[shift:] + valid[:shift]
            calls = {name: (lambda fn=functions[name], x=x: fn(x)) for name in order}
            rounds.append(benchmark(calls, warmup=warmup, repeats=repeats))
        result["timings"][phase] = rounds
        result["summary"][phase] = {
            name: dict(
                median_ms=statistics.median([v for r in rounds for v in r[name]["samples_ms"]]),
                sample_count=sum(len(r[name]["samples_ms"]) for r in rounds),
            )
            for name in valid
        }
    summary = result["summary"]["validation"]
    baseline = summary["compiled_stock"]["median_ms"]
    for name in valid:
        improvement = 1 - summary[name]["median_ms"] / baseline
        summary[name]["improvement_vs_compiled_stock"] = improvement
        if name.startswith("fused_") and summary[name]["median_ms"] < baseline * 0.95:
            result["winners"].append(name)
    return result


def build_functions(mx, kernels, weights):
    gate, up, down = [
        tuple(
            weights[f"model.layers.0.mlp.{proj}.{kind}"] for kind in ("weight", "scales", "biases")
        )
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    options = dict(bits=4, group_size=64)
    gate_up = dict(
        native_stock=lambda x: kernels.stock_gate_up(x, gate, up, **options),
        compiled_stock=lambda x: kernels.compiled_stock_gate_up(x, gate, up, **options),
    )
    for rpg in (1, 2, 4, 8):
        gate_up[f"fused_rpg{rpg}"] = lambda x, rpg=rpg: kernels.compiled_fused_gate_up(
            x, gate, up, rows_per_group=rpg, **options
        )

    def project(activation):
        return mx.quantized_matmul(activation, *down, transpose=True, **options)

    # Captures ONLY validated layer arrays; never loads/requantizes a model.
    mlp = dict(
        native_stock=lambda x: project(gate_up["native_stock"](x)),
        compiled_stock=mx.compile(lambda x: project(gate_up["native_stock"](x))),
    )
    for name, fn in gate_up.items():
        if name.startswith("fused_"):
            mlp[name] = mx.compile(lambda x, fn=fn: project(fn(x)))
    return dict(gate_up=gate_up, mlp=mlp)


def gpu_screen(mx, kernels, benchmark, arrays, args):
    previous, cache = mx.default_device(), None
    try:
        mx.set_default_device(mx.gpu)
        cache = mx.set_cache_limit(256 * 1024 * 1024)
        mx.reset_peak_memory()
        weights = {name: mx.array(value) for name, value in arrays.items()}
        inputs = [
            mx.array(row, dtype=mx.float16).reshape(1, 5120) for row in synthetic_rows(args.seed)
        ]
        mx.eval(*weights.values(), *inputs)
        functions = build_functions(mx, kernels, weights)

        def read(output):
            mx.eval(output)
            return output.reshape(-1).tolist()

        options = dict(repeats=args.repeats, warmup=args.warmup)
        gate = screen_functions(functions["gate_up"], inputs, read, benchmark, **options)
        excluded = [
            name
            for name in functions["mlp"]
            if name.startswith("fused_") and not all(v["passed"] for v in gate["parity"][name])
        ]
        mlp = (
            screen_functions(
                {n: fn for n, fn in functions["mlp"].items() if n not in excluded},
                inputs,
                read,
                benchmark,
                **options,
            )
            if gate["status"] == "screened"
            else dict(status="skipped_gate_up_control_failure")
        )
        mlp["excluded_gate_up_failures"] = excluded
        return dict(
            gate_up=gate,
            mlp=mlp,
            peak_mlx_bytes=mx.get_peak_memory(),
            cache_limit_bytes=256 * 1024 * 1024,
            input_shape=[1, 5120],
            input_fp16_rms=[math.sqrt(sum(v * v for v in read(x)) / 5120) for x in inputs],
        )
    finally:
        if cache is not None:
            mx.set_cache_limit(cache)
        mx.set_default_device(previous)


def source_hashes(package):
    names = ["metal.py", "benchmark.py"] + [
        f"kernels/{p.name}"
        for p in sorted(package.joinpath("kernels").iterdir(), key=lambda p: p.name)
        if p.name.endswith(".metal")
    ]
    return {
        **{name: hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest() for name in names},
        "screen.py": digest(Path(__file__)),
    }


def run_gpu(arrays, args):
    # Every asset file AND raw array has been verified before reaching these imports.
    from importlib.resources import files

    import mlx.core as mx

    from paretoquant import metal
    from paretoquant.benchmark import benchmark_functions, environment

    if not mx.metal.is_available():
        raise RuntimeError("a physical Metal GPU is required")
    sources = source_hashes(files("paretoquant"))
    result = gpu_screen(mx, metal, benchmark_functions, arrays, args)
    return dict(result, environment=environment(), source_sha256=sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="NEW exclusive JSON evidence")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive; warmup must be nonnegative")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    manifest = validate_assets(args.assets)
    evidence = dict(
        manifest=manifest,
        manifest_sha256=digest(args.assets / "manifest.json"),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        activations="SYNTHETIC diagnostic activations; not captured model inputs",
        weights="actual checkpoint packed weights, unchanged",
        scope="layer-0 kernel screen only; no end-to-end speedup or 32B quality proof",
        parity_tolerance=dict(rtol=RTOL, atol=ATOL),
        winner_rule="parity on all three rows; validation median >5% faster than compiled stock",
    )
    with args.output.open("x") as handle:
        try:
            import numpy as np

            arrays = load_arrays(args.assets, manifest, np)
            evidence.update(run_gpu(arrays, args))
            evidence["status"] = "complete"
        except Exception as exc:
            evidence.update(status="error", error=f"{type(exc).__name__}: {exc}")
            json.dump(evidence, handle, indent=2, allow_nan=False)
            raise
        json.dump(evidence, handle, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    main()
