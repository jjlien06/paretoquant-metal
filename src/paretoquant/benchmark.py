"""Synchronized, interleaved wall-clock benchmarks with raw sample retention."""

import importlib.metadata
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from time import perf_counter_ns

import mlx.core as mx


def environment():
    def command(args):
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=5)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "device": mx.device_info(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "git_revision": command(["git", "rev-parse", "HEAD"]),
        "swap_usage": command(["sysctl", "vm.swapusage"]),
    }


def benchmark_functions(functions, *, warmup=5, repeats=30):
    """Time fresh calls, not graph construction or already-evaluated outputs.

    These measurements include Python dispatch, evaluation, and synchronization;
    they are NOT GPU-only device times. Methods are alternated to reduce drift.
    """
    if not functions:
        raise ValueError("at least one function is required")
    if not isinstance(warmup, int) or not isinstance(repeats, int) or warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive integers")
    names = list(functions)
    for _ in range(warmup):
        for fn in functions.values():
            mx.eval(fn())
    mx.synchronize()
    samples = {name: [] for name in names}
    for trial in range(repeats):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        for name in order:
            mx.synchronize()
            start = perf_counter_ns()
            output = functions[name]()
            mx.eval(output)
            mx.synchronize()
            samples[name].append((perf_counter_ns() - start) / 1e6)
    return {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "warmup": warmup,
            "repeats": repeats,
            "measurement": "synchronized_wall_clock",
        }
        for name, values in samples.items()
    }
