"""Intrusive eager-decode diagnostics, never an additive model cost attribution.

Importing this module does not initialize MLX. Instrumentation changes only an
instance's Python class, not its module/parameter dictionary, and is reversible.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def validate_options(
    *,
    decode_steps=16,
    capture_step=0,
    repeats=10,
    warmup=2,
    full_repeats=3,
    full_warmup=1,
    max_prompt_tokens=4096,
):
    bounds = {
        "decode_steps": (decode_steps, 1, 512),
        "repeats": (repeats, 1, 200),
        "warmup": (warmup, 0, 50),
        "full_repeats": (full_repeats, 1, 20),
        "full_warmup": (full_warmup, 0, 10),
        "max_prompt_tokens": (max_prompt_tokens, 1, 32768),
    }
    for name, (value, low, high) in bounds.items():
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if type(capture_step) is not int or not 0 <= capture_step < decode_steps:
        raise ValueError("capture_step must be an integer in [0, decode_steps)")


def check_profile_paths(source, output):
    """Read-only local preflight. Output parent must already exist."""
    from .retune import _no_symlink

    source = Path(source).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    _no_symlink(source)
    _no_symlink(output)
    # Reject symlinks before resolving; then normalize '..' so alternate lexical
    # paths cannot disguise an output inside the saved checkpoint directory.
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError("existing local saved model directory required; no downloads")
    if source == output or source in output.parents:
        raise ValueError("output must not be inside the source model")
    if output.exists():
        raise ValueError("output must be a new nonexistent evidence file")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    if not all((source / name).is_file() for name in ("config.json", "execution_manifest.json")):
        raise ValueError("local config.json and execution_manifest.json required")
    if not list(source.glob("model*.safetensors")):
        raise ValueError("local model*.safetensors required")
    return source, output


def write_evidence(output, report):
    """Create final evidence exclusively; never rewrite previous evidence."""
    import json
    import os

    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(output).open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_admitted_model(source, current, *, stock=False):
    """Fail closed: strict bound manifest, local load, loaded bits, then fusion."""
    import json

    from mlx_lm import load
    from mlx_lm.models.qwen2 import Model

    from .manifest import load_manifest, validate_manifest, validate_model_dispatch
    from .runtime import install_fusion

    config = json.loads((Path(source) / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "qwen2":
        raise ValueError("profiling supports only local Qwen2 models")
    data = load_manifest(Path(source) / "execution_manifest.json")
    dispatch = validate_manifest(source, data, current, strict_runtime=True)
    model, tokenizer = load(
        str(source),
        tokenizer_config={"local_files_only": True, "trust_remote_code": False},
        trust_remote_code=False,
    )
    if type(model) is not Model:
        raise ValueError("profiling supports only exact mlx-lm Qwen2 Model")
    validate_model_dispatch(model, dispatch)
    installed = [] if stock else install_fusion(model, dispatch)
    return (
        model,
        tokenizer,
        {
            "dispatch": dispatch,
            "installed_fusion": installed,
            "stock_override": stock,
            "strict_runtime": True,
        },
    )


def time_operations(operations, *, repeats=10, warmup=2, benchmark=None):
    """Retain overlapping isolated measurements without additive attribution."""
    validate_options(repeats=repeats, warmup=warmup)
    if not operations:
        raise ValueError("no captured operations")
    if benchmark is None:
        from .benchmark import benchmark_functions

        benchmark = benchmark_functions
    timings = benchmark(
        {name: op.function for name, op in operations.items()}, repeats=repeats, warmup=warmup
    )
    groups = {}
    for name, op in operations.items():
        groups.setdefault(op.group, []).append(name)
    names = list(operations)
    return {
        "measurement": "intrusive_capture_isolated_synchronized_wall_clock",
        "additive_cost_attribution": False,
        "group_members": groups,
        "trial_order": [names[i % len(names) :] + names[: i % len(names)] for i in range(repeats)],
        "operations": {
            name: {"group": op.group, "activation": op.metadata, "timing": timings[name]}
            for name, op in operations.items()
        },
        "ranked_isolated_operations": sorted(
            names, key=lambda n: timings[n]["median_ms"], reverse=True
        ),
    }


def environment_snapshot():
    """Begin/end metadata without git operations or a server."""
    import importlib.metadata
    import platform
    import subprocess
    from datetime import datetime, timezone

    import mlx.core as mx

    def command(args):
        try:
            return subprocess.run(
                args, check=True, capture_output=True, text=True, timeout=5
            ).stdout.strip()
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
        "default_device": str(mx.default_device()),
        "swap_usage": command(["sysctl", "vm.swapusage"]),
        "vm_stat": command(["vm_stat"]),
    }


def execution_hashes():
    """Bind runner, profiler, dependencies and installed Qwen2/cache source."""
    import hashlib
    import inspect

    from mlx_lm.models import activations, cache, qwen2

    root = Path(__file__).resolve().parent
    paths = [
        root / name
        for name in (
            "profiling.py",
            "runtime.py",
            "benchmark.py",
            "evaluation.py",
            "manifest.py",
            "metal.py",
            "adapters.py",
            "retune.py",
        )
    ]
    paths.append(root.parents[1] / "scripts" / "profile_decode.py")
    paths.extend(sorted((root / "kernels").glob("*.metal")))
    paths.extend(Path(inspect.getfile(module)) for module in (qwen2, cache, activations))
    result = {}
    for path in paths:
        with path.open("rb") as handle:
            result[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def profile_saved_decode(
    source,
    output,
    *,
    prompt="Explain binary search briefly.",
    decode_steps=16,
    capture_step=0,
    repeats=10,
    warmup=2,
    full_repeats=3,
    full_warmup=1,
    max_prompt_tokens=4096,
    stock=False,
):
    """Run sequential diagnostics on one strictly admitted local saved model.

    No evidence is created before every measurement and read-only integrity
    recheck succeeds. Full-model timing precedes any activation instrumentation.
    """
    from .evaluation import cached_decode_benchmark, reference_schedule
    from .retune import snapshot_hashes

    options = dict(
        decode_steps=decode_steps,
        capture_step=capture_step,
        repeats=repeats,
        warmup=warmup,
        full_repeats=full_repeats,
        full_warmup=full_warmup,
        max_prompt_tokens=max_prompt_tokens,
    )
    validate_options(**options)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("nonempty prompt required")
    if type(stock) is not bool:
        raise ValueError("stock must be a boolean")
    source, output = check_profile_paths(source, output)
    model_hashes = snapshot_hashes(source)
    code_hashes = execution_hashes()
    begin = environment_snapshot()
    model, tokenizer, admission = load_admitted_model(source, begin, stock=stock)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(rendered)
    if not ids or len(ids) > max_prompt_tokens:
        raise ValueError("chat prompt must have 1..max_prompt_tokens tokens; never truncated")
    limit = getattr(getattr(model, "args", None), "max_position_embeddings", None)
    if limit is not None and len(ids) + decode_steps > limit:
        raise ValueError("prompt plus decode schedule exceeds model context limit")
    schedule = reference_schedule(model, ids, steps=decode_steps)
    full = cached_decode_benchmark(
        {"admitted": model},
        ids,
        schedule,
        repeats=full_repeats,
        warmup=full_warmup,
    )["admitted"]
    operations = capture_decode_operations(model, ids, schedule, capture_step=capture_step)
    isolated = time_operations(operations, repeats=repeats, warmup=warmup)
    # Release captured inputs/cache snapshots before environment-end collection.
    del operations
    end = environment_snapshot()
    if snapshot_hashes(source) != model_hashes or execution_hashes() != code_hashes:
        raise ValueError("model or profiling code changed during measurements; no evidence written")
    report = {
        "schema_version": 1,
        "scope": "single_step_eager_qwen2_decode_diagnostics",
        "source_model": str(source),
        "options": options,
        "admission": admission,
        "environment_begin": begin,
        "environment_end": end,
        "hashes": {"model_files": model_hashes, "execution_files": code_hashes},
        "prompt": {
            "user_text": prompt,
            "rendered_chat_template": rendered,
            "template_options": {"tokenize": False, "add_generation_prompt": True},
            "token_ids": ids,
            "token_count": len(ids),
            "truncated": False,
        },
        "schedule": {
            "token_ids": schedule,
            "steps": len(schedule),
            "origin": "greedy_argmax_from_admitted_model_no_eos_stopping",
            "replay": "teacher_forced_each_listed_token_is_fed_after_prefill",
            "capture_step_zero_based": capture_step,
            "capture_token_id": schedule[capture_step],
            "context_tokens_before_capture": len(ids) + capture_step,
        },
        "full_model_uninstrumented": full,
        "isolated_diagnostics": isolated,
        "additive_cost_attribution": False,
        "whole_model_speedup_claim": False,
        "limitations": [
            "Intrusive synchronized activation capture changes lazy graph execution boundaries.",
            "Isolated operations include Python dispatch, evaluation and synchronization.",
            "Attention-inclusive overlaps q/k/v/o and includes KV snapshot replay setup.",
            "Never sum isolated medians or interpret their ratios as whole-model fractions.",
            "One decode-step activation/context snapshot; not representative of every context.",
            "Standard KVCache only; no distributed, quantized or rotating cache profiling.",
            "Warm caches and memory/swap pressure can dominate; compare begin/end environment.",
        ],
    }
    write_evidence(output, report)
    return report


@dataclass
class Operation:
    function: Callable
    group: str
    metadata: dict


def operation_targets(model, *, tied_embeddings):
    """Select Qwen2 named-module boundaries; gate/up are captured together."""
    targets = {}
    for path, module in model.named_modules():
        leaf = path.rsplit(".", 1)[-1]
        group = None
        if leaf == "self_attn":
            group = "attention_inclusive"
        elif ".self_attn." in path and leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
            group = "attention_projection"
        elif leaf == "mlp":
            group = "mlp_gate_up"
        elif leaf == "down_proj":
            group = "mlp_down"
        elif leaf in ("norm", "input_layernorm", "post_attention_layernorm"):
            group = "norm"
        elif leaf == "embed_tokens":
            group = "embedding"
            if tied_embeddings:
                targets[(path, "as_linear")] = "output_projection"
        elif leaf == "lm_head":
            group = "output_projection"
        if group:
            targets[(path, "__call__")] = group
    return targets


def _input_metadata(x):
    return {"input_shape": list(x.shape), "input_dtype": str(x.dtype)}


def gate_up_operations(path, module, x):
    """Replay the stock Qwen2 SwiGLU pair and the admitted custom path, if active.

    No down projection is included. The stock comparator is eager mlx-lm SwiGLU,
    matching Qwen2, not an unrelated compiled microbenchmark implementation.
    """
    from mlx_lm.models.activations import swiglu

    from .runtime import FusedMLP, compiled_fused_gate_up

    gate, up = module.gate_proj, module.up_proj
    gate_call, up_call = gate.__call__, up.__call__
    eligible = (
        isinstance(module, FusedMLP)
        and x.ndim > 0
        and x.size == x.shape[-1]
        and x.dtype == gate.scales.dtype
    )
    metadata = {
        **_input_metadata(x),
        "active_decode_backend": "fused" if eligible else "stock",
        "bits": getattr(gate, "bits", None),
        "group_size": getattr(gate, "group_size", None),
        "includes_down_projection": False,
    }
    result = {
        f"{path}.gate_up_stock": Operation(
            lambda: swiglu(gate_call(x), up_call(x)),
            "mlp_gate_up",
            metadata,
        )
    }
    if eligible:
        gate_weights = (gate.weight, gate.scales, gate.biases)
        up_weights = (up.weight, up.scales, up.biases)
        bits, group_size, rpg = gate.bits, gate.group_size, module.rows_per_group
        result[f"{path}.gate_up_fused"] = Operation(
            lambda: compiled_fused_gate_up(
                x,
                gate_weights,
                up_weights,
                bits=bits,
                group_size=group_size,
                rows_per_group=rpg,
            ),
            "mlp_gate_up",
            {**metadata, "rows_per_group": rpg},
        )
    return result


def capture_decode_operations(
    model,
    prompt_ids,
    schedule,
    *,
    capture_step=0,
    targets=None,
    cache_factory=None,
):
    """Capture real batch-one eager activations at one teacher-forced decode step.

    Attention replays own a fresh standard KVCache snapshot on every call.
    Capture is intrusive: inputs/cache state are evaluated at module boundaries.
    Only shapes/dtypes/cache details are serialized; activation arrays stay live
    in operation closures until isolated timings complete.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    if not prompt_ids or not schedule:
        raise ValueError("nonempty prompt and schedule required")
    if type(capture_step) is not int or not 0 <= capture_step < len(schedule):
        raise ValueError("capture_step must be an integer within the decode schedule")
    if targets is None:
        targets = operation_targets(model, tied_embeddings=model.args.tie_word_embeddings)
    modules = dict(model.named_modules())
    operations = {}
    seen = set()

    def before(path, method, original, args, kwargs):
        key = (path, method)
        if key in seen:
            raise ValueError(f"profiling target executed more than once at one step: {key}")
        seen.add(key)
        group = targets[key]
        x = args[0] if args else kwargs["x"]
        mx.eval(x)
        metadata = _input_metadata(x)
        if group == "mlp_gate_up":
            operations.update(gate_up_operations(path, modules[path], x))
            return
        if group == "attention_inclusive":
            mask = args[1] if len(args) > 1 else kwargs.get("mask")
            cache = args[2] if len(args) > 2 else kwargs.get("cache")
            if type(cache) is not KVCache:
                raise ValueError("isolated attention requires an exact standard KVCache")
            # New array handles pin the pre-call graph across KVCache slice updates.
            keys, values, offset = cache.state
            keys = mx.array(keys) if keys is not None else None
            values = mx.array(values) if values is not None else None
            mx.eval(*(a for a in (keys, values, mask) if a is not None))
            metadata.update(
                cache_offset_before=offset,
                cache_capacity=keys.shape[2] if keys is not None else 0,
                includes_q_k_v_o=True,
                includes_rope_sdpa_cache_update=True,
                includes_cache_snapshot_replay_setup=True,
            )

            def replay():
                clone = KVCache()
                clone.state = (
                    mx.array(keys) if keys is not None else None,
                    mx.array(values) if values is not None else None,
                    offset,
                )
                return original(x, mask, clone)

            function = replay
        else:
            # Captured inputs are evaluated, and each replay constructs fresh outputs.
            def function():
                return original(*args, **kwargs)

        operations[f"{path}.{method}"] = Operation(function, group, metadata)

    cache = (cache_factory or make_prompt_cache)(model)
    mx.eval(model(mx.array([prompt_ids]), cache=cache))
    for token in schedule[:capture_step]:
        mx.eval(model(mx.array([[token]]), cache=cache))
    with instrument_modules(model, targets, before):
        mx.eval(model(mx.array([[schedule[capture_step]]]), cache=cache))
        mx.synchronize()
    if seen != set(targets):
        raise ValueError(f"unexecuted profiling targets: {sorted(set(targets) - seen)}")
    return operations


@contextmanager
def instrument_modules(model, targets, before):
    """Intercept selected methods without registering hooks as model parameters.

    ``before(path, method, original_bound_method, args, kwargs)`` runs before
    each call. Targets map (named-module path, method) to diagnostic groups.
    Do not run concurrent forwards or nested instrumentation on the same model.
    """
    modules = dict(model.named_modules())
    grouped = {}
    for (path, method), group in targets.items():
        if path not in modules or not callable(getattr(modules[path], method, None)):
            raise ValueError(f"missing profiling target: {path}.{method}")
        grouped.setdefault(id(modules[path]), (modules[path], {}))[1][method] = path
    restored = []
    try:
        for module, methods in grouped.values():
            original_type = type(module)
            overrides = {"__slots__": ()}
            for method, path in methods.items():
                original = getattr(module, method)

                def wrapped(self, *args, _path=path, _method=method, _original=original, **kwargs):
                    before(_path, _method, _original, args, kwargs)
                    return _original(*args, **kwargs)

                overrides[method] = wrapped
            instrumented = type(f"Diagnostic_{original_type.__name__}", (original_type,), overrides)
            object.__setattr__(module, "__class__", instrumented)
            restored.append((module, original_type))
        yield
    finally:
        for module, original_type in reversed(restored):
            object.__setattr__(module, "__class__", original_type)
