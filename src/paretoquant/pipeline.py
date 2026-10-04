"""Calibration and measured gate/up precision allocation.

Scope: coupled bias-free gated-MLP pairs. All other eligible weights are fixed
at affine 4-bit. This is not yet a whole-model layer-sharded converter.
"""

from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import quantize_model

from .adapters import require_supported_mlp
from .allocator import Option, Unit
from .benchmark import benchmark_functions, environment
from .metal import compiled_fused_gate_up, compiled_stock_gate_up, stock_gate_up
from .runtime import install_fusion


def model_bytes(model):
    """Resident parameter bytes, not total runtime/KV-cache memory."""
    return sum(value.nbytes for _, value in tree_flatten(model.parameters()))


def mlp_modules(model):
    modules = dict(
        sorted(
            (name, module)
            for name, module in model.named_modules()
            if all(hasattr(module, p) for p in ("gate_proj", "up_proj", "down_proj"))
        )
    )
    for module in modules.values():
        require_supported_mlp(module)
    return modules


def calibrate_inputs(model, tokenizer, texts, *, max_tokens=128, samples_per_text=16):
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("calibration requires nonempty text samples")
    if max_tokens < 2 or samples_per_text < 1:
        raise ValueError("max_tokens >= 2 and samples_per_text >= 1 are required")
    originals = mlp_modules(model)
    if not originals:
        raise ValueError("no supported gated MLP modules found")
    captures = {name: [] for name in originals}

    class Capture(nn.Module):
        def __init__(self, name, original):
            super().__init__()
            self.name = name
            object.__setattr__(self, "original", original)

        def __call__(self, x):
            captures[self.name].append(x.reshape(-1, x.shape[-1])[-samples_per_text:])
            return self.original(x)

    model.update_modules(
        tree_unflatten([(name, Capture(name, module)) for name, module in originals.items()])
    )
    try:
        for text in texts:
            ids = tokenizer.encode(text)[:max_tokens]
            if not ids:
                raise ValueError("calibration text produced no tokens")
            logits = model(mx.array([ids]))
            mx.eval(logits, *(values[-1] for values in captures.values() if values))
    finally:
        model.update_modules(tree_unflatten(list(originals.items())))
    if any(not values for values in captures.values()):
        raise ValueError("some gated MLP modules were not executed during calibration")
    result = {name: mx.concatenate(values, axis=0) for name, values in captures.items()}
    mx.eval(result)
    return result


def profile_units(
    model, inputs, *, bits=(3, 4, 6), group_size=64, repeats=20, warmup=3, verbose=True
):
    if not bits or any(b not in (3, 4, 6) for b in bits) or len(set(bits)) != len(bits):
        raise ValueError("candidate bits must be distinct members of 3, 4, 6")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be nonnegative")
    modules = mlp_modules(model)
    result = {
        "environment": environment(),
        "group_size": group_size,
        "scope": "coupled_gate_up_pairs",
        "units": [],
        "quality_proxy": "relative_MSE_of_gate_up_activation_on_calibration_inputs",
    }
    for name, module in modules.items():
        if name not in inputs:
            raise ValueError(f"missing calibration inputs for {name}")
        if (
            not isinstance(module.gate_proj, nn.Linear)
            or not isinstance(module.up_proj, nn.Linear)
            or "bias" in module.gate_proj
            or "bias" in module.up_proj
        ):
            raise ValueError("profiling requires unquantized, bias-free gate/up linear weights")
        x = inputs[name]
        reference = nn.silu(module.gate_proj(x)) * module.up_proj(x)
        denominator = mx.maximum(mx.mean(reference.astype(mx.float32) ** 2), 1e-12)
        single = x[-1:]
        unit = {
            "name": name,
            "shape": list(module.gate_proj.weight.shape),
            "calibration_input_count": x.shape[0],
            "options": [],
            "measurements": {},
        }
        for bit in bits:
            gate = mx.quantize(module.gate_proj.weight, group_size, bit)
            up = mx.quantize(module.up_proj.weight, group_size, bit)
            mx.eval(gate, up)
            nbytes = sum(a.nbytes for a in (*gate, *up))
            prediction = stock_gate_up(x, gate, up, bits=bit, group_size=group_size)
            loss = (
                mx.mean((prediction.astype(mx.float32) - reference.astype(mx.float32)) ** 2)
                / denominator
            ).item()

            def stock():
                return compiled_stock_gate_up(single, gate, up, bits=bit, group_size=group_size)

            expected = stock()
            mx.eval(expected)
            candidates = {"stock": stock}
            valid = {}
            for rpg in (1, 2, 4, 8):

                def candidate(rpg=rpg):
                    return compiled_fused_gate_up(
                        single, gate, up, bits=bit, group_size=group_size, rows_per_group=rpg
                    )

                actual = candidate()
                mx.eval(actual)
                valid[rpg] = bool(
                    np.allclose(np.array(actual), np.array(expected), rtol=0.01, atol=0.02)
                )
                if valid[rpg]:
                    candidates[f"fused:rpg{rpg}"] = candidate
            tuning = benchmark_functions(candidates, warmup=warmup, repeats=max(2, repeats // 2))
            fused_names = [key for key in tuning if key != "stock"]
            best = (
                min(fused_names, key=lambda key: tuning[key]["median_ms"]) if fused_names else None
            )
            validation_functions = {"stock": stock}
            if best:
                validation_functions[best] = candidates[best]
            # Fresh interleaved measurements after selecting the launch configuration.
            validation = benchmark_functions(validation_functions, warmup=warmup, repeats=repeats)
            unit["measurements"][str(bit)] = {
                "tuning": tuning,
                "validation": validation,
                "numerically_valid_rpg": valid,
            }
            unit["options"].append(
                asdict(
                    Option(
                        f"q{bit}:stock",
                        bit,
                        nbytes,
                        validation["stock"]["median_ms"],
                        loss,
                        "stock",
                    )
                )
            )
            if best and validation[best]["median_ms"] < validation["stock"]["median_ms"] * 0.95:
                # Five percent is a conservative heuristic, not a significance test.
                unit["options"].append(
                    asdict(
                        Option(
                            f"q{bit}:{best}",
                            bit,
                            nbytes,
                            validation[best]["median_ms"],
                            loss,
                            "fused",
                        )
                    )
                )
        result["units"].append(unit)
        if verbose:
            fastest = min(unit["options"], key=lambda o: o["latency_ms"])
            print(
                f"Profiled {name}: best latency {fastest['label']} {fastest['latency_ms']:.4f} ms",
                flush=True,
            )
    return result


def units_from_profile(profile, *, stock_only=False):
    return [
        Unit(
            unit["name"],
            tuple(
                Option(**option)
                for option in unit["options"]
                if not stock_only or option["backend"] == "stock"
            ),
        )
        for unit in profile["units"]
    ]


def apply_plan(model, config, choices, *, group_size=64):
    known = mlp_modules(model)
    if set(choices) != set(known):
        raise ValueError("precision plan must cover exactly the supported gate/up units")
    overrides = {
        f"{name}.{projection}": {"bits": choice.bits, "group_size": group_size, "mode": "affine"}
        for name, choice in choices.items()
        for projection in ("gate_proj", "up_proj")
    }
    model, quantized_config = quantize_model(
        model, config, group_size, 4, quant_predicate=lambda path, module: overrides.get(path, True)
    )
    dispatch = {}
    for name, choice in choices.items():
        rpg = int(choice.label.rsplit("rpg", 1)[1]) if ":rpg" in choice.label else 4
        dispatch[name] = {"backend": choice.backend, "rows_per_group": rpg, "bits": choice.bits}
    install_fusion(model, dispatch)
    mx.eval(model.parameters())
    return model, quantized_config, dispatch
