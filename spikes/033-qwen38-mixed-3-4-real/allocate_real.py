"""Exact per-decoder-layer allocation using measured activation relative MSE.

No model imports or hardware work. Byte counts are tensor payload bytes, not
safetensors headers. The speedup is a bandwidth-model prediction, not timing.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paretoquant.allocator import (  # noqa: E402
    FrontierOverflow, InfeasibleBudget, Option, Unit, allocate,
)

PFX = "language_model.model.layers."
EMBED = "language_model.model.embed_tokens."
HEAD = "language_model.lm_head."
NORM = "language_model.model.norm."
PARTS = ("weight", "scales", "biases")
GIB = 2**30


def layer_of(name):
    if not name.startswith(PFX):
        return None
    token, sep, _ = name[len(PFX):].partition(".")
    if not sep or not token.isdigit():
        raise ValueError(f"Malformed decoder tensor name: {name}")
    return int(token)


def text_tensor(name):
    return name.startswith((PFX, EMBED, HEAD, NORM))


def nbytes(info):
    size = info["end"] - info["start"]
    if size < 0:
        raise ValueError("Negative tensor length")
    return size


def decoder_modules(tensors):
    """Identify quantized decoder linears by their real affine tensor triplets."""
    modules = set()
    for name in tensors:
        if layer_of(name) is None or not name.endswith(".scales"):
            continue
        module = name[:-len(".scales")]
        for part in PARTS:
            if f"{module}.{part}" not in tensors:
                raise ValueError(f"Incomplete quantized module: {module}")
        weight = tensors[module + ".weight"]
        if weight["dtype"] != "U32" or len(weight["shape"]) != 2:
            raise ValueError(f"Not a packed quantized linear: {module}")
        modules.add(module)
    return modules


def allocate_inventory(inventory, sensitivity, max_states=10000,
                       targets_gib=(0.5, 1.0, 1.5, 2.0), expected_layers=64):
    q4, q3 = inventory["q4"]["tensors"], inventory["q3"]["tensors"]
    modules = decoder_modules(q4)
    if modules != decoder_modules(q3):
        raise ValueError("q3/q4 decoder quantized module sets differ")
    layers = set(range(expected_layers))
    if {layer_of(n) for n in q4 if n.startswith(PFX)} != layers:
        raise ValueError(f"Expected exactly decoder layers 0..{expected_layers - 1}")
    if {layer_of(m) for m in modules} != layers:
        raise ValueError("Every decoder layer must have quantized linears")
    errors = {}
    for row in sensitivity["layers"]:
        layer, loss = row["layer"], row["relative_mse"]
        if isinstance(layer, bool) or not isinstance(layer, int) or layer in errors:
            raise ValueError("Sensitivity layer IDs must be unique integers")
        if isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"Invalid relative_mse for layer {layer}")
        errors[layer] = loss
    if set(errors) != layers:
        raise ValueError(f"Sensitivity must cover exactly layers 0..{expected_layers - 1}")
    costs = {i: {"q4": 0, "q3": 0} for i in layers}
    for module in modules:
        layer = layer_of(module)
        assert layer is not None
        for part in PARTS:
            name = f"{module}.{part}"
            for label, tensors in (("q4", q4), ("q3", q3)):
                costs[layer][label] += nbytes(tensors[name])
    units = [Unit(f"L{i:02d}", (
        Option("q4", 4, costs[i]["q4"], 0.0, 0.0),
        Option("q3", 3, costs[i]["q3"], 0.0, errors[i]),
    )) for i in sorted(layers)]
    base_units = sum(c["q4"] for c in costs.values())
    total = sum(nbytes(v) for n, v in q4.items() if text_tensor(n))
    embed = sum(nbytes(v) for n, v in q4.items() if n.startswith(EMBED))
    head = sum(nbytes(v) for n, v in q4.items() if n.startswith(HEAD))
    if not embed or not head or not any(n.startswith(NORM) for n in q4):
        raise ValueError("Inventory requires q4 embedding, lm_head, and final norm")
    fixed = total - base_units
    result = {
        "schema_version": 1, "n_units": expected_layers,
        "loss_metric": "sum of measured per-layer activation relative_mse; q4=0",
        "exact": True, "max_states": max_states,
        "uniform_q4_text_stored_bytes": total, "unit_bytes_q4": base_units,
        "fixed_q4_bytes": fixed, "embed_bytes": embed, "lm_head_bytes": head,
        "uniform_q4_decode_bytes": total - embed,
        "speedup_model": "payload bandwidth only; includes q4 head and norm; excludes embedding read",
        "layer_bytes": {str(i): costs[i] for i in sorted(layers)},
        "plans": [],
    }
    for target in targets_gib:
        saving = int(target * GIB)
        budget = base_units - saving
        item = {"target_savings_gib": target, "target_saved_bytes": saving,
                "budget_bytes": budget}
        if budget < 0:
            item["error"] = "Infeasible savings target exceeds decoder payload"
            result["plans"].append(item)
            continue
        try:
            plan = allocate(units, budget, max_states=max_states)
        except InfeasibleBudget as exc:
            item["error"] = str(exc)
            result["plans"].append(item)
            continue
        # FrontierOverflow intentionally propagates: no approximate fallback.
        selected = [i for i in sorted(layers) if plan.choices[f"L{i:02d}"].label == "q3"]
        actual = base_units - plan.memory_bytes
        stored = fixed + plan.memory_bytes
        item.update({"exact": plan.exact, "q3_layers": selected,
                     "n_q3_layers": len(selected), "unit_bytes": plan.memory_bytes,
                     "q3_layer_bytes": sum(costs[i]["q3"] for i in selected),
                     "saved_bytes_vs_uniform_q4": actual, "actual_saved_bytes": actual,
                     "text_stored_bytes": stored, "predicted_decode_bytes": stored - embed,
                     "predicted_bytes_speedup": (total - embed) / (stored - embed),
                     "activation_relative_mse_loss": plan.loss,
                     "states_considered": plan.states_considered})
        result["plans"].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=10000)
    args = parser.parse_args()
    try:
        result = allocate_inventory(json.loads(args.inventory.read_text()),
                                    json.loads(args.sensitivity.read_text()), args.max_states)
    except (ValueError, KeyError, FrontierOverflow) as exc:
        parser.exit(2, f"Exact allocation failed (no approximation): {exc}\n")
    result["inventory_path"] = str(args.inventory.resolve())
    result["sensitivity_path"] = str(args.sensitivity.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "plans": len(result["plans"])}))


if __name__ == "__main__":
    main()
