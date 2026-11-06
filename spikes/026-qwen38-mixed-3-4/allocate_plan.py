"""Run the ParetoQuant exact allocator over per-layer {q4,q3} options for Qwen3.8-27B.

Loss per option = sum over the layer's quantized linears of rel_out^2 (3-bit vs 4-bit PROXY on synthetic
Gaussian inputs; 4-bit option loss = 0 by construction). Bytes = real packed weight+scales+biases bytes
from the published safetensors headers (non-quantized norms/conv/A_log are identical in both options and
excluded from the unit, but included in the totals below).
Decode speedup is a BANDWIDTH-MODEL PREDICTION: per-token bytes read = text-layer bytes + lm_head bytes
(+ norm/conv); NOT measured.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paretoquant.allocator import InfeasibleBudget, Option, Unit, allocate  # noqa: E402

OUT = ROOT / "results/qwen38-mixed-3-4-plan-v1"
INV = json.loads((OUT / "inventory.json").read_text())
rows = [json.loads(line) for line in (OUT / "per-tensor-error.jsonl").read_text().splitlines() if line.strip()]
by_tensor = {r["tensor"]: r for r in rows}
q4t = INV["q4"]["tensors"]
PFX = "language_model.model.layers."
expected = {t[: -len(".scales")] for t in q4t if t.startswith(PFX) and t.endswith(".scales")}
missing = sorted(expected - set(by_tensor))
if missing:
    sys.exit(f"incomplete measurement: {len(missing)} tensors missing, e.g. {missing[:3]}")

GRAN = sys.argv[1] if len(sys.argv) > 1 else "layer"  # "layer" or "block" (attn vs mlp per layer)
WEIGHT = sys.argv[2] if len(sys.argv) > 2 else "unweighted"  # or "params": rel_out^2 * numel / 1e8


def unit_of(r):
    if GRAN == "layer":
        return f"L{r['layer']:02d}"
    part = "mlp" if ".mlp." in r["tensor"] else "attn"
    return f"L{r['layer']:02d}.{part}"


agg = defaultdict(lambda: {"q4": 0, "q3": 0, "loss": 0.0, "n": 0})
for r in rows:
    a = agg[unit_of(r)]
    a["q4"] += r["bytes_q4"]
    a["q3"] += r["bytes_q3"]
    numel = r["shape"][0] * r["shape"][1]
    a["loss"] += r["rel_out"] ** 2 * (numel / 1e8 if WEIGHT == "params" else 1.0)
    a["n"] += 1
units = [
    Unit(name, (Option("q4", 4, a["q4"], 0.0, 0.0), Option("q3", 3, a["q3"], 0.0, a["loss"])))
    for name, a in sorted(agg.items())
]
unit_q4 = sum(a["q4"] for a in agg.values())
unit_q3 = sum(a["q3"] for a in agg.values())


def tbytes(t):
    return q4t[t]["end"] - q4t[t]["start"]


layer_other = sum(tbytes(t) for t in q4t if t.startswith(PFX)) - unit_q4  # norms, conv, A_log, dt_bias
lm_head = sum(tbytes(f"language_model.lm_head.{p}") for p in ("weight", "scales", "biases"))
final_norm = tbytes("language_model.model.norm.weight")
embed = sum(tbytes(f"language_model.model.embed_tokens.{p}") for p in ("weight", "scales", "biases"))
per_token_fixed = layer_other + lm_head + final_norm  # embedding reads one row; excluded
base_tok = unit_q4 + per_token_fixed

results = {"granularity": GRAN, "loss_weighting": WEIGHT, "n_units": len(units), "unit_bytes_q4": unit_q4, "unit_bytes_q3": unit_q3,
           "layer_nonquantized_bytes": layer_other, "lm_head_bytes": lm_head, "embed_bytes": embed,
           "uniform_q4_text_stored_bytes": unit_q4 + layer_other + lm_head + final_norm + embed,
           "uniform_q3_loss": sum(a["loss"] for a in agg.values()), "plans": []}
max_saving = unit_q4 - unit_q3
for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    budget = unit_q4 - int(round(frac * max_saving))
    try:
        plan = allocate(units, budget, max_states=200000)
    except InfeasibleBudget as e:
        results["plans"].append({"saving_fraction_target": frac, "budget": budget, "error": str(e)})
        continue
    q3_units = sorted(n for n, o in plan.choices.items() if o.label == "q3")
    tok = plan.memory_bytes + per_token_fixed
    results["plans"].append({
        "saving_fraction_target": frac, "budget_bytes": budget, "unit_bytes": plan.memory_bytes,
        "saved_bytes_vs_uniform_q4": unit_q4 - plan.memory_bytes,
        "saved_gib": round((unit_q4 - plan.memory_bytes) / 2**30, 3),
        "proxy_loss": plan.loss, "proxy_loss_fraction_of_uniform_q3": plan.loss / results["uniform_q3_loss"],
        "n_q3_units": len(q3_units), "q3_units": q3_units, "states_considered": plan.states_considered,
        "bandwidth_model_decode_speedup_predicted": base_tok / tok,
    })
name = f"allocator-plans-{GRAN}-{WEIGHT}.json"
(OUT / name).write_text(json.dumps(results, indent=1))
print(json.dumps({k: v for k, v in results.items() if k != "plans"}))
for p in results["plans"]:
    print({k: v for k, v in p.items() if k != "q3_units"}, p.get("q3_units"))
