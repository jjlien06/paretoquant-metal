"""Per-token TP model from measured all-sum overhead. Usage: python model.py RUNS_DIR OUT_JSON"""

import json
import statistics
import sys
from pathlib import Path

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
CALLS = 128  # 64 layers x 2 all-reduces (attn/GDN out-proj + MLP down)
PASS_MS = (96.0, 100.0)  # current 28/36 pipeline forward pass, from parent context
L_M2, L_M4 = 28, 36

per = {}
for f in sorted(runs.glob("*.rank0.json")):
    tag = f.name.split(".rank0")[0]
    r0 = json.loads(f.read_text())
    r1 = json.loads((runs / f"{tag}.rank1.json").read_text())
    for m in r0["results"]:
        a, b = r0["results"][m], r1["results"][m]
        nocomm = max(a["token_nocomm_ms"]["p50"], b["token_nocomm_ms"]["p50"])
        comm = max(a["token_comm_ms"]["p50"], b["token_comm_ms"]["p50"])
        per.setdefault(m, []).append(
            {
                "tag": tag,
                "stream": r0["stream"],
                "token_comm_p50_ms": comm,
                "token_nocomm_p50_max_rank_ms": nocomm,
                "allreduce_overhead_in_chain_ms": (comm - nocomm) / CALLS,
                "percall_comm_p50_ms": [a["percall_comm_ms"]["p50"], b["percall_comm_ms"]["p50"]],
                "percall_comm_p90_ms": [a["percall_comm_ms"]["p90"], b["percall_comm_ms"]["p90"]],
                "percall_nocomm_p50_ms": [a["percall_nocomm_ms"]["p50"], b["percall_nocomm_ms"]["p50"]],
                "percall_overhead_p50_ms": max(a["percall_comm_ms"]["p50"], b["percall_comm_ms"]["p50"])
                - max(a["percall_nocomm_ms"]["p50"], b["percall_nocomm_ms"]["p50"]),
                "ranks_agree_sha": a["final_comm_sha256"] == b["final_comm_sha256"],
            }
        )

# Bandwidth ratio from the same runs' compute-only chain (1x width, same 4-bit qmm on both GPUs).
m2_bw = statistics.median(json.loads(f.read_text())["results"]["1"]["token_nocomm_ms"]["p50"] for f in runs.glob("*.rank0.json"))
m4_bw = statistics.median(json.loads(f.read_text())["results"]["1"]["token_nocomm_ms"]["p50"] for f in runs.glob("*.rank1.json"))
ratio = m4_bw / m2_bw  # M4 time / M2 time for equal bytes

model = {"m4_over_m2_time_ratio_measured_qmm": ratio, "cases": []}
for pass_ms in PASS_MS:
    t2 = pass_ms / (L_M2 + L_M4 * ratio)  # ms per full layer on M2 (comm in pipeline ignored -> optimistic for TP)
    t4 = t2 * ratio
    even = 64 * max(t2, t4) / 2
    prop = 64 * t2 * t4 / (t2 + t4)  # bandwidth-proportional column split
    case = {
        "pipeline_pass_ms": pass_ms,
        "m2_layer_ms": t2,
        "m4_layer_ms": t4,
        "tp_compute_even_split_ms": even,
        "tp_compute_proportional_split_ms": prop,
        "breakeven_allreduce_ms_even": (pass_ms - even) / CALLS,
        "breakeven_allreduce_ms_proportional": (pass_ms - prop) / CALLS,
        "tp_pass_ms_with_measured": {},
    }
    for m, rows in per.items():
        for row in rows:
            lat = row["allreduce_overhead_in_chain_ms"]
            case["tp_pass_ms_with_measured"][f"{m}x/{row['tag']}"] = {
                "allreduce_ms": lat,
                "proportional": prop + CALLS * lat,
                "even": even + CALLS * lat,
            }
    model["cases"].append(case)

json.dump({"measured": per, "model": model}, open(out, "w"), indent=1)
print(json.dumps({"measured": per, "model": model}, indent=1))
