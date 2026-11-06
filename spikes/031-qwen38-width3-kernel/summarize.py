"""031 summary: per task type geomeans over (case, sample) pairs from analyze.py output.

Ratios: adaptive(031 tuned width-3) vs depth-1 (021 profile, same process). Decode = post-first (N-1)/(wall-first);
whole = depth1 wall / adaptive wall. Equal-weight per case: samples of a case are first geomeaned, then cases.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def main():
    src, out = sys.argv[1], sys.argv[2]
    rep = json.loads(Path(src).read_text())
    rows = [r for r in rep["per_case"] if not r["warmup"]]
    per_case = defaultdict(list)
    for r in rows:
        per_case[r["case_id"]].append(r)
    keys = ("adaptive_vs_mtp_decode", "adaptive_vs_mtp_whole", "mtp_decode_speedup", "adaptive_decode_speedup",
            "mtp_whole_speedup", "adaptive_whole_speedup")
    case_gm = {c: {k: gm([r[k] for r in rs]) for k in keys} | {"samples": len(rs),
               "identical": all(r[m]["identical_to_native"] for r in rs for m in ("mtp", "adaptive")),
               "width3_rounds": [r["adaptive"]["width3_rounds"] for r in rs],
               "rounds": [r["adaptive"]["rounds"] for r in rs]} for c, rs in per_case.items()}
    types = ("code", "math", "structured", "explanation", "spanish", "chinese")
    by_type = {t: {k: gm([v[k] for c, v in case_gm.items() if c.startswith(t)]) for k in keys}
               | {"cases": sorted(c for c in case_gm if c.startswith(t))} for t in types}
    by_len = {L: {k: gm([v[k] for c, v in case_gm.items() if c.endswith(L)]) for k in keys} for L in ("short", "512", "1536")}
    res = {"source": src, "cases": len(case_gm), "pairs": len(rows),
           "all_outputs_identical_to_native": rep["all_outputs_identical_to_native"],
           "overall": {k: gm([v[k] for v in case_gm.values()]) for k in keys},
           "by_task_type": by_type, "by_prompt_length": by_len, "per_case": case_gm,
           "verify_wall_ms": rep["verify_wall_ms"], "round_cost_ms_median": rep["round_cost_ms_median"]}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("cases", "pairs", "all_outputs_identical_to_native", "overall", "round_cost_ms_median")}, indent=1))
    for t, v in by_type.items():
        print(t, {k: round(v[k], 3) for k in ("adaptive_vs_mtp_decode", "adaptive_vs_mtp_whole")})
    for c, v in sorted(case_gm.items()):
        print(c, v["samples"], round(v["adaptive_vs_mtp_decode"], 3), round(v["adaptive_vs_mtp_whole"], 3), v["identical"])


if __name__ == "__main__":
    main()
