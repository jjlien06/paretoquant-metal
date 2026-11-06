"""MODELLED tau selection from the tau=0 calibration rounds (analysis-cal.*.json).

Single-threshold rule v2: compute chained d2 only if p1 >= tau; verify width3 only if
p2 >= tau. Costs: measured medians (mtp period, adaptive width3 period, adaptive width2
:= mtp verify2 + adaptive overhead); rounds held fixed. In-sample on the 6 cal cases.
"""
import json, math, sys
from collections import defaultdict
base = sys.argv[1]
R = [r for r in json.load(open(base + ".rounds.json")) if r["p2"] is not None and r["width"] == 3]
c = json.load(open(base + ".json"))["round_cost_ms_median"]
p1m, p3, p2a = c["depth1_mtp_round_period"], c["adaptive_round_period_w3"], c["modelled_adaptive_width2_round"]
rows = []
for t1 in [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
    for t2 in [0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99]:
        pc = defaultdict(lambda: [0, 0, 0, 0])
        for r in R:
            a1, a2, x = r["accepted"] >= 1, r["accepted"] >= 2, pc[r["case_id"]]
            if r["p1"] < t1:
                x[0] += 1 + a1; x[1] += p1m
            elif r["p2"] >= t2:
                x[0] += 1 + a1 + (a1 and a2); x[1] += p3
            else:
                x[0] += 1 + a1; x[1] += p2a
            x[2] += 1 + a1; x[3] += p1m
        ratios = {k: (v[0] / v[1]) / (v[2] / v[3]) for k, v in pc.items()}
        rows.append({"tau_p1": t1, "tau_p2": t2,
                     "modelled_vs_depth1_gmean": math.exp(sum(map(math.log, ratios.values())) / len(ratios)),
                     "per_case": ratios})
d2 = defaultdict(lambda: [0, 0])
for r in R:
    if r["accepted"] >= 1:
        b = "p2>=0.99" if r["p2"] >= 0.99 else ("0.95-0.99" if r["p2"] >= 0.95 else ("0.8-0.95" if r["p2"] >= 0.8 else ("0.5-0.8" if r["p2"] >= 0.5 else "<0.5")))
        d2[b][0] += 1; d2[b][1] += r["accepted"] >= 2
out = {"label": "MODELLED (not measured)", "costs_ms": [p1m, p3, p2a], "n_rounds": len(R),
       "P(d2 accepted | d1 accepted) by p2 bucket": {k: {"n": v[0], "rate": v[1] / v[0]} for k, v in d2.items()},
       "chosen": {"tau": 0.9, "rule": "single tau for both p1 prefilter and p2 gate"},
       "grid": sorted(rows, key=lambda r: -r["modelled_vs_depth1_gmean"])}
json.dump(out, open(base.rsplit("/", 1)[0] + "/tau-selection.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("costs_ms", "n_rounds", "P(d2 accepted | d1 accepted) by p2 bucket")}, indent=1))
for r in out["grid"][:5]: print(r["tau_p1"], r["tau_p2"], round(r["modelled_vs_depth1_gmean"], 4))
print([ (r["tau_p1"], r["tau_p2"], round(r["modelled_vs_depth1_gmean"],4)) for r in out["grid"] if r["tau_p1"]==0 and r["tau_p2"]==0])
