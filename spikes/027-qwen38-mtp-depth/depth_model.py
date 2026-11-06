"""Expected-speedup model for depth-k MTP drafting, from archived broad-v1 data.

Inputs (read-only):
  results/qwen38-broad-v1/results.json         per-request accepted_drafts,
      verification_rounds, post_first_tps (72 measured requests, 18 cases)
  results/qwen38-broad-v1/batch-0*-v1/actors/*/stderr   rank0 `rank_action`
      timestamps -> per-round span decomposition (measured MTP requests only)
Output: results/qwen38-mtp-depth-assessment-v1/depth-model.json

Everything beyond k=1 is a MODEL with stated assumptions, not a measurement.
"""

import collections
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BROAD = ROOT / "results/qwen38-broad-v1"


def rank0_actions(path):
    with open(path) as f:
        for line in f:
            if line.startswith('{"phase": "rank_action"'):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("rank") == 0 and e.get("method") == "mtp" and not e.get("warmup"):
                    yield e


def spans():
    s = collections.defaultdict(list)
    trans = collections.Counter()
    files = sorted(glob.glob(str(BROAD / "batch-0*-v1/actors/*/stderr")))
    for p in files:
        by = collections.defaultdict(list)
        for e in rank0_actions(p):
            by[e["id"]].append(e)
        for es in by.values():
            es.sort(key=lambda e: e["sequence"])
            flags = []
            for i, e in enumerate(es):
                prev = es[i - 1] if i else None
                if e["op"] == "snapshot" and prev is not None:
                    dt = e["timestamp"] - prev["timestamp"]
                    if prev["op"] == "commit_prefix":
                        s["draft_plus_snapshot_after_reject"].append(dt)
                    elif prev["op"] == "verify":
                        s["advance_plus_draft_plus_snapshot_after_accept"].append(dt)
                elif e["op"] == "verify":
                    s["verify_width2"].append(e["timestamp"] - prev["timestamp"])
                    nxt = es[i + 1] if i + 1 < len(es) else None
                    flags.append(0 if nxt is not None and nxt["op"] == "commit_prefix" else 1)
                elif e["op"] == "commit_prefix":
                    s["commit_prefix1"].append(e["timestamp"] - prev["timestamp"])
            snaps = [e["timestamp"] for e in es if e["op"] == "snapshot"]
            s["round_period"].extend(b - a for a, b in zip(snaps, snaps[1:]))
            f = flags[:-1]  # final round: terminal commit is ambiguous
            for x, y in zip(f, f[1:]):
                trans[(x, y)] += 1
    med = {k: st.median(v) for k, v in s.items()}
    counts = {k: len(v) for k, v in s.items()}
    p_aa = trans[(1, 1)] / (trans[(1, 1)] + trans[(1, 0)])
    p_ar = trans[(0, 1)] / (trans[(0, 1)] + trans[(0, 0)])
    return med, counts, {"P(accept|prev accept)": p_aa, "P(accept|prev reject)": p_ar,
                         "pairs": sum(trans.values())}, [str(Path(p).relative_to(ROOT)) for p in files]


def expected_tokens(alphas):
    e, prod = 1.0, 1.0
    for a in alphas:
        prod *= a
        e += prod
    return e, prod


def main():
    res = json.loads((BROAD / "results.json").read_text())
    reqs = [r for r in res["requests"] if not r["warmup"]]
    med, counts, corr, files = spans()
    t_v2 = med["verify_width2"]
    t_ds = med["draft_plus_snapshot_after_reject"]
    t_adv = med["advance_plus_draft_plus_snapshot_after_accept"] - t_ds
    t_commit = med["commit_prefix1"]
    native_steps = [1 / r["post_first_tps"] for r in reqs if r["method"] == "native"]
    t_native = st.median(native_steps)
    dw_measured = t_v2 - t_native  # width1->width2 increment incl. vocab rows, slot-QMV
    cases = collections.defaultdict(lambda: {"mtp": [], "native": []})
    for r in reqs:
        cases[r["case_id"]][r["method"]].append(r)
    scen_dw = {"dw_full": dw_measured, "dw_half": dw_measured / 2}
    scen_decay = {"iid": 1.0, "decay0.9": 0.9, "decay0.8": 0.8}
    rows = []
    for cid, m in sorted(cases.items()):
        acc = sum(r["accepted_drafts"] for r in m["mtp"])
        rnd = sum(r["verification_rounds"] for r in m["mtp"])
        a = acc / rnd
        tn = 1 / st.median(r["post_first_tps"] for r in m["native"])
        obs = st.median(r["post_first_tps"] for r in m["mtp"]) * tn
        row = {"case": cid, "alpha_depth1": a, "native_step_s": tn,
               "observed_depth1_speedup": obs, "model": {}}
        for dn, dw in scen_dw.items():
            for rn, r_ in scen_decay.items():
                for k in (1, 2, 3):
                    alphas = [min(1.0, a * r_ ** i) for i in range(k)]
                    e, all_acc = expected_tokens(alphas)
                    t_round = (t_v2 + (k - 1) * dw + t_ds + (k - 1) * t_ds
                               + (1 - (1 - alphas[0])) * t_adv + (1 - all_acc) * t_commit)
                    row["model"][f"{dn}/{rn}/k{k}"] = {
                        "expected_tokens_per_round": e, "round_s": t_round,
                        "speedup_vs_native": e * tn / t_round}
        rows.append(row)

    def gmean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    keys = rows[0]["model"].keys()
    agg = {k: gmean([r["model"][k]["speedup_vs_native"] for r in rows]) for k in keys}
    out = {
        "inputs": {"results_json": "results/qwen38-broad-v1/results.json",
                   "actor_stderr_files": files, "measured_requests": len(reqs)},
        "measured_depth1": {
            "pooled_alpha": sum(r["accepted_drafts"] for r in reqs if r["method"] == "mtp")
            / sum(r["verification_rounds"] for r in reqs if r["method"] == "mtp"),
            "span_medians_s": med, "span_counts": counts,
            "derived_s": {"verify_width2": t_v2, "draft_plus_snapshot": t_ds,
                          "mtp_advance": t_adv, "commit_prefix1": t_commit,
                          "native_step_median": t_native,
                          "width2_minus_native_step": dw_measured},
            "acceptance_autocorrelation_consecutive_rounds": corr,
            "equal_case_gmean_observed_depth1_speedup": gmean(
                [r["observed_depth1_speedup"] for r in rows]),
        },
        "assumptions": [
            "Rank0 rank_action timestamps are local inclusive wall spans; verify span includes "
            "both-host pipeline + rank1 ACK.",
            "Width-(k+1) verify cost = measured width2 verify + (k-1)*dw; dw_full = measured "
            "width2 verify - native width1 step (linear extrapolation), dw_half = optimistic.",
            "Each extra chained draft costs the measured draft+snapshot span (upper bound; "
            "snapshot is per round, so this overcharges slightly).",
            "Acceptance of draft i = alpha*r^(i-1); r=1 is i.i.d. geometric. Depth>=2 MTP "
            "acceptance is UNMEASURED: Qwen3.x ships one MTP layer; chaining it on its own "
            "hidden output is expected to lose acceptance (r<1) but the size is unknown.",
            "Catch-up of j accepted drafts into the head KV is one batched head call costing the "
            "measured depth-1 advance span; commit cost is the measured prefix-1 commit.",
        ],
        "equal_case_gmean_model_speedup_vs_native": agg,
        "cases": rows,
    }
    dest = ROOT / "results/qwen38-mtp-depth-assessment-v1/depth-model.json"
    if dest.exists() and "--force" not in sys.argv:
        raise SystemExit("exists")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({"measured": out["measured_depth1"]["derived_s"],
                      "alpha": out["measured_depth1"]["pooled_alpha"],
                      "corr": corr,
                      "obs_gmean": out["measured_depth1"]["equal_case_gmean_observed_depth1_speedup"],
                      "agg": agg}, indent=1))


if __name__ == "__main__":
    main()
