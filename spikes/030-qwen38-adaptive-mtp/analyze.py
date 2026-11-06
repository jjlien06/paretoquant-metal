"""030 analysis: paired native / depth1-MTP / adaptive results + per-round tau model.

usage: analyze.py OUT_JSON RUN_DIR [RUN_DIR ...]
Reads each run's rank0.stdout (final per-request results) and rank0.stderr (events).
Measured: per-request outputs, identity vs same-process native, warm post-first and
whole-request rates, per-round verify/period by width. Modelled (labelled): tau sweep.
"""

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def events(path):
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("{") and ('"rank_action"' in line or '"adaptive_round"' in line):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def load(run):
    run = Path(run)
    out = json.loads((run / "rank0.stdout").read_text())
    reqs = out["requests"]
    by_id = {r["id"]: r for r in reqs}
    rounds = defaultdict(list)
    actions = defaultdict(list)
    for e in events(run / "rank0.stderr"):
        if e["phase"] == "adaptive_round":
            rounds[e["id"]].append(e)
        else:
            actions[e["id"]].append(e)
    # per-round verify wall (snapshot ACK -> verify ACK) and round period (verify -> verify)
    timing = {}
    for rid, acts in actions.items():
        v2, v3, periods = [], [], defaultdict(list)
        last_verify = None
        for prev, cur in zip(acts, acts[1:]):
            if cur["op"] == "verify" and prev["op"] == "snapshot":
                d = cur["timestamp"] - prev["timestamp"]
                (v2 if len(cur["tokens"]) == 2 else v3).append(d)
            if cur["op"] == "verify":
                if last_verify is not None:
                    periods[len(last_verify["tokens"])].append(
                        cur["timestamp"] - last_verify["timestamp"]
                    )
                last_verify = cur
        timing[rid] = {"verify_w2": v2, "verify_w3": v3, "period_by_prev_width": dict(periods)}
    return run.name, reqs, by_id, rounds, timing


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def main():
    out_path, runs = sys.argv[1], sys.argv[2:]
    rows, round_rows, all_timing = [], [], {}
    for run in runs:
        name, reqs, by_id, rounds, timing = load(run)
        cases = defaultdict(dict)
        for r in reqs:
            cases[(r["case_id"], r["warmup"], r["sample_repeat"])][r["method"]] = r
        for (case, warm, rep), m in cases.items():
            nat = m["native"]
            row = {"run": name, "case_id": case, "warmup": warm, "sample_repeat": rep}
            for meth, r in m.items():
                row[meth] = {
                    "id": r["id"],
                    "tokens": len(r["token_ids"]),
                    "finish_reason": r["finish_reason"],
                    "identical_to_native": r["token_ids"] == nat["token_ids"]
                    and r["finish_reason"] == nat["finish_reason"],
                    "post_first_tps": r["post_first_tokens_per_second"],
                    "wall_tps": r["wall_tokens_per_second"],
                    "wall_seconds": r["wall_seconds"],
                    "first_token_seconds": r["first_token_seconds"],
                    "rounds": r.get("verification_rounds"),
                    "accepted": r.get("accepted_drafts"),
                    "width3_rounds": r.get("width3_rounds"),
                    "adaptive_tau": r.get("adaptive_tau"),
                }
            for meth in ("mtp", "adaptive"):
                row[f"{meth}_decode_speedup"] = (
                    row[meth]["post_first_tps"] / row["native"]["post_first_tps"]
                )
                row[f"{meth}_whole_speedup"] = row["native"]["wall_seconds"] / row[meth]["wall_seconds"]
            row["adaptive_vs_mtp_decode"] = row["adaptive"]["post_first_tps"] / row["mtp"]["post_first_tps"]
            row["adaptive_vs_mtp_whole"] = row["mtp"]["wall_seconds"] / row["adaptive"]["wall_seconds"]
            rows.append(row)
            for e in rounds.get(m["adaptive"]["id"], []):
                round_rows.append({"run": name, "case_id": case, "warmup": warm, **{
                    k: e[k] for k in ("round", "width", "keep", "accepted", "p1", "p2", "tau",
                                      "verify_seconds", "commit_seconds", "timestamp")}})
            for meth in ("mtp", "adaptive"):
                all_timing[f"{name}/{m[meth]['id']}"] = timing.get(m[meth]["id"], {})
    measured = [r for r in rows if not r["warmup"]]
    ident = all(r[m]["identical_to_native"] for r in rows for m in ("mtp", "adaptive"))

    def agg(sel):
        sel = list(sel)
        return {
            "cases": len(sel),
            **{k: gmean([r[k] for r in sel]) for k in (
                "mtp_decode_speedup", "adaptive_decode_speedup", "mtp_whole_speedup",
                "adaptive_whole_speedup", "adaptive_vs_mtp_decode", "adaptive_vs_mtp_whole")},
        }

    def pooled(key, sel_width):
        xs = [x for t in all_timing.values() for x in t.get(key, [])]
        return {"n": len(xs), "median_ms": 1000 * st.median(xs) if xs else None}

    report = {
        "runs": [Path(r).name for r in runs],
        "all_outputs_identical_to_native": ident,
        "per_case": rows,
        "equal_case_geomean_all": agg(measured),
        "equal_case_geomean_code": agg(r for r in measured if r["case_id"].startswith("code")),
        "verify_wall_ms": {
            "width2_all_requests": pooled("verify_w2", 2),
            "width3_all_requests": pooled("verify_w3", 3),
            "convention": "rank0 snapshot-ACK -> verify-ACK wall (includes rank1 + transfer)",
        },
    }
    # Tau model (MODELLED) from adaptive rounds where d2 was evaluated (p2 not None).
    def med(method, key, width=None):
        xs = []
        for k, t in all_timing.items():
            if f"-{method}-" not in k:
                continue
            if key == "period":
                xs.extend(t.get("period_by_prev_width", {}).get(width, []))
            else:
                xs.extend(t.get(key, []))
        return (st.median(xs) if xs else None), len(xs)

    v2, n_v2 = med("mtp", "verify_w2")
    p2_mtp, n_p2 = med("mtp", "period", 2)
    v3, n_v3 = med("adaptive", "verify_w3")
    p3, n_p3 = med("adaptive", "period", 3)
    per2 = per3 = None
    if None not in (v2, p2_mtp, v3, p3):
        per3 = p3
        # adaptive width2 round still drafts d2 (to read p2): verify2 + adaptive overhead
        per2 = v2 + (p3 - v3)
    report["round_cost_ms_median"] = {
        "depth1_mtp_verify_w2": v2 and 1000 * v2, "n": n_v2,
        "depth1_mtp_round_period": p2_mtp and 1000 * p2_mtp, "n_period": n_p2,
        "adaptive_verify_w3": v3 and 1000 * v3, "n_w3": n_v3,
        "adaptive_round_period_w3": p3 and 1000 * p3, "n_period_w3": n_p3,
        "modelled_adaptive_width2_round": per2 and 1000 * per2,
    }
    model = []
    cal = [r for r in round_rows if r["p2"] is not None and r["width"] == 3]
    if cal and per2 and per3:
        for tau in [x / 20 for x in range(0, 21)]:
            per_case = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
            for r in cal:
                a1 = r["accepted"] >= 1
                a2 = r["accepted"] >= 2
                c = per_case[r["case_id"]]
                if r["p2"] >= tau:
                    c[0] += 1 + a1 + (a1 and a2)
                    c[1] += per3
                else:
                    c[0] += 1 + a1
                    c[1] += per2
                c[2] += 1 + a1  # depth1-equivalent tokens on the same rounds
                c[3] += p2_mtp
            ratios = {k: (v[0] / v[1]) / (v[2] / v[3]) for k, v in per_case.items()}
            model.append({"tau": tau, "modelled_vs_depth1_equal_case_gmean": gmean(ratios.values()),
                          "per_case": ratios,
                          "fraction_width3": sum(r["p2"] >= tau for r in cal) / len(cal)})
    report["tau_model_MODELLED"] = {
        "assumption": "rounds from tau=0 calibration (d2 always evaluated); depth1 round := "
                      "measured mtp period; width3 := measured adaptive period; adaptive width2 "
                      ":= mtp verify2 + adaptive non-verify overhead; round boundaries fixed",
        "rows": model,
    }
    Path(out_path).write_text(json.dumps(report, indent=1))
    Path(out_path).with_suffix(".rounds.json").write_text(json.dumps(round_rows))
    print(json.dumps({k: report[k] for k in (
        "all_outputs_identical_to_native", "equal_case_geomean_all", "equal_case_geomean_code",
        "verify_wall_ms", "round_cost_ms_median")}, indent=1))
    for r in measured:
        print(r["case_id"], round(r["native"]["post_first_tps"], 2),
              round(r["mtp"]["post_first_tps"], 2), round(r["adaptive"]["post_first_tps"], 2),
              "w3", r["adaptive"]["width3_rounds"], "/", r["adaptive"]["rounds"],
              "acc", r["mtp"]["accepted"], r["adaptive"]["accepted"])
    for m in model:
        print("tau", m["tau"], round(m["modelled_vs_depth1_equal_case_gmean"], 4),
              "w3frac", round(m["fraction_width3"], 3))


if __name__ == "__main__":
    main()
