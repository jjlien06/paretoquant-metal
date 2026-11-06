"""Offline depth-2/3 chained-MTP acceptance from 029 rank0 actor logs.

Reads results/qwen38-mtp-depth2-acceptance-v1/part-*/ :
  actors/*/stderr  rank0 `depth_chain` events (d1, chained d2, d3 per round) and
                   `rank_action` verify events (cross-check d1 == verified draft),
                   `request_done` request_result (final emitted token IDs).
Greedy decoding => emitted tokens are the target's greedy sequence (checked equal to
the same-process native request), so chained-draft correctness is decidable offline.

Writes acceptance.json and depth2-model.json (depth_model.py round-cost model with the
measured per-case conditional acceptances instead of the assumed decay r).
"""

import collections
import glob
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/qwen38-mtp-depth2-acceptance-v1"


def events(path):
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect():
    reqs = {}
    for p in sorted(glob.glob(str(OUT / "part-*/actors/*/stderr"))):
        ev = [e for e in events(p) if e.get("rank") == 0 and "id" in e]
        if not ev:
            continue
        by = collections.defaultdict(list)
        for e in ev:
            by[e["id"]].append(e)
        for rid, es in by.items():
            done = [e for e in es if e.get("phase") == "request_done"]
            if len(done) != 1:
                raise ValueError(f"missing request_done {rid}")
            r = done[0]["request_result"]
            reqs[rid] = {
                "case_id": es[0]["case_id"], "method": es[0]["method"], "part": p,
                "token_ids": r["token_ids"], "finish_reason": r["finish_reason"],
                "result": r,
                "chain": [e for e in es if e.get("phase") == "depth_chain"],
                "verifies": [e["tokens"] for e in es
                             if e.get("phase") == "rank_action" and e.get("op") == "verify"],
            }
    return reqs


def simulate(tokens, chain, verifies, eos_terminal):
    if len(chain) != len(verifies):
        raise ValueError("depth_chain count != verify count")
    i, rows = 1, []
    for k, (c, v) in enumerate(zip(chain, verifies)):
        if v != [c["y"], c["d1"]] or tokens[i - 1] != c["y"]:
            raise ValueError("draft log does not bind to verified round/emitted progress")
        a1 = c["d1"] == tokens[i]
        row = {"round": k, "pos": i, "d1": c["d1"], "a1": a1}
        if a1 and i + 1 < len(tokens):
            row["a2"] = c["chain"][0] == tokens[i + 1]
            if row["a2"] and i + 2 < len(tokens):
                row["a3"] = c["chain"][1] == tokens[i + 2]
        rows.append(row)
        i += 2 if a1 else 1
    # last accepted round can be terminal (only d emitted, no bonus): i overshoots by 1
    if i not in (len(tokens), len(tokens) + 1):
        raise ValueError(f"simulation length mismatch {i} vs {len(tokens)}")
    return rows


def rates(rows):
    n1 = len(rows)
    a1 = sum(r["a1"] for r in rows)
    n2 = [r for r in rows if "a2" in r]
    a2 = sum(r["a2"] for r in n2)
    n3 = [r for r in rows if "a3" in r]
    a3 = sum(r["a3"] for r in n3)
    return {"rounds": n1, "a1": a1, "alpha1": a1 / n1 if n1 else None,
            "a2_given_a1_n": len(n2), "a2": a2, "p_a2_given_a1": a2 / len(n2) if n2 else None,
            "a3_given_a2_n": len(n3), "a3": a3, "p_a3_given_a2": a3 / len(n3) if n3 else None}


def main():
    reqs = collect()
    cases = collections.defaultdict(dict)
    for rid, r in reqs.items():
        cases[r["case_id"]][r["method"]] = r
    per_case, pooled_rows, parity = {}, [], {}
    for cid, m in sorted(cases.items()):
        mtp, nat = m["mtp"], m["native"]
        parity[cid] = {"token_ids_equal": mtp["token_ids"] == nat["token_ids"],
                       "finish_equal": mtp["finish_reason"] == nat["finish_reason"],
                       "tokens": len(mtp["token_ids"]),
                       "accepted_drafts": mtp["result"]["accepted_drafts"],
                       "verification_rounds": mtp["result"]["verification_rounds"]}
        rows = simulate(mtp["token_ids"], mtp["chain"], mtp["verifies"], None)
        if sum(r["a1"] for r in rows) != mtp["result"]["accepted_drafts"]:
            raise ValueError("simulated accepts != runtime accepted_drafts")
        per_case[cid] = rates(rows)
        pooled_rows += rows
    pooled = rates(pooled_rows)
    eq = {k: st.mean(c[k] for c in per_case.values() if c[k] is not None)
          for k in ("alpha1", "p_a2_given_a1", "p_a3_given_a2")}
    # archived broad-v1 depth-1 parity for the same prompts (first N tokens)
    broad = json.loads((ROOT / "results/qwen38-broad-v1/results.json").read_text())
    arch = {}
    for r in broad["requests"]:
        if not r["warmup"] and "token_ids" in r:
            arch.setdefault(r["case_id"], r["token_ids"])
    for cid in parity:
        if cid in arch:
            parity[cid]["equals_archived_broad_v1"] = (
                arch[cid] == cases[cid]["mtp"]["token_ids"])
    acc = {"per_case": per_case, "pooled": pooled, "equal_case_mean": eq, "parity": parity,
           "requests": len(reqs),
           "definitions": {
               "alpha1": "P(d1 == emitted token) over depth-1 rounds",
               "p_a2_given_a1": "P(chained d2 == emitted token at pos+1 | d1 accepted, pos+1 "
                                "emitted); chain = head(norm(h1), d1) per upstream mlx-vlm",
               "p_a3_given_a2": "P(d3 == token at pos+2 | d1,d2 correct)",
               "caveat": "rounds start at depth-1 round boundaries; a depth-2 runtime would "
                         "start rounds at different positions"}}
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1))

    # depth model with measured acceptances
    spec = importlib.util.spec_from_file_location(
        "dm", ROOT / "spikes/027-qwen38-mtp-depth/depth_model.py")
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    med, _, _, _ = dm.spans()
    res = [r for r in json.loads((ROOT / "results/qwen38-broad-v1/results.json").read_text())
           ["requests"] if not r["warmup"]]
    t_v2 = med["verify_width2"]
    t_ds = med["draft_plus_snapshot_after_reject"]
    t_adv = med["advance_plus_draft_plus_snapshot_after_accept"] - t_ds
    t_commit = med["commit_prefix1"]
    t_native = st.median(1 / r["post_first_tps"] for r in res if r["method"] == "native")
    dw = t_v2 - t_native
    model_rows = []
    for cid, c in sorted(per_case.items()):
        nat = [r for r in res if r["case_id"] == cid and r["method"] == "native"]
        tn = 1 / st.median(r["post_first_tps"] for r in nat)
        row = {"case": cid, "alpha1": c["alpha1"], "p_a2_given_a1": c["p_a2_given_a1"],
               "p_a3_given_a2": c["p_a3_given_a2"], "model": {}}
        for dn, d in (("dw_full", dw), ("dw_half", dw / 2)):
            for k in (1, 2, 3):
                cond = [c["alpha1"], c["p_a2_given_a1"] or 0.0, c["p_a3_given_a2"] or 0.0][:k]
                e, all_acc = dm.expected_tokens(cond)
                t = (t_v2 + (k - 1) * d + k * t_ds + cond[0] * t_adv + (1 - all_acc) * t_commit)
                row["model"][f"{dn}/k{k}"] = {"expected_tokens_per_round": e, "round_s": t,
                                              "speedup_vs_native": e * tn / t}
        model_rows.append(row)
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))  # noqa: E731
    agg = {k: gm([r["model"][k]["speedup_vs_native"] for r in model_rows])
           for k in model_rows[0]["model"]}
    out = {"timing_source": "broad-v1 rank0 rank_action medians via 027 depth_model.spans()",
           "derived_s": {"verify_width2": t_v2, "draft_plus_snapshot": t_ds, "advance": t_adv,
                         "commit": t_commit, "native_step": t_native, "dw_full": dw},
           "equal_case_gmean_speedup_vs_native": agg,
           "depth2_over_depth1": {d: agg[f"{d}/k2"] / agg[f"{d}/k1"] for d in ("dw_full", "dw_half")},
           "depth3_over_depth1": {d: agg[f"{d}/k3"] / agg[f"{d}/k1"] for d in ("dw_full", "dw_half")},
           "cases": model_rows,
           "assumptions": "027 depth_model round-cost formula; extra draft costs one "
                          "draft+snapshot span; verify(k+1)=verify2+(k-1)*dw; acceptance "
                          "conditionals are the measured per-case values (cases not in this "
                          "run are absent)."}
    (OUT / "depth2-model.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"pooled": pooled, "equal_case_mean": eq,
                      "parity_all": all(p["token_ids_equal"] and p["finish_equal"]
                                        for p in parity.values()),
                      "archived_equal": {k: v.get("equals_archived_broad_v1") for k, v in parity.items()},
                      "per_case": {k: (round(v["alpha1"], 3), v["p_a2_given_a1"] and round(v["p_a2_given_a1"], 3),
                                       v["p_a3_given_a2"] and round(v["p_a3_given_a2"], 3)) for k, v in per_case.items()},
                      "agg": agg, "d2/d1": out["depth2_over_depth1"],
                      "d3/d1": out["depth3_over_depth1"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
