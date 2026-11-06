# Qwen3.8-27B chained MTP depth-2/3 acceptance (measured), v1

Date 2026-11-05. Real two-Mac run: M2 Pro rank0 (final 28 layers + MTP head, native verifier),
M4 mini rank1 (first 36 layers, slot-QMV), group32, prefix rollback, exact target rev
`10c35caa…`, sidecar `b643c01b…`. Workload: all 18 broad-v1 cases (6 task types × short/512/1536
prompts), frozen broad-v1 prompts and caps (128/256), one native + one MTP request per case
in the same process (3 processes: `part-a` 9 cases, `part-b` 6, `part-c` 3).

## Method

Code: `spikes/029-qwen38-mtp-depth2-acceptance/` (copy of the 023 schedule runner; only
`mtp_head.py` and `mlx_backend.py` changed). Runnable deployed snapshots `…-a/-b/-c` differ only
in `benchmark_schedule.json`; deployed to fresh mini dirs `~/Projects/paretoquant-dual-host/spikes/029-qwen38-mtp-depth2-acceptance-{a,b,c}`
with full-file SHA-256 readback match (same `SOURCES` registry names, so no registry change).

- `BoundHead.forward_chain` = the unchanged head op, also returning the head's own
  **normalized** output `m.norm(h)`. Convention checked against upstream mlx-vlm Qwen3.5 MTP
  drafter (`spikes/017-qwen38-mtp/primary_sources/vlm_head.py`: `_forward_hidden` returns
  `self.norm(h)`; `draft_block` feeds it back as `h_prev` with the next drafted token).
- `MTPBackend.draft`: computes d1 exactly as before, then takes an evaluated snapshot of the head
  KV, chains d2 = head(norm h1, d1) and d3 = head(norm h2, d2), restores the snapshot (byte-identical
  buffers; plain `trim` leaves stale rows — caught by `test_depth_chain.py`), logs a `depth_chain`
  event. Verify/commit/target caches untouched.
- Offline (`analyze_depth2.py`): bind each log to its verify action (`[y,d1]`) and emitted progress,
  check simulated accepts == runtime `accepted_drafts`, then score d2 against the emitted token at
  pos+1 given d1 accepted, d3 at pos+2 given d1,d2 correct.

## Correctness / admission
- All 3 supervisors `status: success`: same-process small+full strict numerical gates, rank actions
  agree, `paired_greedy_tokens_and_stop_agree: true`, listeners closed, owned groups gone.
- 18/18 MTP outputs == same-process native outputs token-for-token, and == archived broad-v1 IDs.
- Post-run: no task processes/listeners on either host (remaining loopback listeners are pre-existing
  unrelated processes). M2 pressure hit warning(2) briefly during part-a (allowed); normal at end.

## Results (`acceptance.json`)

Pooled: α1 = 1391/1627 = **0.855** (matches broad-v1 0.855);
**P(A2|A1) = 1098/1382 = 0.795**; P(A3|A2) = 869/1092 = 0.796.
Equal-case means: α1 0.860, P(A2|A1) **0.781**, P(A3|A2) 0.768.

| case | rounds | α1 | P(A2\|A1) | P(A3\|A2) |
|---|---:|---:|---:|---:|
| structured-short/512/1536 | 64/64/127 | 1.00/1.00/0.98 | 0.97/0.95/0.94 | 0.98/1.00/0.91 |
| code-short/512/1536 | 66/66/132 | 0.92/0.94/0.93 | 0.90/0.90/0.89 | 0.82/0.87/0.87 |
| math-short/512/1536 | 67/68/134 | 0.91/0.88/0.90 | 0.90/0.81/0.88 | 0.93/0.77/0.83 |
| spanish-short/512/1536 | 73/72/143 | 0.75/0.76/0.79 | 0.70/0.69/0.74 | 0.63/0.71/0.71 |
| explanation-short/512/1536 | 71/70/122 | 0.79/0.81/0.75 | 0.64/0.56/0.68 | 0.75/0.52/0.63 |
| chinese-short/512/1536 | 73/69/146 | 0.74/0.84/0.75 | 0.56/0.66/0.69 | 0.62/0.63/0.64 |

Bimodal: code/math/structured ≥0.81 (9/18 cases), prose/Spanish/Chinese 0.56–0.74.

## Predicted speedup (`depth2-model.json`, model not measurement)
027 round-cost model, broad-v1 timing medians, measured per-case conditionals; equal-case gmean
vs native warm decode:

| | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| Δw = 27.6 ms (linear) | 1.43 | 1.59 | 1.62 |
| Δw = 13.8 ms (optimistic) | 1.43 | 1.70 | 1.81 |

Depth 2 over today's depth 1: **+10.9% (Δw full) to +18.4% (Δw half)**; depth 3 adds only ~2–7% more.

## Verdict vs go/no-go gate (P(A2|A1) ≥ ~0.80)
Pooled 0.795 / equal-case 0.781: **just below the gate overall**; passes clearly for
code/math/structured, fails for prose and non-English. Not a clean go for uniform depth 2.

Caveats: chained rounds are scored at depth-1 round boundaries (a depth-2 runtime starts rounds at
different positions); one sample per case, greedy, caps 128/256; width-3 verify cost on the 27B
two-host path and M4 width-3 numerics are still unmeasured, so speedups are modelled.

## Reproduce
    results/qwen38-mtp-depth2-acceptance-v1/run-part.sh a|b|c   # needs fresh output dirs
    .venv/bin/python -B spikes/029-qwen38-mtp-depth2-acceptance/analyze_depth2.py
