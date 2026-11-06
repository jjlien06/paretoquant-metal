# Qwen3.8 MTP draft depth 2–3: assessment v1

Scope: code analysis, a model built from archived data, and a small-model (0.8B) prototype of the
verify/rollback mechanics on the local M2 Pro GPU. **No 27B load, no two-host run, no MTP-head
chaining test, no speed measurement of depth ≥ 2.**

## 1. What limits the current 021 fast path to depth 1

Every layer has width 2 hard-coded. The Metal kernel is the one exception.

| Layer | Hard limit (spikes/021-qwen38-prefix-rollback) | Needed for depth k / width k+1 |
|---|---|---|
| Controller | `speculative.py:26-55`: one `draft`, `verify([y, d])`, requires 2 predictions, `commit_prefix(1)` only | Accept the longest matching prefix n ≤ k, emit `d1..dn + pred[n]`, `commit_prefix(len(emitted))`, keep the EOS/cap terminal-not-consumed convention (prototype: `spikes/027…/speculative_k.py`) |
| MTP head | `mtp_head.py:83-98`: `BoundHead.forward` returns only logits, not the head's hidden state. The sidecar has `mtp_num_hidden_layers: 1` | Chained drafting: draft i+1 = head(head_hidden_i, d_i), using the same single layer. Requires returning `m.norm(h)` (assumed convention, **unverified**). Head KV gets k-1 speculative entries; trim them and catch up j accepted pairs with verified target hidden in one batched call |
| Rank protocol | `rank_control.py:30-31,54`: `verify` requires exactly 2 tokens; `commit_prefix` ∈ {0,1,2} | `verify` of 2..k+1 tokens and commit 0..W, bound to the last verify width |
| Backend | `mlx_backend.py:146`: a width other than 2 is sent as plain `forward` (no capture, no verifier mode) | Route widths 2..k+1 to `verify` |
| GDN prefix rollback | `prefix_rollback.py:27,133,138,145-146`: capture rejects shape ≠ (1,2); keep ∈ {0,1,2}; KV offset is pre+2; only the keep==1 recurrence rerun exists | General W: rerun the recurrence on `args[:, :keep]`, restore conv tail `conv_input[:, keep:keep+K-1]`, trim KV by W−keep (prototype: `prefix_rollback_k.py`) |
| Verifier numerics | `verifier.py:31,54,73`: the singleton-QMM / slot-QMV substitution fires only when `x.shape[1]==2` (width 3 would silently run native batched QMM) | Substitute at the verify width W (prototype: `verifier_k.py`) |
| Slot-QMV kernel | `slot_qmv.py:56` and `slot_qmv_header.metal:460` already admit **1..9 slots** | No code limit. Width 3/4 needs per-GPU numerical gates and timing |
| Admission gates | `verification.py:124,273`: forced windows of 2, prefixes (0,1,2) | Gates at every used width W and keep 0..W, on both physical GPUs, small and full |

## 2. Expected speedup (model, not measurement)

Source data: `results/qwen38-broad-v1/results.json` (72 measured requests, 18 cases) and rank0
`rank_action` timestamps from `results/qwen38-broad-v1/batch-0{0..5}-v1/actors/*/stderr`.
Script: `spikes/027-qwen38-mtp-depth/depth_model.py`. Output: `depth-model.json`.

Measured inputs at depth 1, medians:
- Width-2 verify: 162.6 ms
- Draft + snapshot: 11.8 ms
- MTP advance: 2.2 ms
- Prefix-1 commit: 4.7 ms (only after a rejection)
- Round period: 177.1 ms
- Native width-1 step: 134.9 ms

Pooled acceptance α = 2782/3254 = 0.855. Per case it ranges from 0.74 (chinese-short) to 1.00
(structured). Consecutive rounds show mild positive correlation: P(A|A) = 0.868 and P(A|R) = 0.772.

Expected emitted tokens per round = 1 + Σ_{j=1..k} Π_{i≤j} α_i. Round time = verify(k+1) + k·draft +
advance + P(any reject)·commit, where verify(k+1) = 162.6 + (k−1)·Δw. Δw = 27.6 ms is the measured
width1→width2 increment, or half that as an optimistic case. Draft i acceptance is α·r^(i−1); r = 1 is
i.i.d. geometric. Chained-head acceptance at depth ≥ 2 has never been measured here.

Model check at k = 1: the equal-case geometric mean is 1.433× versus the observed 1.429×.

Equal-case geometric-mean speedup vs native warm decode:

| Scenario | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| Δw = 27.6 ms, i.i.d. (r=1) | 1.43 | 1.63 | 1.72 |
| Δw = 27.6 ms, r=0.9 | 1.43 | 1.58 | 1.58 |
| Δw = 27.6 ms, r=0.8 | 1.43 | 1.54 | 1.47 |
| Δw = 13.8 ms, i.i.d. | 1.43 | 1.75 | 1.93 |
| Δw = 13.8 ms, r=0.8 | 1.43 | 1.64 | 1.64 |

Interpretation:
- Depth 2 is the robust choice: +7% to +22% over today's depth 1 across these scenarios.
- Depth 3 helps only if chained acceptance decays slowly (r ≳ 0.9) or wider verifies are cheap. At
  r = 0.8 with linear Δw it loses to depth 2.
- High-acceptance cases (structured, code, math) gain the most. The model gives up to ~2.1–2.4× for
  structured output at k = 3.
- The external llama.cpp observation (n-max 3, ~1.33×, acceptance 50–96%) came from a different
  machine and engine. It is consistent with depth 3 not being a large win at moderate acceptance.

These are warm post-first decode estimates only. TTFT and whole-request time are unaffected or worse,
since prefill is unchanged.

## 3. Prototype: depth-k verify and rollback on the 0.8B control (`spikes/027-qwen38-mtp-depth/`)

Model: `models/control-qwen35-08b-da28692b…`, Qwen3.5 0.8B Q4, same qwen3_5 hybrid. It has no MTP
head, so the drafter is a stand-in: a replay of sequential greedy tokens with deterministic forced
mismatches at rates 0, 0.3, 0.6 and 1.0.

- `small-gpu-gates.json` (M2 Pro GPU, mlx 0.32.3 / mlx-lm 0.32.0):
  - Capture gates at widths 2, 3 and 4 × modes native / singleton-qmm / slot-qmv. Each gate compares
    same-width instrumented vs plain, then every keep 0..W vs native sequential width-1 caches, the
    slot logits and a 1-token continuation.
  - Result: 9/9 gates and 36/36 prefix rows pass the strict 1e-6/1e-5 tolerance. Worst absolute
    difference is 0.0 (bit-exact).
  - End-to-end: 72 runs (2 prompts × 3 modes × depth 1/2/3 × 4 mismatch rates, 48-token cap). 72/72
    have tokens identical to sequential greedy, and 72/72 final hybrid caches (KV, conv, recurrent)
    are strictly equal (max abs 0.0). Across runs, 1,521 rollback commits occurred.
  - The slot-QMV kernel was actually exercised at slots 3 and 4 (substitution counters are in the
    receipt).
  - `small-gpu-gates-run1-prepathfix.json`: identical earlier run, before a model-path constant edit.
- `pytest-gpu-m2.xml`: 318 passed. This covers 302 symbolic control-flow cases (depth 0–4, EOS, caps,
  mismatch rates) plus 16 small-model tests. Four of those are negative controls that the gate catches:
  wrong keep, sabotaged conv tail, no rollback, and width mismatch.
- `pytest-cpu-subset.{xml,log}`: CPU subset (width-3 gate, depth-2 e2e with a 6-token cap, one
  negative control). 3 passed in 548 s; CPU runs about 4 s per width-1 token on this model.

## Blockers and what is NOT established

1. **Chained MTP acceptance at depth ≥ 2 is unmeasured.** It is the dominant uncertainty. Measuring it
   needs the 27B target (or its archived hidden states) plus the sidecar head, which is outside this
   box. An offline acceptance screen needs saved per-position target hidden states, which the archived
   receipts do not contain.
2. **Width-3/4 verify cost on the two-host 27B is unmeasured.** Δw is extrapolated.
3. **27B / M4 numerics are not admitted.** Native width-2 matched on M2 but diverged on M4 for the
   full model. The bit-exact small-model M2 results here do not admit width 3/4 on M4 or on the full
   checkpoint. Slot-QMV at 3/4 slots needs physical gates on both GPUs.
4. The distributed protocol (rank_control, Rank1Target, supervisor, auditor `mtp_audit.py` which
   assumes one draft per round) is not ported. The prototype is single-host.
