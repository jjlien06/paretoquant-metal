# 025 frozen integration surface (CPU implementation; physical gates unexecuted)

## Command-line flags

`supervise.py` and `probe.py`: `--overlapped-prefill` (default false),
`--require-normal-pressure` (default false), and `--pressure-policy` with choices
`normal` (default) / `noncritical`. Overlap capability plus an effective pressure
guard are mandatory for BOTH serial/overlap rows. Default policy requires the
legacy normal flag; explicitly opted-in noncritical forces monitoring with that
flag false and rejects a conflicting true flag. Continue to pass
`--hidden-only-prefill --hidden-gate-chunk-size 256 --prefix-rollback`, fixed
split28/36/group32 and native/slot-qmv. `--overlapped-prefill` means admit/run
both-mode gates, not force all requests to overlap. `supervise.py --dry-run`
validates CPU-only options/source registry without transport, writes or MLX.

## Schedules

`schema = "qwen38-prefill-overlap-schedule-v1"`. Top-level exact keys:
`schema, requests`. Each request's exact keys:

```text
id, case_id, method, prompt, max_tokens, warmup, repeat,
prefill_mode, source_request_id
```

`method` is always `mtp`. `prefill_mode` is `serial` or `overlap`.
`source_request_id` is the original source slot ID, NOT a native execution label.
Each `(case_id,warmup,repeat)` has exactly one row of each mode. IDs and source
slot IDs are unique within the schedule.1..32 rows. Other fields/limits and exact
prompt/cap policy remain unchanged. Old `qwen38-workload-schedule-v1` admission
and default serial behavior remain supported. New schema requires capability
flag before hardware. Every request constructs a fresh backend/target/cache.

## Rank results and incremental phase envelopes

Per-request fields on BOTH rank results, `request_before`, new `request_start`
(immediately after request_before), `request_done` and ALL in-request events:

```text
id, case_id, method="mtp", warmup, max_tokens,
repeat=ordinal, sample_repeat=original repeat,
eval_group_size, prompt_sha256, prefill_mode, source_request_id
```

Both rank results also carry `request_nonce` and `cache_epoch` (same fresh UUID,
reconciled with reset/actions). Existing independent local rank/memory/wall fields
are retained. Rank1's local memory/wall is never overwritten by rank0's values.

Process `process_start` and top-level rank result: `overlapped_prefill` and
`pressure_policy` and truthful `require_normal_pressure`. `verification` and
`numerical_admission_done` also carry both pressure fields. `verification` includes `overlapped_prefill`,
`prefill_modes=["serial","overlap"]`, plus `request_prefill_modes`, the ordered
list of exact `{id,prefill_mode,source_request_id}` maps for new-schema rows.
`numerical_admission_done` carries those same fields before request timers.

Each group stores `small.overlapped_prefill`, `full.overlapped_prefill` and
bounded `peer.{small,full}.overlapped_prefill` summaries. Gate schema:
`qwen38-prefill-overlap-gate-v1`; required fields include `physical_gpu=true`,
`scope`, `modes`, `chunk_size=256`, exact `prompt_lengths` (small:
1/2/255/256/257/511/512/513; full:513), rank/owned layers, eval group, tolerances,
all cache parts and full output/head/continuation metrics. These fields are
NOT authorization: the live Admission additionally requires a model-bound
process record produced by the actual physical gate. CPU fixture JSON cannot
mint it. Existing numerical gates remain independently required.

## Compound and credit frames

Actual Command keys remain `request, sequence, op, tokens`. New opcode
`prefill_pipeline`: full nonempty prompt, IDs<=2048, including final-token
metadata. All other token caps remain256; frame cap65536. Prefix intervals are
`range(0,len(prompt)-1,256)`, final singleton outside compound. One-token prompts
bypass it. Exactly one compound ACK with unchanged ACK shape after the prefix.

Receive credit exact keys/types:

```json
{"schema":"qwen38-prefill-receive-credit-v1","request_nonce":"<fresh UUID>",
 "command_sequence":1,"chunk_index":0,"physical_start":0,"physical_end":256}
```

Only exact strings/integers; bool-as-int, unknown keys, stale nonce/sequence,
wrong chunk/start/end fail closed. Rank0 sends after evaluated hidden receive,
before any local graph. Rank1 waits after evaluated hidden send before next
embedding. No final per-chunk completion ACK. No normal error/reset frame after
raw-channel failure; owned bounded actor supervision cleans up.

## Source-bound causal stream schema

All internal chunk/native progress events are wrapped by
`qwen38-prefill-overlap-stream-v1` and bind:

```text
schema, rank, request_nonce, cache_epoch=request_nonce, command_sequence,
prefill_mode="overlap", chunk_index, physical_start, physical_end,
stream_payload, stream_sequence, stream_previous, stream_digest
```

`stream_payload` is the exact pre-envelope event; every payload key must equal
its top-level value with exact type. `stream_sequence` is consecutive from0.
Seed = SHA256(compact JSON `[request,sequence,full_prompt_ids,rank]`). Each digest
is SHA256(previous hex + sorted compact JSON payload). This binds the full trusted
prompt and performed ordering, not only action counts. The verifier reconstructs
the digest AND checks state grammar; resealing swapped/omitted/replaced credits
or chunks does not make a valid transcript. Source hashes must be pinned by an
external preregistration, never chosen by the receipt.

Rank0 per chunk, in order:

```text
pipeline_chunk_start
handoff_receive_before -> evaluated recv -> handoff_receive_done(shape,dtype)
pipeline_receive_evaluated -> exact credit send -> pipeline_credit_sent
native block/layer completions -> hidden_norm_done -> pipeline_target_done
shifted head eval -> pipeline_head_done(head_position=start,head_offset=end,shifted_tokens)
```

Rank1 per chunk:

```text
pipeline_chunk_start -> native embedding/block/layer evaluations
handoff_send_done(shape,dtype) -> pipeline_send_evaluated
validated exact credit -> pipeline_credit_received
```

Both finish with chained `pipeline_prefix_done(prefix_offset=len(prompt)-1)`.
Rank0 additionally emits `pipeline_drained(request_nonce,command_sequence,
prefill_mode,pending_handoffs=0)` AFTER consuming the final compound ACK and
BEFORE recording the compound rank_action / entering ordinary final forward.
Each internal chunk must contain every owned `layer_done` once, in original
order. Rank0 must normalize hidden but NOT emit target-vocab head_done inside
prefix; final ordinary forward still projects. Monotonic timestamps remain
local rank clocks, not cross-host GPU-overlap or cable-transfer attribution.

Public CPU APIs:

```python
verify_overlap_stream(events, command, rank, *, owned_layers, require_native=True,
                      hidden_size=None, dtype=None, eval_group_size=None)
verify_request_modes(spec, result, events, *, enabled)
verify_compound_drain(events, command, rank)
verify_overlap_gate(evidence, *, expected_layout, scope, eval_group_size)
metadata_layout(options, *, scope, rank)  # hash-bound JSON, no MLX import
pilot_budget_check(receipts, *, requests_observed, requests_planned)
```

`require_native=False` exists ONLY for explicit symbolic protocol fixtures;
controller calls the default strict native-evaluation grammar with independently
trusted hidden size/dtype/group size. Receipt digests
are not proof against compromised Python and cannot replace physical numerical
admission. No simultaneous-Metal-kernel claim is made.

## Both-host actor pressure receipts (v2 repair interface)

NEW public API (v1 files remain historical, not current proof):

```python
policy_name(options)  # namespace or dict; validate normal/default or noncritical
pressure_required(options)  # legacy exact bool OR policy==noncritical
PressureGuard(enabled, *, policy="normal", ...)
run_owned(..., require_normal_pressure=False, pressure_policy="normal", ...)
expected_pressure_binding(status, *, nonce, host_role, actor_role, expected_policy="normal")
verify_pressure(receipt, *, expected_binding=None, actor_result=None, expected_policy="normal")
verify_actor_pressures(receipts, *, expected_bindings=None, expected_policy="normal")
```

`expected_pressure_binding` consumes independently retrieved `startup`/`pids`
records plus the controller-admitted job nonce and roles; NEVER derive expected
identity from `result.pressure.binding`. Role pairs are `local/rank0`,
`remote/rank1`, `remote/server` (opaque labels, no addresses or keys). It checks
startup/pids/result origin and child-start reconciliation. The returned binding
has exactly `host_role, actor_role, nonce, actor_pid, child_pid`. Source hashes
must still match the externally frozen registry. `verify_pressure` requires both
external binding and independent actor `result` with same-origin execution
boundaries. `verify_actor_pressures` requires all three distinct expected actors,
receipts ordered correspondingly, and rejects replayed nonces or per-host PIDs.
Controller/workload entry points accept keyword `expected_pressure_bindings`;
raw adapters must reconstruct them from their trusted per-actor job descriptors
and startup/pids records, not result self-claims. Missing bindings fail closed
when pressure is required. Supply policy from externally admitted options, NEVER
from the receipt. Default pressure opt-out remains supported outside overlap;
noncritical cannot disable monitoring. All three jobs include `pressure_policy`.
For noncritical, actor startup/result MUST explicitly match `pressure_policy` and
legacy `require_normal_pressure=false`; process/result/verification/live-admission
flags also match on each rank. Default normal/v2 verification remains backward
compatible with legacy records lacking the new policy field.

Normal actor pressure receipts retain schema `qwen38-owned-pressure-v2` unchanged.
Noncritical requires `qwen38-owned-pressure-v3`, explicit `policy="noncritical"`,
`allowed_levels=[1,2]` and actual exact integer1/2 observations. V2 or a missing
policy is never accepted under noncritical; v3 is never accepted under normal.
Critical4 and all malformed/missing/error measurements fail closed in either mode.
Both schemas retain interval0.5s, exact sampler
timeout0.25s, cap2048 and exact grammar `pre_spawn, owned_child*, post_child`.
Finite nonnegative strictly increasing sample times, positive total duration,
policy-allowed exact integer levels, gaps<=1s are mandatory. Actor captures its monotonic
origin BEFORE guard/setup, and records actual child start, gated execution release,
and observed child end using that origin. Coverage must bracket the child interval
and end within1s of independently reported actor elapsed. Long source-copy/setup
before the first sample is allowed, not mistaken for missing child coverage.
Actor rechecks stop/deadline after the initial sampler before Popen and before
releasing the gated bootstrap. Scope stays sampled-only, not continuous pressure
certification. No unrelated/reaped/reused PID signalling is authorized.

## Mandatory real pilot evidence budget

The collector must call `pilot_budget_check` on the THREE actual actor status
objects containing full exact `stdout`/`stderr` strings after its separate pilot.
It enforces64MiB/log and128MiB retrieval and a conservative byte-count projection
for the next planned schedule (1..32 requests/process). Record its returned
receipt before full execution. This implementation has no actual pilot; it
cannot claim a budget pass. External harness owns freeze/deploy/health/output
comparison and exact84-row population/paired estimator. Pilot rows replace none.
