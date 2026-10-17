# Raw025 integration v2 (CPU implementation, not approval)

`integration025.py` consumes the actual025 producer format without translating it
into the older proposed `protocol.py` normalization. Original v1 entry points,
fixtures, estimator, frozen workloads and historical receipts are preserved.

## Inert freeze, plan, collect

From the repository root, use `.venv/bin/python` and newly allocated directories:

```sh
.venv/bin/python -m benchmarks.qwen38_overlap.cli freeze025 \
  --out-dir NEW_FREEZE \
  --target-config-sha256 EXTERNAL_TARGET_CONFIG_SHA256 \
  --target-index-sha256 EXTERNAL_TARGET_INDEX_SHA256 \
  --small-metadata SMALL_METADATA_DIRECTORY \
  --small-config-sha256 EXTERNAL_SMALL_CONFIG_SHA256 \
  --small-index-sha256 EXTERNAL_SMALL_INDEX_SHA256

.venv/bin/python -m benchmarks.qwen38_overlap.cli run025 \
  --manifest NEW_FREEZE/integration-freeze.json \
  --manifest-sha256 EXTERNAL_FREEZE_SHA256 --phase pilot --out-dir NEW_TRIAL

.venv/bin/python -m benchmarks.qwen38_overlap.cli collect025 \
  --manifest NEW_FREEZE/integration-freeze.json \
  --manifest-sha256 EXTERNAL_FREEZE_SHA256 \
  --raw-dir TRIAL/raw --trusted EXTERNAL_TRUST.json \
  --trusted-sha256 EXTERNAL_TRUST_SHA256 --out-dir NEW_REPORT
```

`freeze025` snapshots every literal `remote_actor.SOURCES` entry, including the
controller, all registered transitive Python/config/Metal dependencies and one
exact schedule per process. It also pins the exact top-level harness module set,
independently hashed small/full config/index JSON, and complete numerical plan.
Its phase order is pilot, batch-00 through batch-05. Pilot has six requests; the
six full batches have 14 each (84 total, 72 measured, 12 required warmups).
Both methods are truthfully `mtp`, with explicit serial/overlap `prefill_mode`.
Original prompt bytes, mode order, caps, repeats and source slots are unchanged.

A freeze contains `execution_permitted=false`, not a deployment/health receipt.
`run025` without `--execute` validates source bytes/registry and real producer
schedule grammar, prints public arguments, never reads a private profile, and
does not create the trial directory. Freeze must be recreated after any runtime
or harness source change, including the pressure-v2 repair. CPU smoke freezes
must not be reused as parent-reviewed GPU admission.

`collect025` takes matching repeated raw-directory/trust-path/trust-hash triples
(up to six). Reports use the original median-of-paired-ratios reducer: all six
long cases independently need >=30% TTFT and >=15% request-wall reductions. Full
reports keep all18 cases and regressions. A standalone pilot reports its two
measured long cases plus exact six-row/warmup coverage; it cannot be a full-suite
pass. Invalid inputs retain hash-bound public attempted rows as failed, not
repaired successes. TEST data returns exit2 even when all detached checks pass.
JSON/CSV/Markdown reports contain public metrics/hashes, not private startup args.

### Explicit amended memory policy

Normal is still the default (`--pressure-policy normal` is equivalent); its
manifest fields and `--require-normal-pressure` plan flag remain unchanged.
To create a **new**, non-approved amended freeze, append to `freeze025`:

```sh
--pressure-policy noncritical \
--memory-acceptance results/qwen38-prefill-overlap-v1/memory-acceptance-amendment-v1.json \
--memory-acceptance-sha256 a0341821118cfd83d45f0f8979152aaffae3f296b372f8dc29b58aacc35d409f
```

The amendment schema is `qwen38-overlap-memory-acceptance-amendment-v1`, with
`pressure_policy=noncritical`, `allowed_pressure_levels=[1,2]`,
`abort_pressure_levels=[4]`, and the original frozen `original_acceptance_sha256`.
Explanatory fields are retained, not rejected. The manifest contains explicit
`pressure_policy` and `memory_acceptance.{bytes_utf8,sha256}`. Freeze/read/run/raw
audit independently rehash/decode those exact bytes and recheck the original
acceptance binding; a receipt cannot select or escalate its own expected policy.

The amended plan sends `--pressure-policy noncritical`, **not**
`--require-normal-pressure`. Both serial and overlap still require an effective
guard. Actual runtime `policy_name`/`pressure_required` helpers validate controller,
actor startup/result/job and rank/process flags. The pressure helper calls receive
external `expected_policy`, and completed-trial receives the independently bound
options. Noncritical uses `qwen38-owned-pressure-v3` with explicit
`policy=noncritical`, `allowed_levels=[1,2]`, `required=true`; normal remains v2.
Both-host identity, sampler timeout/interval, full execution coverage, deadlines,
numerics, output equality, placement, all84 rows and all18 estimators are unchanged.
Critical4, unknown/missing sampler data and owned-cleanup failures still reject.

Noncritical external trust must explicitly include `pressure_policy=noncritical`
both at top level and in **each** `actor_contexts` record. Physical normal trust
must likewise explicitly select `normal`; old normal TEST fixtures stay compatible.
JSON audits/reports retain per-actor observed levels and warning-sample counts;
Markdown labels warning observations as **not normal-pressure qualification**.
Old failed normal trials are not reclassified. See `NONCRITICAL_API_CONTRACT.json`.
Fresh combined runtime/loader/harness parent review is required before any
physical trial; this CPU amendment support is not hardware admission.

## Actual producer fields (not normalized aliases)

| Input | Producer fields used directly |
|---|---|
| `rank{0,1}.receipt.json`, `server.receipt.json` | `job_status`: `nonce`, `startup`, `pids`, `result`, `source_sha256`, complete JSON-lines `stdout`/`stderr` |
| Actor startup/PID/result | `actor_pid`, `child_pid`, `pgid`, `nonce`, `host_role`, `actor_role`, `actor_started_monotonic`, `require_normal_pressure`; PID `child_start_seconds`; result `execution.{child_start_seconds,gate_release_seconds,child_end_seconds}`, `elapsed`, `reason`, exact integer `returncode`, `error_type`, log completeness/cap fields, owned `cleanup` |
| Pressure | `qwen38-owned-pressure-v2`, `binding`, `actor_started_monotonic`, `interval_seconds`, `sampler_timeout_seconds`, `sample_cap`, `samples.{stage,elapsed_seconds,level}`; actual public `expected_pressure_binding` / `verify_actor_pressures`, not a context-free pass bit |
| Rank process evidence | stderr `process_start` with `pid`, rank, full `source_sha256` and actual overlap/pressure/hidden flags; numerical admission before request boundaries |
| Requests | `repeat` is process ordinal, `sample_repeat` is original repeat; `id`, `case_id`, `method`, `warmup`, `max_tokens`, `prompt_sha256`, `prompt_token_ids`, `prefill_mode`, `source_request_id`, `local_rank`, `request_nonce`, `cache_epoch`, `token_ids`, `finish_reason`, actual `first_token_seconds` / `wall_seconds`, independent local rank memory/wall and MTP counters |
| Actions | actual `rank_action`: `request`, `sequence`, `op`, `tokens`, `ok`; `prefill_pipeline` uses full prompt including singleton metadata, while native prefix intervals omit final token |
| Causal stream | untouched `qwen38-prefill-overlap-stream-v1` envelopes/payloads and chain; real `Command`, `verify_overlap_stream(require_native=True)`, `verify_compound_drain`; rank0 `pipeline_drained` precedes compound action and final ordinary forward |
| Gates | `verification.groups["32"].{small,full}` width windows, prefix rollback, `hidden_only_prefill`, `overlapped_prefill`, actual peer summaries, plus metadata/mode binding; canonical `supervise.verify_completed_trial` receives independent pressure bindings |
| Controller | `startup.json`, `result.json`, `schedule_admission.json`, exact source/controller/peer readback, `actors.json`, all three `*.job.json`, `listeners.cleanup.json`; final owned actor/service cleanup plus each controller transport/control cleanup is checked, not merely `status=success` |

Raw dictionaries and original files are not rewritten. The adapter checks all
actor/rank identity, sources, numerical inventory, request grammar, output and
cleanup prerequisites before exposing successful rows. It separately calls
`pilot_budget_check` on all three complete raw status dictionaries: actual UTF-8
log/retrieval size and conservative projection for a 14-row batch, not an estimate
invented from synthetic measurements. A TEST budget result is only a TEST result.

## Independent trust manifest

The caller must authenticate an external file SHA256 before passing its contents
to `audit_raw_trial(root, trusted, freeze_manifest)`. CLI does that readback.
A receipt selecting its own expected values is **not** independent trust.

Required schema: `qwen38-overlap-raw025-trusted-v2`. Required fields:

- `artifact_kind`: exactly `TEST_FIXTURE_NOT_MODEL_RESULTS` or `PHYSICAL_RECEIPT`.
- `phase`, exact externally frozen `manifest_sha256`.
- `raw_artifacts`: independently sealed basename -> SHA256, including every
  `RAW_REQUIRED` input. Exact source bytes are additionally re-read from
  `raw/sources` and compared with the external phase registry, not reported hashes.
- `prompt_token_ids`: independent tokenizer admission for every ordered row;
  never taken from submitted rank output as its own reference.
- `native_outputs`: one independent original native `{token_ids,finish_reason}`
  per row. Matching current serial/overlap ranks is not a native reference.
- `tensor_sha256_by_rank`: independent native per-tensor SHA256 maps with exact
  checkpoint-derived ownership. Actual rank loading and replicated hashes must
  agree with these, and receiver metadata may not contain weight files.
- `actor_contexts`: rank0/rank1/server records independently bound by the parent:
  `host_role`, `actor_role`, `nonce`, `actor_pid`, `child_pid`,
  `actor_started_monotonic`, `execution` and `elapsed`. The adapter reconciles
  these against separate startup/PID/result documents and controller job/remote
  ownership before deriving pressure bindings. A short healthy timeline from
  another job/host or interval is not accepted.
- `listener_endpoints`: independent ordered `[address,port]` pairs for local rank,
  remote rank, remote weight server. Must match admitted host addresses and the
  exact closed endpoints in `listeners.cleanup.json`, with three distinct ports
  by endpoint identity.
- For physical acceptance only, `parent_review`: `approved=true`, exact `phase`
  and `manifest_sha256`, and external64-hex `review_receipt_sha256`. Authenticating
  that review/trust against the independent parent's records is caller-owned;
  an arbitrary JSON claim or hash computed by the producer is not approval.

The physical gate is wired, not a constant-false placeholder: successful raw
checks + exact external trust + physical provenance + matching parent review can
produce `physical_accepted=true`. **Every TEST input remains false.** This is a
detached source-bound receipt audit, not a CPU method to mint the runtime's live
model/group-bound admission capability. No real physical positive receipt has
been generated or admitted by this implementation task.

## Parent-only execution API

`supervisor025.guarded_run(manifest_path, sha, phase, out_dir, execute=True,
profile_path=..., parent_guard=..., runner=..., pilot_admission=...)` is the only
new executable seam. Normal CLI invocation has no parent guard and fails closed.
The parent guard must explicitly approve this exact external manifest and phase,
using `PARENT_REVIEWED_PHYSICAL_APPROVAL`; fresh host/bridge/headroom checks,
single hardware ownership, independent code review and source deployment/readback
remain the parent's responsibility, never inferred from historical preflight.

Only after guard approval is the private JSON profile read. Its exact whitelist:
`remote_source`, `ssh_key`, `ssh_target`, `local_address`, `remote_address`,
`ssh_host_key_alias`, `small_model_remote`, `sidecar`, `remote_root_by_phase`.
`remote_source` is the single pinned target-weight checkpoint directory; it is
passed unchanged as `--remote-source` in every phase. `remote_root_by_phase` maps
all seven phases to distinct immutable code snapshot directories; the selected
entry becomes `--remote-root`, used to locate `remote_actor.py` and its imports.
The real producer parser and pure `ssh_argv` builder are exercised by regression
tests to prevent swapping code roots and weight sources.
Profile data is never returned, written into the public plan or accepted as
arbitrary extra argv. Real025 parser/validation runs with sanitized failures;
parent guard gets an isolated plan copy. Actual controller raw logs remain private
owner-controlled inputs, not published reports. No saved profile is guessed or
loaded from historical logs.

Full batches require `pilot_admission=(raw_dir, external_trust_path, trust_sha)`;
the complete real source-bound pilot and byte-budget checks run again. A passed
boolean/dict cannot authorize them. TEST authorization requires an injected mock
runner and cannot fall back to the real subprocess. Controller exit0 is never
receipt acceptance: return status is `raw_receipt_audit_required`. Timeouts or
exceptions are reported with cleanup explicitly unverified, not silently passed.

## Numerical expected plan and CPU proof limits

Dimensions and cache class order derive independently from small/full config/index
bytes: every original layer, local layer numbering, KV keys/values/offsets, GDN
convolution history and F32 recurrent state, normalized hidden/full vocabulary,
shifted head KV/dtype, final singleton and both continuations. Small native
width/prefix/hidden gates own the whole local control model; small overlap uses
the actual two-rank split. Full uses28/36. Width prompt is explicitly five token
IDs `[1,2,3,4,5]`, forced IDs6..11; no benchmark prompt changes. Full overlap and
hidden gates cover length513 with **two full256-token prefix chunks**, not257.
Original atol1e-6/rtol1e-5 remain unchanged.

CPU tests reconstruct explicit TEST grammar through real025 `Worker`/`Command`,
pipeline/credit stream helpers, pressure sampler/receipt helper and `job_status`.
Numerical TEST shapes use the producer's fixture helpers, never GPU arrays. Some
old loading identity is read from preserved024 receipts, without treating024 as
025 or changing historical files. TEST `physical_gpu` fields are grammar inputs,
not executed measurement or live capability. Rehashed semantic mutations must
still fail. The final CPU receipt records actual counts/commands/source hashes.

Remaining physical blockers: parent independent review, final immutable sources,
current both-host health/deployment readback, actual small two-rank + same-loaded
full513/chunk256 gates, original-native output/tensor trust, real pilot/evidence
budget, full84 rows, all six performance targets and verified owned cleanup. No
GPU/model import, SSH/network, install/build, signals, deployment or commit is part
of this CPU task.
