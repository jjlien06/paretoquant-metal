# Proposed normalized CPU evidence contract v1

Status: `pending_runtime025_adapter_and_independent_review`.

**These are normalization API fields, not undocumented claims about sibling025's
raw fields.** `schema.json` supplies structural envelopes only; it deliberately
cannot certify physical evidence. There is no admitted raw025 adapter in this
implementation. A future adapter must preserve all raw evidence, authenticate its
source/hash bindings independently, and reject unknown or unrepresentable fields.
It must not manufacture completion events or treat equal action counts as proof.

## Trust inputs (independent of receipts)

The CLI requires external trusted-manifest path/SHA256, schedule path/SHA256,
and each normalized record path/SHA256. A trusted input contains:

- `schema: qwen38-overlap-trusted-input-v1` and explicit `artifact_kind`:
  `TEST_FIXTURE_NOT_MODEL_RESULTS` or `PHYSICAL_RECEIPT`.
- `sources`: complete exact relative source-name -> SHA256 map; independently
  frozen `required_source_names` with exactly that set. Every corresponding
  `sources/` file is read/hash-checked and extra/unregistered files are rejected.
  For physical inputs the known legacy runtime/helper/config/Metal/supervisor
  minimum is mandatory. It is not a claim to include all new025 dependencies.
- `configuration`: exactly the fixed configuration in `evidence.CONFIGURATION`,
  including both-mode capability flags and `require_normal_pressure: true`.
- `metadata_sha256`: independently pinned target config/index/sidecar metadata
  bindings (and all additional metadata required by the parent). Both rank and
  controller normalized values must agree exactly.
- `tensor_sha256_by_rank`: exact expected tensor names AND values for each rank,
  copied from the trusted native-control snapshot, not from submitted receipt
  inventory. Replicated entries must agree. Physical union must be1847 tensors.
- `guard_source_sha256`: externally frozen guard implementation source. It must
  occur in the source map; physical inputs bind `remote_actor.py` specifically.
- `max_pressure_gap_seconds`: predeclared finite maximum in [0.5,1.0] seconds;
  samples requested at exactly0.5 seconds. This is sampled evidence, not a claim
  that pressure could not change between samples.
- `requests`: exact ordered batch request-ID set, with independent prompt IDs,
  original native output `token_ids` and `finish_reason`, and the expected binding
  below. Expected prompt IDs require tokenizer/chat-template provenance from the
  parent. This CPU code does not independently retokenize; current S/O agreement
  cannot substitute for original native output reference.
- `numerical_plan_by_rank`: independent architecture-derived gate -> boundary ->
  exact component specification list. Do not derive this plan from submitted
  metric rows. Parent architecture/plan derivation is still pending integration.
- `raw_artifacts`: externally sealed raw relative artifact-name -> SHA256 map.
  Physical inputs require rank0/rank1/server receipts, startup/final, schedule
  admission and source/controller hash receipts. Hash agreement alone is not a
  lossless raw-to-normalized semantic check; that is an explicit adapter blocker.

`qwen38-overlap-trusted-suite-v1` wraps `schedule_sha256` and `batches` indexed by
decimal batch index. Each nested manifest supplies its exact schedule-specific
source set and expected14 requests. Per-batch `benchmark_schedule.json` source
hashes differ; runtime/code hashes must be frozen consistently before actual GPU.
No receipt gets to nominate this trusted map.

## Normalized record envelope

`schema: qwen38-overlap-normalized-record-v1` includes artifact kind, `batch_index`,
configuration/sources/metadata, exact cleanup, two ordered `ranks`, and an ordered
`request_evidence` list. A record is one loaded batch (full14 or pilot6).

Each rank includes rank, actor nonce, source hashes, configuration, metadata,
verifier mode, exact tensor hashes, independent positive parameter bytes,
`log_complete: true`, actual zero integer returncode, `reason: completed`, positive
request-local peak bytes and at least8GiB reserve. Cleanup must prove owned
process exit/listener closure and no unrelated signalling. These are proposed
normalized attestations; the pending raw adapter must verify actual PID/process,
listener, source, startup, placement, tensor and allocation evidence rather than
copy booleans. No CPU audit creates a live numerical capability.

### Sampled watchdog

Each rank has an independent `execution_interval` (`spawn_seconds`, `end_seconds`)
and a `pressure_guard` with:

- `require_normal_pressure: true`, exact `guard_source_sha256`,
  `interval_seconds: 0.5`, `spawn_seconds`, `execution_end_seconds`.
- `samples`: finite ordered `monotonic_seconds`, integer `pressure: 1`, and
  explicit `read_error: null` in every sample. No missing/boolean measurement,
  warning2, critical4, read error, duplicate time or oversized gap is admitted.
- A pre-spawn sample and a final sample covering the full actor interval;
  guard/actor interval must match, all successive gaps stay within the frozen
  maximum, and boundary samples must not be stale. At most10000 samples per rank.

The eventual raw adapter must bind these intervals to actual actor watchdog
start/spawn/exit receipts on BOTH hosts (and requests inside them), not to a
receipt author's shortened convenient interval. Field names above remain pending
coordination with sibling025's documented actual watchdog output.

## Request / prefix transcript

The externally trusted per-request binding contains:

`id`, `ordinal`, `request_nonce`, `cache_epoch`, `prefill_mode`, two `actor_nonces`,
exact `prompt_token_ids`, expected complete per-rank `layers`, `hidden_size`, and
`dtype`. Request nonce and cache epoch cannot be reused within a batch.

Each normalized prefix event has exactly the expected fields plus a finite,
nonnegative rank-local integer `t_ns`. Shared fields are `request_id`, ordinal,
rank, actor/request nonce, cache epoch, mode, phase, command sequence. Clock order
is checked only within a rank; there is no inter-host offset subtraction.

Chunk fields are exact `chunk_index`, `physical_start`, `physical_end`, and
`tokens`. Derive bounds exclusively from the independent prompt IDs:
`range(0,len(prompt)-1,256)`; the final singleton is never a prefix chunk. Hidden
send/receive completions additionally require `[1,width,hidden_size]` and expected
dtype. `target_done` requires the full independently expected owned layer list.
Rank0 head completion requires exact shifted tokens `prompt[start+1:end+1]`,
absolute `head_position=start`, and `head_offset=end`.

Overlap grammar after reset(seq0):

- Both: `pipeline_begin`(seq1).
- Rank0 each chunk: `hidden_receive_done`, `credit_send`, `target_done`,
  `hidden_norm_done`, `head_shift_done` (all compoundseq1).
- Rank1 each chunk: `target_done`, `hidden_send_done`, `credit_receive` (seq1).
- Exactly one final `pipeline_ack_receive` / `pipeline_ack_send` (seq1), then
  `drain_done`(seq1, active_chunks0/pending_handoffs0), then ordinary
  `final_forward_done`(seq2, exact final singleton and target_offset prompt length).

Serial grammar keeps the old per-chunk commands (seq1,2,...):

- Rank0: receive, target completion, hidden normalization, **serial ACK receive**,
  then shifted-head completion. This matches024 `_command` returning after ACK.
- Rank1: target completion, send completion, serial ACK send.
- Drain after all prefixes, then the unchanged final singleton forward at the
  next command sequence. No pipeline begin/credit/compound ACK in serial mode.

One-token prompts skip prefixes and compound/credits completely: reset, drain,
ordinary final singleton. Unknown, missing, duplicate, reordered, stale-mode /
nonce / epoch / command / bound, surplus, wrong-token or shifted-head events
fail, even if counts agree. This proves a normalized recorded causal grammar
only; pending source/adapter checks must ensure each completion is a real
synchronized seam. It is not simultaneous GPU-kernel occupancy evidence.

## Decode grammar

`decode_actions` on each rank start immediately after final-forward sequence and
have exact actor/request/cache/mode/ordinal binding. It faithfully replays024
prefix-rollback MTP's snapshot -> width2 verify -> conditional commit_prefix1 ->
finish grammar against recorded output progress. A verify must consume the
current emitted token, accepted drafts/bonus/terminal-EOS cases advance the exact
cursor, and actual rounds/teacher/accepted counters must agree. No aliases,
missing/surplus actions, forged counters or same-count wrong tokens are accepted.
This cannot validate unrecorded teacher logits; physical numeric gates stay
separate.

## Mandatory numerical evidence

For physical normalized inputs the expected gate IDs include all small lengths
1,2,256,257,258,511,512,513 at chunk256; full lengths1,258,513 at chunk256; both
small/full width2 prefix0/1/2 gates; small hidden-only and full hidden-only
native256 gates. Full513 is mandatory because it actually has two256 prefixes,
not the single prefix of257. These normalized gate IDs are not claims about raw
runtime key names.

Every independently expected gate boundary must be present exactly once with
its exact inventory. Each component is keyed by `(tree,layer,part)` and requires
independently expected shape/dtype/offset. KV keys/values/**offset**, GDN
`state.0.0` convolution and `state.0.1` recurrence, head keys/values/offset, hidden,
full logits and continuation components must not be omitted or replacement-
duplicated. Which arrays/shapes apply must come from trusted checkpoint config.

Metrics require both reference and candidate finite, unchanged atol1e-6/rtol1e-5,
finite nonnegative max absolute error/tolerance/elementwise error ratio, ratio<=1,
and consistent extrema. A submitted `strict_pass: true` cannot override invalid
or missing metrics. Recorded summaries are not independent recomputation of
arrays and cannot grant a live capability; the raw numeric-plan/adapter review
and fresh physical gates remain mandatory.

## TEST fixtures and acceptance

The synthetic TEST receipt deliberately uses a tiny artificial source/tensor /
cache plan, not real checkpoints or original native output arrays. Physical
registry/1847-tensor requirements are not claimed exercised by it. Fixtures are
clearly labeled and never produce `accepted: true`. `accepted` is false for ALL
inputs while the integration status above is pending. Timing coverage and numeric
ratios remain independently visible even when evidence is rejected.
