# 023 — paired same-loaded-model workload benchmark

Isolated copy of 021, **including its batched snapshot integration**. All numerical,
cache, prefix-capture, head, tensor-loading and Metal kernel sources remain byte-for-byte
021. No edits to 012/019/021/022 or prior results. CPU fixtures are not physical
checkpoint admission or throughput evidence. This implementation performs no GPU,
SSH, deployment or full checkpoint loading; those belong to the parent hardware owner.

## Admission and exact interface

The ordinary CLI remains bounded to 1–3 repeats, 1–96 generated tokens and deadline
≤240 seconds. The new mode is explicit:

```sh
<venv-python> <isolated-local-023>/supervise.py \
  <all existing trusted checkpoint/metadata/sidecar/SSH/bridge options> \
  --benchmark-schedule --benchmark-schedule-sha256 "$SCHEDULE_SHA256" \
  --split 28 36 --execution phased --eval-group-size 32 --prefix-rollback \
  --repeats 14 --max-tokens 256 --deadline 900 \
  --prompt 'Explain briefly how a rainbow forms.' \
  --rank0-verifier native --rank1-verifier slot-qmv
```

The example is an interface, not an executed hardware benchmark. Supply the actual
batch count and largest request cap; these CLI values **must exactly match** the
schedule. No evaluation-group schedule is allowed in this mode. Original trusted
revision is `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; native BF16/Q4,
checkpoint/sidecar hashes, inventory and strict tolerances are unchanged. Verifier
modes remain explicit trusted per-rank options; no fallback is introduced.

`--benchmark-schedule` is boolean on **both supervise.py and probe.py**. It always
reads the fixed, source-registered `benchmark_schedule.json` next to the executing
script. There is no arbitrary schedule-path option. Current checked-in file has
an empty request list and **cannot be admitted for a live run**.

Schema:

```json
{"schema":"qwen38-workload-schedule-v1","requests":[
  {"id":"case-native-r0","case_id":"case","method":"native",
   "prompt":"Exact UTF-8 prompt","max_tokens":128,"warmup":false,"repeat":0},
  {"id":"case-mtp-r0","case_id":"case","method":"mtp",
   "prompt":"Exact UTF-8 prompt","max_tokens":128,"warmup":false,"repeat":0}
]}
```

Requirements: 1–32 ordered requests; unique IDs; exact schema/no ignored fields or
duplicate JSON keys; nonempty bounded strings; integer caps 1–512; boolean warmup;
integer repeat 0–31. Each `(case_id,warmup,repeat)` has exactly one native and one
MTP request, with identical exact prompt/cap for that case. Order is never sorted
or inferred: ABBA and other parent-chosen ordering is followed exactly. Tokenized
prompt plus **full requested cap** must be ≤4096; prefill remains 256. Raw `tokens:`
prompts retain the original probe semantics.

### Freeze and deployment

1. Copy this directory to a NEW immutable per-batch local deployment directory.
2. In that copy only, replace `benchmark_schedule.json` with the frozen parent batch.
3. Hash its **exact file bytes**, not reserialized JSON; pass that SHA on the CLI.
4. Deploy every `remote_actor.SOURCES` entry **plus `supervise.py`** to a fresh remote
   directory; retain the original repository/venv directory layout required by
   `ssh_argv`. In particular deploy the three new `workload_*.py` modules and JSON.
5. Read back the exact registry hashes. Parent, controller, both actors and both
   actual rank process-start receipts must agree. The controller and actors make
   private immutable source snapshots before executing. The controller separately
   snapshots/hashes itself and writes `schedule_admission.json` containing the
   exact schedule and locally trusted tokenizer-derived prompt token IDs.
6. Use a new output directory. Keep an outer timeout long enough for the ≤900s
   internal trial **plus bounded owned-worker cleanup**. Do not modify source while
   a batch is running; do not replace the main working-tree empty template.

All six parent frozen schedules (14 requests each, 84 total) were read and passed
schema/exact-byte admission without modifying them. No live run was performed.

## Runtime and evidence

Every process still runs unchanged small/full width2 and prefix0/1/2 strict gates,
including instrumented-vs-native width2 logits, normalized hidden and all live
hybrid caches at `atol=1e-6, rtol=1e-5`, before any benchmark request. Full gates
use the short CLI `--prompt` seed, not the longest scheduled workload. Gate-only
mode still returns before loading the sidecar or starting requests.

The sidecar is loaded once on rank0 **and remains resident even during native
requests**; this is disclosed in top-level and per-request results. Native requests
on BOTH ranks branch into original collective `phased_generate`, with native
width-one greedy decode and no MTP backend/capture/QMV invocation. MTP requests
use existing prefix rollback. Each request starts fresh target/MTP caches, resets
local MLX peak memory, and synchronizes at the end; prior request backends are
released. A common synchronize/GC step runs after **both** methods, outside the
reported request timing, before the next request peak reset. No parameter or
kernel changes are made.

Every result preserves `token_ids`, `text`, `wall_seconds`, `first_token_seconds`,
`prompt_tokens`, `finish_reason`, `accepted_drafts`, `verification_rounds`, and
`teacher_forwards`; native speculative counters are explicitly zero. Added fields:

- `id`, `case_id`, `method`, `warmup`, `max_tokens`, `prompt_sha256` (exact UTF-8),
  `prompt_token_ids`, `sample_repeat` (schema repeat).
- `repeat` remains zero-based **request ordinal**, never the sample repeat.
- `local_rank`, `local_peak_mlx_bytes`, `local_rank_wall_seconds`: independent local
  receipts, including final synchronization. Rank1's memory/local wall is NOT
  overwritten by the rank0 MTP result transfer.
- `wall_tokens_per_second`, `post_first_tokens_per_second`, `timing_convention`,
  numerical admission and sidecar residency disclosures.

MTP `wall_seconds`/first-token metrics transferred to rank1 are explicitly the
rank0 generation metrics; use rank1's `local_rank_wall_seconds` for its observed
local wall. Native generation metrics are independently measured on each rank.
Post-first TPS is `(emitted_tokens-1)/(wall_seconds-first_token_seconds)` when more
than one token is emitted; otherwise null. **Emitted EOS is included**. Setup,
gates and head loading are excluded from generation wall; MTP finish ACK is included.

`request_before` and `request_done` on both ranks bind the complete ordered spec.
Each flushed `request_done` carries that rank's full `request_result`, retaining
partial evidence in actor stderr if a later request times out. Partial batches are
never marked successful. `workload_evidence.verify_schedule_trial` is dedicated to
this mode: verifies exact prompt token IDs against admitted local tokenization,
all IDs/caps/warmup/repeats/hash, both actual ranks, exact boundaries, native layer
and token completion, MTP commands/reset/finish/prefill256/counters, source registry,
loading inventory, replicated tensor digests, original numerical receipts, small
checkpoint binding, owned cleanup, and pairwise greedy **tokens and stop** equality.
The original `verify_trial` defaults are not loosened; its unchanged inventory and
cleanup checks are reused on individual actual request slices rather than comparing
unrelated workloads to one global token sequence.

Final `result.json` adds `phase_timing_by_rank` and `request_phase_timing_by_rank`.
These summarize existing timestamp pairs (block/head/handoff/etc.) and command
completion-to-completion `inter_action_until_<op>` intervals. They are observed
**local_rank wall including actual synchronization**, nested/overlapping and NOT
additive GPU attribution. Inter-action spans include intervening draft/control
work; they are not isolated kernel timings. No kernel instrumentation is added.

## Explicit evidence limits

023 only: each actor stdout/stderr disk log is capped at **64 MiB**, serialized
retrieval responses and controller response admission at **128 MiB**; actor job
runtime is ≤900s. Exceeding a log cap stops the owned worker, records `log_cap`,
`log_cap_exceeded=true`, `log_evidence_complete=false`, and fails the trial.
Bounded drain overflow similarly reports `drain_incomplete`; oversized retrieval
fails closed with evidence retained on the actor. No successful result silently
truncates evidence. Live transport can drop forwarded bytes only as already
explicitly counted by `forwarded_bytes_dropped`; authoritative actor disk logs
remain the verification source.

## CPU validation

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch .venv/bin/python -c \
 'import mlx.core as mx,pytest; mx.set_default_device(mx.cpu); raise SystemExit(pytest.main(["spikes/023-qwen38-workload-bench","-q","-k","not gpu","--junitxml=spikes/023-qwen38-workload-bench/cpu-tests.xml","--basetemp=/Users/jeremylien/.hermes/cache/scratch/pq023-final-tests"]))'
.venv/bin/ruff check spikes/023-qwen38-workload-bench
```

Tests include genuine tiny packed native hybrid CPU execution, independent fresh
native/MTP requests and original strict gates, adversarial evidence mutations,
real bounded subprocess cleanup/source snapshots, and explicit cap failures.
Distributed receipt envelopes are labeled fixtures, not physical two-rank evidence.
See `TDD.md`, `cpu-tests.xml`, and `cpu-regression.txt` for observed verification.
Final result: **109 passed, 1 GPU test deselected**; Ruff passes. The mixed ABBA
regression also exercises genuine tiny sharded CPU models on two ranks with only
physical send/recv replaced by queues. `changed_source_sha256.json` records the
eight new/modified registered-runtime/controller source hashes; recalculate the
schedule digest after replacing the empty template in an isolated deployment.
