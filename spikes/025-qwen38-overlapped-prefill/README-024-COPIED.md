# 024 — opt-in hidden-only MTP prefill

## Parent physical validation update

The implementation-only status below is historical. Parent physical testing is
now recorded in `results/qwen38-broad-v1/prefill-experiment/`: both-host strict
small/full/native256 gates passed, and all42reference/candidate/reference requests
matched corresponding native/reference token IDs and stop reasons. Complete-request
throughput improved5.57% on long code and3.01% on long explanation versus the
same-source after control; short requests were effectively flat. This is retained
as an opt-in small-screen optimization, not a universal/default speed claim or a
new18-case broad result. See that directory's README and verification receipts.

## Original CPU implementation record (historical)

**CPU-validated candidate, not a measured acceleration.** Isolated copy of
`023-qwen38-workload-bench`; only this new directory was modified. No GPU, SSH,
deployment, checkpoint loading, running batch polling, original result edits or
commit was performed here. Physical small/full admission and paired timing belong
to the parent hardware owner.

## One optimization, default reference unchanged

`--hidden-only-prefill` is explicit on both `supervise.py` and `probe.py`, default
**off**. When enabled, nonfinal MTP prompt chunks use the new bounded
`prefill_hidden` command. It executes the same native target layers and evaluates
the same hybrid cache states, then executes the existing full-position native
`core.norm(h)` and evaluates it, returning `(None, normalized_h)` on rank0. It
never invokes the unused target vocabulary projection. Rank1 still sends its
native target activations and returns `None`; it owns no MTP head or output
projection. Every normalized position, not just the last or greedy position,
feeds the unchanged shifted-token MTP-head recurrence.

The final prompt token still uses ordinary `forward`, producing full logits and
normalized hidden. A one-token prompt emits no empty prefill command. Decode,
width2 verification, numerical modes, prefix0/1/2 rollback, head KV catch-up,
snapshot arithmetic and all Metal sources remain unchanged. The default MTP
control still projects nonfinal chunks, exactly as 023. Native scheduled requests
still enter the original collective `phased_generate` path on both ranks.
**No prefill overlap, transport, ownership, fusion, custom numerics or grouping
optimization is included.**

## Mandatory live numerical admission

An enabled backend requires a live `Admission` bound to that exact loaded model,
verifier mode and enabled option. Previous receipts cannot authorize a process.
Before requests, **both hosts** must pass the original small/full width2 and
prefix gates, plus new same-process hidden-prefill equivalence gates at unchanged
`atol=1e-6, rtol=1e-5`:

- Small loaded checkpoint on each host: prompt lengths **256, 257, 511, 512, 513**,
  native chunk width256, all positions normalized. Its standalone head is an
  explicitly deterministic, quantized native-head fixture matching that small
  architecture, not a claim that a target-sized sidecar fits the small model.
- Same-loaded full target shards: two forced chunks of
  `--hidden-gate-chunk-size N`, default256, bounded integer1..256, followed by the
  final prompt token and two native continuation tokens. Prompt length is `2*N+1`.
- Reducing N **does not bypass native width256**: it additionally requires a
  257-token same-loaded full-shard case with one nonfinal256-token chunk, recorded
  under `representative_256`. Default N=256 already covers two native256 chunks.
  This option changes proof workload only; benchmark prefill stays256.

Each case executes three fresh target cache trees: unchanged reference MTP
prefill (all-vocabulary output), optimized hidden-only prefill, and native
cache-only prefill. Every chunk compares **all normalized hidden positions**,
every live recurrent/KV cache against reference and native, and shifted-pair
head KV keys/values/offset. Final-token and continuation checks compare complete
vocabulary logits, normalized hidden, argmax, target caches, head logits and head
cache continuity. The actual pinned sidecar is loaded once on rank0 before the
full hidden gate, including in enabled **gate-only mode**. It is never loaded on
rank1. Bounded peer summaries exchange admission, not large per-layer trees;
both independent local target-cache attestations must pass.

Required receipts: top-level rank and `verification` fields
`hidden_only_prefill`, `hidden_gate_chunk_size`; each evaluation group's
`small.hidden_only_prefill`, `full.hidden_only_prefill`, and peer summaries;
source-bound `process_start`; and pre-request `numerical_admission_done` with
matching options. Hidden gates use schema `hidden-only-prefill-v1`. The verifier
requires complete shapes/chunks/offsets/native/head/continuation evidence, not a
passed bit. It replays bounded request/sequence/finish state, binds exact prompt
chunk256 operations, and checks rank0 `hidden_norm_done` rather than `head_done`
for each nonfinal hidden command. Final and decode forwards still require
`head_done`; rank1 must have neither output phase.

### Exact cache-part admission boundary (review fix)

Layer coverage alone is insufficient. Every applicable chunk, final target
token and continuation now requires the **exact unique `(layer, part)` inventory**
in both `caches` and `native_caches`: native `ArraysCache(size=2)` contributes
`state.0.0` (convolution) and `state.0.1` (recurrent); native `KVCache` contributes
`keys`, `values`, **and `offset`**. Head receipts require all three KV parts.
Missing, duplicate, replacement-duplicate and unknown parts fail closed.

Required shapes come from native cache classes and live architecture dimensions,
not the submitted metric rows: convolution `[1,K-1,2*Hk*Dk+Hv*Dv]`, recurrent
`[1,Hv,Dv,Dk]`, KV `[1,Hkv,known_prefix_length,Dhead]`. Recurrent state is float32;
activation/head dtypes come from the warmed native reference. Metrics now record
actual dtype, finiteness of **both** arrays, maximum absolute error, maximum
elementwise tolerance and maximum error/tolerance ratio. Admission independently
checks their consistency and requires ratio <=1, with unchanged
`atol=1e-6, rtol=1e-5`; offsets must be exact integral prefix positions with zero
error. A forged `strict_pass=true` cannot override an invalid metric.

Live admission obtains its expected layout from a process-local producer record,
bound to the exact full model object and ownership; reduced-width native256
receipts must also belong to that model. Only metadata/receipts are retained,
not model weights or native cache arrays. Detached JSON cannot mint this live
capability. The external controller separately derives layouts from independently
hash-verified small/target checkpoint metadata and the pinned native cache-class
schedule; it never infers required inventory from the evidence itself. CPU tests
exercise float32 and bfloat16 target/head combinations, including the small
float32 fixture head and float32 recurrent state.

This validates complete, source/process-bound metric **receipts**, not raw-array
recomputation or cryptographic proof of execution. Arbitrary in-process Python
compromise is outside that boundary. It does not claim physical full-checkpoint
correctness; both hosts' small/full/native256 gates remain mandatory before work.

## Exact parent-owned interface

Use fresh immutable experiment copies, not this working template. The registered
`benchmark_schedule.json` remains empty and cannot admit a scheduled live run.
The following is an interface example, **not an executed hardware command**.
All uppercase shell variables must refer to the parent's independently trusted
current paths, hashes and bridge addresses:

```sh
PY=/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python
LOCAL_024=/absolute/frozen/local/spikes/024-experiment

common=(
  --remote-root "$REMOTE_024"
  --remote-source "$REMOTE_CHECKPOINT"
  --local-metadata "$LOCAL_METADATA"
  --output "$NEW_OUTPUT"
  --ssh-key /Users/jeremylien/.ssh/paretoquant_mini_ed25519
  --ssh-target "$SSH_TARGET"
  --ssh-host-key-alias 'fe80::56:c507:c02a:a630%bridge0'
  --local-address "$LOCAL_BRIDGE_ADDRESS"
  --remote-address "$REMOTE_BRIDGE_ADDRESS"
  --split 28 36 --execution phased --eval-group-size 32 --prefix-rollback
  --rank0-verifier native --rank1-verifier slot-qmv
  --target-revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818
  --target-config-sha256 "$TARGET_CONFIG_SHA256"
  --target-index-sha256 "$TARGET_INDEX_SHA256"
  --sidecar "$LOCAL_SIDECAR"
  --small-model-local "$LOCAL_SMALL_CHECKPOINT"
  --small-model-remote "$REMOTE_SMALL_CHECKPOINT"
  --small-config-sha256 "$SMALL_CONFIG_SHA256"
  --small-index-sha256 "$SMALL_INDEX_SHA256"
  --prompt 'tokens:[1,2,3,4,5]'
)

# Mandatory enabled admission-only run; no generated workload requests.
"$PY" "$LOCAL_024/supervise.py" "${common[@]}" \
  --hidden-only-prefill --hidden-gate-chunk-size 256 \
  --verification-gate-only --max-tokens 5 --repeats 1 --deadline 900

# In a NEW experiment/output, with the parent's frozen schedule copied into
# benchmark_schedule.json BEFORE freezing/hashing/deployment:
"$PY" "$LOCAL_024/supervise.py" "${common[@]}" \
  --hidden-only-prefill --hidden-gate-chunk-size 256 \
  --benchmark-schedule --benchmark-schedule-sha256 "$SCHEDULE_SHA256" \
  --max-tokens "$SCHEDULE_MAX_CAP" --repeats "$SCHEDULE_REQUEST_COUNT" \
  --deadline 900
```

For the contemporaneous **024 reference control**, omit `--hidden-only-prefill`
(and the gate chunk option), retain the identical frozen workload, source set,
placement, groups, explicit per-rank modes, prompts/caps and fresh-cache policy.
Also retain the immutable023 control and exact per-case output equality against
it and native; no physical comparison has been performed by this implementation.
Ordinary default runs retain the inherited240s controller cap; enabled admission
or scheduled runs may use up to900s. Allow the existing bounded cleanup outside
that internal deadline. No gate wall/runtime budget is claimed from CPU timings.

Deploy **every fixed `remote_actor.SOURCES` entry**, now including all `.py`,
`.json` and `.metal` files in this directory, including the supervisor and tests.
Read back hashes on both hosts and verify actor/source snapshots and exact
schedule bytes before launch. `source-sha256.txt` records the current registry;
`changed-source-sha256.txt` records differences against023 with before/after
hashes. Manifests use `.txt` to avoid impossible self-referential source hashes.
Recompute them after inserting a schedule or changing any registered source.

## Evidence hardening and remaining integration

024 checks each scheduled request's `verifier_mode` against the trusted native
or per-rank MTP mode, and requires `wall_tokens_per_second == emitted/wall_seconds`
in addition to the existing post-first timing check.

The parent's actual independent `benchmarks/qwen38/mtp_audit.py` and its74 tests
are now copied into024 and registered. Formatting changes only: both copied
files' Python ASTs are verified identical to the parent. The real protocol
spelling remains `prefill_hidden`. `workload_evidence.verify_schedule_trial`
mandatorily calls `validate_mtp_actions` on **each scheduled MTP request on both
ranks**, with explicit trusted mode, target EOS IDs and hidden-prefill option.
The audit binds reset/chunk256/final/snapshot/verify/current emitted token,
prefix commit/terminal/EOS grammar, exact accepted/round/teacher counters and
finish to recorded output progress. Three integration regressions reproduce
the independent review's zero-round/five-output, changed-acceptance-counter and
wrong-current-token mutations while keeping both owners/results consistent.
All were observed to pass the old verifier and are now rejected. Existing
per-request mode and wall-TPS checks remain intact.

The auditor does **not** prove teacher greedy predictions: those require logits
or equivalent numerical evidence not present in action receipts. It supplements,
never replaces, numerical/loading/inventory/cleanup gates. No original023 source
or raw results were edited, and no archived physical batch was re-audited here.

## Verified CPU proof and limits

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch .venv/bin/python -c \
 'import mlx.core as mx; mx.set_default_device(mx.cpu); import pytest; raise SystemExit(pytest.main(["spikes/024-qwen38-hidden-prefill","-q","-k","not gpu","-p","no:cacheprovider","--junitxml=spikes/024-qwen38-hidden-prefill/cpu-cache-tests.xml","--basetemp=/Users/jeremylien/.hermes/cache/scratch/pq024-final-cache-cpu"]))'
.venv/bin/ruff check spikes/024-qwen38-hidden-prefill
.venv/bin/ruff format --check spikes/024-qwen38-hidden-prefill
```

Observed after the review fixes: **331 passed, 1 skipped, 1 GPU test deselected
in22.70s**; Ruff check and formatting pass (58 files reported by Ruff). The skip is the
copied auditor's explicitly opt-in archived-batch test; no physical audit ran.
The original130 CPU tests were also isolated and rerun: **130 passed,
4 deselected in18.85s** (the GPU test plus3 new review regressions excluded).
CPU is selected before pytest collection and in every new MLX fixture;
queue workers select their own CPU streams. No GPU test was executed.
`cpu-cache-tests.xml`, `cpu-cache-regression.txt`, `cpu-original-regression.txt`,
`TDD.md`, and `cpu-cache-inventory-proof.txt` retain current actual proof.
The earlier `cpu-tests.xml`, `cpu-regression.txt`, and
`cpu-hidden-prefill-proof.txt` are preserved as the original130-test snapshot,
not current-source authorization. Tests include all
normalized positions (and deliberate early-position corruption), zero nonfinal
target projection calls with final projection preserved, all boundary lengths,
one-token prompts, head corruption/offset continuity, admission refusal,
command bounds/sequences/finish, option/source receipts, and actual two-rank
queue-backed native/reference/optimized schedules at full gate widths4 and256.
Controller/watchdog envelopes in unit tests are explicitly fixtures, not physical
loading or cleanup receipts.

Unchanged-byte checks against023 include `mtp_head.py`, `speculative.py`,
`verifier.py`, `prefix_rollback.py`, `cache_state.py`, `slot_qmv.py` and both Metal
files. Existing token/inventory/loading/width/prefix tests remain included.

## Verdict: PARTIAL

CPU behavior and opt-in control/admission paths are exercised. Full-checkpoint
physical correctness, memory headroom, bounded gate duration, old/native output
parity and end-to-end speed remain **unmeasured**. The motivating diagnosis only
attributes roughly one second of long-prefill waste to the unused projection;
much of the larger gap is serialized command/prefill scheduling, deliberately
untouched here. No large speedup or deployment readiness is claimed.
