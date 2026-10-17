# Qwen3.8 prefill-overlap benchmark harness

**025 integration v2 is implemented separately in `integration025.py`,
`numerical025.py`, and `supervisor025.py`; see `INTEGRATION025.md` for actual raw
producer fields, inert freeze/run-plan/collector CLIs, and the parent-only execute
boundary. No CPU implementation receipt authorizes hardware or claims speedup.**

025 `freeze025` defaults to the unchanged normal-pressure contract. The explicit
`--pressure-policy noncritical` requires `--memory-acceptance PATH` and its external
`--memory-acceptance-sha256 SHA`; the freeze binds the amendment's exact bytes and
original acceptance SHA. Only this amended contract admits sampled warning2;
critical4, unknown/missing samples, coverage gaps and incomplete cleanup still
fail. Reports label noncritical policy and actual warning observations, never a
normal-pressure qualification. Fresh combined parent review remains required.

## Preserved v1 interface documentation

The following sections describe the unchanged v1 proposed-normalization interface;
its integration-pending verdicts still apply to **v1** `collect`/`audit`, not the
new lossless `collect025` adapter.

**CPU-only implementation. No model results, live admission, GPU approval, deployment,
SSH, or physical overlap claim. Independent review and runtime025 integration are
pending.** Nothing in this directory authorizes a hardware run.

## Safe entry points

From the repository root, using the existing project Python:

```sh
.venv/bin/python -m benchmarks.qwen38_overlap.cli plan
.venv/bin/python -m benchmarks.qwen38_overlap.cli plan --out-dir NEW_FREEZE_DIRECTORY
.venv/bin/python -m pytest benchmarks/qwen38_overlap/tests -q \
  --import-mode=importlib -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache benchmarks/qwen38_overlap
```

`plan` without `--out-dir` reads inputs and prints a deterministic summary; it does
not write. Explicit freeze/report directories must be new, exclusive and free of
symlink ancestors. No command has an executable/SSH/profile-argv option. `--execute`
exists only on `plan`: without a parent-injected guarded supervisor it fails with
`parent_only_guarded_supervisor_unavailable`. The injected callback receives the
non-approved plan and mandatory flags, never an automatic subprocess fallback.
The callback must enforce admission; its CPU test is only a recording function.

### Collect / audit

```sh
.venv/bin/python -m benchmarks.qwen38_overlap.cli collect \
  --schedule FREEZE/schedule.json --schedule-sha256 EXTERNAL_SCHEDULE_SHA256 \
  --trusted TRUSTED.json --trusted-sha256 EXTERNAL_TRUSTED_MANIFEST_SHA256 \
  --record TRIAL/normalized-record.json --record-sha256 EXTERNAL_RECORD_SHA256 \
  --out-dir NEW_REPORT_DIRECTORY
```

`audit` accepts the same arguments and repeats the same source, schedule, receipt,
action, output, numeric, memory, pressure and cleanup checks. Repeat `--record` and
`--record-sha256` in matching order for up to six batches. The trusted suite form
uses schema `qwen38-overlap-trusted-suite-v1`, top-level `schedule_sha256`, and a
`batches` mapping of decimal batch indexes to independently frozen trusted inputs.
A single trusted input works for one batch. Both forms need an independently
supplied expected manifest/hash, not a registry chosen by a receipt. Raw records
are bounded to 32MiB each; trust to 32MiB, schedule to 8MiB, six records maximum.
Raw receipt readback remains bounded to 128MiB per file; source files to 64MiB.
Duplicate JSON keys, NaN/Inf/overflow JSON floats, symlinks and nonregular inputs
are rejected. Artifacts and sources are relative, contained snapshot paths.

**Current collect/audit status is never physical acceptance:** a valid normalized
contract returns `contract_valid_integration_pending`, `accepted: false`, and
exit 2. A failed or partial collection retains attempted public request rows and
missing IDs; it cannot become a complete-suite pass. The CLI publishes hash
bindings, not private profiles, account/SSH arguments or arbitrary raw receipt
bodies. Original inputs are read-only and are never repaired or overwritten.

### Analyze

```sh
.venv/bin/python -m benchmarks.qwen38_overlap.cli analyze \
  --schedule FREEZE/schedule.json --schedule-sha256 EXTERNAL_SCHEDULE_SHA256 \
  --input COLLECTED_ROWS.json --input-sha256 EXTERNAL_INPUT_SHA256 \
  --out-dir NEW_DESCRIPTIVE_REPORT_DIRECTORY
```

Input is an object with `rows` or `attempted_rows` and an explicit artifact kind.
The analyzer deliberately does not trust any input's claimed evidence pass. Its
`evidence_validation` is `not_evaluated`, and `accepted` is always false. Exit 0
means a complete descriptive report, **not** physical acceptance or meeting a
performance goal. Missing/invalid timing rows produce exit 2. JSON, formula-safe
CSV, Markdown, and an artifact hash manifest are emitted only with explicit
`--out-dir`. Every original case remains in the case table; invalid/missing pairs
have null estimates, not imputed values. Raw failed input JSON remains on disk.

## Frozen workload and estimator

- Historical manifest SHA256:
  `b20a600c3889d62cdf15a7faa6c2ca90611084ceae15b9a8dea7f7823d5a29e6`.
- Benchmark design SHA256:
  `710703230b23461193337e78c8a38e1c72675e19f6d747c614c14c7211ac95f0`.
- `qwen38-prefill-overlap-schedule-v1` keeps every original v1 row field. Both
  actual `method` fields are `mtp`. Original native ORDER slots map to `serial`;
  original MTP slots map to `overlap`. `source_request_id` preserves the original
  slot ID; a derived ID is distinct and unique. An ID containing historical
  `native` text is an order-slot reference, never a native execution claim.
- Six same-loaded batches of 14 requests: 84 total, 72 measured, 12 queue warmups.
  Exact original ABBA/BAAB order and repeats 0,0,1,1 remain. Prompts and caps are
  not retokenized, rewritten, padded, shortened, or relabeled. `structured-1536`
  remains that ID with **1535 actual formatted prompt tokens**.
- Separate six-request pilot: original queue warmups S/O (cap32), code-1536 S/O
  repeat0, structured-1536 O/S repeat0. Pilot rows never replace full-suite rows.
- Both modes use hidden-only MTP + prefix rollback. Fixed revision
  `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, 28/36 placement, group32, native /
  slot-qmv verifier modes, unchanged Q4/BF16 arithmetic and chunk256.
- Loading and live gates precede request timing. Two short warmups per loaded
  batch are excluded from latency estimates but required for exact coverage.
  There is no first-long-shape exclusion. TTFT and request wall are rank0 inclusive
  metrics; phase times are non-additive. Never subtract clocks across hosts.
- EOS IDs248044/248046 are output events. No EOS may precede the last output ID.
  `stop` requires final EOS; `length` requires no EOS and the exact original cap.
  A complete length-capped request is not a naturally finished answer.
- Pair each metric by **case and repeat ID**; use the arithmetic median of the
  two candidate/reference ratios, not the ratio of medians. Reduction is
  `1 - median_ratio`, not throughput gain. Both ratios and their range are kept.
- Each of **all six long IDs** must independently have TTFT ratio <=0.70 and wall
  ratio <=0.85, unrounded. Equality passes. No pooled override. All 18 case rows
  and every observed per-repeat regression are reported. Missing, duplicate,
  extra, failed or invalid rows prevent complete numerical coverage. Numerical
  timing validity is separate from evidence validity and physical admission.
- Two repeats are descriptive, not confidence intervals or tail/general-workload
  guarantees. Neither sample fixtures nor favorable timing rows mint provenance.

## Evidence API and honest integration boundary

See `CONTRACT.md`. The current `protocol.py` event names are an explicitly
**proposed normalized CPU contract**, not an assertion that sibling025 already
emits those names. The sibling source was still mutable and its new compound /
credit / pressure receipt fields were not documented at implementation time.
Do not feed its raw receipts through an invented field renamer or claim this
implements its raw adapter. Existing024 decode grammar was reused faithfully in
`audit_decode`; serial ACK drains before shifted-head update, whereas overlap
uses credits before local target work and one final compound ACK after the prefix.

Before any actual collection/acceptance the parent must:

1. Obtain sibling025's documented fields, implement a source-bound lossless raw
   adapter, bind actual actor start/request/cache epochs, final ACK/drain and
   sampled pressure intervals, and test genuine positive receipts plus mutations.
2. Derive every numerical expected inventory from independently pinned small/full
   checkpoint architecture, including native/serial/overlap target caches, GDN
   convolution/recurrent arrays, KV keys/values/offsets, normalized hidden, full
   final/continuation logits, shifted head caches/draft logits and continuation.
   CPU fixture layouts and passed bits are not a substitute.
3. Freeze the complete actual transitive runtime/helper/config/Metal registry,
   supervisor, adapter, collector, reducer and per-batch schedule hashes externally.
   The hardcoded legacy minimum registry is a floor, **not** the new complete set.
4. Bind exact original native output IDs/stops and independently admitted prompt
   IDs to the expected request set; agreement between current S/O alone is not
   enough. This CPU harness does not retokenize or execute teacher logits.
5. Independently review CPU artifacts, then conduct parent-owned healthy-host live
   small/full/513-at256 gates, bounded pilot, full84 collection and cleanup. No
   continuous-pressure or simultaneous-Metal-kernel claim follows from logs.

No installation, network, MLX import, model loading, SSH, signalling, commit,
source deployment or existing source/benchmark/result edit was performed here.
`cpu-harness-implementation.json` and `tdd-log.json` record actual commands and
limitations, not expected/fabricated test output. Sample reports are TEST fixtures.
