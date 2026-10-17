# 025 — bounded receiver-credit overlapped MTP prefill

**New CPU-tested experiment; no physical GPU gate, deployment or speed claim.**
This is an isolated fork of 024. `README-024-COPIED.md`, `TDD.md`, the copied
source/hash manifests and copied CPU receipts are **024 history**, not evidence
for 025. Original 024 sources/results and the benchmark sibling are untouched.

## Runtime contract

- Default serial path is retained. `--overlapped-prefill` enables fresh live
  two-mode admission; it does not force every request into overlap.
- New strict schedule schema: `qwen38-prefill-overlap-schedule-v1`; every row has
  `method="mtp"`, `prefill_mode="serial"|"overlap"`, and `source_request_id` for
  the original ordering slot. Other original row fields are unchanged. The old
  native/MTP schema remains supported. New-schema schedules without the flag
  reject before model/transport execution. `benchmark_schedule.json` stays the
  copied empty template; the parent/sibling must freeze exact schedules separately.
- `prefill_pipeline` contains full prompt IDs (including final-token metadata),
  cap 2048. Every other opcode retains cap 256; control frames remain 65536 bytes.
  Both ranks derive `range(0,len(prompt)-1,256)`. Arithmetic stays native.
- Rank0 sends one exact nonce/command/chunk/start/end credit **after evaluated
  hidden receive and before local graph construction**. Rank1 evaluates hidden
  send, then validates credit before the next embedding. This bounds two logical
  chunks / one pending handoff under eager and zero-capacity rendezvous wires.
- One compound ACK is drained only after rank0 finishes every target/head prefix
  update. Final prompt token uses ordinary `forward` and its ordinary ACK,
  outside the compound. One-token requests bypass the compound entirely.
- No runtime producer threads, extra streams, sockets or async transport. A raw
  stream failure is fatal; no JSON error ACK or reset is inserted into tensors.

## Admission and evidence

`overlap_gate.pipeline_gate` is wired after existing width/prefix/hidden gates:
physical two-rank small lengths 1/2/255/256/257/511/512/513, then same-loaded full
513 at chunk256. It executes the actual backend/Worker/Channel paths with fresh
serial and overlap caches; compares all normalized hidden positions, every native
cache part, shifted head KV, final logits/argmax and two forced continuations.
Unchanged atol1e-6/rtol1e-5. The live capability is model/group-bound in a private
process record; detached JSON and the symbolic CPU test Admission cannot mint it.
**The GPU path is implemented/wired but has NOT been executed in this task.**

`INTERFACE-OVERLAP.md` specifies flags, receipt fields and source-bound causal
stream grammar for the independent collector. The stream auditor checks performed
receive/credit ordering plus exact per-chunk layer/head coverage, not just counts
or digest syntax. It rejects resealed equal-count causal forgeries. This proves
recorded boundaries under trusted executed source, not simultaneous Metal kernels
or cryptographic proof against compromised in-process Python.

## Both-host sampled pressure watchdog

The legacy `--require-normal-pressure` guard remains strict level1 and uses its
unchanged `qwen38-owned-pressure-v2` receipt schema. Policy defaults to `normal`;
historical default monitoring opt-out is retained outside overlap. Overlap always
requires an effective watchdog on rank0, rank1 AND the remote weight-server actor.

Explicit user-authorized `--pressure-policy noncritical` is a separate opt-in,
never a new default. **Omit `--require-normal-pressure`** in this mode: selecting
both is rejected. Noncritical forces monitoring even with the legacy flag false,
permits exact sampled integer levels1/2, and aborts on critical4, invalid/missing
or timed-out samples. It emits `qwen38-owned-pressure-v3`, explicit
`policy="noncritical"`, `allowed_levels=[1,2]`, and actual observations (never
normalized to1). Actor/controller startup and results, probe process-start,
verification and live-admission records disclose policy and the truthful false
legacy flag. External verification selects `expected_policy`; receipts cannot
select their own acceptance criterion. V3 cannot pass the default normal checker.

Both modes preserve0.5s sampling,0.25s sampler timeout,2048 sample cap,1s maximum
gap, exact job/nonce/PID/role binding, actual execution coverage, all stop/deadline
checks and ownership-safe cleanup. This is sampled evidence, not crash prevention
or continuous pressure certification. No global memory/wired policies, numerical
tolerances or unrelated applications are changed by this policy implementation.

`TDD-NONCRITICAL.md` and the NEW parent result
`results/qwen38-prefill-overlap-v1/noncritical-runtime-implementation.json` record
CPU-only implementation evidence. Archived trials and strict-normal criteria are
unchanged; neither warning-tolerant success nor hardware admission is claimed.
Fresh independent source review and immutable deployment/readback are required
before the parent hardware owner may execute the new sources.

## CPU verification

Use the literal project interpreter and import guard:

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch \
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python \
  cpu_only_runner.py --all-cpu -q --junitxml=cpu-overlap-tests.xml
```

The runner forbids MLX/model imports, network and signals. Old MLX-dependent
modules/functions and GPU/process-signalling fixtures are explicitly excluded or
skipped, never represented as passed. Archived original sources remain byte-exact;
one live v2 test double now explicitly asserts the default expected policy.
`TDD-OVERLAP.md`, `overlap-command-history.jsonl` and the new parent result
`cpu-runtime-implementation.json` record actual ordered RED/GREEN commands and
final permitted-suite counts. Symbolic stages exercise exact backend class
bodies and real Command/Worker/Channel semantics; they are not numerical proofs.

`supervise.py --dry-run` validates options and source hashes without creating
outputs, loading models or touching SSH. Guarded `--help`/dry-run were exercised.

## Still required from parent/independent reviewers

Independent CPU/source review; immutable source/schedule deployment/readback;
healthy both-host admission; actual physical small/full/native256 gates; exact
metadata/output/cleanup assurance; bounded matched pilot and actual log-budget
check (`overlap_evidence.pilot_budget_check`) before larger schedules; full frozen
18-case acceptance. Caps remain64MiB per actor log and128MiB retrieval. Nothing
here establishes the requested30% TTFT /15% full-request reductions.
