# M4 speculative-loop integration — partial, not acceleration-admitted

## Verdict

The opt-in speculative component runs on actual GPU models. The original
single-token decoder and guarded full32B deployment remain unchanged.

- 80 tiny random-weight packed-FP16 GPU cases passed token IDs, stop reason,
  every-layer cache offsets, live teacher KV and accounting checks.
- Six real mixed-3/4-bit 0.5B self-draft cases matched the sequential teacher's
  token IDs, stop reason and offsets, but **all six failed the predeclared live
  KV tolerance gate** (`rtol=0.002`, `atol=0.002`). The gate was not loosened.
- No actual 32B teacher run, two-physical-rank speculative deployment, quality
  evaluation, or end-to-end speedup is established by this archive.

## Component contracts

`speculative_decode.py` is separate from `phased_generate`. It validates identical
shared tokenizer ID semantics, requires an explicitly configured draft tokenizer
on both ranks, runs the draft only on rank1/M4 for two-rank use, and keeps CPU
proposal/teacher packets outside lazy local GPU work. Packets contain strict
round/offset/count metadata, fixed K+4 int32 storage and -1 padding; padding is
never fed to a model.

At each round both caches hold prefix S and the next input token is pending.
The teacher verifies current+proposals with all-position logits. Matching proposals
are accepted only through the first mismatch or confirmed EOS, with an unmasked
teacher correction/bonus. Both teacher and draft cache offsets are restored to
S+emitted count, including all-accepted draft catch-up and final zero-proposal
rounds. Source tests cover stale-tail overwrite and every retained layer.

The acceptance algorithm agrees with the teacher's **batched verification**
argmax. That does not prove equality with the teacher's sequential arithmetic at
other shapes or checkpoint weights. The component is not deployed in the guarded
32B runner; CLI/admission/source-inventory integration remains separate.

## Actual tiny GPU gate

The standalone parent-owned fixture ran 80 cases:

  tied/untied embeddings × eval groups1/8 × K1/2/4/8 × five scenarios

Scenarios: identical draft weights, different genuine randomly initialized draft
weights, confirmed first-token EOS, a later EOS ID, and a one-token generation cap.
All 16 different-draft cases exercised rejection. These are real 3-layer hidden64,
vocab128, group32, 4-bit FP16 Qwen2 computations; fixture token names and weights
are synthetic and are **not checkpoint inference or quality evidence**.

The final test passed with a peak MLX allocation of 1,254,044 bytes. Its owned
actor completed in 2.338 seconds with process-group cleanup verified. This is
an execution bound, not an inference throughput measurement.

The first fixture attempt failed because the test treated the integer offset in
MLX KVCache.state as an array. The pinned implementation returns keys, values,
offset. The fixture now compares the two live tensor prefixes and separately checks
all offsets. The failed actor receipt is preserved in `failed-fixture-v1/`;
no decoder source change or tolerance change was needed for that fixture repair.

## Real checkpoint gate and numerical difference

On the existing authentic 260,563,712-byte mixed-3/4-bit 0.5B checkpoint, two prompts
were tested at K1/4/8 with fresh caches, unchanged weights, group8, max32 tokens.
The same loaded model served as teacher and draft. This self-draft experiment
validates integration, not the proposed small-draft/32B-teacher acceptance or speed.

All six outputs and stop reasons matched the original sequential teacher. The
arithmetic prompt stopped after nine EOS-inclusive tokens. The other prompt
reached its 32-token cap; output content is not a quality claim. Loaded model
config/source inventories were bound and unchanged before/after execution.

Nevertheless, all six cases failed strict live-KV numerical parity. A separate
K4 arithmetic-prompt diagnostic found:

- Prefilled prompt KV prefixes were exact (maximum absolute error zero).
- The serial native draft KV passed the same tolerance at all 24 layers.
- The batched teacher KV exceeded tolerance in layers1–23; layer0 passed.
- Maximum observed cache-element absolute difference was 0.125.
- Teacher/draft/control offsets and generated token IDs still agreed.

This localizes the discrepancy to the batched teacher decode path, not the draft's
serial cache or initial prefill. Width-dependent attention/math is a hypothesis,
not a proved cause. It does not prove a rollback defect, output quality loss, or
future greedy equality. Preserve the failed gate until the numerical mechanism
and intended equivalence contract are resolved. Do not silently relax tolerance.

## Gates, supervision and evidence

Parent CPU tests: 269 passed, two explicit optional skips, 1,383 subtests passed;
full Ruff check and format passed for spikes007/009. One skip is CPU F16 admission,
the other is this opt-in GPU fixture. CPU doubles are not model/timing evidence.
The component's independent CPU review passed: 16 unittest tests, 1,331 subtests,
400 prefix-sensitive cases, 48 blocking two-rank cases, five admission checks and
one catch-up timing check. Frozen, live and archived source hashes match. Exact
verdict/probes are in `review/`. This is control-flow/accounting approval only,
not GPU numerical equivalence, performance or deployment approval. The real
checkpoint's failed cache gate remains unchanged.

Every real GPU run used explicit GPU selection, a 256 MiB allocator-cache limit,
a reset peak counter, bounded owned actors and verified process/group cleanup.
Device/cache settings were restored. Peak allocation is not residency or proof
of no paging; GPU-driver state was not inspected.

Full32B remains paused: the latest parent preflight recorded elevated M2 memory
pressure, 8,817 MiB swap use and about 4.5 GiB free boot-volume disk. Other user
workloads were not terminated. No source was pushed or full32B weights changed.

`tiny-gpu/`, `real-checkpoint/` and `cache-diagnostic/` preserve exact JSON outputs,
stdout/stderr and actor receipts. `source/` and `admission/` preserve executed
sources, harnesses and read-back deployment identities. `verified-summary.json`
separates passing token agreement from failed cache parity; `integrity.json`
hashes the raw evidence. No weight binaries are archived.
