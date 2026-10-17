# M4 width-one teacher verification — diagnostic passes, batching still rejected

## Verdict: PARTIAL

The new opt-in `verification_mode="sequential"` control passed the original
live-KV gate on the authentic mixed-3/4-bit 0.5B checkpoint. The default batch
path reproduced its cache-parity failure. This is numerical diagnosis, not a
speculative acceleration result. No two-rank or 32B speculative run occurred.

## Real checkpoint comparison

Same loaded teacher/draft weights, two prompts, K1/4/8, evaluation group8,
prefill chunks32, fresh caches and max32 EOS-inclusive token events. Only teacher
verification mode changes between each pair. The arithmetic prompt stops after
nine events including EOS; the Moon prompt reaches the 32-token limit.

| Gate | Batched teacher | Width-one teacher |
|---|---:|---:|
| Complete numerical/accounting gate | 0/6 | 6/6 |
| Generated IDs match original sequential decoder | 6/6 | 6/6 |
| Stop reason and every-layer cache offsets match | 6/6 | 6/6 |
| Full live teacher key/value prefixes within tolerance | 0/6 | 6/6 |
| Full live self-draft key/value prefixes within tolerance | 6/6 | 6/6 |
| Request/round/call accounting matches | 6/6 | 6/6 |
| Largest teacher cache absolute difference | 0.125 | 0.0 |

Both paths use unchanged `rtol=0.002`, `atol=0.002`; no failed gate was relaxed.
Every retained layer's live keys and values are compared separately from integer
offsets. Complete checkpoint-file SHA256 inventories match before and after.
Loaded parameters total 260,563,712 bytes; peak MLX allocation was 344,176,992 bytes.
These quantities are not an OS-residency or no-paging attestation.

Changing teacher verification width removes the observed discrepancy in these
six cases while preserving acceptance, trimming, catch-up and EOS code. This
strengthens localization to width-dependent teacher execution, but does not
identify a particular attention or quantized-matmul kernel as the root cause.
Matching these short outputs does not establish quality or future batch equality.

## Actual packed-GPU fixtures and CPU gate

The preserved real tiny Qwen2 GPU fixture was rerun in both modes: 80/80 batch
and 80/80 sequential cases passed. Each mode covers tied/untied embeddings,
evaluation groups1/8, K1/2/4/8, matched and different genuine random draft weights,
first/later EOS and a one-token cap. These are actual 4-bit FP16 computations on
synthetic 3-layer weights, not authentic checkpoint or distributed evidence.
A wrapper additionally verifies receipt mode and individual teacher-forward counts.

Parent CPU gate: 36 passed, one explicit opt-in GPU skip and 2,653 subtests passed.
Ruff check and all six Python-file format checks passed. This gate was observed
before staging; it is separate from the physical GPU runs.

## Why this does not accelerate the target

Sequential verification evaluates every current/proposal input with its own
width-one teacher forward. It removes teacher amortization, and can evaluate
later rejected proposals before trimming. Fewer verification rounds are therefore
not fewer teacher forwards. No latency/throughput comparison or speedup is claimed.
The exact-string opt-in does not change the default `"batch"` behavior, the original
`phased_generate` decoder, or the guarded full32B deployment. The batched numerical
admission gate remains failed.

## Supervision, provenance and remaining blockers

All three owned actors completed and their children/process groups were cleaned
up. Tiny actors had 120-second deadlines; the real-checkpoint actor had a
180-second deadline. Device/cache settings were restored, and the allocator-cache
limit was 256 MiB. GPU-driver state was not inspected.

Sources were exclusively staged on the M4 and read back byte-exactly before and
after execution. `speculative_decode.py` SHA256:
`25dd62cb80b4d9bcabe78d9a396615f8a658936799908cc0b10b153f2b65b856`.
The guarded full32B source inventory was read back unchanged after each actor.
Earlier failed fixtures/checkpoint evidence was left intact in the older archive.
The earlier independent component review applies to its batch baseline only;
no independent post-extension review or distributed deployment approval is claimed.

Thunderbolt SSH succeeded after rediscovering the mini's changed bridge-scoped
address and verifying its previously pinned host key. The last post-run preflight
showed M4 pressure level1, 229.06 MiB swap and 90 GiB free disk. The MacBook remained
at pressure level2 with 12,630.50 MiB swap and only 753 MiB free disk. Full32B work
remains paused; unrelated user workloads were not terminated and no model was
downloaded or weight binary archived.

`evidence/` preserves exact reports, stdout/stderr, actor/process receipts,
deployment and readback attestations. `source/` preserves the executed modules
and diagnostic drivers. `verified-summary.json` separates passing width-one
controls from failed batching. `integrity.json` hashes the archive contents.
