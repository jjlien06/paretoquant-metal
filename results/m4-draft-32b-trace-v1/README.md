# M4 small-draft / archived 32B trajectory diagnostic

## Verdict: promising prerequisite, not accelerated 32B inference

A genuine GPU run of the existing ParetoQuant mixed-3/4-bit Qwen2.5-0.5B model on
Apple M4 proposed tokens along the archived original 32B response. **The 32B target
was not loaded or executed.** The target side was an immutable archived 41-token
trajectory, including EOS, with its original formatted prompt and pinned tokenizer.

| Proposed block K | Hypothetical verification rounds | Matching prefix tokens / proposed tokens | Total draft-only time, s |
|---|---:|---:|---:|
| 1 | 23 | 19 / 23 | 0.704 |
| 2 | 18 | 24 / 34 | 0.565 |
| 4 | 13 | 29 / 45 | 0.502 |
| 8 | 12 | 30 / 75 | 0.611 |

Each round teacher-forces the already emitted archived prefix, recomputes a fresh
draft KV cache, and checks only the matching proposal prefix against archived IDs.
A mismatch advances through the archived replacement token, not rejected draft IDs.
Matching EOS emits no hypothetical bonus after EOS. Every proposal/window/count
is retained. All four walks consume exactly the same 41 archived token events.

K=4 used 13 hypothetical verification rounds with 29 accepted proposals, versus
41 one-token target events in the original receipt. This **is not a 3x speedup**:
actual batched 32B verification cost, acceptance under batched target arithmetic,
retained draft KV behavior, two-rank packet/rollback costs and full-request wall
latency remain unmeasured. K=8 only saved one more hypothetical round and drafted
more rejected tokens. Neither K is performance-admitted on these counts alone.

Timings include fresh prefix/cache recomputation and synchronization, exclude model
loading, and run sequentially K1→K2→K4→K8 with runtime warming carryover. There is no
randomized repetition, confidence interval, or optimized draft-decode claim.

## Identity and bounded execution

- Draft: authentic existing mixed 3/4-bit 0.5B, 260,563,712 loaded parameter bytes,
  628 tensors; not uniform4 and not a new quantization of the target.
- Target receipt binds model revision `2938092373e5f97b95538884112085364c2da315`,
  exact config bytes and the immutable reference digest. This is parent-provided
  receipt consistency, not a new independent proof of target execution.
- Every shared tokenizer ID and every used prompt/reference/proposal ID was checked
  for identical token strings and valid model vocabulary range; no retokenization.
- Target directory contains copied tokenizer/config/index only, no weight files.
- Draft files and target metadata were hashed before/after actual model execution.
  Loaded-layout accounting is recorded; no independent packed-array extraction claim.
- Parent explicitly selected GPU, bounded allocator cache to 256 MiB, reset peak
  before the run and restored device/cache settings afterward.
- Bounded actor deadline 180 s, outer limit 300 s; actual process completed in 3.871 s,
  with owned group cleanup verified. The runner's library seam was used; actor CLI
  entrypoint admission was not widened.
- Environment and active/peak MLX allocation are in `draft/draft-v1.json`; allocation
  is not OS residency or proof of no paging. Minor deprecated device-info warning
  remains in the original stderr rather than being hidden.

## Batched numerical prerequisite

An opt-in all-position-logit head extension preserves the existing last-position
default. Its CPU suite passed 55 tests with one explicit optional F16 skip.

On actual M4, the original packed-FP16 parity fixture body passed 16 tied/untied,
group-size and all-position combinations at **the same input width**. This alone
cannot establish batched-vs-sequential equivalence.

A separate genuine tiny packed-FP16 GPU check compared verification batches of
K+1 positions with sequential single-token calls on the same forced IDs and fresh
prefilled caches. **16 cases** (tied/untied × groups1/8 × K1/2/4/8) passed all-position
argmax, logits and live-KV tolerance checks (`rtol=0.002`, `atol=0.002`); observed
maximum absolute logit difference was zero. This remains a tiny-fixture result,
**not full32B parity or an exact-greedy guarantee for other weights/shapes**.

## Remaining gates

A separate speculative decoder component is being built/tested; it is not part of
this executed diagnostic. It must validate proposal/teacher packets, synchronize
both cache offsets, trim rejected suffixes, handle all-accepted draft catch-up,
EOS and generation caps, and preserve CPU communication boundaries.

Full32B runs are paused: the M2 currently reports elevated memory pressure and only
approximately 5.7 GiB free disk, unlike the earlier idle-host trials. Do not make
an unsafe heavy retry or infer a speedup without real target/control measurements.

`executed-source/` and `executed-phased/` preserve actual diagnostic/head sources.
`draft/`, `tiny-batch-sequential/`, `tiny-same-width/` retain results and actor receipts;
`admission/` retains exact asset/receipt identities; `integrity.json` hashes the files.
The diagnostic's 10 CPU tests and full lint/format pass after parent style cleanup.
Independent diagnostic/head review passed for its narrow scope; exact verdict is
in `review/draft-review-verdict.json`. It is separate from full-engine approval.
Later component integration and its failed real-checkpoint cache gate are recorded
in `../m4-speculative-integration-v1/`; they do not alter this executed diagnostic.
