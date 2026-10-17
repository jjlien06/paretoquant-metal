# Qwen3.8-27B: warm decoding above 10 tokens/s on the two Macs

## Verified result

Exact target: `mlx-community/Qwen3.8-27B-4bit`, revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.

The new opt-in `021-qwen38-prefix-rollback` runtime exceeded 10 **warmed post-first
output token events/s** on both tested prompts, using the original BF16/Q4 target
weights. It is not a universal 10 tokens/s floor, cold-start claim, whole-request
throughput claim, task-quality evaluation or production-service certification.

| Same-placement comparison | Native warm decode | New warm decode | Gain |
|---|---:|---:|---:|
| Count 1 through 20 | 7.375 tok/s | **11.779 tok/s** | **59.7%** |
| Two-sentence technical explanation | 7.582 tok/s | **10.240 tok/s** | **35.1%** |

These are short, separately loaded process trials, not randomized statistical
proof. The first request of each process is excluded from warm medians. Counting
has two warm observations; prose combines four warm observations from two starts.
All warm samples were >=10: counting11.730–11.828, prose10.022–10.274.

The saved launcher was then executed end-to-end again. Its three real explanation
requests measured **10.336 / 10.432 / 10.472 tok/s**, with numerical admission,
matching rank actions, complete target inventory and verified worker/listener
cleanup. This independent launcher execution is not silently folded into the
original pilot. Evidence directory:
`../qwen38-live-1792161172-39091/`.

Metric: `(generated_events - 1) / (request_wall - first_token_latency)`, including
emitted EOS and final rank ACK, excluding model loading, numerical admission and
prefill/first-token latency. Whole-request rates are lower. One earlier first-use
prose request measured9.967 decode tok/s; an unrelated candidate's cold prefill
stalled for96s before completing. Do not promise a cold or universal minimum.

## What made the difference

- Native pretrained **one-token MTP drafting**, verified two tokens at once.
  The pinned head is238,934,137 bytes and shares target embedding/output weights.
- A **BF16 slot-shared affine QMV** on the M4 verifier, derived from the earlier
  ParetoQuant experiment and pinned Apple MLX helpers. Native M4 multirow QMM
  changed arithmetic; this candidate instead preserved singleton results.
- **GDN accepted-prefix rollback**: rejected proposals rerun only the captured
  recurrent update, repair convolution history and trim attention KV. This avoids
  replaying every target layer. The technical prose case removed seven full-model
  replays per request:35 teacher forwards became28, with identical output.
- **Batched cache-copy evaluation**: snapshot/restore still completes all copied
  leaves before mutation, but uses one evaluation barrier rather than one per leaf.
- Placement: **28 M2 Pro layers / 36 M4 layers**, group32. This is not an equal-speed
  partition; capacity and actual stage cost determine placement.

The counting request accepted35/35 proposed tokens; prose accepted21/28. Acceptance
is prompt-dependent, so the counting gain must not be generalized to all requests.
MTP head behavior is not attributed to the custom kernel, and isolated projection
ratios are not presented as complete-model speedups.

## Numerical and identity verification

- All1,847 original text-tensor payload digests matched the native controls; no
  requantization, FP16 conversion or target-weight edits were adopted.
- Every counting completion matched the native70-event sequence including EOS.
  Every explanation completion matched the native50-event sequence including EOS.
  Both physical ranks agreed on token IDs and stop reasons.
- Before generation, each process reruns actual small-checkpoint and full27B gates:
  forced width2 versus singleton full logits, exact argmax, every owned live KV,
  convolution/recurrent state, scalar metadata and offsets.
- Instrumented/uninstrumented same-width execution and prefix0/1/2 commit plus
  continuation passed unchanged `atol=1e-6, rtol=1e-5`; the reported full-target
  logits/cache/hidden comparisons were **bit-exact**, not merely within tolerance.
- Original native M4 width2 failure is preserved. Neither matching short tokens
  nor a claim of speculative losslessness was used to bypass failed numerics.
- Parent final021 suite:95 tests passed locally and on the mini, with clean Ruff.
  The fixture tests alone are not the physical gates; actual trial receipts are.
- Independent read-only019,021 and022 reviews reported no blocking security/logic
  issues; additional mixed-acceptance/KV-boundary test coverage was suggested.

## Run it

From the project root, with both Macs unlocked and the same bridge addresses:

```sh
bash results/qwen38-speed-v2/run-fast.sh
```

An optional first argument replaces the tested explanation prompt:

```sh
bash results/qwen38-speed-v2/run-fast.sh 'Explain briefly how a rainbow forms.'
```

The launcher was actually executed successfully. It makes three fresh-cache
requests, prints responses/decode rates, and creates a new exclusive evidence
directory. It is a bounded CLI, **not** a persistent chat server. Internal deadline
is220s; give its terminal at least310s. Setup/numerical gates make process startup
substantially longer than the request decoding time.

`fast-profile-v2.json` contains exact source hashes, checkpoint/head identity,
trusted metadata hashes, split, verifier modes and remote deployment path.
Never bypass a failed source/numerical/memory gate. Rediscover bridge addresses
if the cable reconnects, preserving the pinned SSH host identity. Recheck both
hosts' pressure, swap and disk before heavy trials; do not kill unrelated apps.
Observed final full-model peak MLX allocation was about7.4GiB M2 and9.1GiB M4;
allocator bytes are not total OS residency, and swap existed.

## Preserved alternatives and failures

-015 fast synchronization:7.722→7.772 tok/s, no useful demonstrated gain; flag off.
  Upstream warns of possible GPU deadlock, so it is not adopted.
-016 exact FP16 rejected BF16 values not representable exactly in FP16.
-020 explicitly rounded FP16 passed the small physical trial but the first large
  trial rejected legitimate BF16 `A_log`; later eligibility repair remains a
  separate unadopted experiment, not evidence for the winning native-BF16 result.
-44/20 and later36/28 large-block native prefills timed out under the later live
  conditions. A pending local cleanup resolved on post-hoc process/listener
  readback, with the original failure unchanged. No unrelated process was killed.
-019 initial generation passed numerical gates but failed MTP metadata binding
  because upstream `ModelArgs` mutated the caller's RoPE dictionary. Strict TDD
  reproduced it; deep-copying constructor input fixed it without weakening binding.

## Evidence

- `final-verification-v2.json`: programmatic final identity/output/numerical/cleanup
  checks and all raw samples used for the table.
- `fast-profile-v2.json`, `run-fast.sh`: saved opt-in configuration and tested CLI.
- `021-prefix-full-counting-v2/`, `021-prefix-full-prose-v1/`,
  `021-prefix-full-prose-confirm-v2/`: final full-target candidate receipts.
- `native-counting-28x36-g32-v1/`, `native-explanation-28x36-g32-v1/`: same-placement
  native control receipts, exact tensor hashes and generated IDs.
- `019-controlled-comparison-v1.json`: intermediate MTP without optimized rollback.
- `018-operator-{m2,m4}-v1.json`, `width2-slot-small-m4-v2.json`: kernel/component
  evidence, not full-generation rates.
- `research-external.md`: indexed X leads and primary implementation research;
  external benchmark claims were not used as local measurements.

All completed final trials verified owned workers and listeners closed. Downloads
remain intact. No commits/pushes or persistent inference server were created.
