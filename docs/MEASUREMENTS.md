# Measurement ledger

All measurements below were executed on the local M2 Pro with 16 GB unified memory.
Parameter storage, local gate/up timings, practical engine throughput, and quality are different quantities.

## Evidence index

- Original 0.5B mixed run: `results/m2pro-mixed-v2/`.
- Original positive same-weight replay: `results/m2pro-paired-v3/`.
- Sealed legacy-dispatch parity replay: `results/m2pro-paired-release-v4/`.
- Fresh saved-weight retuning: `results/m2pro-saved-retune-v1/`.
- Fresh retuned-dispatch paired replay: `results/m2pro-paired-retuned-v5/`.
- Six-path compiled cached-decode runs: `results/m2pro-compiled-paired-v1/` and `results/m2pro-compiled-paired-v2/`.
- Actual native/compiled autoregressive requests: `results/m2pro-compiled-autoregressive-v2/`.
- Larger-model profile/construction: `results/m2pro-1.5b-mixed-v2/`.
- Practical engine comparison: `results/framework-head-to-head-v1/`.
- Held-out quality: `results/quality-wikitext-prefix16384-v1/`.
- Exact surrogate sweeps: `results/portfolio-sweep-0.5b-v1/` and `results/portfolio-sweep-1.5b-v1/`.

## Retuned same-weight replay

Fresh retuning ignored old execution choices, retained all weight/config/tokenizer bytes,
validated four launch configurations on 64 captured inputs per pair, and selected 18 fused pairs.
Eight candidate launches failed numerical tolerance across layers 20 and 23; those layers stayed stock.
Other stock fallbacks missed the fresh 5% local margin. No bits or other projections changed.

21 paired trials per context, 64 teacher-forced decode steps, same saved weights/schedules:

| Context | Latency ratio stock/fused | Paired bootstrap 95% interval |
|---|---:|---|
| short | 1.037006x | [1.021699, 1.050716] |
| medium | 1.041444x | [0.999557, 1.054900] |
| long | 1.050541x | [1.034338, 1.065620] |

Ratios above one indicate higher cached decode throughput. These intervals cover only the sampled trials.
Check the exact interval, not a rounded `1.000`, before describing a case as excluding parity.
The earlier stale-dispatch rerun contained parity in every interval and is retained, not discarded.
Local dispatch tuning is not proof of universal end-to-end acceleration.

## Compiled decode and actual requests

Both six-path cached-decode runs use unchanged mixed weights and 18 selected fused
pairs. Each retains 432 measured path trials plus 36 warmups; 24 paired trials per
context, 64 fixed schedule tokens. Stock/fused and dynamic/fixed/compiled controls
separate cache policy, graph compilation, and custom MLP contribution.

The second run's combined stock-eager/fused-compiled ratios are
1.285094x [1.267641,1.301666], 1.261018x [1.238254,1.286979], and
1.262617x [1.249584,1.275418]. These are warmed teacher-forced cached decode,
excluding prefill, padding, sampling and first compilation; not practical requests.
The first run is retained separately, not pooled or replaced.

A separate actual MLX-LM/compiled generation protocol included prefill, sampling
and text decoding, with models loaded and compiled decoders reused. It retained
144 measured requests (12/path/context), each exactly 64 generated tokens. Native
and compiled tokens matched within each backend. Compilation's incremental benefit
versus existing native fused generation was 1.021511x [1.014905,1.027051],
1.007712x [1.004304,1.011335], and 1.003929x [0.998911,1.006193]; long includes parity.
Combined fused/compiled versus native stock was 1.094577x / 1.081632x / 1.059324x.
Compiled stock alone was slower on the long context (0.990885x).

Native MLX-LM already overlaps token work. A whole-prompt versus split-final-token
prefill mismatch initially failed fused token equality; matching native prefill
fixed that check before publication. First-call observations are not cold-process
benchmarks. The larger synchronous gain does not transfer to real-request throughput.
See `docs/COMPILED_DECODE.md` for full controls and exact archived source bindings.

## Larger model and quality

The 1.5B profile selected stock for every pair; both precision variants generated real text.
Uniform4 used 828.311523 MiB; mixed used 775.811523 MiB (6.338195% reduction).
In the final sequential timing blocks, uniform4 measured 130.080364 cached tok/s and mixed 129.226619.
That does not establish a larger-model speedup; earlier block timing varied and remains under `artifacts/`.

For each size, FP16 reference, uniform4, and mixed stock scored the same 16,384 next-token targets
over 64 windows from a pinned WikiText-2 raw test export. Export preserves all 4,358 rows, including
1,467 blanks; joining, tokenizer hashes, window masks, and target accounting are recorded.
Inputs have at most 511 tokens; tokenizing the full stream can emit a tokenizer length warning,
but the full 299,078-token stream is never passed to the model in one forward operation.

This is noncanonical bounded-context prefix perplexity, not full WikiText benchmark equivalence
or task accuracy. The calibration texts are separate authored data; pretraining contamination is not audited.

| Model | FP16 PPL | Uniform4 PPL | Mixed PPL |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 13.957981 | 16.576891 | 18.409162 |
| Qwen2.5-1.5B | 9.268064 | 10.577767 | 11.519860 |

## Environment and comparisons

These runs used a heavily used desktop with substantial pre-existing swap. The raw environment
records are authoritative; results are not controlled idle-machine laboratory estimates.
Do not conflate paired same-weight replay with sequential practical engine blocks.
vLLM Metal is MLX-backed, with HTTP/scheduler overhead and different versions.
TensorFlow uses native Keras Qwen graph generation with FP16 and explicit CPU cache updates.
Ollama uses different quantized weights. None establishes equal task accuracy or a batched-serving win.

### Historical code and current admission

`results/framework-head-to-head-v1/executed_runner.py` and
`executed_tensorflow_probe.py` preserve the exact source hashes recorded in the
historical measurements. The Ollama executed runner is also archived unchanged.
These are historical evidence, not integrity-hardened live entry points.

The vLLM process was launched against the same local checkpoint and `/v1/models`
reported its expected root/context; local fingerprints and launch evidence are not
a server-side loaded-weight attestation. The old runner asserted serving settings
without checking them through the API. Current live runners verify bound manifests
before timing, reject conflicting server root/context metadata, and label unavailable
loaded-weight, prefix-cache and scheduler observations `null`/unverified. Use current
`spikes/` scripts and an explicit bound `--model` for new experiments.

Each sweep directory now includes byte-exact `source-profile.json`, verified against
its recorded SHA-256. The 1.5B sweep used v1 profiling, not the later v2 profile used
for model construction and quality; those different experiments are not conflated.

## Scope limits

No 7B/14B model, layer-sharded reference loading, standard task-accuracy evaluation, hardware
counter attribution, two-physical-Mac execution, or iPhone worker is implemented or claimed.
GitHub CPU correctness/lint/packaging passed on Ubuntu with Python 3.11 and 3.13
in run `37230028838` for commit `0be5c1836105596d8994aae0f4510848432fb09b`.
That CI does not execute Metal kernels or hardware performance trials. See DESIGN.md
and QUALITY.md for method details.
