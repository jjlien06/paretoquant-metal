# Broad Qwen3.8 benchmark: preregistered protocol

## Scope

Run the exact `mlx-community/Qwen3.8-27B-4bit` revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` across the M2 Pro and M4, with unchanged native BF16 computation and packed Q4 weights. Preserve the previous pilot as historical evidence, not a baseline for this experiment.

## Frozen workload design

`workloads/manifest.json` records 18 authored cases: Python code, technical explanation, combinatorial math, Chinese, Spanish and structured extraction, each at short, approximately512 and approximately1536 token formatted prompt lengths. Actual tokenizer lengths are recorded. Long prompts contain an explicitly synthetic distracting archive; this is context-length performance stress, not a standard reasoning or accuracy benchmark.

Output caps are128 events at short/512 contexts and256 at1536. Respect actual EOS; do not force padded output to reach a cap. Caps include EOS when emitted. Two measured fresh-cache repetitions of each method/case use ABBA or BAAB order. Six process starts each have one32-event warmup per method and three cases:72 measured requests plus12 warmups,84 planned total.

## Comparison

Use the SAME loaded target shards, native placement28M2/36M4 and group32. Native baseline executes stock width-one phased generation, not MTP/capture/custom two-slot verification. MTP uses the existing strict-admitted slot-QMV verifier, native pretrained MTP head and prefix rollback. The unused MTP head remains resident during native requests: report this explicitly, and do not interpret this experiment as independent minimal-memory residency for the baseline.

Each process reruns the original physical small and full checkpoint full-logit/cache/hidden and prefix0/1/2 continuation gates, unchanged tolerances1e-6/1e-5. Every case/method/repeat must match native output IDs and stop reason; fail closed on disagreement. Retain every failure and incomplete case. Same-rank agreement is not sufficient.

## Metrics

- First-token latency including prefill, request wall, full-request event throughput.
- Post-first event throughput `(N-1)/(wall-first_token)`, with final synchronization/ACK convention explicit. Excludes loading, admission and head setup.
- First loaded-model requests/warmups separately; the first new longer prefill shape is not silently removed.
- Native/MTP paired ratios percase/repeat and descriptive variation; equal-case geometric mean, not pooled tok/s.
- MTP accepted proposals/verification rounds, teacher forwards and rejections.
- Independently recorded per-rank peak MLX allocation plus host pressure/swap/disk snapshots. MLX allocator bytes are not total system residency.
- Monotonic phase/operation timings, labeled inclusive wall/communication observations, not additive GPU kernel attribution.

## Controls and safety

Parent process is the sole GPU hardware owner. No unrelated applications are stopped. Check both hosts' pressure, swap, disk and Thunderbolt routes before each batch. Pinned SSH host verification remains required. Deploy to new immutable directories with read-back source hashes, including each schedule. Use bounded deadlines and nonce-owned workers, requiring process and listener cleanup before another batch. No persistent model server, git commit or push is authorized.

## Decision rule

Do not target10tok/s by choosing easy prompts or deleting slow samples. Publish the complete distribution and per-case paired improvement. Diagnose a bottleneck from this suite before selecting ONE isolated optimization. Keep the original measured runtime and evidence unchanged. Repeat representative affected workloads against a contemporaneous control, preserve numerical checks, and reject any optimization without measured end-to-end benefit. A broad failure is a reported finding, not permission to weaken correctness tolerances.

## Attribution

Pretrained target/MTP architecture and weights are upstream Qwen/MLX-community. MLX and MLX-LM supply native model arithmetic, Metal infrastructure and communication primitives. ParetoQuant-specific integration includes two-Mac orchestration, width-stable verification adaptation, accepted-prefix recurrent-cache restoration, snapshot scheduling, and this controlled benchmark. Existing custom kernel helpers retain their upstream attribution. Do not claim to have invented MTP, model quantization or pipeline parallelism.

## Status

Protocol and tokenized workloads are frozen before physical execution. No broad-suite performance result is implied by this document.
