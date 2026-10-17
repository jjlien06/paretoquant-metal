# Qwen3.8-27B: two-Mac overlapped-prefill full-suite result

**Complete: measured targets met and final independent result audit passed 481/481 checks.** All six designated long cases clear both thresholds individually; all eighteen scheduled cases are retained, including short-request regressions.

## Configuration and comparison

- Target: unchanged `mlx-community/Qwen3.8-27B-4bit`, revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; Q4 affine target with original BF16/floating components and separately pinned pretrained MTP head.
- Hardware: M2 Pro MacBook Pro16GB plus M4 Mac mini16GB, Thunderbolt IP; group32. M4 owns layers0–35; M2 Pro owns layers36–63. Both compared methods share placement, tensor identity, MTP/rollback/QMV settings and fresh request caches.
- Comparator: the previously validated **serialized hidden-only MTP prefill**, not native autoregressive decoding or the older discarded-logit path. Candidate: compound prefix scheduling with evaluated-receive credits, bounded in-flight chunks, one prefix ACK and unchanged final prompt-token/decode execution.
- Population: six categories × three context lengths =18 measured cases; two serial/overlap repeat pairs per case.84 physical requests =12 warmups +72 measured. The earlier separate six-request pilot is excluded.
- Estimator: `100 * (1 - median(overlap_elapsed / serial_elapsed))`, paired by repeat. This is not the ratio of median times, throughput increase, a pooled average or a confidence bound.
- Timers: loaded-model request first-token and complete-request elapsed, including the normal request completion protocol; model loading and untimed numerical admission excluded. Numerical admission warms extra shapes. This is not cold-start performance.

## Required long-case targets

| Case | Actual prompt tokens | Output events | TTFT reduction | Request reduction | ≥30% / ≥15% |
|---|---:|---:|---:|---:|---|
| code-1536 | 1536 | 256 | 38.39% | 19.77% | PASS |
| explanation-1536 | 1536 | 214 | 37.12% | 20.12% | PASS |
| math-1536 | 1536 | 256 | 35.29% | 17.87% | PASS |
| chinese-1536 | 1536 | 256 | 34.10% | 16.38% | PASS |
| spanish-1536 | 1536 | 256 | 34.38% | 16.90% | PASS |
| structured-1536 | 1535 | 253 | 39.48% | 19.48% | PASS |

The preserved identifier `structured-1536` has1535 actual formatted prompt tokens; neither the identifier nor the input was repaired to manufacture1536. Outputs ending before the256-event cap are retained, not padded.

## All eighteen cases

Positive reductions mean faster; negative values mean slower. Non-long cases have no30%/15% acceptance requirement.

| Case | Prompt tokens | TTFT reduction | Request reduction |
|---|---:|---:|---:|
| code-short | 55 | -0.283% | -0.275% |
| code-512 | 512 | +24.155% | +8.827% |
| code-1536 | 1536 | +38.394% | +19.767% |
| explanation-short | 54 | -0.169% | -0.051% |
| explanation-512 | 512 | +25.938% | +11.218% |
| explanation-1536 | 1536 | +37.124% | +20.123% |
| math-short | 74 | -0.329% | -0.154% |
| math-512 | 512 | +22.818% | +9.527% |
| math-1536 | 1536 | +35.293% | +17.872% |
| chinese-short | 51 | +0.178% | -0.118% |
| chinese-512 | 512 | +22.069% | +8.885% |
| chinese-1536 | 1536 | +34.097% | +16.380% |
| spanish-short | 63 | -0.293% | -0.303% |
| spanish-512 | 512 | +22.286% | +8.836% |
| spanish-1536 | 1536 | +34.385% | +16.905% |
| structured-short | 96 | -1.485% | -0.599% |
| structured-512 | 512 | +25.060% | +11.045% |
| structured-1536 | 1535 | +39.475% | +19.483% |

Whole-request latency improved in12/18 cases and regressed slightly in6/18 (all short requests). Worst whole-request regression:0.599%; worst TTFT regression:1.485%. No losing case is omitted.

## Correctness, bounded execution and memory

- Exact measured serial/overlap token IDs and stop reasons match the independently retained original native outputs. Source/checkpoint/per-rank tensor and replicated-weight identities are verified by the raw025 collector.
- Every batch ran same-process small and full target gates on both GPUs before requests, including513-token prompts with two full256-token prefix chunks, original cache-part inventories, logits, continuation and `atol=1e-6, rtol=1e-5`. Received head metrics are not a second independent head execution on the peer.
- Compound-operation command/credit/chunk/drain causality and bounded in-flight chunks are audited. Local event spans do not establish synchronized GPU-kernel overlap or a network-only attribution.
- User-authorized opt-in noncritical policy records actual pressure1/2 samples; 95 warning2 samples across all batch actors. No critical4 observation was admitted. Critical/invalid/missing/gapped samples and deadlines remain aborts. This is **not** strict-normal pressure qualification and cannot guarantee immunity from OS/GPU-driver stalls.
- Maximum recorded per-request MLX allocator peaks, across both methods/warmups: rank0/M2 Pro: 8,310,217,996 bytes (7.739GiB), rank1/M4: 9,776,345,011 bytes (9.105GiB). These are allocator figures, not RSS/system residency or minimum baseline memory.
- Two explicit streamed-payload reference releases run in both modes. They reduce Python payload overlap only; this experiment does not isolate physical memory savings or prove a multi-GB reduction.
- Every batch records owned actor/child cleanup and three explicit listener refusals. Fresh post-suite read-only checks find all36 recorded actor/child PIDs absent, all18 recorded endpoint probes explicitly refused, and both Macs at pressure1. No unrelated processes were stopped. See `post-suite-health.json`.

## Actual exercised commands

From the repository root, with the already-admitted exact immutable deployment and private profile present:

```sh
export TMPDIR=/Users/jeremylien/.hermes/cache/scratch
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B results/qwen38-prefill-overlap-v1/parent-run-full-suite-noncritical-v1.py
```

Actual process `proc_66b3fa491a79` exited0. The controller serializes six bounded14-request batches and audits each before starting the next. Do not rerun by silently reusing executed immutable remote snapshots: remote bytecode additions can fail exact-directory admission. Preserve old evidence and prepare fresh admitted snapshots first. Private addresses/key paths are intentionally absent from this report.

## Evidence and provenance

- `report.json`: complete actual raw025 collection and per-case analysis.
- `summary.json`, `case-table.json`, `cases.csv`: all-case tables; full raw samples remain in JSON.
- `batch-00-complete.json` through `batch-05-complete.json`: exact physical trial, external source/native trust, audit/report hashes and cleanup links.
- `plan.json`: frozen runner/auditor/controller hashes, policy,84-request plan and pilot exclusion.
- `post-suite-health.json`: fresh post-execution host and cleanup observations.
- Parent result auditor: `../parent-audit-noncritical-trial-v2.py`.
- Independent source approval: `../noncritical-combined-independent-review.json`; separate earlier pilot result approval: `../noncritical-pilot-independent-audit.json`.
- Final full-suite independent result audit: `../noncritical-full-suite-independent-audit.json`, **481/481 checks passed**, no blocking security concerns or logic errors. The independent collector replay exactly reproduces the published report. This is a receipt/result audit, not cryptographic execution attestation or statistical-confidence certification.
- Final parent acceptance and refreshed publication hashes: `final-acceptance.json`. The earlier `publication-verification.json` preserves its pre-review snapshot, including the README hash before this completion update.

Report SHA256: `ec0e3cf9e9bf21d1cba20af4988ca6e22d5ab92a2fe2199d910bc01a01ccf67b`.
Manifest SHA256: `161349a091468906a7c699c83a9224a388403e2d3542675770e8635184f812eb`.
Memory-amendment SHA256: `a0341821118cfd83d45f0f8979152aaffae3f296b372f8dc29b58aacc35d409f`.

## Attribution and limitations

The project contribution here is source-bound heterogeneous two-host scheduling/credit flow control, streamed execution/lifetime handling and a strict reproducible end-to-end measurement/audit path. Qwen/MLX-community supply pretrained target/head weights and model architecture; Apple MLX/MLX-LM supply native hybrid operators, caches, Metal execution and distributed primitives. The slot-shared QMV adapts attributed upstream Metal helpers. This work does not invent or train MTP, reimplement the entire model, or establish a new quantization-quality result. See `../../../benchmarks/qwen38/ATTRIBUTION.md` and `../../../spikes/018-qwen38-slot-qmv/ATTRIBUTION.md`.

Two repeat pairs per case provide descriptive observations only: no reliable p95, confidence guarantee, arbitrary-prompt equivalence or universal latency floor. Authored/synthetically padded contexts are performance stress, not coding accuracy or long-document reasoning scores. No visual inference, autonomous tool-use, persistent service or24–48h reliability is qualified by this result. Sources/evidence remain local; this task does not establish a GitHub push. Hash binding is not cryptographic execution attestation.
