# Qwen3.8-27B on two16GB Macs: broad workload benchmark

## Verified conclusion

The native-MTP runtime improves warmed decoding across the18 tested cases, but **does not provide a dependable10tokens/s floor and does not always improve complete-request time**.

- Exact unchanged target: `mlx-community/Qwen3.8-27B-4bit`, revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
- Placement:28final layers on the M2 Pro,36initial layers on the M4; group32; fresh request caches; native BF16 computation, original affine Q4 weights.
- Six authored task types × three prompt lengths;72 measured requests plus12explicit warmups =84 total, across six process starts. Each method/case has two measured repetitions with counterbalanced order.
- **42.9% higher warm post-first decode throughput**, equal-case geometric mean of median paired ratios.
- **12.8% higher whole-request speed**, same aggregation convention.13of18cases improved complete-request time; five regressed.
- Measured MTP decode range **9.49–11.34tok/s**.11of36measured MTP requests were below10.
- Peak request-reported MLX allocation across the suite: approximately **7.86GiB M2 /9.10GiB M4**. This is allocator accounting, not total OS residency. The sidecar stays resident during native baseline measurements.
- All native/MTP output IDs and stop reasons matched within each case. All1,847text-tensor payload hashes match the prior same-placement native control. No requantization or FP16 conversion.
- Six completed batches, no failed scheduled requests; verified process/listener cleanup on both hosts. Failures in earlier independent evidence reviews are retained separately rather than hidden.

These are **descriptive observations, not confidence intervals or production reliability**. Two repeats per method/case do not establish a worst-case speed or p95latency.

## Per-case results

See `observed-table.md` for all18case medians, native/MTP TTFT, acceptance and whole-request ratios. Complete samples, min/max/sample standard deviations, per-rank memory, warmups, actual output lengths/text and paired ratios are in `all-six-collected/{evidence,analysis}.json`.

Representative findings:

| Case | Native warm decode | MTP warm decode | Whole-request ratio |
|---|---:|---:|---:|
| Code, short prompt |7.58tok/s|11.24tok/s|1.426×|
| Code,1536prompt tokens |7.45tok/s|11.04tok/s|1.020×|
| Explanation,1536prompt tokens |7.35tok/s|9.69tok/s|0.894×|
| Chinese,1536prompt tokens |6.73tok/s|9.60tok/s|0.964×|
| Structured extraction, short |7.44tok/s|11.30tok/s|1.432×|

Ratios above1favor MTP. Whole-request ratios use native wall/MTP wall per repeat, then median; they are not computed from the rounded displayed rates.

## Why faster decoding sometimes loses overall

On the long code case, native TTFT was15.33/14.83s versus25.05/25.48s with MTP. The post-first decode gain barely offset that penalty. Long explanations finished more slowly despite faster decoding.

The saved source and phase evidence identify two distinct mechanisms:

1. Nonfinal MTP prompt chunks calculate full target vocabulary scores that are discarded. On the long code case, rank0target-head phase spans totaled about1.09s with MTP versus0.004s natively. These spans include normalization/evaluation/reporting; they are not isolated GPU projection timings.
2. MTP's per-chunk command/ACK flow waits for rank0and head work before rank1starts the next chunk. Native prefill can overlap the pipeline stages. Saved rank1send-complete→next-embedding gaps were about1.85–2.24s with MTP versus sub-millisecond natively. Receive spans include upstream compute and waiting—not just network transfer.

The inclusive phase observations overlap and must not be added as disjoint kernel attribution. The unused projection explains only part of the TTFT gap; serialization is a separate remaining bottleneck.

## Targeted optimization status

`spikes/024-qwen38-hidden-prefill` isolates removal of unused nonfinal target vocabulary projection. It preserves every normalized hidden position, native target caches, shifted MTP-head recurrence and final-token projection. It does **not** change pipeline overlap, model precision, target weights or decoding strategy.

The candidate requires strengthened full-cache inventory and action-progress admission, physical both-host small/full gates and contemporaneous reference→candidate→reference comparison. The frozen experiment protocol is `prefill-experiment/protocol.json`; it reuses the broad suite's exact short-explanation, long-code and long-explanation prompts/caps.

**Physical follow-up completed:** the hidden-only-prefill candidate passed both-host gates and42reference/candidate/reference requests, matching corresponding native/reference output IDs and stops. Against the same-source after control, complete-request throughput improved5.57% on long code and3.01% on long explanation; short requests were effectively flat. Retained as an opt-in, limited-scope optimization—not a new default or a replacement for the original broad-suite numbers. See `prefill-experiment/README.md`, `comparison.json` and `physical-verification.json`.

## Verification, including failed reviews

- Original runtime physical small/full width2verification, live KV/recurrent/convolution state, normalized hidden, prefix0/1/2and continuation gates were run per process at unchanged1e-6/1e-5tolerances.
- Same-placement native/MTP outputs match and all1,847target-tensor hashes agree with the native control (`checkpoint-and-cleanup-readback.json`).
- Independent review found two original023evidence-checking holes: action sequences were not fully tied to emitted progress, and request-level verifier mode was not checked. These are reproduced verifier defects—not evidence that GPU work was omitted. Original code/results remain byte-preserved.
- Separate strict `mtp_audit.py` rejects the reported forgeries and rechecks every actual rank transcript. All84requests/168rank-request records passed; see `strict-action-audit.json`. Its scope is recorded progress, counters, modes and rank equality—not unlogged teacher predictions or new physical numerical admission.
- Final independent read-only review reproduced all six audits, normalized rows and final statistics with no blocking issues (`independent-review-final.json`). It explicitly does **not** approve the original023generation verifier.
- The benchmark helpers/analyzer/action auditor passed276tests with one skipped optional test and clean Ruff at that review. CSV export formula handling and quoted-prompt worker discovery were fixed with observed RED→GREEN regressions.
- Original failure dispositions live in `review-disposition.json` and `benchmarks/qwen38/review-fixes.json`. Candidate024's separate review/fix history is not conflated with the completed broad run.

## Reproduce

Read `../../benchmarks/qwen38/README.md` for the exercised benchmark command, dry-run, collection and strict action-audit commands. Use an exclusive new output directory, retain strict SSH host verification and inspect current memory/network admission. The runner is bounded and serial, not a persistent inference service.

Frozen inputs are in `workloads/`; exact hashes were recorded before execution in `preregistration.json`. `PROTOCOL.md` distinguishes native/MTP paths, warmups, actual caps/EOS and failure treatment. `all-six-trials.json` links the six raw directories to their independently retained deployment-source hashes. `broad-verification.json` records the composed verification rather than rewriting original receipts.

## Limits and attribution

- Authored synthetic performance suite, not standardized accuracy evaluation. Longer contexts contain synthetic distracting archive padding.64of72measured requests stopped at their output cap; partial code/math/JSON responses are not evidence of completed task quality.
- Contexts stop at approximately1536prompt tokens and outputs at256events. No long-duration thermal soak, arbitrary context length, sampled decoding, vision, concurrent serving or production reliability claim.
- Baseline is native width-one **phased two-Mac execution**, not every possible MLX configuration. Sidecar residency is matched, not optimized away for native.
- Qwen/MLX-community supply pretrained models and MTP head; Apple MLX/MLX-LM supply native arithmetic, Metal infrastructure and communication. Project contributions are distributed integration, width-stable verification adaptation, hybrid-prefix cache handling, scheduling and correctness/performance evidence. See `../../benchmarks/qwen38/ATTRIBUTION.md` for source-derived kernel attribution.
- No git commit/push or persistent service was created. Model downloads remain intact.
