# 27B bounded overlapped-prefill experiment

## Frozen objective

This is a new experiment, not an achieved speedup. Compare serialized hidden-only-prefill MTP with an opt-in bounded overlapped-prefill candidate on the same executed source, unchanged exact27B Q4/BF16 weights, 28/36 M2/M4 placement, group32 and native/slot-QMV verifier modes. Both methods retain the physically admitted hidden-only-prefill optimization and accepted-prefix rollback. Do not use the older all-vocabulary-projection MTP path as the only control.

Machine-readable acceptance, immutable prompts/caps and the historical90-file preservation snapshot are in `acceptance.json`. The exact six target cases are:

- `code-1536`
- `explanation-1536`
- `math-1536`
- `chinese-1536`
- `spanish-1536`
- `structured-1536`

Each target case must independently meet both: at least30% lower first-token elapsed latency and at least15% lower complete-request elapsed latency. For each metric, compute `1 - median(candidate_elapsed/reference_elapsed)` over matched repeat IDs. A favorable pooled average cannot replace a failed target case. Ratios are elapsed-time reductions, not throughput gains.

## Required verification

- Full18-case population: six categories × short/512/1536 prompt targets, retaining exact original prompt bytes, formatted-token checks, output caps and explicit warmup/first-shape conventions. These are authored performance workloads with synthetic distracting context, not task-accuracy proof.
- Planned full-suite workload:72 measured requests and12 short warmups,84 total. None are claimed executed. Mode encoding and exact ordered schedules will be frozen after the read-only design review and before any GPU execution. Same-loaded counterbalanced serial/overlap requests are preferred; never label serializedMTP as native width-one.
- Native-matching output IDs/counts/stops, every applicable hidden/logit/KV/GDN/convolution/offset component, shifted MTP-head cache and continuation, existing width2/prefix gates and actual256-token chunk coverage. Original atol1e-6/rtol1e-5 remain unchanged. Fresh live admission is required; archived approval cannot mint capability.
- Explicit resource bounds: at mosttwo simultaneous pipeline chunks andone pending handoff; existing64KiB control frame cap. The command-level prompt bound must be predeclared and tested if compound prefill is selected. No unbounded queue or reliance on a nonblocking-send assumption.
- Both hosts must be at normal pressure1 before loading and throughout execution; warning2/critical4 abort the owned trial. Preserve failed attempts, source/checkpoint evidence, exact request identities and verified owned-process/listener cleanup. Do not terminate unrelated apps.
- Keep all18 cases in the final table, including slower or unchanged cases. Missing, duplicate, failed or unaudited requests prevent a complete-suite verdict. Two repeats are descriptive observations, not a tail-latency guarantee.

## Ordered admission

1. Read-only dependency-graph and benchmark-schema review.
2. Isolated scheduler/control implementation with vertical RED→GREEN tests, synchronous-rendezvous CPU transport tests, stale/out-of-order/cancel/failure tests and adversarial evidence validation. Preserve all original code/results.
3. Independent CPU review, immutable source/schedule snapshots and exact readback.
4. Fresh host/link/headroom checks, then physical small two-host and same-loaded full-target cache/logit/native256 gates.
5. Bounded matched pilot; proceed to full18-case rerun only when correctness, memory and evidence retrieval pass. Retain any failed performance target rather than loosening it.
6. Independently audit all requests and target reductions, publish raw data and the actually exercised command, then refresh cleanup/health.

## Initial live blocker

`initial-health.json` records warning pressure2 on the laptop and normal pressure1 on the mini. No GPU/model loading was performed for this experiment. CPU-only inspection can continue. These are dated samples, not permanent machine-state claims; recheck immediately before a hardware trial.

## Scope

No vision/CUDA/persistent-service work, checkpoint/precision changes, fourthGGUF-RPC load, unrelated cleanup or unrequested GitHub push. Prior ParetoQuant dirty/untracked work and the separate MLX-vision candidate are preserved.
