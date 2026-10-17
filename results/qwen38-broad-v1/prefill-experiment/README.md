# Hidden-only MTP prefill: measured follow-up

## Verdict

**Retain as an opt-in optimization with a modest observed benefit, not a new default or a universal speed guarantee.** The full18-case broad-suite results still describe023, not this follow-up candidate.

The candidate removes unused target-vocabulary projection on nonfinal MTP prompt chunks. It retains all normalized hidden positions, native target caches, shifted-token MTP-head recurrence and final-token projection. It does not restore pipeline overlap or change weights, precision, speculative width or rollback.

## Actual execution

Three successful physical processes ran in order: `reference-before-v2` → `candidate-v1` → `reference-after-v1`. Each ran the same14-request schedule: two32-event warmups and two native/MTP repetitions for a short explanation,1536-token code prompt and1536-token explanation. All42 requests completed with exact corresponding output-ID and stop-reason agreement. All1,847target-tensor hashes match the original native control.

The candidate passed mandatory same-process small/full hidden-prefill gates, reduced forced chunks of8 **plus an actual native256 full-shard case**, original strict width2/prefix gates, and shifted-head cache/continuation comparisons. Across these hidden-prefill receipts,3802rank0and4126rank1metrics were strict-passing and exact. Tolerances remained1e-6/1e-5. Independent action audits passed all three trials. Owned processes/listeners closed and both hosts returned to normal memory pressure.

A preliminary `reference-before-v1` attempt failed before GPU execution with an exclusive duplicate supervisor-copy error. It remains preserved. A regression fixed unique source copying while keeping the supervisor last for its dedicated digest; subsequent trials used new immutable deployments.

## Measured MTP results

Values are medians of two measured requests, not statistical confidence bounds.

| Case | Before TTFT s | Candidate TTFT s | After TTFT s | Before wall s | Candidate wall s | After wall s | Whole-request throughput gain vs after |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short explanation |1.246|1.197|1.250|14.090|13.830|13.836|0.05%|
| Code,1536prompt tokens |28.265|25.505|28.447|52.027|49.164|51.904|5.57%|
| Explanation,1536prompt tokens |26.444|23.888|25.408|48.055|45.433|46.801|3.01%|

Against the arithmetic mean of before/after median TTFT, the long code and explanation TTFT reductions were **10.06% and7.86%**. Corresponding whole-request throughput ratios were **1.0570× and1.0439×**. The equal-case geometric mean over all three cases was1.0366×; short-request complete time was effectively flat against the same-source after control.

Long-request rank0MLX peak allocation fell by approximately121MiB. Rank1peak was effectively unchanged. This is observed allocator accounting, not total OS residency or an isolated traffic measurement. The removed256×248320BF16 score tensor's logical payload is121.25MiB, but do not assign every timing change to that tensor alone.

Raw samples and calculated conventions are in `comparison.json`; no slow or first-long-shape sample was discarded.

## Controls and limitations

- This is a small, separately loaded reference/candidate/reference screen, not randomized same-process old/new pairing. No confidence interval or reliable tail estimate is claimed.
- Candidate and after-control sources are byte-identical. The before control preceded the final receipt-consistency/unsupported-mode rejection guards and three test-file changes; all13checked runtime arithmetic files are identical. Exact source differences are disclosed in `source-comparison.json`.
- The candidate performs additional mandatory hidden-prefill admission before timing. This can change warmup/thermal state. Both methods have explicit short warmups, and first new long-shape samples remain measured; shape and system-state variation are not eliminated.
- Decode speed was largely unchanged; this is a TTFT/whole-request optimization. Cross-rank prefill serialization remains the larger architectural bottleneck.
- No full18-case rerun was performed with this candidate. Do not silently apply its small-screen percentages to the earlier broad aggregate or claim it fixes the10tok/s floor.
- The candidate's original CPU/implementation-only notes are historical. Physical proof is in this directory, not inferred from green tests.

## Reproduce the exact enabled trial

The deployed local/remote source snapshots remain immutable and the model downloads remain intact. Recheck current bridge addresses, strict pinned SSH identity and host memory before running. From the repository root, use a new absolute output directory:

```sh
NEW_OUTPUT="$PWD/results/qwen38-hidden-prefill-rerun-$(date +%s)"
TMPDIR=/Users/jeremylien/.hermes/cache/scratch \
.venv/bin/python -u -c 'import json,subprocess,sys; from pathlib import Path; cmd=json.loads(Path("results/qwen38-broad-v1/prefill-experiment/candidate-v1-command.json").read_text()); cmd[cmd.index("--output")+1]=sys.argv[1]; raise SystemExit(subprocess.run(cmd).returncode)' "$NEW_OUTPUT"
```

This replays the exact exercised controller arguments with only a fresh output path; every source, numerical, inventory, schedule and cleanup gate reruns. Give the terminal at least1020seconds for the bounded900-second controller plus cleanup. It is not a persistent service. Use `reference-after-v1-command.json` in the same command for the exact same-source disabled control. Existing output directories are rejected, not overwritten.

For another host or changed network, adapt the trusted profile and redeploy through the documented source-binding workflow instead of weakening checks. Full command options and admission requirements remain in `spikes/024-qwen38-hidden-prefill/README.md`.

## Evidence

- `protocol.json`, `schedule.json`: frozen follow-up design and exact prompts/caps.
- Three raw trial directories and their deployment/command/normalized JSON records.
- `*-action-audit.json`: separate complete transcript audits.
- `physical-verification.json`, `final-code-review.json`, `independent-physical-review.json`: physical and code proof, with assurance boundaries. Independent read-only review verified7928exact hidden-prefill metrics, original width/prefix evidence, target/source identity and corresponding native output parity. It preceded the final after-reference comparison; the performance arithmetic is recorded separately.
- `source-comparison.json`: identical candidate/after runtime and explicit before differences.
- `comparison.json`: complete measured samples and calculated ratios.
- `final-health.json`: healthy idle host readback after all three trials.

Upstream models, MTP and native kernels are not claimed as new inventions. See `benchmarks/qwen38/ATTRIBUTION.md` for project-specific integration versus source-derived components.
