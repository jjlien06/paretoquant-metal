# Reproducible Qwen3.8 two-Mac benchmark

This benchmark compares native width-one phased decoding against native-MTP speculation on the exact same loaded Qwen3.8-27B shards. It is not a benchmark of Strata, tensor parallelism, an unmodified single-host asynchronous MLX generator, or model quality.

## Prerequisites

- The two already-provisioned16GB M2 Pro/M4 hosts, connected over Thunderbolt Bridge.
- The pinned target/head/control artifacts and approved SSH identity in `results/qwen38-speed-v2/fast-profile-v2.json`.
- Project `.venv` with the recorded MLX/MLX-LM/tokenizer versions.
- Healthy memory pressure and at least8GiB free disk on each host. The operator checks current bridge routes, pressure, disk and existing inference workers. It never closes unrelated applications.

The saved profile is machine-specific. Validate current bridge addresses after reconnecting, retaining strict pinned SSH host verification; do not simply disable a failed check.

## Inspect the complete plan without touching hardware

From the repository root:

```sh
.venv/bin/python benchmarks/qwen38/run_bench.py \
  --profile results/qwen38-speed-v2/fast-profile-v2.json \
  --workloads results/qwen38-broad-v1/workloads \
  --runtime spikes/023-qwen38-workload-bench \
  --output "$PWD/results/qwen38-broad-rerun-NEW" \
  --plan
```

`--plan` validates the six frozen schedules and local paths without creating an output directory, SSH, or model execution. Use a new exclusive output path for every real run.

## Run the physical suite

Use the same command without `--plan`. The operator serializes six bounded processes, each with14 scheduled fresh-cache requests; it stops on the first failed preflight, deployment, execution, verification or cleanup. Each process reruns same-loaded-model numerical admission before measuring requests. Allow roughly an hour of uninterrupted hardware availability; actual duration varies. There is no persistent inference server.

Native and MTP requests are counterbalanced within each process. The sidecar remains resident during the native baseline; this controls same-process residency but does not measure a separately optimized minimal-memory native configuration.

The command writes deployment/source hashes, pre/post health, raw actor receipts, request tokens/text, timings, per-rank memory and phase evidence. The normalized analysis expects all84 planned requests; selecting `--batches` does not silently reduce the full-suite denominator. Partial or failed runs retain missing IDs and suppress full-suite aggregate claims.

**Additional certification requirement:** independent review found an insufficient generation/action binding check in the original023 verifier. Preserve it as historical code; run the separate strict action auditor on the raw receipts before treating a new run as certified:

```sh
.venv/bin/python benchmarks/qwen38/mtp_audit.py \
  /absolute/path/to/trial-00 /absolute/path/to/trial-01 \
  /absolute/path/to/trial-02 /absolute/path/to/trial-03 \
  /absolute/path/to/trial-04 /absolute/path/to/trial-05 \
  --output NEW-strict-action-audit.json
```

Use the actual trial paths in the operator's `trial_manifest.json`. Audit output is exclusive; original trial files remain unchanged. The recorded broad run passed this additional audit for all84 requests on both ranks (`results/qwen38-broad-v1/strict-action-audit.json`). This checks recorded generation progression, mode, counters and rank agreement; it does not recover unlogged teacher predictions or replace physical numerical gates. Do not equate a successful original `result.json` with independent review approval.

## CPU tests and analysis

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch \
  .venv/bin/python -m pytest benchmarks/qwen38 -q
.venv/bin/python -m ruff check benchmarks/qwen38

.venv/bin/python benchmarks/qwen38/analyze.py EVIDENCE.json \
  --json NEW-analysis.json --csv NEW-analysis.csv \
  --markdown NEW-analysis.md --exclusive
```

Add `--allow-partial` only to publish an explicitly partial report. `analyze.py` performs no runtime attestation: it consumes rank-reconciled evidence; the operator and independent audit establish checkpoint/source/protocol/numerical/cleanup provenance. CSV spreadsheet formula escapes affect presentation only, not JSON evidence.

## Workloads and interpretation

- Six authored tasks: Python code, technical explanations, combinatorial math, Chinese, Spanish, and structured extraction.
- Each task at its short natural length, approximately512 and1536 formatted prompt tokens.
- Synthetic engineering archive padding stresses context length; it is not a standard long-context reasoning test.
- Output caps128/256 events; actual EOS stops count. Capped partial answers are not successful quality evaluations.
- Two measured repeats per method/case, plus explicit warmups:72 measured requests and12 warmups.
- Report TTFT, request wall/throughput, warmed post-first decode, acceptance, per-rank MLX peak, failures and variation separately.
- Equal-case geometric-mean speedups are descriptive; no cross-case pooled tok/s and no reliable tail-latency claim from two repeats.

## Ownership and limitations

See [ATTRIBUTION.md](ATTRIBUTION.md) for upstream Qwen, MLX and kernel sources versus project-specific integration. The frozen protocol and raw runs live in `results/qwen38-broad-v1/`; the original two-prompt pilot remains in `results/qwen38-speed-v2/`.

The measured prefill bottleneck motivated isolated `024-qwen38-hidden-prefill`. Its strengthened cache/action admission and both-host physical gates passed, followed by42reference/candidate/reference requests with exact output agreement. It remains opt-in: long code/explanation complete-request throughput improved5.57%/3.01% versus the same-source after control, while short requests were effectively flat. See `results/qwen38-broad-v1/prefill-experiment/README.md` for the exact exercised replay command, raw evidence and limitations. This limited follow-up does not replace the original18-case results.
