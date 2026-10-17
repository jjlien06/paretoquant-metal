# Actual-weight 32B fused-kernel screen — October 5, 2026

## Verdict: retain stock; do not integrate these candidates

The existing packed 4-bit Metal gate/up kernel was exercised on both physical GPUs
with authentic layer-0 packed tensors from pinned Qwen2.5-32B-Instruct-4bit.
Three synthetic unit-RMS FP16 activation rows were used; these are **not captured
model activations**. No checkpoint weights, math, or tolerances were changed.

| Physical GPU | Compiled stock gate/up median, ms | Best fused launch | Fused median, ms | Observed median latency reduction |
|---|---:|---|---:|---:|
| M2 Pro | 1.0855 | fused_rpg1 | 1.0486 | 3.40% |
| M4 | 1.8831 | fused_rpg1 | 1.8271 | 2.97% |

Every fused gate/up launch passed the preset elementwise tolerance on all three
inputs, but **none exceeded the strict 5% validation selection margin**.
All four fused full-MLP variants failed the same preset tolerance on two of three
inputs on **each host**, so none was timed or selected at full-MLP scope. Failed
parity evidence remains in the JSON. The stock down projection was identical across
controls and candidates; native and equally compiled stock controls passed.

The tolerance was predeclared `rtol=0.01`, `atol=0.02`, with finite values required;
it was not relaxed after seeing results. A small relative MSE does not override an
elementwise failure. There is no custom-fused 32B generation, quality or speedup claim.

## Measurement scope

- Both GPUs used the same verified nine `.npy` arrays and manifest hash, approximately
  228 MiB; no full model/shard was loaded for this isolated diagnostic.
- Input is `[1,5120]`; gate/up output width 27648; affine4 group size 64.
- Each phase has 72 raw samples per surviving method: three inputs × 24 synchronized,
  rotated, fresh-call measurements. Warmup/compilation is outside timed samples.
- Separate tuning and validation phases are retained; >5% is a diagnostic margin,
  not statistical significance or a confidence interval.
- Gate/up and full-MLP native/compiled controls remain separate.
- Peak MLX allocation approximately 228 MiB is not total process memory or OS residency.
- Deadline 120 s, outer tool limit 220 s; the existing single-owner bounded actor library
  supervised each diagnostic. Both completed and verified their owned process groups gone.
  This library diagnostic did not widen the actor CLI's entrypoint admission.
- Runtime MLX 0.32.3 / MLX-LM 0.32.0 / Python 3.13.2 on both physical hosts.
- M4 installed-source hashes match the M2 run exactly. The M4 copied package has no
  Git checkout identity (`git_revision: null`); explicit executed-source hashes are retained.

## Evidence

`m2.json` / `m4.json` include all parity checks, failures, raw timings, complete manifest,
actual FP16 RMS, source hashes, environment and memory. `*-actor/` proves bounded
completion/owned-group cleanup. `executed-source/` and `executed-package/` preserve
the executed harness and unchanged kernel package; `integrity.json` hashes them.
The harness's CPU verification passed **10 tests and 13 subtests**, plus lint/format.
Independent review **passed for this isolated diagnostic only**, recomputing all
1,152 timing samples per host and verifying all nine local file/raw hashes without
GPU imports. `review/` retains the verdict and descriptive paired-sample summary;
its empirical percentiles are not confidence intervals or statistical significance.
The reviewer did not independently redo checkpoint extraction/config provenance.
The separate parent crosscheck matches retained config bytes and every layer-0 raw
hash against the published successful full-model rank-1 tensor inventory. This is
not a new independent extraction attestation. No kernel rollout is approved.

The admitted full-32B pilot optimization is instead local evaluation grouping using
stock operators: [full-model evidence](../m2pro-m4-dual-host-grouped-32b-v1/README.md).
