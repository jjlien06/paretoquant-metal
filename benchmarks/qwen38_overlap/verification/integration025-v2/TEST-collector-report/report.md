# Qwen3.8 overlap descriptive report

Artifact kind: TEST_FIXTURE_NOT_MODEL_RESULTS

**Not physical acceptance.** Evidence status: source_bound_raw025_valid

Two repeats are descriptive, not confidence intervals or tail-latency guarantees.
Elapsed reduction = 1 - median(paired candidate/reference ratios). No pooled pass override.

Loading/gates excluded; two queue warmups per batch included in coverage but not estimators.
No first-long-shape exclusion. Inclusive rank0 TTFT/request wall; phases are non-additive.
Both modes MTP hidden-only + prefix rollback; serialized MTP is never native width-one.
Original caps retained. EOS248044/248046 counts as an output event; length-capped is not natural EOS.
Pressure evidence is sampled, never a continuous guarantee. No simultaneous GPU-kernel claim.

| Case | TTFT ratio | Wall ratio | Complete | Long thresholds |
|---|---:|---:|---|---|
| code-1536 | 1.0 | 1.0 | True | False |
| structured-1536 | 1.0 | 1.0 | True | False |

## Regressions (all observed repeat ratios >1)

None among the valid recorded pairs; missing/invalid pairs are not gains.

Coverage complete: True; missing=0, failed=0, duplicate=0, extra=0.

Full-suite acceptance: **false**. Runtime adapter and independent review remain pending.
