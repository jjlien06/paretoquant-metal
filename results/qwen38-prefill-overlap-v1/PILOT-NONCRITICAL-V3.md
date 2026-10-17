# Audited 27B overlapped-prefill pilot

**Scope: a successful six-request pilot, not the completed eighteen-case study.**

Both methods use hidden-only-prefill MTP on the unchanged Q4/BF16 target, split28/36 and group32. The comparator is serialized hidden-only MTP, not a slower discarded-projection path or native decode.

| Long workload | Serial TTFT (s) | Overlap TTFT (s) | TTFT reduction | Serial request (s) | Overlap request (s) | Request reduction |
|---|---:|---:|---:|---:|---:|---:|
| code-1536 | 25.89 | 15.25 | 41.1% | 49.87 | 39.27 | 21.3% |
| structured-1536 | 25.94 | 15.27 | 41.1% | 49.14 | 38.47 | 21.7% |

- Actual population: 6 requests = 2 warmups + 4 measured requests; one serial/overlap pair per measured case.
- Both tested cases meet the30% TTFT and15% request-latency thresholds individually. This is not a pooled mean or a confidence interval.
- Exact output IDs/stops match the archived original native references. The same-process small/full513-token, chunk256 logit/cache/continuation gates passed at unchanged atol1e-6/rtol1e-5.
- Source/weight identities, receiver-credit/causal transcripts, three-actor pressure coverage, owned cleanup and three explicit closed-endpoint refusals passed the actual raw025 audit.
- Under the explicit user amendment, warning2 is allowed and recorded. Critical4/missing sampling/deadline failures remain aborts. This is not normal-pressure qualification or a guarantee against OS/driver stalls.
- The two Python-buffer release changes run in both compared methods; the pilot does not isolate their physical memory savings. No multi-GB memory-saving claim is made.
- Untimed same-process numerical gates precede request measurements and warm shapes. Loading/setup is excluded; no cold-start or24/7 reliability claim.
- Pilot log/retrieval budget passed conservative projection for fourteen requests per full-suite batch.

## Durable evidence

- Trial: `physical-pilot-noncritical-v3-1792272408272316000/`
- Parent audit: `physical-pilot-noncritical-v3-1792272408272316000-raw-audit.json`
- Analysis: `physical-pilot-noncritical-v3-1792272408272316000-report.json`
- External native/source/actor trust: `physical-pilot-noncritical-v3-1792272408272316000-external-trust.json`
- Freeze SHA256: `161349a091468906a7c699c83a9224a388403e2d3542675770e8635184f812eb`
- User memory-amendment SHA256: `a0341821118cfd83d45f0f8979152aaffae3f296b372f8dc29b58aacc35d409f`
- Independent source approval: `noncritical-combined-independent-review.json`.

Independent pilot-result review passed61 checks and reproduced the raw audit, numerical/cache coverage, native-output bindings and timing reductions. Its report is `noncritical-pilot-independent-audit.json`, SHA256 `63e8c8cb269ce2a24f8231ca9875375b3d3db3824a3ad2f4f0585ede5e20fc8d`. Full-suite execution is running under `parent-run-full-suite-noncritical-v1.py`, with each batch requiring real receipt audit/cleanup before the next. All eighteen cases and all six long-case targets remain required before closing the milestone. Upstream attribution remains in `../../benchmarks/qwen38/ATTRIBUTION.md`.
