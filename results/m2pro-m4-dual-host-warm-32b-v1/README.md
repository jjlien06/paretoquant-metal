# 32B cold versus loaded-model reuse: physical baseline

Same pinned 4-bit weights, same approved phased stock-operator source, same prompt,
and fresh KV cache for every request. Three requests ran in one bounded two-host
process lifetime. All produced the same 41 token IDs including EOS, matching the
previous full-model trial. Tensor inventory and process/listener cleanup passed.

| Request | First token (s) | Total request (s) | Post-first-token events/s |
|---|---:|---:|---:|
| 0 | 40.100068 | 49.102576 | 4.443206 |
| 1 | 1.032018 | 9.059603 | 4.982819 |
| 2 | 1.235463 | 8.918178 | 5.206493 |

The first request was cold with respect to this process; subsequent requests reused
loaded weights and runtime state, not the previous KV prefix. This demonstrates
that keeping the model loaded materially improves observed request latency. It
is not a new fused-kernel speedup or a controlled attribution to shader compilation
versus residency/page faults. Same-prompt repetition is not diverse-workload proof.

Setup/load/first-use costs still exist. Every worker was stopped at completion;
this is not an always-on service. Raw redacted actor receipts and the exact source
hashes are retained. No weights, keys, session tokens or job payloads are included.
