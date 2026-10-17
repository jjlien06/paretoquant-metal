# 40 Gb/s cable and unchanged full32B rerun — October 6, 2026

## Result

The upgraded physical link improves bulk-transfer smoke throughput, but no
material warm decode improvement appeared in this short full-model rerun.

| Measurement | Earlier reported/archived result | Current result |
|---|---:|---:|
| Negotiated physical link | Previously reported 20 Gb/s | 40 Gb/s on both hosts |
| M2 Pro → M4 TCP smoke median | Previously reported 18.93 Gb/s | 37.552 Gb/s |
| M4 → M2 Pro TCP smoke median | Previously reported 18.93 Gb/s | 37.317 Gb/s |
| Group8 warm post-first token events/s | 6.270 | 6.314 |
| Group8 warm request wall median | 7.451 s | 7.466 s |

Current transfer medians are approximately 1.98×/1.97× the previous reported
smoke rate. Historical warm decode throughput differs by only +0.707%, while
warm request wall time is essentially unchanged. This is **not a controlled
same-session cable A/B**, a statistical equivalence result or a universal inference
claim. The earlier bandwidth figures come from the verified prior session report;
the previous physical cable and its original transfer harness were not rerun.
The historical inference rows are recomputed from the preserved group8 receipt.

## Bulk-transfer method

A standalone standard-library Python TCP probe binds the actual Thunderbolt
Bridge IPv4 addresses, validates the peer and verifies routing through `bridge0`.
No payload files or large model allocations are used. Each direction has:

- One separate 16 MiB SHA256-verified integrity transfer, not a timing sample.
- One 256 MiB untimed-selection warmup (raw elapsed is retained).
- Three 512 MiB timed single-stream payload transfers.

Buffers are fixed at 1 MiB; requested socket send/receive buffers are 4 MiB.
The client timer includes payload and final acknowledgement, excluding connection
and initial control setup. Every byte count agrees with the server. Both separate
integrity digests match. This is short **smoke throughput**, not maximum bandwidth
or concurrent bidirectional saturation. Transport is peer-bound plaintext over
the trusted direct cable, not an encrypted/authenticated service.

The owned server and client actors completed; child/process groups and the
listener were verified gone. The server deadline was 90 seconds and client
60 seconds. Exact executed probe SHA256 and read-back remote receipts are retained.
Live `system_profiler` confirms the connected link at 40 Gb/s on both actual Macs;
`link-attestation.json` excludes unrelated unconnected ports' advertised limits.

## Full-model correctness and performance

Before the large trial, a physical two-rank authentic mixed-3/4-bit 0.5B control
passed with the historical nine EOS-inclusive token IDs and unchanged per-rank
tensor digests. Then the existing Qwen2.5-32B-Instruct-4bit checkpoint ran with:

- Pinned revision `2938092373e5f97b95538884112085364c2da315`.
- Original reverse contiguous split `[26,38]`: M2 layers38–63, M4 layers0–37.
- Unchanged independently reviewed grouped decoder, evaluation group8.
- Original prompt `What is 2 plus 2? Explain briefly.`, greedy, max64 token events.
- Four loaded-model requests on both ranks, each with a fresh KV cache; no prefix reuse.

Every request reproduced the same historical 41 EOS-inclusive token IDs and stop.
The 1,671 unique tensor digests and seven replicated tensors agree with the
historical rank receipts; trusted metadata/ownership/completeness checks passed.
Sources remain byte-exact to the 20-file reviewed snapshot, including the original
all-position-logit-free grouped runtime. Unreviewed speculative code was not used.
SSH's changed bridge address was handled through a transport adapter retaining
the previously pinned host-key alias, without changing actor/decoder source.

Rank0 measurements:

| Request | Wall, s | TTFT, s | Post-first events/s |
|---|---:|---:|---:|
| First loaded-model request | 27.387 | 20.820 | 6.091 |
| Warm1 | 7.459 | 1.124 | 6.315 |
| Warm2 | 7.466 | 1.131 | 6.314 |
| Warm3 | 7.573 | 1.139 | 6.217 |

Post-first throughput is `(token events including EOS - 1)/(wall - TTFT)`.
Request timings exclude weight loading and controller setup. First-use request
latency is not a complete cold-start measurement and cannot be attributed to
cable speed: process/driver cache, paging, thermals and other conditions were
not independently controlled. Native asynchronous engine TPS is not claimed.
The full supervised trial, including setup/loading and all four requests,
completed in 70.457 seconds with a 240-second deadline and 4 GiB disk reserve.
All owned actors/control transports and all listeners passed cleanup verification.

## Scope, resources and provenance

Both hosts passed live normal-pressure/disk preflights before each inference
trial. Swap was present and increased on the mini; no no-paging, residency,
automatic pooled32GB or resource-free-machine claim is made. Unrelated user
processes were not terminated. No checkpoint was downloaded, requantized or copied
to a new disk artifact on the MacBook; its rank was streamed into memory.

The faster cable is demonstrated for bulk data, not admitted as a decode
acceleration. No new custom32B kernel, speculative32B run, quality evaluation,
engine comparison or public push occurred. Existing speculative batching's
failed numerical gate remains separate and unchanged.

`evidence/` preserves the executed diagnostic/controller, physical small/full
trial snapshots, complete reports, per-rank raw records, source and tensor
bindings, actor/process receipts, preflights and cleanup. `verified-summary.json`
recomputes counts/medians/comparison formulas from exact records. `integrity.json`
hashes the archive. Generated weight-service tokens are redacted; no private key
contents or weight binaries are included.
