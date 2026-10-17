# Attribution and scope of the Qwen3.8 two-Mac benchmark

## Upstream components

- **Qwen / MLX-community:** pretrained Qwen3.8-27B target, original Q4 affine checkpoint, pretrained standalone MTP head and model architecture. This work does not train the head, invent multi-token prediction, or introduce a new 27B model.
- **Apple MLX / MLX-LM:** native Qwen hybrid decoder, GatedDeltaNet recurrence, attention and convolution operators, quantized layers, cache abstractions, Metal execution and distributed communication primitives.
- **Apple MLX quantized Metal helpers:** the slot-shared QMV uses adapted MIT-licensed upstream packing/vector/dot helpers and source-derived scheduling. See `spikes/018-qwen38-slot-qmv/ATTRIBUTION.md` and its upstream license. Do not describe every line of the kernel as original work.
- **Referenced native MTP implementations:** the sidecar adapter follows the normalized-target-hidden, shifted-token and absolute-normalization contract inspected in the pinned primary sources documented in `results/qwen38-speed-v2/research-external.md` and `spikes/019-qwen38-mtp-distributed/README.md`.

## Project-specific implementation

- Bounded, source-attested execution and streamed shard loading across two physically separate Macs with heterogeneous Apple GPUs, independent memory and Thunderbolt networking.
- Explicit numerical verification modes and slot-sharing adaptation, required because the native M4 width-two path failed the single-token full-logit/cache comparison.
- Distributed MTP control and native hybrid accepted-prefix restoration: recurrent-only prefix replay, convolution history repair and full-attention KV trimming, with same-process numerical gates and native-matching greedy output checks.
- Snapshot-copy evaluation batching while retaining mutation isolation.
- Workload scheduling and same-loaded-model native/MTP paired measurements; strict case/request/source/identity verification, failure retention, and descriptive analysis.

## What the performance claim can mean

The comparison is between a native width-one **phased two-Mac** baseline and the project-specific MTP runtime at the same layer placement and evaluation group size, using unchanged target weights. It is not a comparison against all MLX execution modes, every production inference engine, Strata, or an optimally configured single larger machine.

The sidecar remains resident during both methods for a controlled same-process comparison. Separate-process minimal-residency baseline memory has not been measured by this suite. MLX allocation peaks are not total OS residency.

The suite uses authored tasks and synthetic context padding. Matching native output IDs establishes equivalence on the observed requests, not task accuracy, arbitrary-prompt equivalence, long-context reasoning, or production reliability. Two repeats per method/case provide descriptive variation, not dependable tail-latency estimates. Expose truncated-at-cap outputs and EOS stops separately.

An optimization is a **candidate** until physical same-process correctness admission and a contemporaneous end-to-end comparison pass. Keep failed candidates and failed/incomplete batches in the report rather than presenting their absence as success.
