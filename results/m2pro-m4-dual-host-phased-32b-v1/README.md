# Two-host phased 32B generation: completed physical smoke trial

On October 5, 2026, the pinned `mlx-community/Qwen2.5-32B-Instruct-4bit`
revision `2938092373e5f97b95538884112085364c2da315` completed a genuine request
across an Apple M2 Pro and Apple M4, each with 16 GiB unified memory.

## What actually ran

- Strict two-rank MLX ring over Thunderbolt Bridge.
- Reverse pipeline split `[26,38]`: M2 Pro rank 0 owned layers 38–63; M4 rank 1
  owned layers 0–37. Progress records cover all 64 original layers.
- Full checkpoint remained on the mini. The MacBook streamed only its owned
  tensors into a tokenizer/config-only directory, without receiver weight files.
- Both ranks returned the same 41 token IDs including EOS, with `finish_reason=stop`.
- All 1,671 trusted-index tensor names were covered; all seven replicated tensor
  digests agreed. Rank inventories contained 683 and 995 tensors, respectively.
- Source hashes agreed across physical hosts. Owned processes were reaped,
  groups were gone, and all three trial listeners were verified closed.

Prompt: `What is 2 plus 2? Explain briefly.`

Actual output:

> 2 plus 2 equals 4. This is a basic arithmetic operation where you add two numbers together. In this case, you're adding the number 2 to itself, resulting in 4.

## Measured timings and allocations

| Metric | M2 Pro / rank 0 | M4 / rank 1 |
|---|---:|---:|
| Stored parameter bytes | 8,007,776,256 | 11,299,411,968 |
| Peak MLX allocation bytes | 8,044,165,296 | 11,348,383,920 |
| Request wall time, seconds | 64.877646 | 64.910617 |
| First-token time, seconds | 56.522993 | 56.559480 |
| Whole-request token events/second, including EOS | 0.631959 | 0.631638 |

The rank-0 interval after the first token was 8.354653 seconds for the remaining
40 token events: 4.787751 events/second. This is not the whole-request throughput
and does not include initial prefill/first-token latency. Total supervised trial
elapsed time, including setup/loading/verification/cleanup, was 89.179153 seconds.
These are one cold-process smoke request, not repeated performance measurements.
MLX peaks are allocator observations, not OS residency, process RSS or total
system memory. The MacBook was already swapping before the run.

## Correctness and safety controls

`small/` is a preceding real two-GPU 0.5B request using the same supervised phased
execution and source. Its nine token IDs, including EOS, exactly matched the
previous streamed run's single-host control tokens; all 24 small-model layers
were observed and cleanup passed.

The durable supervisor and conservative single-owner process-group cleanup were
independently reviewed. `review/retry-review-owned-verdict.json` approves the
preceding frozen source. The sole later production delta increased the strict
closed-port connection probe timeout from 0.2 to 2.0 seconds: real peer refusal
took approximately one second. That bounded timeout-only change was parent
reviewed and regression tested; it was not included in the independent verdict.
Both subsequent physical small and 32B trials passed every success/cleanup gate.

The final native spike gate passed 188 tests and 48 subtests, with one optional
CPU F16 skip, clean Ruff checks, and native packed-F16 GPU parity coverage.

## Evidence and provenance

- `qwen32/verified-summary.json`: independently aggregated inventory, phases,
  request timings, allocations and cleanup facts.
- `qwen32/*.receipt.json`: exact redacted actor receipts, startup/process/source
  identities, complete output records, and persisted progress logs.
- `qwen32/*.record.json`: parsed final records extracted from those receipts.
- `qwen32/metadata/`: exact admitted config and safetensors index, not weights.
- `small/`: analogous physical small-model receipts and control comparison.
- `executed-source/`: frozen source/tests used by the successful trials.
- `review/`: ownership review verdict, exact later timeout delta and admission.
- `integrity.json`: SHA-256 inventory for the archived evidence and source.

No ephemeral authentication values, SSH keys, job payloads, or model-weight
payloads are included. Earlier failed experiments remain unchanged in
`../m2pro-m4-dual-host-32b-attempts-v1/`.

## Limits

This proves one completed full-model two-host generation, not production
reliability, longer-context capacity, task accuracy, standalone 32B logit parity,
a distributed speedup or an automatic contiguous 32 GiB pool. Both ranks consume
the rank-0 selected-token broadcast, so agreeing token IDs alone are not an
independent full-model correctness control. The small-model controls and exact
large inventory/phase evidence provide separate checks.

The execution uses stock MLX-LM quantized operators with explicit synchronized
Qwen2 layer evaluation and CPU communication handoffs. It does not use the
project's custom fused Metal kernels or compiled decoder. Legacy wrapper call
counters remain zero because those wrappers do not instrument explicit phased
handoffs; the progress events record the actual handoffs instead.

The trusted direct-cable tensor service is session-authenticated plaintext, not
TLS. Cleanup verifies owned processes/groups/listeners, not GPU-driver internals.
