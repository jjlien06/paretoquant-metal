# Full 32B grouped-evaluation pilot — October 5, 2026

## Result

Opt-in local GPU evaluation blocks reduce synchronization boundaries without changing
checkpoint tensors or upstream model mathematics. Eight-layer blocks completed genuine
full Qwen2.5-32B-Instruct-4bit generation across the same physical M2 Pro / M4 hosts.

| Variant | Warm samples | Median post-first token events/s | Median warm request wall, s | First request TTFT, s |
|---|---:|---:|---:|---:|
| Stock scheduling, before | 3 | 5.614 | 8.266 | 57.817 |
| Group 4, earlier pilot | 2 | 6.038 | 7.740 | 33.277 |
| Group 8 | 3 | 6.270 | 7.451 | 28.848 |
| Stock scheduling, after | 3 | 5.486 | 8.436 | 40.614 |

Group 8 measured **6.270 events/s** against a pooled
**5.573 events/s** for bracketing stock-scheduling
controls: **12.51% observed higher warmed decode throughput**.
Against the before/after control medians separately, the observed gains were
11.67% / 14.29%.
The earlier group-4-vs-original-baseline 18.5% observation was unpaired and used an
older/slower control; it is not the final comparison here.

This is a short-prompt, separate-process, bracketed pilot, **not a randomized,
in-process paired benchmark or a statistical/reliability guarantee**. Three warm
group-8 samples and six warm bracketing controls are insufficient for universal
claims. There is no speedup claim over single-host MLX, vLLM, or TensorFlow.
The control is this same two-host runner with `eval_group_size=1`, not an independent
upstream native asynchronous generation engine.

## Identity and correctness gates

- Pinned revision: `2938092373e5f97b95538884112085364c2da315`.
- Split remains `[26, 38]`: M2 layers 38–63; M4 layers 0–37.
- Same prompt: `What is 2 plus 2? Explain briefly.`; greedy generation, cap 64.
- All **15 requests** in the four archived full-32B trials produced the same 41 token
  IDs including EOS as the published original completion, on both ranks.
- Fresh KV cache for every request; no prompt-prefix KV reuse. Weights remain loaded
  between requests within each bounded process; no persistent serving product is claimed.
- Every per-rank tensor digest matches the published original; trusted completeness,
  replicated equality, exact source agreement, actor cleanup and closed listeners passed.
- All original owned layer indices have successful completion evidence. Group 8
  includes final ragged blocks and evaluates every live block KV state plus hidden output.
- Peak MLX allocation for group 8: M2 **7.547 GiB**, M4 **10.618 GiB**. This is not
  OS residency, process RSS, proof of no swapping, or an automatically pooled 32 GB.
- Native tiny packed-FP16 GPU and CPU fixtures cover logits/cache parity for groups
  1/2/4/8; physical small group-4 and group-8 controls match existing single-host tokens.
- Equal greedy tokens for this prompt are not a 32B quality/perplexity evaluation.

## Scope and limits

`--eval-group-size 8 --execution phased` enables the measured path; default is still
1, with admission restricted to exact integer groups 1/2/4/8. Communication is eagerly
resolved outside each local GPU graph. Grouped `layer_done` events certify completion
only; they do not provide individual layer timings within a block.

The faster path uses **stock upstream quantized operators**, not custom gate/up fusion,
requantization, or the project's compiled full-model decoder. The separate actual-weight
kernel screen found no admitted fused MLP candidate on either physical GPU:
[negative kernel evidence](../m2pro-m4-32b-kernel-screen-v1/README.md).

Cold first-use latency remains large and varied. Do not attribute the different cold
TTFT values to grouping: OS paging, process start, driver/kernel first use and thermal
conditions were not isolated. Warm reuse is separate from scheduling improvement.

## Evidence and verification

`grouped-comparison-summary.json` has all ranks' cold/warm rows, unchanged-weight/token
assertions, source agreement, exact definitions and limitations. For each warm request,
post-first throughput is `(token events including EOS - 1) / (wall - TTFT)`.

`group4/`, `stock-before/`, `group8/`, `stock-after/` retain original receipts, exact
source hashes and verified cleanup. `small-control/`, `small-group8/` retain physical
controls. `executed-source/` is the independently approved 20-file snapshot;
`review/verdict.json` approves scheduling/admission, not speedup or full-model quality.
`integrity.json` remains immutable; `integrity-v2.json` hashes the later additions.

Parent combined gate: **778 passed, 15 explicit optional skips, 65 subtests**;
Ruff lint passed across core/tests/new spikes; all 24 spike Python files pass formatting.
Five unrelated core/test format failures are byte-exact to HEAD and were left untouched.
Details: `parent-quality-gate.json`. Optional integration skips are not counted as passes.
