# Qwen3.8-27B real-activation mixed 3/4-bit quality study

Owner: Jeremy. Date: 2026-11-06. Source: `spikes/033-qwen38-mixed-3-4-real/`.

## Result

Completed real-activation calibration, exact ParetoQuant per-layer allocation,
two mixed text-checkpoint builds with source-byte checks, and matched held-out
perplexity for uniform4 / both mixed plans / uniform3 on the M4 mini. These are
bounded quality/storage results, **not measured decode acceleration** or a general
quality benchmark. No two-Mac mixed-checkpoint inference was attempted.

| Checkpoint | Perplexity | Absolute delta vs q4 | Relative delta vs q4 | Mean NLL | Scored tokens |
|---|---:|---:|---:|---:|---:|
| Uniform 4-bit | 12.560952 | +0.000000 | +0.000% | 2.53059298 | 6,120 |
| Mixed target 0.5 GiB | 12.667576 | +0.106624 | +0.849% | 2.53904568 | 6,120 |
| Mixed target 1.5 GiB | 12.981188 | +0.420235 | +3.346% | 2.56350120 | 6,120 |
| Uniform published 3-bit | 14.307830 | +1.746878 | +13.907% | 2.66080696 | 6,120 |

Perplexity is exp(mean next-token negative log likelihood); lower is better.
The source uniform3 baseline also demotes embedding and output head; mixed plans
keep them q4. Therefore mixed-vs-uniform3 differences do not isolate decoder-layer
selection alone. No equal-budget random/synthetic-ranking baseline was evaluated,
so this is **not proof that the allocator beats another selection policy**.

## Amended Stage 1 gate and original failure

Jeremy explicitly authorized a BF16 fidelity gate: every per-position argmax ID
identical, maximum absolute logit difference <=0.125, mean absolute difference
<=0.001; report per-layer hidden errors. The original strict failure is preserved:
17-token control, max difference 0.0625, mean 0.000583972549, argmax all matched,
failed atol=rtol=1e-5 (`remote-evidence/small-parity-r2.json`). It was not relabelled.

Cheap fix: `stream_v2.py` projects all positions together, matching the reference
row count; decoder projections already process each complete 256-position sequence.
On the native control model, lengths **17, 64, 256 all became bit-exact**:
max/mean logit differences 0; every per-position argmax identical; every one of
24 layers had relative hidden RMSE=0 and max hidden difference=0 at every length.
The separate instrumented hidden-capture witness also matched pristine native
logits exactly. BF16 output/hidden dtype was asserted. Thus the fixed candidate
passes the original strict gate as well as the explicitly amended gate on these
control cases. Full 27B native forward was not possible alongside chat and was
not claimed. See `small-parity-v2.json`, `gate-amendment.json` and
`continuation-evidence/small-v2.log`.

## Method / exact source identities

- q4: `mlx-community/Qwen3.8-27B-4bit` revision
  `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
- q3: `lukaskremla/Qwen3.8-27B-3bit-MLX` revision
  `b3503b31b36125107e12a2b57fbe690c37bfc053`.
- Native `mlx_lm.models.qwen3_5` text classes; hybrid `DecoderLayer` chooses GDN
  or full attention. Native SSM/causal attention mask helpers, positions reset per
  independent sequence, no cross-sequence cache, native evaluation mode. Model
  arithmetic was not reimplemented. MLX 0.32.3 / MLX-LM 0.32.0 on the mini.
- Each process keeps only one decoder layer resident, evaluates output before
  releasing its weights, and carries the hidden tensor forward. It does not
  construct/evaluate the full 27B model. Embedding and final head load separately.
- Both current source checkpoints' **complete shard sizes and SHA-256** were
  independently checked against their pinned HF LFS metadata after evaluation.
  Both have 2,180 source tensors; text-only models use 1,847. All 353 decoder/final
  norm floating tensors were byte-identical between sources. There are 496
  quantized decoder linear modules, with affine group64 layouts.

The q3 artifacts were downloaded once (12,719,810,876 bytes including tokenizer
and metadata) and retained on the mini. This work uses source checkpoint tensors;
it does not claim to have designed the upstream 3-bit packing or independently
verified BF16 requantization equivalence for every tensor.

## Real-activation sensitivity

Pinned dataset `Salesforce/wikitext` revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, WikiText-2 raw:
first **3,072 train tokens**, 12 independent 256-token windows. Tokenizer is the
project's `models/qwen3.8-27b-4bit-tokenizer-only/tokenizer.json`, SHA-256
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

For each original decoder layer i, compute q4 output on actual q4 hidden states
h_i, then substitute q3 **quantized linear tensors only**, keeping the same q4
floating parameters and identical h_i. Record

    sensitivity_i = sum || layer3_i(h_i) - layer4_i(h_i) ||^2
                    / sum || layer4_i(h_i) ||^2

Only q4 outputs advance calibration to the next layer. No Gaussian proxy,
Fisher approximation or final-logit KL is claimed. Relative-to-residual error
and raw numerator/denominator are also retained.

| Sensitivity statistic (relative MSE) | Value |
|---|---:|
| Minimum | 0.0012270420 |
| Median | 0.0059288806 |
| Maximum | 0.0158638556 |
| Max/min ratio | 12.928535 |
| 90th/10th percentile ratio | 5.147802 |
| Coefficient of variation | 0.537598 |

Predeclared spread gate: interdecile ratio >=1.5 and CV>=0.15: **passed**.
Least sensitive five layers: [1, 15, 13, 11, 14].
Most sensitive five layers (descending): [63, 51, 50, 62, 59].
Full 64-layer rows: `sensitivity-v2.json`; raw incremental rows and pressure log
are preserved under `continuation-evidence/`. This local error ranking is a
calibration heuristic, not proof of downstream task importance or additive quality.

## Exact ParetoQuant allocation / storage

The project's unchanged `paretoquant.allocator.allocate` receives 64 units, each
with {q4,q3} choices. q4 loss=0; q3 loss is the measured relative layer MSE;
bytes are actual packed weight/scales/biases payloads from downloaded headers.
All non-quantized tensors, embedding and head are fixed q4. Exact frontier limit
10,000; **all four runs completed exactly**, no approximation or timeout fallback.

Uniform-q4 text tensor payload: 15,132,802,048 bytes.
Headers/tokenizer assets are excluded from these parameter-payload totals.

| Target savings GiB | q3 layers | Actual bytes saved | Actual GiB saved | Mixed text payload bytes | Predicted decode speedup | Built/evaluated |
|---|---:|---:|---:|---:|---:|---|
| 0.5 | 12 | 570,716,160 | 0.531521 | 14,562,085,888 | 1.041216x | yes |
| 1.0 | 23 | 1,094,901,760 | 1.019707 | 14,037,900,288 | 1.082183x | no |
| 1.5 | 34 | 1,620,459,520 | 1.509171 | 13,512,342,528 | 1.126626x | yes |
| 2.0 | 45 | 2,148,761,600 | 2.001190 | 12,984,040,448 | 1.175139x | no |

Prediction = q4 decode payload / mixed decode payload, including decoder floats,
final norm and q4 output head, excluding embedding read. It assumes byte-bandwidth
limitation; it is **not measured tokens/s**, and ignores kernel overhead,
communication, MTP acceptance and workload effects.

All selected layer lists and exact states-considered counts:
`allocator-plans-v2.json`. Build 0.5 GiB-target layer list:
[1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16].
Build 1.5 GiB-target layer list:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 29, 30, 38, 41, 42, 44, 45, 57].

## Mixed checkpoint assembly / reload

Both are text-only, one safetensors shard per decoder layer plus separate
embedding/head/norm shards. Copy uses <=8 MiB CPU chunks, no weight conversion.
`config.json` retains affine global defaults and explicit module-path overrides.
The builder checks actual output headers and independently reads back **every
one of 1,847 tensor hashes** against the selected source bytes. Evaluation then
reloads every tensor using its per-module bit setting and native quantized MLX
classes, and its actual loaded tensor hashes match the build manifest and source
uniform-checkpoint evaluation hashes. This is layer-streamed reload/inference;
no monolithic full-27B reload or two-host runner admission was attempted.

Mini checkpoint paths:

    /Users/jeremylien/Projects/paretoquant-dual-host/spikes/033-qwen38-mixed-3-4-real/checkpoint-mixed05
    /Users/jeremylien/Projects/paretoquant-dual-host/spikes/033-qwen38-mixed-3-4-real/checkpoint-mixed15

Manifests/configs/indexes copied locally under
`continuation-evidence/checkpoint-mixed05/` and `checkpoint-mixed15/`.
Weights remain on the mini, not in local git.

## Quality protocol and limitations

Held-out **test split** first 6,144 tokenizer IDs, disjoint source artifact from
train calibration. Exactly 24 independent nonoverlapping windows ×256 tokens;
reset GDN/attention positions/state between windows, no BOS, score positions1..255
using preceding tokens: **6,120 scored targets per checkpoint**. Head projects
all 256 positions together; NLL is accumulated in F32 from BF16 model outputs.
The same actual token digest and sequence population were checked for all four
runs. This is a capped short-context WikiText prefix, **not canonical full-corpus
WikiText perplexity**, held-out task accuracy, long-context evaluation or a
reliability guarantee. Natural-language phrases can repeat across dataset splits.

No speed conclusion follows from these layer-streamed wall times, which include
repeated CPU file loading and are deliberately not throughput benchmarks.

## Evidence, resources and cleanup

Independent post-hoc checker `assure_v2.py` returned an explicit **passed** verdict:
`continuation-evidence/final-assurance-v2.json`. It recomputed exact allocation,
checked coverage/counts/digests, reconciled calibration/source/build/loaded tensor
hashes, verified source LFS shards and checked guarded process exits. This audit
supplements, rather than replaces, physical small-model numerical admission.
The initial reviewer held claims until these linkage/count checks were available.
The strengthened checker and actual receipts received an independent **APPROVE**
for the bounded storage/PPL claims (`independent-review-v2.txt`), with no
measured-speed or general-quality claim. That reviewer did not rerun hardware
checks and excluded cleanup, which is verified separately below.
The 14 allocation/build CPU tests and 4 AST-extracted evaluation-contract tests
passed; these synthetic test fixtures are not counted as model measurements.

Heavy MLX work ran only on the M4 mini. The watchdog permitted pressure1/2 and
aborted on other/invalid readings, missing RSS, deadline, excessive gaps or sampled
RSS>=3 GiB. Each heavy step invokes `memory_pressure` before loading. No concurrent
experiment GPU workers were used; the pre-existing live chat remained untouched.
Maximum actor-reported process RSS high-water across calibration/evaluations/control:
**1,800,257,536 bytes (1.677 GiB)**. Maximum watchdog-sampled RSS:
1,454,014,464 bytes (1.354 GiB). Peak MLX allocation:
1,159,498,754 bytes (1.080 GiB), not total system residency. Monitoring
is sampled evidence, not continuous proof or a crash-immunity guarantee.

Both final cleanup readbacks found no owned experiment processes. The pre-existing
chat ranks remained present and local TCP8780 still listened. Receipts:
`cleanup-mini-final.json`, `cleanup-m2-final.json`. No unrelated processes killed,
no git commit. All added repo artifacts are under the requested new spike/results
directories. Earlier failure reports, logs and status.json were retained rather
than overwritten as successful historical calls. `final-summary-v2.json` is the
current machine-readable result.

## Remaining work

1. Two-Mac mixed-checkpoint numerical admission and timing with contemporary q4
   controls; no reuse of archived peak rates as speed evidence.
2. Actual per-rank stored bytes, replicated embedding/head costs and placement
   admission under live memory pressure.
3. Larger/varied quality population and equal-budget selection-policy controls;
   decoder-only-q3 (fixed q4 embedding/head) comparator if isolating selection gains.
4. Native full-load/token-generation integration when resource availability permits.

Reproducibility entry points and exact command arguments are in the new spike's
`README.md`; deployed source/data hashes and copied remote evidence transfers are
bound by their JSON manifests.
