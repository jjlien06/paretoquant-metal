# 013 BF16 TP investigation — candidate rejected

No tolerance changes, xfails, remote runs, full-model TP, commits, or changes to 011/012/014. All tiny GPU tests completed before the parent's bracketing benchmark notice; no further GPU tests were started.

## Preserved baseline

`baseline-archive-v1/` contains byte-exact copies of the original runtime, tests, README and complete evidence directory, with SHA256 manifest and read-only files/directories. Manifest verification passed; active original `evidence/` bytes also remain unchanged. All new numerical reports use exclusive-create writes in `evidence-v2/`.

## Identical-input local projection probe

Seed 731, quantized affine 4-bit 256-to-128 row projection, group size 32, BF16 inputs/scales/biases, one token:

| Against native unsharded BF16 | max abs | nonexact / 128 |
|---|---:|---:|
| BF16 partial matmuls + BF16 sum | 0.0078125 | 51 |
| Already-rounded BF16 partials + F32 sum, cast back | 0.0078125 | 51 |
| F32-input/scales/biases partial matmuls + F32 sum, cast back | 0.0078125 | 114 |

A partial differs between BF16 and F32 before reduction (max 0.005216836929321289). Thus casting an already-rounded output is too late, but this observation alone does not isolate rounding from matmul/dequantization arithmetic. F32 full versus F32 partial sum differs by only 1.1920928955078125e-7, with **zero** differences after BF16 cast. However, explicit `mx.dequantize` with BF16 versus exactly cast F32 scales/biases differs at 13668 entries, max 0.0001220703125. The higher-precision path is not arithmetically identical to native BF16. This probe does not establish the first divergent operator of the complete model or uniquely identify MLX kernel internals.

## Optional implementation

`--high-precision-row-reductions` enables `precision_v2.py`. Default native path is unchanged. It preserves all stored packed/floating parameter values and names; only row projections use F32 casts of input/scales/affine biases, F32 quantized matmul, F32 all_sum, then cast back. MLP down and FA output projections own their reduction/cast. GDN retains its native forward/reduction; its output projection returns F32 and a thin subclass casts only after the native reduction returns. Column projections, attention, recurrent kernels and caches are not promoted.

Unit test was observed red for missing precision module, then green (2 passed), including exact parameter and dtype preservation on loaded F32/BF16 fixtures. Original tests plus unit tests: **10 passed, 1 failed**, retaining original BF16 failure.

## Actual two-local-process original gates

Two ranks, GPU producers, scoped CPU collectives, five chunked-prefill/decode forwards, all logits and every cache state. F32 original atol=2e-5/rtol=2e-4; BF16 original atol=0.03/rtol=0.03.

| Mode | Dtype | Logit steps passed per rank | max abs | Cache checks passed rank0/rank1 |
|---|---|---:|---:|---:|
| Native | F32 | 5/5 | 1.5497207641601562e-6 | 50/50, 50/50 |
| Candidate | F32 | 5/5 | 1.5497207641601562e-6 | 50/50, 50/50 |
| Native | BF16 | 2/5 | 0.046875 | 50/50, 50/50 |
| Candidate | BF16 | 2/5 | 0.0400390625 | 50/50, **49/50** |

Both BF16 paths fail logit steps 1,2,4 (zero-based); candidate also fails rank1 step0 layer2 conv cache, max abs 0.032958984375. Initial verbal shorthand that every candidate cache passed was incorrect; parsed rank1 report exposes this failure. No shape/dtype comparison failed. Native/candidate source and rank-loading attestations match exactly. All ranks/modes generated [1,89,63], which does not override numerical rejection or prove single-host greedy equivalence.

Integration suite: **2 passed, 2 failed**. Candidate is rejected and must not be enabled for full-model TP. Source snapshot, process exit receipts, original tolerance results and programmatic summary are retained in `evidence-v2/`.

## Reproduction workflow

Use the repository `.venv/bin/python`. Run `repro_row_v2.py` only with a new output filename/directory (existing evidence deliberately refuses overwrite); then run `test_row_v2.py` and `test_integration_v2.py` with a fresh scratch basetemp. Integration uses fixed exclusive evidence filenames: archive or explicitly select a fresh version in the harness before rerunning. Keep native negative control visible. Do not rerun GPU work during other agents' performance trials.
