# Slot-shared single-token-arithmetic affine QMV candidate

**Independent array-kernel experiment only. Not admitted to a decoder. Metal has
not been compiled or run. No numerical-parity or performance claim.**

## Interface and admission

```python
slot_qmv(x, packed_weights, scales, affine_bias, *, bits, group_size=64)
```

- `x`: `(T, K)` or `(1, T, K)`, `1 <= T <= 9`. Width one is an oracle/control;
  weight sharing is intended for widths 2–9.
- Packed weights: exactly `(N, K * bits // 32)`, uint32.
- Scales and **quantization affine bias**: exactly `(N, K // group_size)`, each
  exactly the input dtype, float16 or float32.
- `bits`: built-in int 3 or 4; group size: built-in int 32, 64, or 128.
  Booleans, floats, strings, int subclasses, nonpositive/noncanonical dimensions,
  K not divisible by the group size, N<8, and N not divisible by 8 are rejected.
- All metadata shapes are tuples of positive built-in ints. Indexing is
  conservatively limited to signed 32-bit expressions: `K*bits`, packed matrix
  bytes, affine group count, input elements, and output elements each <=2^31-1.
- Output: `(T, N)` or `(1, T, N)`, preserving input rank and dtype. The Metal
  memory contract is flattened slot-major `(T, N)`; singleton batch is only shape.
- Logical computation is affine `x @ dequantize(packed_weights, scales,
  affine_bias).T`, but the kernel intentionally does **not** dequantize first.
- No transpose/mode/batching options, ragged rows, fallback, same-token activation
  fusion, learned output-bias fusion, or change to activation/attention semantics.
  A projection's separate learned output bias remains an external native NN add.
- `shape_plan(ArrayMeta(...), ...)` is pure stdlib host admission used by the real
  wrapper. Public array metadata is admitted **before** the lazy `_kernel()`
  getter imports MLX. Admitted objects must then actually be MLX arrays before
  launch. The generated-kernel builder uses `ensure_row_contiguous=True`, so
  noncontiguous logical arrays rely on MLX's explicit materialization contract,
  not an unhandled custom stride path; its costs/behavior need GPU validation.

## Arithmetic and intended sharing

Packing factor is eight for both admitted bit widths; bytes per pack are three
or four. Fast eligibility is exactly `N % 8 == 0 and K % 512 == 0`: two packs per
lane, 512-value K blocks. Otherwise use one pack per lane and 256-value blocks.
The slow loop processes `k < K - block_size` with ordinary helpers, then uses
safe vector/dot helpers for the final step **even when that step is a full block**.

Each thread has one stock-QMV accumulator sequence per slot/output row. It stages
raw packed bytes once per row/K step into aligned thread uint32 storage, then
uses the same bytes and affine parameters for every slot. Original input-T group
sum expressions are copied unchanged (no eager float casts of every input), with
float vector/dot/accumulator storage. Integer dot is combined as
`scale * accum + sum * affine_bias`; reduction remains 32-lane `simd_sum` followed
by the original T cast. See `ATTRIBUTION.md` for exact source pins/adaptations.

Threadgroup `(32, 2, 1)` and grid `(32, N//4, 1)` produce one threadgroup per eight
output rows. Lane is threadgroup-local x; SIMD group is local y. All slots are
handled by that same group, not a slot grid axis. Source staging is an intention,
not evidence that the compiler avoids reloads or register spilling.

## CPU verification

From `/Users/jeremylien/Projects/paretoquant-metal`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts= -p no:cacheprovider --confcutdir=spikes/010-width-stable-verification spikes/010-width-stable-verification/test_slot_qmv.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python spikes/010-width-stable-verification/cpu_smoke.py
.venv/bin/ruff check --no-cache spikes/010-width-stable-verification/{slot_qmv.py,test_slot_qmv.py,cpu_smoke.py}
.venv/bin/ruff format --check --no-cache spikes/010-width-stable-verification/{slot_qmv.py,test_slot_qmv.py,cpu_smoke.py}
```

`TDD_EVIDENCE.jsonl` contains actual command arguments, working directory,
exit codes, stdout, and stderr from **18 vertical RED→GREEN behavior slices**.
Each RED was executed before its production change; each GREEN ran the accumulated
CPU suite. Later regression-only tests added coverage without production changes.
The fake runtime tests exercise host admission, generated-kernel construction,
and launch arguments only; they cannot prove Metal arithmetic or GPU behavior.
`cpu_smoke.py` additionally runs without pytest. The final verification artifact
and SHA manifest give final counts and file hashes.

All writes in this task are new files in this directory. The pre-existing dirty
repository was not repaired, staged, or committed. No actual MLX, NumPy, model
imports, GPU use, SSH, network calls, installs, downloads, credentials, or runtime
modification occurred. Only this isolated CPU suite ran; the wider project suite
may import array runtimes and was deliberately excluded.

## Localization boundary (parent-provided, not measured by this worker)

The parent's held-equal actual-checkpoint SDPA replay covered four cases / 24
layers with same-mode cache/logit witnesses. Even with identical Q/K/V, width 2
SDPA was bit-exact throughout, width 5 independently failed the existing .002
threshold in many layers, and width 9 differed but remained within .002 throughout.
**A projection-only fix is therefore insufficient for full verifier admission.**
A separate worker owns slot-stable causal attention and its own GPU gate. This
candidate supplies no attention component and makes no claim that fixing QMV
alone repairs full KV/logit/acceptance gates. The parent must evaluate those gates
with a separately admitted attention component where needed. Its reserved files
are untouched by this worker.

## Exact acceptance gates — ALL PENDING, parent-owned

1. **Audit and compile:** independently audit pinned helper equivalence and both
   QMV schedules; compile on each target GPU/runtime. Exercise bits 3/4, group
   sizes 32/64/128, float16/float32, widths 1–9, both admitted ranks, and contiguous
   versus noncontiguous views. Include K=32/64/128 with compatible group sizes,
   slow full-last-block K=256/768/4864, partial tail K=384/896, fast K=512/5120/27648,
   N=8 and real projection/head sizes. Verify bounds, alignment, lane/SIMD mapping,
   dtype, and shape. Reject any unhandled case; do not hide it behind a fallback.
2. **Array numerical gate:** compare each candidate slot against the same input
   passed to native single-token affine `mx.quantized_matmul` (not width-dependent
   native wide/QMM). Require bit-exact width-one control and slot-versus-width-one
   candidate stability across widths 2–9 for deterministic finite fixtures and
   genuine checkpoint projection inputs. Test extremes/zeros and representative
   affine biases; audit any compiler-induced arithmetic change. Preserve the
   parent's existing elementary rtol=0.002/atol=0.002 gates as additional checks,
   never loosen them to obtain a pass.
3. **Real checkpoint teacher gate (requires separately gated attention where
   indicated by the localization above):** parent alone may replace teacher-verification
   affine quantized-matmul calls in an isolated harness, leaving separate output
   biases, all other operations, and decoder/runtime untouched. Replay the four
   genuine checkpoint cases from the localization work and cover widths 2, 5,
   and 9 for both prompts; include the real small mixed 3/4-bit checkpoint
   (hidden 896, intermediate 4864, head 151936) and full 32B stock 4-bit gs64
   (hidden 5120, intermediate 27648, head 152064). Observer-instrumented and
   uninstrumented cache/logit outputs must remain bit-exact within each mode.
   Every layer's complete KV gate must pass unchanged rtol=0.002/atol=0.002;
   validate full logits, acceptance decisions, and replayed verifier trajectories,
   not merely selected tokens or the first layer. Existing six-case batched and
   six-case width-one verifier suites must retain their established semantics;
   all candidate cases must pass. Any mismatch blocks admission.
4. **Measured benefit:** only after numerical gates, parent benchmarks warm,
   synchronized repeated runs on both target hosts and the full checkpoint.
   Compare stock batched verification and serialized native width-one controls
   at matched inputs/cache state; report kernel and end-to-end accepted-token/s
   distributions including launch/materialization overhead. Inspect register
   pressure/spills and actual memory traffic to corroborate intended weight reuse.
   Require repeatable latency/accepted-throughput improvement with unchanged
   acceptance and numerical gates. No source-level load count is a speed result.
5. **Deployment decision:** parent audits and explicitly approves before any GPU
   deployment or decoder integration. This folder supplies no adapter, patch,
   decoder hook, checkpoint loader, or deployment automation.
