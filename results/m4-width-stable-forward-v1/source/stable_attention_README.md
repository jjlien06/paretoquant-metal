# Isolated slot-stable causal attention candidate

**Status: unadmitted candidate.** No numerical parity, GPU correctness, model
correctness, speedup, maximum context, or residency claim is established here.
This is the attention half of a possible batched-projection/shared-weight-read
candidate, not an admitted faster decoder. No existing runtime or public README
is changed; the component performs no global patching or decoder integration.

## Interface and contract

```python
stable_causal_attention(queries, keys, values, *, cache, scale, mask, sinks=None)
```

The caller has already applied Q/K RoPE and updated the native floating KV cache
**once for the full query width**. Pass the returned valid-length K/V tensors,
not the cache's possibly padded allocation. This component never updates,
trims, reads cache storage, or assigns its offset. It reads only the cache's
`offset` and checks for the unsupported `bits` attribute.

Validation precedes the lazy MLX import:

- Q shape `(1, Hq, L, D)`; identical K/V shapes `(1, Hkv, K, D)`.
- All dimensions positive Python integers (not bool), batch one, equal head
  dimension, `Hq % Hkv == 0`, `1 <= L <= 9`, and `K >= L`.
- Matching float16 or float32. Metadata accepts exact `float16`, `float32`,
  `mlx.core.float16`, and `mlx.core.float32` names. The wrapper uses `str(dtype)`;
  aliases, suffix matching, float64, bfloat16, and mixed precision are rejected.
- A non-None cache without `bits`, whose Python-int `offset` equals `K`.
  Quantized caches and rotated/truncated histories with `offset != K` are
  rejected. This is a layout/offset contract, **not** a class-identity or tensor
  contents proof; callers must provide the native floating-cache outputs in
  chronological order, not a misleading metadata-compatible object.
- `scale` is a positive finite Python int/float, never bool. Integers too large
  for finite native-real conversion are rejected, rather than reaching MLX.
- `sinks` is strictly None. For `L >= 2`, `mask` is exactly the built-in string
  `'causal'`, not a string subclass, array mask, or None. For `L == 1`, None or
  exactly `'causal'` is allowed.

Invalid metadata raises `ValueError` before `_load_mx()` runs. The stdlib-only
`plan_causal_prefixes(...)` validates the same metadata used by the wrapper and
returns each exclusive prefix end; no array values are inspected.

For query position `i`, prefix end is `K - L + i + 1`. At widths 2–9:

1. Make `queries[..., i:i+1, :]` contiguous.
2. Slice `keys[..., :end, :]` and `values[..., :end, :]`.
3. Call **`mx.fast.scaled_dot_product_attention` directly**, with the supplied
   scale, `mask=None`, and `sinks=None`.
4. Concatenate the outputs along `axis=-2`.

The direct native call avoids recursion if a parent experiment temporarily
patches a Qwen2/base attention helper. At width one, delegate one original
native call with the original arrays, original scale, and **supplied mask**;
no slicing, contiguous conversion, or concatenation occurs.

## Evidence boundaries

`test_stable_attention.py` is CPU-only. Import guards block MLX, MLX-LM, and
NumPy through both standard imports and importlib, and assert they are not
loaded. Lazy-loader tests deliberately attempt imports that the guards stop;
no actual runtime/model module is imported. Tests exercise real metadata
validation and the real wrapper. Scoped replacement of the private `_load_mx`
uses unavoidable slicing/call doubles with no numerical values. These tests
prove **structural prefix selection, native call shape, output order, width-one
passthrough, dtype-name handling, and cache immutability**, not attention math.
Covered layouts include GQA ratios 7 and 5 and head dimensions 64 and 128.

`stable-attention-TDD.json` retains actual command outputs from nine vertical
RED→GREEN slices plus final regressions and Ruff checks. Initial failures are
intentional feature/contract failures, not fabricated output. No GPU, network,
SSH, model execution, dependency installation, or Git write is involved.

Run only the assigned CPU module from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest -p no:cacheprovider -q spikes/010-width-stable-verification/test_stable_attention.py
.venv/bin/ruff check --no-cache spikes/010-width-stable-verification/stable_attention.py spikes/010-width-stable-verification/test_stable_attention.py
.venv/bin/ruff format --no-cache --check spikes/010-width-stable-verification/stable_attention.py spikes/010-width-stable-verification/test_stable_attention.py
```

## Required admission gates (pending; parent-owned)

1. Real MLX 0.32.3 / MLX-LM 0.32.0 GPU replay with held-equal genuine checkpoint
   Q/K/V: compare each query's native width-one baseline across widths 1/2/5/9,
   cache-prefix lengths, float16/float32, GQA ratios 7/5, and dimensions 64/128.
   Record bit-exactness separately from any unchanged numerical tolerance;
   validate slicing/contiguity does not change the intended arithmetic route.
2. Native cache/storage/offset integrity, valid-length outputs, repeated calls,
   and future-token isolation on actual arrays, including explicit rejection
   of unsupported cache/mask/sink configurations.
3. Parent-controlled isolated teacher/decoder integration together with the
   independent slot-stable projection candidate. Replay genuine layer outputs,
   natural caches/logits, greedy outputs, and verifier decisions against the
   fixed stock width-one baseline. Batched QKV projection differences are an
   independent issue; this attention component alone cannot resolve them.
4. Whole-pipeline, synchronized, warmed-up, matched-work GPU benchmarks on the
   intended models/hosts. Include all attention calls, projection, verification,
   cache handling, allocation, and graph evaluation; report memory and throughput
   alongside correctness. No local microbenchmark establishes decoder speed.

Overhead risks: L native SDPA dispatches instead of one, repeated prefix reads,
contiguous query materialization, concatenation/allocation, and graph-building
costs. No prefix reuse or fused attention kernel is implemented. Dispatch and
memory costs may outweigh any projection/shared-weight-read benefit; width-one
passthrough does not imply wider calls are faster. No admission until these
numerical and end-to-end gates pass.
