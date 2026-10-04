# Retune a saved mixed-precision model

Retuning measures the **current saved weights on the current Metal runtime**. It
changes execution dispatch, not quantization. It does not reallocate bits,
dequantize/requantize, cast the loaded model, or serialize new weight tensors.
Only exact Qwen2 SiLU gate/up pairs with saved affine 3-, 4-, or 6-bit weights
and group sizes 32, 64, or 128 are supported. Unsupported quantization modes,
conflicting metadata, incompatible pairs, and unverified MLP types are rejected.

## Run (exclusive GPU use)

From the repository, using its existing environment:

```sh
.venv/bin/python scripts/retune_model.py \
  --model artifacts/m2pro-mixed-v2-sealed/model \
  --output artifacts/m2pro-saved-retune-v1 \
  --repeats 20 --warmup 3 --min-improvement 0.05
```

Run only while other GPU work and performance trials are idle. `--output` must
be a **new, nonexistent directory**, not even an existing empty directory. The
model is copied to `<output>/model`; source and output must be disjoint. Source
files, directories, output paths, and their ancestors must not be symlinks.
Nonregular source entries are also refused. No source files are written.

Optionally provide `--calibration /absolute/path/calibration.json`, a local JSON
array of nonempty strings. Without it the shipped `data/calibration.json` is used.
There is no model download, dependency installation, forced-fusion option, or
precision-selection option. If publication fails, inspect the incomplete new
output directory; the tool will not overwrite it on another attempt.

## What is measured

1. Direct local MLX loaders restore stock Qwen2 modules. **Original dispatch is
   deliberately ignored**, including legacy dispatch and stale device/runtime
   choices. An existing schema-2 source's model/config SHA-256 binding must still
   match: retuning is not a way to repair a corrupt seal. Its old kernels and
   runtime are not required to match because those are being freshly profiled.
2. `calibrate_inputs` captures real layer inputs on the saved quantized model,
   with at most 128 tokens/text and the last 16 inputs/text. Modules are restored
   after capture. Model parameter objects and resident bytes are checked unchanged.
3. Every `rows_per_group` value in **1, 2, 4, 8** is numerically checked against
   stock affine gate/up on **all captured token inputs**, using `rtol=0.01` and
   `atol=0.02`. Nonfinite outputs, shape mismatch, tolerance failure, or kernel
   errors exclude that candidate. Errors are relative to the saved-weight stock
   operation, **not** an unquantized teacher or a model-quality metric.
4. Both stock and custom paths use the existing compiled functions with explicit
   dynamic weight arguments. Valid candidates and stock are tuned using
   `benchmark_functions`, then only stock versus the fastest valid fused launch
   receive a **separate fresh interleaved validation round**. Each phase uses the
   requested repeats and warmup. Timings use the last captured single-token input.
5. Fusion is selected only when the fresh candidate sample median is strictly
   below `stock_median * (1 - min_improvement)`. The default margin is 5%; an
   explicit `--min-improvement 0` still requires a strict win. Invalid/slow
   candidates leave stock selected. **Zero fused pairs is a valid result.**

These are synchronized wall-clock **local gate/up pair timings**, including host
execution/evaluation/synchronization, not GPU-only times. The margin is a heuristic,
not statistical significance. Tuning can overfit; fresh validation reduces that
risk but does not eliminate measurement noise. Single-token input choice, cache
state, concurrency, power, and thermal conditions can affect the result.

This does **not** profile custom down projections or attention and makes **no
whole-model speedup guarantee**. No full-model decode benchmark is performed.
Whole-model paired replay and quality checks are separate required evidence
before claiming an end-to-end benefit. Runtime fusion retains the existing stock
fallback for unsupported/non-single-token shapes.

## Output and provenance

- `<output>/model/`: byte-copied weights, config, tokenizer, and other source files.
- `<output>/model/execution_manifest.json`: a new strict **schema-2** execution
  manifest, produced by `create_manifest` and read back with `validate_manifest`.
  It binds the copied saved weights/config and shipped kernels to freshly observed
  device, MLX, and mlx-lm metadata.
- `<output>/retune_profile.json`: raw tuning and validation samples, medians,
  numerical errors for all four launches, per-pair dispatch, counts, capture
  counts/dtypes/input hashes, calibration text/settings/file hashes, fresh
  start/end environment, shipped kernel hashes, profiling source-code hashes,
  and original source seal/model binding.
- `<output>/provenance/source_execution_manifest.json`: exact original manifest
  bytes, when present. It is archived for lineage, never activated.

`source_file_sha256 == copied_source_file_sha256` verifies the **complete copied
snapshot before manifest replacement**. `source_payload_sha256 == new_payload_sha256`
verifies every final file except `execution_manifest.json`, including tokenizers
and all weight shards. `new_file_sha256` records final model files, including the
intentionally new execution manifest; therefore its manifest hash is **not**
claimed identical to the original. The exact original manifest and its hash
remain preserved in provenance. The source is rehashed before publication and
again afterward; a change aborts publication. Profiling source-code hashes and
hardware/runtime identity are also checked unchanged during the run. These are
integrity checks, not cryptographic author signatures or transaction-safe locking
against a concurrent adversarial filesystem writer.

## Tests

CPU policy, numerical diagnostics, local calibration, and output safety:

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch \
  .venv/bin/python -m pytest tests/test_retune.py -m 'not metal' -q
```

Opt-in tiny saved-Qwen GPU integration (actual 3-, 4-, and 6-bit kernels,
real captured inputs, byte-preserving copy, and strict manifest readback):

```sh
PARETOQUANT_RETUNE_GPU=1 TMPDIR=/Users/jeremylien/.hermes/cache/scratch \
  .venv/bin/python -m pytest tests/test_retune.py -q
```

Tiny random model/tokenizer fixtures test plumbing and numerics only. They are
not evidence of language-model quality or a real-model speedup.

## Recorded local run

The command above was actually executed on October 4, 2026 on Apple M2 Pro,
macOS 26.5.2, Python 3.13.2, MLX 0.32.3, and mlx-lm 0.32.0. Its artifact is
`artifacts/m2pro-saved-retune-v1/retune_profile.json`:

- All 24 pairs retained their saved precision: 16 at 3-bit, 8 at 4-bit.
- 18 fused pairs were selected (14 at 3-bit, 4 at 4-bit); 6 remained stock.
- 88 launch candidates passed numerical validation. All four launches failed
  tolerance on `model.layers.20.mlp` and `model.layers.23.mlp`, so those pairs
  remained stock; four other stock pairs did not clear the fresh 5% margin.
- Every pair was numerically checked on 64 captured inputs; every measured
  method retained 20 raw samples per phase after 3 warmups.
- The 260,563,712 resident parameter bytes, all model payload/tokenizer hashes,
  original manifest archive, strict schema-2 readback, and source-code hashes
  were verified unchanged/as bound.

This run's selected local pair median reductions ranged from 5.61% to 22.35%.
They are **not a whole-model speedup**. The recorded environment also reports
substantial swap usage; inspect the raw environment and replicate under your
own conditions rather than treating these launch choices as universal. No
custom down/attention or full-model decode benchmark was run here.
