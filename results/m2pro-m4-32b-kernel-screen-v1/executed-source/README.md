# 008 — actual-weight 32B fused-kernel screen

CPU-tested harness; **physical-GPU verdict pending**. This does not change the
ParetoQuant kernel, packing, quantization, or checkpoint. It loads only the nine
layer-0 MLP arrays (~228 MiB), never the full model.

## Run on a physical Apple Silicon host

From the repository root, using the parent's existing model-runtime environment:

```sh
.venv/bin/python spikes/008-32b-kernel-screen/screen.py \
  --assets /absolute/path/to/qwen32-kernel-assets \
  --output /absolute/path/to/NEW-screen-evidence.json \
  --repeats 24 --warmup 3 --seed 0
```

The output directory must already exist. Existing files, directories, and dangling
symlinks at `--output` are rejected **before GPU imports**; exclusive creation also
prevents a race from overwriting evidence. No evidence file is read or reused.
Validation errors precede output creation. Later errors leave an explicitly
`status: error` JSON with the real exception, not invented benchmark samples.
Use a new output path for each run. No installations or downloads are performed.

`--help` and importing `screen.py` require only the Python standard library.
Successful execution additionally requires the existing NumPy, MLX, and
`paretoquant` environment. MLX and package GPU code are imported only after every
NPY hash, actual array shape/dtype, and raw-array hash has passed validation.

## Asset contract

`ASSETS/manifest.json` uses the parent's exact schema:

- `configuration`: exactly `{"bits": 4, "group_size": 64}`.
- `revision`: pinned 40-character lowercase hexadecimal commit.
- `source_config_sha256`: 64-character lowercase hexadecimal hash.
- `tensors`: exactly the nine original names
  `model.layers.0.mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`.
- Each entry has `file`, `shape`, `dtype`, `raw_sha256`, `npy_sha256`, and
  `original_name` equal to its mapping key. `raw_sha256` hashes C-order array bytes.
  Additional descriptive manifest fields (e.g. `model`, `layer`) are retained.

Only unique basename `.npy` files are allowed: no path components or symlinks.
The manifest is bounded to 1 MiB, each tensor file to 256 MiB, and all tensor files
together to 300 MiB. NPY loading uses `allow_pickle=False`; metadata cannot replace
actual dtype/shape checking. Dtypes are NumPy `dtype.str`, not human-readable names:

| Projection | Packed weight `<u4` | Scales and biases `<f2` |
|---|---|---|
| gate/up | `[27648, 640]` | `[27648, 80]` |
| down | `[5120, 3456]` | `[5120, 432]` |

## Predeclared screen protocol

- Three independent deterministic Gaussian input rows, seeds `seed`, `seed+1`,
  `seed+2`, each `[1, 5120]`. RMS is normalized to 1 in Python double precision
  **before FP16 rounding**; actual FP16 RMS is recorded.
- **SYNTHETIC diagnostic activations**, actual unchanged checkpoint weights.
  These are not captured model inputs, an end-to-end benchmark, or a 32B quality
  evaluation. Do not infer full-model speedup from this screen.
- Gate/up: native stock, equally compiled stock, compiled fused packed kernel at
  `rows_per_group=1/2/4/8`, using the existing public wrappers.
- Whole layer-0 MLP: native stock, **whole-MLP compiled stock**, and each
  **whole-MLP compiled fused** path. All use exactly the same stock
  `mx.quantized_matmul` down projection (4 bits, group size 64, transpose true).
  Compilation captures only the already validated arrays. No trace count is
  presented as a runtime call count.
- Before timing, every method is checked on all three inputs against both native
  stock and compiled stock: all elements finite, `rtol=0.01`, `atol=0.02`.
  Evidence retains failures, relative MSE, and max absolute error. Failed fused
  methods are never timed; a gate/up failure also excludes that candidate from
  MLP timing. Failed stock controls abort that stage's timing.
- All surviving methods are traced/evaluated during parity before warmup.
  Existing `paretoquant.benchmark.benchmark_functions` performs fresh-call,
  synchronized, rotating-order measurements; the harness also rotates the
  starting method between input rounds and phases.
- Separate **tuning** and **validation** calls each retain three rounds of 24 raw
  samples per method by default (72 per phase), with three warmups per round.
  Both stages test all surviving candidates, not only the tuning favorite.
  A stage's winner must pass every parity gate and have a pooled **validation**
  median latency improvement strictly greater than 5% over that stage's equally
  compiled stock. Tuning alone cannot select a winner. This threshold is a
  diagnostic rule, not a confidence interval or noise-significance claim.
- Measurements include Python dispatch, evaluation, and synchronization, not
  GPU-only device time. Native control timings remain separate in the JSON.
- Evidence includes the complete manifest and its hash, CLI arguments, actual
  environment via the existing helper, peak MLX allocation, and installed-source
  hashes for `metal.py`, all `kernels/*.metal`, `benchmark.py`, and `screen.py`.
- GPU is explicitly selected, cache is limited to 256 MiB, and the previous
  default device and cache limit are restored in `finally`. No wired-memory
  limit is changed.

## CPU verification

```sh
python3 -m unittest discover -s spikes/008-32b-kernel-screen -v
.venv/bin/python -m ruff check spikes/008-32b-kernel-screen
python3 spikes/008-32b-kernel-screen/screen.py --help
```

The standard-library suite uses tiny fixture bytes and explicit CPU test doubles,
not NumPy, MLX, checkpoints, or a GPU. It covers fail-closed metadata/path/hash/size
validation, actual-array rechecks, finite parity gates, independent unit-RMS
inputs, fair whole-MLP controls, raw-sample retention, validation-only selection,
GPU state restoration, and exclusive/error evidence behavior. Test-double timings
are only internal unit-test values and are never checkpoint benchmark evidence.

## Verdict: PARTIAL

CPU protocol/validation tests pass, Ruff is clean, and the real CLI help executes.
No GPU timing or correctness result has been produced by this CPU-only worker.
The parent must run this exact harness on the two physical hosts and inspect both
parity results and fresh validation samples before making a kernel decision.
