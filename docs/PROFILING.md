# Eager decode operation diagnostics

`scripts/profile_decode.py` profiles one strictly admitted, local saved Qwen2
model. This is a **bottleneck investigation tool**, not a whole-model cost
attribution or a speedup claim. No model is downloaded and no server is started.
Run it sequentially, never alongside another MLX/Metal experiment.

## Run

From `/Users/jeremylien/Projects/paretoquant-metal`, with an existing output parent
and a **new** output filename:

```bash
.venv/bin/python scripts/profile_decode.py \
  --model artifacts/m2pro-saved-retune-v1/model \
  --output artifacts/m2pro-decode-diagnostics-v1.json \
  --prompt 'Explain binary search briefly.' \
  --decode-steps 4 --capture-step 0 \
  --repeats 3 --warmup 1 --full-repeats 2 --full-warmup 1
```

Start small when memory/swap pressure is high. Defaults are 16 decode steps,
10 isolated samples, 2 isolated warmups, 3 full-model samples, and 1 full-model
warmup. Later steps can be investigated with `--capture-step` (zero-based).
Only one activation/context snapshot is retained per invocation.

`--stock` suppresses fusion installation, **not** manifest validation. A saved
mixed-precision stock model still needs its valid, bound execution manifest.
Separate stock/fused runs generate their own schedules; they are not a paired
speed comparison unless the retained schedules and conditions actually match.

## Admission and evidence safety

- Require an existing local model directory, config, safetensors weights, and
  schema-2 execution manifest. No legacy admission, remote model identifiers,
  symlinks, remote tokenizer files, or custom remote Python code.
- Strictly validate manifest artifact/kernel hashes and hardware/MLX/mlx-lm
  identity **before loading**. Require exact installed mlx-lm Qwen2 `Model`,
  validate loaded projection precision, then install the admitted dispatch.
  Errors fail closed; there is no silent stock fallback.
- Apply the saved source tokenizer's user chat template with
  `tokenize=False, add_generation_prompt=True`, then its `encode` method, as
  used by the existing runtime CLI. Record the user text, rendered template,
  options, and every prompt token ID. An overlong prompt fails; it is not cut.
- Record the greedy schedule, every fed token ID, decode-step counts, captured
  step/token, and context length before that step. Greedy schedule creation
  does not stop at EOS; replay feeds every listed token after prefill.
- Hash every source-model file, including config, all weights, manifest,
  tokenizer, and provenance. Hash the profiler, runner, runtime dependencies,
  kernels, and installed Qwen2/cache/activation source. Recheck both sets after
  measurements and refuse to write if anything changed.
- Collect begin/end environment metadata (device/runtime, time, swap and
  `vm_stat`); do not invoke git. Hashing itself can affect page-cache pressure.
- The output must be outside the model directory, with an already-existing
  parent. Serialize finite JSON first, then create the final file using
  exclusive `open("x")`, flush and fsync. Never overwrite existing evidence.
  This is exclusive creation, **not transactional publication**: interruption
  or disk failure during the final write can leave a new partial file.

Integer limits: decode steps 1–512; isolated repeats 1–200; isolated warmup
0–50; full repeats 1–20; full warmup 0–10; max prompt tokens 1–32768;
capture step 0 through decode steps minus one. Booleans are not accepted as
integers by the Python API. Prompt plus decode schedule must fit the configured
model context limit.

## What is measured

1. Generate the schedule on the admitted model.
2. Measure the **uninstrumented** complete cached decode using the existing
   `cached_decode_benchmark`: separate prefill/decode samples, fresh prompt
   caches per trial, teacher-forced tokens, no token-sampling time.
3. Prefill a fresh standard KV cache and replay up to the selected step.
4. Capture actual eager module inputs at that step. Capture evaluates inputs
   and relevant cache state at module boundaries, so it is **intrusive** and
   alters lazy graph execution/fusion boundaries.
5. Restore all instrumented module classes, then benchmark isolated operations
   using fresh calls with evaluated inputs, warmups, rotating trial order,
   `mx.eval`, and synchronization. Keep every timing sample and trial order.

The instrumentation uses temporary per-instance Python subclasses. Hook
closures never enter the MLX module/parameter dictionary; weights and module
objects are not replaced. A `finally` restores original classes even if setup,
forward, evaluation, or capture fails. Concurrent forwards or nested profiling
on the same model are unsupported.

### Diagnostic groups

| Group | Boundary / replay |
|---|---|
| `attention_inclusive` | Actual attention input, original attention forward, including q/k/v/o, RoPE, SDPA and KV update. Every call reconstructs an independent pre-call standard `KVCache` snapshot at the original offset/capacity; setup overhead is included. |
| `attention_projection` | q, k, v and o projection calls separately, using their actual captured inputs. |
| `mlp_gate_up` | Eager stock Qwen2 SwiGLU of gate/up on actual post-attention-norm input. For an installed fusion that really meets the runtime single-token/dtype dispatch condition, also replay the compiled custom gate/up on exactly that input and dynamic weights. |
| `mlp_down` | Original down projection on its actual activation (including the custom gate/up activation on fused layers). |
| `embedding` | Actual single-token embedding lookup. |
| `output_projection` | Untied `lm_head`, or the tied embedding's `as_linear` method. |
| `norm` | Input, post-attention and final RMS norms, separately. |

The stock gate/up comparator is the installed Qwen2 eager SwiGLU path; the
custom path uses the existing compiled fused operation because that is what
`FusedMLP` dispatches. Compilation/dispatch policies are therefore those of the
actual implementations, not artificially matched microbenchmark policies.
Only the manifest-selected custom launch is timed: this tool does not retune,
change precision, install new dispatch, or certify custom numerical accuracy.

## Interpretation

`isolated_diagnostics.operations` retains each operation's group, input
shape/dtype, dispatch/cache details, and raw synchronized wall-clock samples.
`group_members` is an index, not a group cost. `ranked_isolated_operations`
sorts individual isolated medians to suggest where to investigate next.

**Never sum these medians or divide them by full-model time to assert a
whole-model fraction.** Inclusive attention overlaps its projection samples;
stock/custom gate/up are alternatives, not additive work. Isolated timings
include Python, evaluation, synchronization, and replay setup, not GPU-only
kernel duration. They lose whole-model graph fusion, resource overlap and
cache locality. They cannot explain a complete execution-time budget.

The full-model result is recorded separately, before instrumentation, with no
additive-cost or whole-model-speedup claim. Examine swap and memory pressure,
retain multiple raw samples, and repeat under controlled conditions before
choosing an optimization. Longer-context attention, KV allocation boundaries,
and decode steps beyond the captured snapshot may behave differently.

Not covered: bare SDPA/RoPE kernel breakdown, distributed/pipeline models,
quantized/rotating KV caches, compiled whole-model decode, arbitrary model
architectures, or persisted activation tensor dumps. Activation arrays remain
live only until isolated measurements finish; evidence stores their metadata,
not their numerical contents.

## Verified local runs

The parent subsequently exercised the profiler on the real retuned 0.5B checkpoint:
237 isolated operations, 18 installed fused MLP pairs, all seven diagnostic groups.
The opt-in native integration test passed. Raw profiles and exact executed source
are archived under `results/m2pro-decode-diagnostics-v1/`, `v2/`, and `v3/`.

- v1: capture step 0, 3 samples/operation, 4 uninstrumented decode steps.
- v2: capture step 8, 12 samples/operation, 16 uninstrumented decode steps.
- v3: capture step 1, 3 samples/operation, after path-normalization hardening.

The tied vocabulary output projection was the largest individual isolated
operation in v1 and v2: 0.634584 ms and 0.622813 ms respectively. Inclusive
attention calls followed (roughly 0.30 ms each), but overlap their separately
measured q/k/v/o projections and include cache-clone setup. These suggest output
projection and attention as further optimization candidates; they do not establish
a whole-model fraction or prove either dominates total model latency.

A regression test reproduced a lexical `..` path disguising an output inside the
checkpoint. The live preflight now rejects symlinks first and resolves parent
components before checking overlap. Earlier normal-path evidence remains byte-exact;
archived executed source is historical evidence, not a hardened live entry point.
CPU test device changes are restored after each test so diagnostic tests do not
silently move subsequent native tests onto CPU.

## Tests

CPU logic/toy-module tests (no saved-model load or custom Metal dispatch):

```bash
.venv/bin/python -m pytest tests/test_profiling.py -q -m 'not metal' \
  --basetemp /Users/jeremylien/.hermes/cache/scratch/paretoquant-profiling-tests
.venv/bin/ruff check src/paretoquant/profiling.py scripts/profile_decode.py tests/test_profiling.py
```

A native integration test is explicitly opt-in and was not run by the profiler
implementation agent. The parent may run it **sequentially**, with a new scratch
test directory and the correct saved model/runtime:

```bash
PARETOQUANT_PROFILE_NATIVE_MODEL="$PWD/artifacts/m2pro-saved-retune-v1/model" \
  .venv/bin/python -m pytest tests/test_profiling.py::test_opt_in_local_native_profile -q \
  --basetemp /Users/jeremylien/.hermes/cache/scratch/paretoquant-profile-native-tests
```

The native test intentionally uses one decode step and one timing sample: it
checks real topology/evidence plumbing, not a stable performance result.
