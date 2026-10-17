# 020: Qwen3.8 finite FP16 rounding diagnostic

New isolated candidate derived from 016. **Opt-in practical numerics tradeoff,
not the same arithmetic, not precision equivalence, and no quality guarantee.**
CPU tests establish admission/diagnostic behavior only. Physical small-checkpoint
quality and full-model counting/control benchmarks remain the parent's gates;
no throughput result or >=10 tokens/s acceptance is established here.

## Modes and CLI

Both `supervise.py` and `probe.py` accept:

- `--compute-dtype native` (default): unchanged 016 native behavior.
- `--compute-dtype float16`: unchanged 016 exact BF16->F16->BF16 bit-roundtrip
  requirement; finite but unrepresentable values still fail closed.
- `--compute-dtype float16-rounded`: new explicit `finite-fp16-rounding-v1`
  policy, accepting finite rounding, including underflow to zero. Source NaN/Inf,
  FP16 overflow and nonfinite results are rejected before generation.

For the parent's existing admitted supervisor invocation, use the **020**
`supervise.py` and an immutable **020** remote source deployment, retaining the
original Qwen3.8 packed checkpoint, source identities, ownership and controls:

```sh
.venv/bin/python spikes/020-qwen38-fp16-rounded/supervise.py \
  [existing admitted host/checkpoint/output arguments] \
  --compute-dtype float16-rounded --execution phased --eval-group-size 32
```

The square-bracket argument above is a placeholder, not a runnable host recipe.
For a direct probe use `--mode generate --compute-dtype float16-rounded --phased`
with the existing streaming-loading arguments (`--stream-local` or weight server).
All non-native modes require attested streaming loading. Both ranks receive the
requested mode, exchange distinct mode/dtype codes (`rounded=[2,16]`, exact=[1,16]),
and retain observed hidden/handoff dtype evidence. Mixing rounded and exact ranks
is rejected even though both use FP16 arrays.

## Data and evidence contract

No checkpoint rewriting or repacking. Only eligible BF16 runtime arrays become
FP16. Packed U32 words, F32 parameters and existing F16 arrays are untouched.
In **rounded mode**, protected `A_log` retains its exact original array, dtype and
bytes, whether its source storage is BF16 or F32. The actual Qwen3.8 checkpoint
stores `A_log` in BF16 even though the model constructor initializes it as F32;
`Model.cast_predicate` excludes it from casting, not from legitimate loading.
The supervisor independently expects this unchanged layout and excludes `A_log`
from converted tensor names, digests and numerical counts. Forged protected casts
or conversion evidence are rejected. Other BF16 predicate exclusions still fail
explicitly: the supervisor cannot independently bind arbitrary model predicates.
Exact `float16` mode keeps its existing protected-BF16 rejection, and native mode
remains unchanged. A failed transform is not a usable model: discard it/process
as in 016.

Original `weight_loading.tensor_sha256` identities remain hashes of serialized
source payloads, before casting. `compute.source_parameter_layout` and
`compute.runtime_parameter_layout` retain exact names, shapes, dtypes and byte
sizes. Existing source, inventory, replicated tensor and runtime digest checks
remain; no attestation redesign or gate removal.

Each converted tensor appears in `compute.conversion_diagnostics` in the rank
JSON receipt, with:

- `source_sha256`: raw BF16 storage bytes (uint16 view), checked against the
  unchanged loader source digest.
- `runtime_sha256`: raw FP16 bytes, also in `converted_tensor_sha256`.
- `element_count`, `numerically_changed_count` (original vs FP16 in F64),
  `roundtrip_exact_count` (bit-exact BF16 roundtrip).
- `max_abs_error`, `max_rel_error`: CPU F64 diagnostics; relative error uses
  `abs(runtime-source)/abs(source)` and is zero for unchanged zero source values.

These are load/setup diagnostics, not timed generation metrics. Global
`roundtrip_exact` is derived from all tensor counts; `precision_equivalent` is
always **false** in either FP16 mode. A roundtrip-exact tensor is not evidence of
identical dequantization/kernel/activation/accumulation arithmetic or task quality.
The phase receipt SHA binds the complete diagnostics, and supervisor admission
checks policy, shapes/counts, finite nonnegative errors, pre/post digests, actual
dtype and both-rank agreement. This is a rounding diagnostic, not a quality gate.

## CPU verification

Strict TDD: the tiny BF16 Conv1d `1e-9` witness first failed for the absent rounded
mode; implementation then passed with recorded underflow drift and CPU forward.
Rounded CLI/trial acceptance was separately observed failing before gate support.
Exact-mode rejection and native controls remain covered. For the eligibility
repair, a minimal CPU BF16 `A_log` plus BF16 norm witness first reproduced the
protected-parameter rejection; after the selection fix, a real streamed tiny
`qwen3_5.Model` with BF16 `A_log` separately reproduced the supervisor-layout
rejection before its fix. Tests now retain protected BF16/F32 object identity,
raw bytes and source digest, cast only eligible parameters, and reject forged
protected layouts or conversion entries. The selected CPU suite passes **101
tests**; lint and formatting checks pass. No numerical tolerances changed.

```sh
.venv/bin/python -m pytest \
  spikes/020-qwen38-fp16-rounded/test_rounded.py \
  spikes/020-qwen38-fp16-rounded/test_compute_dtype.py \
  spikes/020-qwen38-fp16-rounded/test_compute_receipts.py \
  spikes/020-qwen38-fp16-rounded/test_schedule.py -q
.venv/bin/ruff check spikes/020-qwen38-fp16-rounded
.venv/bin/ruff format --check spikes/020-qwen38-fp16-rounded
```

Only these selected CPU tests were run for 020. Copied GPU/long-block tests are
retained for later physical admission, not claimed as executed here. No GPU or
SSH was used for this implementation. The registry still includes
`compute_dtype.py`; deploy the complete existing `remote_actor.SOURCES` set.

016 sources and `results/qwen38-speed-v2/016-small-fp16-v1` failure evidence are
untouched. `preserved-016-sha256.json` records their pre-work digests; verification
compares those exact paths after implementation. Never relabel the preserved
016 exact-roundtrip failure as a success of this different policy. The original
`results/qwen38-speed-v2/020-full-fp16-rounded-v1` pre-generation protected-BF16
failure and its archived actor sources also remain unchanged. Repeat physical
admission with a **new immutable remote v2 deployment** of the current registered
sources plus `supervise.py`, and a fresh output directory; never overwrite the
previous runtime-source manifest, actor snapshot or failed receipt.
