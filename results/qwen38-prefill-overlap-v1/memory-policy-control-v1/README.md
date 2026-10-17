# Wired-policy control v1 — ready for narrow read-only review

**CONTROL_NOT_MODEL_GATE. CPU implementation/verification only. No physical trial has run.**

This diagnostic compares the process-local MLX policies `recommended` and `unwired` (exactly zero), without model/checkpoint loading. It does not attribute the earlier warning to wiring, claim that setting a budget preallocates/pins that budget, or support speedup/capacity conclusions. The previous limit is measured, never assumed to be zero. Installed MLX version/device information is recorded by the physical entry; a version string does not cryptographically bind a wheel to a source tag.

## Files and scope

- `fixture.py`: inert on import; injected pure runner, bounded parser, fixed physical entry.
- `harness.py`: inert on import; parent staging/admission/owned execution/collection.
- `test_control.py`: **25 CPU tests**; fake backend and sampler only. The collector test's synthetic TEST receipts are temporary and are not physical measurements.
- `run_cpu_tests.py`: prohibits MLX/NumPy/model imports, subprocess/exec/fork, sockets/network and signals during CPU tests; emits `cpu-receipt.json`.
- `cpu-receipt.json`, `cpu-test-output-reviewed.txt`: current successful CPU evidence and hashes.
- `tdd-01-*` through `tdd-18-*`: actual vertical RED/GREEN outputs, not reconstructed transcripts. Seven supplemental error-path regressions exercise the previously test-driven runner. `tdd-15-green.txt` preserves an intermediate test expectation mismatch; `tdd-15-green-final.txt` is the corrected pass. `cpu-test-output.txt` preserves an initial guard compatibility failure (replacing the socket class broke stdlib SSL imports); the fixed guard preserves that class and blocks socket creation through its audit hook. Earlier 24-test output is retained separately.

Only this new directory was written. The original 025 actor/pressure guard, 024, benchmarks and historical results were not edited. No GPU, MLX import, network, SSH, subprocess child or signal operation was executed by the CPU suite. Shell tools launched the Python CPU tests; the suite itself launched no processes.

## Bounds and semantics

| Item | Physical entry |
|---|---|
| Modes | exactly `recommended`, `unwired` |
| Retained MLX arrays | exactly 64 MiB, 64 × 1 MiB U32 chunks |
| Allocator ceiling | 192 MiB, `mx.set_memory_limit(..., relaxed=False)`; cache limit zero |
| Work | finite 64 allocation iterations + 64 retained-buffer checks |
| Child work/import budget | at most 30 seconds, cooperative checks |
| Parent-owned actor deadline | at most 45 seconds, original watchdog |
| Host payload | bounded by construction; conservative logical budget 136 MiB including possible retained input/copy lifetimes, transient chunks and bounded observations; **not a process-RSS/OS/driver bound** |
| OS observations | at most 66 samples; vm_stat ≤64 KiB, ≤64 fields, ≤64-character field names; required counters and exact pressure `1` |

Small payloads/chunks are available only to the injected Python test seam. The physical CLI exposes no size/iteration override. A chunk is deterministic SHAKE-256 output interpreted as little-endian U32, not repeated zeros or a Python list of millions of integers. Each generated CPU byte sequence is compared exactly with the MLX→CPU roundtrip, SHA-256 and first/last U32 endpoints. All retained arrays are checked again after the last allocation. No model, checkpoint, distributed initialization, compile or kernel benchmark is involved.

The adapter requests the strict allocator API; an unsupported installed API fails rather than falling back to a relaxed ceiling. Counters are checked before allocation, after each chunk and after cleanup. The child requires an initially empty active allocator. It records requested wired limit, setter-returned old limit, idempotent readback, and restored idempotent readback. Cleanup independently attempts synchronization, releasing only its retained arrays, garbage collection/cache clearing, synchronization and restoration of old wired/cache/memory limits. Cleanup failures make the outcome fail. Abrupt OS termination cannot promise Python `finally`; the parent's original process/group watchdog receipt remains authoritative. Cooperative deadlines cannot interrupt a stuck MLX/C call or cleanup; the outer watchdog owns that case.

`physical_runtime()` alone imports MLX/NumPy and runs read-only `/usr/bin/vm_stat`, `sysctl -n kern.memorystatus_vm_pressure_level`, `sysctl -n hw.memsize` and `ps -o rss= -p CHILD_PID`, each with a 0.25-second command timeout and bounded accepted output. `ps` RSS is labeled RSS, **not physical footprint**. No sysctl writes, root operations or system/global configuration changes exist. The original actor independently samples normal pressure before spawn, during execution and after exit. Parent passive observation is preferable for finer-grained host/footprint context; it must not replace the original required watchdog. Global wired-page deltas are noisy observations, not a pass criterion or a causality test.

## Repeat the CPU verification

From this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B run_cpu_tests.py
```

Actual current result: **25 tests, 0 failures, 0 errors, 0 forbidden attempts**. `cpu-receipt.json` hashes the four local Python files, the two originals, and every selected RED/GREEN log. These are audit hashes, **not parent admission or physical approval**. Each TDD log was produced with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B test_control.py > tdd-NN-red.txt 2>&1
# implement only that slice, then:
PYTHONDONTWRITEBYTECODE=1 python3 -B test_control.py > tdd-NN-green.txt 2>&1
```

Cycles cover config, strict sampling, end-to-end policy/roundtrip/restoration, retained corruption, bool readback, residual allocation, fixed CLI, strict allocator adapter, physical entry injection, exact staging, scoped original `run_owned`, external admission, original module loading, independent pressure collection, admitted parent CLI, bounded VM fields, guarded receipt runner and conservative host payload lifetimes.

## Parent-only staging and admission (NOT executed here)

Independently review `fixture.py` and `harness.py`, verify the original actor/guard against **parent-owned approved hashes**, and inspect the installed MLX API before admitting physical work. Do not derive permission by trusting whatever files the child/staging directory currently contains. The controller owns host identity/role mapping, Python environment selection, transport, job ownership and execution order.

`harness.stage` is CPU-only and writes a **new** private staging directory containing exactly:

```text
fixture.py
remote_actor.py    # byte-exact original 025 source
pressure_guard.py  # byte-exact original 025 source; required by run_owned
```

Call it from the parent with independently approved values:

```python
# h is the reviewed harness module loaded by the parent.
# APPROVED is the parent's independent review/admission record, NOT child output.
hashes = h.stage(
    STAGED_ROOT,                       # must not exist; its parent must exist
    original_root=REPO / 'spikes/025-qwen38-overlapped-prefill',
    expected_original_hashes=APPROVED['original_sha256'],
    expected_fixture_sha256=APPROVED['source_sha256']['fixture.py'],
)
```

For the remote host the parent can transport these three files byte-exactly, plus the separately reviewed `harness.py`. The harness is deliberately not an extra child-source dependency, but its own hash is mandatory in external admission. Verify transferred bytes before invoking anything. No staging/transport/SSH command is run by this diagnostic.

Before each trial the parent creates `STAGED_ROOT/admission.json` (or passes `--admission PATH`) with **exactly** these keys:

```json
{
  "schema": "wired-policy-control-admission-v1",
  "nonce": "<fresh literal 32 lowercase UUID hex characters>",
  "mode": "recommended",
  "host_role": "local",
  "source_sha256": {
    "fixture.py": "<parent-approved 64 lowercase hex>",
    "remote_actor.py": "<parent-approved original 64 lowercase hex>",
    "pressure_guard.py": "<parent-approved original 64 lowercase hex>"
  },
  "original_sha256": {
    "remote_actor.py": "<same independently approved original hash>",
    "pressure_guard.py": "<same independently approved original hash>"
  },
  "harness_sha256": "<parent-approved reviewed harness hash>"
}
```

Placeholders above are deliberately not a usable admission. Use `uuid.uuid4().hex` for a fresh nonce; hyphenated or uppercase tokens are rejected, not normalized. Host role/mode/nonce must match the external admission literally. Hash sets must be exact; symlink sources, oversize files and changed bytes fail before execution. Admission is external and no automatic self-approval is provided.

## Actual parent invocation (NOT executed here)

Run in the parent's selected MLX-capable Python environment **on the intended host**:

```sh
"$MLX_PYTHON" -B /path/to/reviewed/harness.py \
  --root "$STAGED_ROOT" \
  --jobs "$EXISTING_PRIVATE_JOB_PARENT/$FRESH_TRIAL_DIRECTORY" \
  --mode recommended --host-role local --nonce "$FRESH_UUID_HEX"
```

`--jobs` is the exact **new per-trial directory**, not an existing jobs directory; its parent must already exist. Default actor deadline is 45 seconds and `--deadline` may only lower it. Do **not** invoke `fixture.py` directly, use the original actor CLI (it does not admit this fixture), or substitute an unchecked subprocess/SSH launch. The function equivalent is:

```python
result = h.execute(
    STAGED_ROOT, NEW_TRIAL_DIRECTORY,
    mode='recommended', host_role='local', nonce=FRESH_UUID_HEX,
    admission=PARENT_CREATED_ADMISSION, deadline=45,
)
```

The executor verifies external hashes first, executes the verified original actor/guard bytes, scopes the original actor's `SOURCES` override to the three-file registry and restores it in `finally`. It calls the **unchanged original `run_owned`**, with expected hashes, nonce/host/actor labels and `require_normal_pressure=True`. Its temporary pressure-module binding is also restored. Use one synchronous parent owner only; the helper is not a concurrent launch service.

The original actor snapshots the three source files, gates and owns the child, samples normal pressure, enforces deadline and retains stdout/stderr/startup/PID/result/source receipts. Collection re-reads those exact targets; uses the original `expected_pressure_binding`/`verify_pressure` with parent-admitted identity plus independent startup/PID/execution records; checks cleanup completion, child identity, fixed physical bounds, all roundtrip rows, and wired restoration. It rejects failed or incomplete outcomes rather than stamping them approved. On failure, inspect the retained trial directory; do not discard it or automatically retry.

## Frozen parent execution plan

Execute **M2/local first, then M4/remote, serially**. Within **each** host:

1. `recommended` (repeat 1)
2. `unwired` (repeat 1)
3. `unwired` (repeat 2)
4. `recommended` (repeat 2)

Use a fresh parent admission/nonce and new job directory for every trial. No parallelism or automatic full-model retry. Stop if sampled pressure is not `1`, or any trial/restoration/cleanup is incomplete; the controller decides whether/when pressure recovery permits another action. Results remain **CONTROL_NOT_MODEL_GATE**, including fully successful controls. This fixture cannot establish the original load-warning cause, model feasibility, a speedup or increased capacity.
