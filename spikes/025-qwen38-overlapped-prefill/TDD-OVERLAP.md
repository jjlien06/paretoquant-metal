# 025 vertical implementation ledger — not 024 history

`TDD.md` is the unmodified copied024 record. This file and
`overlap-command-history.jsonl` are the new025 CPU implementation history.
The raw ledger preserves every actual guarded invocation, timestamp and exit
code, including intentional RED runs and fixture/runner mistakes. It must not
be read as a list of only passing trials.

## First tracer bullet: real Command / Worker / Channel

Actual initial command:

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch \
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python \
  cpu_only_runner.py -q test_overlap_protocol.py -x
```

RED:1 failed because the real Command rejected the missing `prefill_pipeline`
opcode. The written test already required rank1 chunk1 to progress while rank0
chunk0 completion was held, using real Worker/serve_request and Channel framing
on a condition-variable **zero-capacity rendezvous** wire. It was not a queue-only
or flag-only test. GREEN:the exact same command returned1 passed after the bounded
compound/receiver-credit tracer bullet. The receive credit was mandatory from
its first implementation; no uncredited compound was adopted.

The eager transport regression then holds rank0 while requiring chunk1 progress
and proving chunk2 cannot begin. A CPU-only source mutation removes the actual
receiver-credit wait:the same real seam exceeds the bound and is killed by the
witness, with bounded thread cleanup. Boundary regressions cover
1/2/255/256/257/511/512/513/2048, exact shifted head IDs/absolute positions,
ordinary final singleton and final ACK drain. Original cache arithmetic is
opaque to the symbolic stage fixture, not reimplemented.

## Subsequent vertical RED → GREEN slices

Actual ordered command arrays and return codes are in the raw JSONL and copied
into the new parent `cpu-runtime-implementation.json`. The feature REDs include:

- Worker accepted wrong final singleton/snapshot/finish/reset after compound;
  raw failure allowed reset; duplicate control JSON keys were accepted. GREEN
  enforces exact final metadata and poisoned raw-stream state.
- Native receive hook/per-request backend mode wiring was absent. GREEN puts
  only a receive-stage hook after evaluated receive, before native graphs, and
  routes actual backend/target loops through the compound helpers.
- New two-MTP-mode schedule/schema and explicit capability admission were absent.
  GREEN preserves old native/MTP schema/default serial behavior while admitting
  exact new rows, per-request mode/source identity and pre-hardware flag checks.
- Actor lacked required pre-spawn/periodic pressure guard. GREEN uses injected
  sampler/Popen CPU tests:unhealthy initial sample cannot spawn; later unhealthy
  sample stops only the owned unreaped child through the existing cleanup seam.
- Gate/source API and strict full513 inventory checker were absent. GREEN wires
  fresh physical two-mode small/full gates and model-bound capability records;
  detached CPU fixtures exercise inventories, shapes/dtypes, offsets, tolerance
  consistency and actual513/256 schedule requirements. **GPU gates not executed.**
- Action auditor lacked compound grammar. GREEN preserves exact final singleton
  outside compound and unchanged speculative output/commit grammar.
- Actual backend class-body test caught wrong-mode command sent before rejection.
  GREEN rejects before transport and forbids reset after raw failure.
- Dry-run flag was absent. GREEN executes option/source checks without live run,
  output creation, SSH or MLX.
- Missing pressure observation gaps and gate-only pressure receipts were accepted.
  GREEN fails late observations and requires both-host actor timelines.
- Bool-as-int shape/shift metrics and resealed missing block completion were
  accepted. GREEN validates exact types and full native block/layer/receive/
  credit/head causal grammar, even when an adversary recomputes every digest.
- Enabled CLI admitted wrong rank math/placement; new schedule admitted2049 IDs.
  GREEN fixes the frozen28/36/group32/native-slot profile and new-study2048 cap.
- Direct runtime entry imported MLX before rejecting a missing capability flag;
  GREEN performs the pure flag check first. Compound drain audit was missing;
  GREEN binds exact nonce/sequence/integer occupancy and requires prefix/head
  completion before drain, then compound action before final ordinary receive.

Extra regressions exercise malformed opcode/credit/ranges/bools, stale nonce,
wrong enabled flag, drop, cancel, timeout and stage failure on actual Worker/
Channel wires under eager AND rendezvous transport. None emits a JSON ACK into
a failed raw-hidden stream; every test joins its owned threads with finite bounds.

## Verification scope and honest failures

- `cpu_only_runner.py` hard-forbids MLX/model imports, network and process signals.
  The full permitted suite preserves every original test file; MLX-dependent
  modules/functions and owner-only GPU/process-signalling fixtures are explicitly
  excluded/skipped, not relabelled passed. The final XML/raw ledger provides the
  exact counts and exclusions.
- Earlier failed runs include a symbolic score fixture lacking slicing, a mocked
  bootstrap pipe lacking its inherited reader, and a CPU discovery filter that
  initially missed function-local MLX imports. Those failures are preserved in
  the ledger. Fixing them did not run MLX, hardware or real sysctl measurements.
- A frozen normalized AST fingerprint verifies native phased-forward arithmetic
  against024 after removing only evidence emits and the receive hook. Protected
  head/rollback/cache/QMV/Metal source hashes are checked against the copied base.
- Ruff formatting/checks, guarded real CLI help/dry-run, the focused suite and the
  full permitted CPU suite were actually executed. No physical GPU gate, source
  deployment, checkpoint load, pilot, performance acceptance or independent
  approval is claimed. Final local source hashes are not remote deployment proof.

## Pressure/ownership blocking repair — protocol v2

The earlier v1 pressure checker is NOT current admission. Historical result JSON,
`TDD.md`, original024 sources and baseline benchmark files remain immutable.
`test_pressure_guard.py::test_pressure_timeline_missing_periodic_samples_rejected`
is now an explicitly v2 positive fixture with independent identity/execution
fields; its old unbound positive fixture is not carried forward as approval.

New tracer bullets in `test_pressure_v2.py`, run before each corresponding repair:

1. Bound coverage API absent and historical0.01s unbound timeline accepted:
   **2 RED**, then **2 GREEN**. Coverage now brackets source-recorded Popen/gated
   release/observed child completion and reconciles final sampling with actor
   elapsed on one actor origin. A long pre-child setup interval is legitimate.
2. Timeout/stage/timestamp/binding grammar: **10 RED,17 already passing**;
   then **27 GREEN**, including missing/wrong timeout, unknown/repeated stages,
   negative/duplicate/nonfinite time, bool PID and malformed evidence structures.
3. Stop/deadline appearing inside initial sampler: **3 RED** proved Popen was
   reached; **3 GREEN** proved no spawn and correct stop/deadline reason.
4. Actual `run_owned` producer lacked bound fields: **1 RED**, then **31 GREEN**
   cumulatively. Mocked actor now produces externally verifiable receipts; later
   parametrization exercises local/rank0, remote/rank1 and remote/server roles.
5. Startup/PID/admitted nonce reconciliation absent: **1 RED ->1 GREEN**.
   Triplicated other-job and900s/short-coverage controller admission plus distinct
   both-host bindings: **6 RED ->38 GREEN** cumulative.
6. Gated bootstrap released after PID-write stop/deadline: **2 RED** observed
   `b'1'`; **40 GREEN** cumulative proved no gate release in these cases.
7. Job/role admission and actual mocked controller descriptor/status propagation:
   **2 RED ->42 GREEN** cumulative. Registry/workload forwarding:
   **2 RED ->44 GREEN** cumulative.
8. Exact identity type coercion/completion checks: **4 RED ->64 GREEN** across
   then-current focused pressure/CLI suite. Zero child interval and coerced clock
   origin were each separately observed RED, then GREEN after their own fixes.

Actual invocation arguments/return codes are in the append-only
`overlap-command-history.jsonl`; final full stdout/commands and final extra
boundary RED/GREEN logs are under `pressure-fix-evidence/`. Initial RED stdout
was returned by tools, not retrospectively synthesized as a test log.

Final CPU verification additionally audits `subprocess.Popen`, `os.system` and
`os.posix_spawn`; no test child, sysctl, SSH, network, GPU or signal was launched.
Four historical subprocess-CLI tests are explicitly deselected rather than run
outside the guard. One initial full scan hit the audit refusal on the additional
`test_standalone_cli_and_exact_registry`; the retry explicitly excluded it.
The nine compatible mocked tests from the original scratch independent repro
also pass. Re-running its five historical vulnerability assertions as if they
were positive v2 acceptance tests would be incorrect; the new v2 regressions
invert those assertions. This execution is implementation verification, NOT a
fresh independent review. Both new CPU independent review and owner-only live
gated-bootstrap/pressure hardware verification remain mandatory.

