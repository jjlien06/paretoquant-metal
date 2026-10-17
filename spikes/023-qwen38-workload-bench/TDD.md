# Observed RED → GREEN record

All commands ran from `/Users/jeremylien/Projects/paretoquant-metal` with the
repository `.venv/bin/python`, in isolated 023 pytest processes. No GPU, SSH,
deployment or full model loading was performed. Tests creating MLX models explicitly
select `mx.cpu`; final full regression also selects CPU before pytest collection
and deselects the one GPU test. Only tiny fixture models run on CPU.

Each implementation slice followed its observed failing test; later slices were
not implemented in advance. These are actual tool-observed counts/reasons, not
fabricated benchmark output. Full failure stacks remain in session tool output;
this file records concise observed command/output receipts.

| Vertical slice / command suffix after `.venv/bin/python -m pytest spikes/023-qwen38-workload-bench/` | RED observed | GREEN observed |
|---|---|---|
| `test_workload_schedule.py -q` | 2 failed: missing `workload_schedule`; registry missing fixed schedule/module | 2 passed in 0.03s |
| `test_workload_runtime.py -q` | 1 failed: `run_loaded()` unexpected `request_specs` | 1 passed in 0.94s; genuine tiny CPU native/MTP two-workload execution |
| `test_cli_schedule.py -q` | 1 failed: inherited `Invalid deadline/token/repeat bounds` at 900s | 1 passed in 0.02s |
| `test_workload_evidence.py -q` | 1 failed: missing `workload_evidence` | 1 passed in 1.03s; strict gates and adversarial evidence mutations |
| `test_workload_caps.py -q` | 2 failed: missing `LOG_CAP_BYTES`/`RETRIEVAL_CAP_BYTES` | 2 passed in 0.22s; real subprocess cap stop and owned cleanup |
| `test_dispatch_schedule.py -q` | 1 failed: missing `verify_completed_trial` | 1 passed in 0.89s |
| `test_evidence_hardening.py -q` | 1 failed: registry mutation `DID NOT RAISE ValueError` | 1 passed in 0.97s |
| `test_workload_timings.py -q` | 1 failed: missing `request_phase_timing_by_rank` | combined timing/snapshot check: 3 passed in 0.98s |
| `test_unbound_schedule.py -q` | 1 failed: unbound boundary `DID NOT RAISE ValueError` | 1 passed in 0.88s |
| `test_workload_cleanup.py -q` | 1 failed: common outside-timing GC receipt `[] != ['request_done','request_done']` | 1 passed in 0.88s; MTP-before-native cleanup and next-peak-reset ordering |

The final three RED/GREEN cycles explicitly used:

```sh
TMPDIR=/Users/jeremylien/.hermes/cache/scratch .venv/bin/python -m pytest \
  spikes/023-qwen38-workload-bench/<test-file> -q \
  --basetemp=/Users/jeremylien/.hermes/cache/scratch/pq023-<unique-cycle>
```

Initial cycles inherited the host's `/var/folders/...` pytest temp default; an
intermediate regression used an owned `.test-tmp` inside 023. After the parent
identified the TMPDIR discrepancy, all subsequent pytest commands explicitly set
TMPDIR/basetemp to the mandated scratch location. No unrelated temp files were
removed. The intermediate 023-owned test temp directory was removed only after
its tests finished.

Regression progression: `102 passed, 1 deselected in 2.39s`; after hardening,
`103 passed, 1 deselected in 2.50s`; after timing, snapshot/retrieval-cap and unbound
boundary coverage, `107 passed, 1 deselected in 2.62s`, repeated with cache-disabled
pytest and scratch TMPDIR: `107 passed, 1 deselected in 3.12s`. Subsequent common-GC
fairness coverage is included in the final regression output file.
Final full CPU regression after common-GC fairness and genuine two-rank mixed
schedule coverage: **109 passed, 1 deselected in 3.06s**. The two-rank CPU ABBA
regression separately passed in 0.95s using real tiny sharded models and original
Channel framing, replacing only physical send/recv with queues; no frames remained.

Formatting/import cleanup was performed only after GREEN. The first Ruff pass
reported import/format/line-length issues; these were fixed without changing
numerical kernels. Final `.venv/bin/ruff check spikes/023-qwen38-workload-bench`
returned **`All checks passed!`**. Final CPU command and output are recorded in
README / `cpu-regression.txt`, with JUnit evidence in `cpu-tests.xml`.

Additional existing-behavior regression tests exercise real immutable actor
snapshots (`probe.py --help`, no model loading), exact source/schedule hashes,
successful owned cleanup, and serialized response-cap rejection. These were added
as tests of the already-green snapshot/cap implementation, not claimed as new
production-code tracer bullets.
The additional two-rank ABBA regression similarly exercises the already-green
mixed runtime; it required no production changes.

The six frozen parent workload schedules passed exact-byte/schema loading:
6 batches × 14 requests = 84 requests; source JSONs were only read. No physical
throughput, checkpoint numerical admission, or completed broad benchmark is claimed.
