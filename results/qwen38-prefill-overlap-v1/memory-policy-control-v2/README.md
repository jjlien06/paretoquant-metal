# Memory-policy control v2: installed-API compatibility

**CONTROL_NOT_MODEL_GATE. No arrays or physical control cases have run yet.**

V1 and all its test evidence remain intact. A parent preflight against installed
MLX 0.32.3 reproduced a `TypeError`: `set_memory_limit` accepts one integer and
does not accept `relaxed=False`. No arrays were allocated during that preflight.
The API documentation installed in that exact environment describes the limit
as a guideline, not a hard allocator ceiling.

## Narrow correction

- `harness.py` is byte-identical to v1.
- The adapter calls the installed single-argument API directly, without a hidden
  exception fallback. Backend evidence labels it `advisory-with-observed-cap`.
- Retained data is structurally bounded to 64 × 1 MiB U32 arrays. Active/cache/peak
  counters must remain at or below 192 MiB at the recorded checkpoints. This is
  **not** a strict underlying allocator limit or a process-RSS bound.
- The original independent pressure1 watchdog, 45-second actor deadline,
  30-second cooperative fixture budget, exact roundtrip checks, source/nonce/role
  binding, cleanup and policy restoration are unchanged.
- No system-wide setting, target model, target split, tolerance or user memory
  acceptance condition is changed. This revises only the diagnostic's own
  earlier, unsupported strict-allocator claim.

The signature regression failed against v1 and passed after this correction.
All 25 guarded CPU tests pass with zero forbidden-operation attempts in
`cpu-1792270414331422000-verification.json`; the detailed output filename and
exact current source hashes are recorded there. The initial v2 runner-interface
regression is
preserved in `cpu-verification.json` and `cpu-output.txt`, not relabelled a pass.

Run fresh CPU verification from this directory:

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B run_cpu_tests.py
```

Each invocation creates exclusive timestamped output/receipt files. Physical
staging/admission/CLI and fixed recommended/unwired/unwired/recommended order
are unchanged from `../memory-policy-control-v1/README.md`, except admission
must bind **these v2 fixture bytes**, not the v1 fixture. Parent independent
review and current both-host health checks remain prerequisites. No capacity,
model numerical-admission, causality or speedup conclusion follows from this
control, even if all eight planned physical cases complete.
