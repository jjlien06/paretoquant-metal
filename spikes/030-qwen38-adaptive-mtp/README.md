# 030 — adaptive depth-1/2 MTP drafting (27B, two hosts)

This spike is a copy of the 029/023 schedule runner, ported to width 3. Results are in
`results/qwen38-adaptive-mtp-v1/README.md`.

## Changes from 029
- **`verifier.py`**: singleton-QMM and slot-QMV substitution now runs at the explicit verify
  width W in {2, 3}, not only at width 2.
- **`prefix_rollback.py`**: `Capture(width=W)` and `commit(keep)` for keep 0..W. This is the
  general rule ported from 027:
  - rerun the recurrence on `args[:, :keep]`;
  - restore the conv tail;
  - trim the KV cache by W−keep.
- **`rank_control.py`**:
  - `verify` accepts 2 or 3 tokens.
  - `commit_prefix` accepts 0..3, and keep must be ≤ the last verify width (enforced by
    Worker).
- **`mlx_backend.py`**:
  - `draft` is the original 021 depth-1 draft; the 029 chain logging is removed.
  - `draft_adaptive` gives d1 and p1, plus the chained d2 and p2. It uses
    `head(m.norm(h1), d1)` under an evaluated snapshot/restore of the head KV, and only runs
    if p1 ≥ min_p1.
  - Widths 2 and 3 route to `verify`.
- **`speculative.py`**: adds `generate_adaptive(tau)`. Each round:
  - verify `[y, d1, d2]` if p1 ≥ tau and p2 ≥ tau, otherwise `[y, d1]`;
  - accept the longest prefix;
  - commit prefix keep = len(emitted), with the same bounded collective command on both
    ranks;
  - catch up the head KV from verified hidden states;
  - log an `adaptive_round` event.
- **`verification.py` / `mtp_runtime.py`**:
  - When `--adaptive-tau` is set, the same-process width-3 window and prefix0..3 gates run
    on small and full shards.
  - `Admission(adaptive=True)` requires both ranks to pass.
- **Schedule (`workload_schedule.py`)**: the `…-adaptive-v1` schedule requires one
  native/mtp/adaptive triple per case.
- **Admission and receipts (`probe.py`, `supervise.py`, `workload_evidence.py`)**:
  `--adaptive-tau` is bound into the rank receipts and gate receipts. The supervisor
  re-verifies the width-3 evidence, and checks per request:
  - `width3_rounds`;
  - the verify-width bounds;
  - `accepted_drafts ≤ 2×rounds`.
- **New files**:
  - `test_adaptive.py` and a rewritten `test_depth_chain.py`;
  - `make_run.py` (exclusive local and remote snapshot, with SHA-256 readback);
  - `analyze.py`, `tau_select.py`.

## Tests
`pytest --ignore=test_integration.py`: 283 passed (CPU).
