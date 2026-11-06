# 027 — depth-k MTP drafting: verify/rollback mechanics prototype

Single-host, small-model only. Generalizes 021's width-2 / prefix 0-1-2 machinery to
width W = k+1 ≤ 5 and prefix keep 0..W. 021 is untouched; slot_qmv files are byte copies.

- `speculative_k.py` — greedy depth-k controller (longest accepted prefix, bonus/correction,
  EOS/cap terminal-not-consumed cache convention, `commit_prefix(len(emitted))`).
- `prefix_rollback_k.py` — `CaptureK(model, caches, width)`: native GDN observation and
  recurrence-only prefix commit for any keep.
- `verifier_k.py` — native / singleton-qmm / slot-qmv numerical modes at width W.
- `small_harness.py` — 0.8B backend, replay stand-in drafter, strict gates.
- `depth_model.py` — expected-speedup model from archived broad-v1 data.
- `run_small_gates.py` — writes receipts to `results/qwen38-mtp-depth-assessment-v1/`.

Run (repo root):

    .venv/bin/python spikes/027-qwen38-mtp-depth/run_small_gates.py --device gpu --out results/qwen38-mtp-depth-assessment-v1/<new>.json
    cd spikes/027-qwen38-mtp-depth && ../../.venv/bin/python -m pytest -q -p no:cacheprovider test_control_flow.py test_small_model.py
    .venv/bin/python spikes/027-qwen38-mtp-depth/depth_model.py --force

Not covered here: chained MTP head (no head on the 0.8B), distributed rank protocol, 27B, M4.
See `results/qwen38-mtp-depth-assessment-v1/README.md`.
