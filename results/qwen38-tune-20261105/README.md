# Qwen3.8-27B two-Mac tuning pilot — 2026-11-05 (~30 min)

Saved fast profile (`results/qwen38-speed-v2/run-fast.sh`: MTP prefix-rollback, phased, slot-QMV on M4) with current
bridge addresses (M2 169.254.201.199, mini 169.254.130.250; pinned host key alias), via `run.sh`. Same prompt,
96 max tokens, 3 repeats per process; warm post-first decode tok/s. Each run passed tokens_agree / strict numerical
admission / cleanup. Split = (M2 final layers, M4 initial layers).

| Run | split | group | post-first tok/s (3 repeats) |
|---|---|---|---|
| control2 | 28/36 | 32 | 9.995, 10.301, 10.348 |
| s32_32 | 32/32 | 32 | 10.365, 10.464, 10.446 |
| s34_30 | 34/30 | 32 | 9.890, 9.966, 9.881 |
| control3 | 28/36 | 32 | 10.261, 10.397, 10.437 |
| g16 | 28/36 | 16 | 10.053, 10.233, 10.271 |
| s30_34 | 30/34 | 32 | 9.800, 9.843, 9.817 |

Conclusion: the saved 28/36 group-32 profile is already at a plateau (~10.3–10.4 warm). 32/32 is +~0.5% vs the
bracketing controls, within run-to-run noise; not adopted. Single prompt only.

Failures kept: control1 (local small-control model had been pruned from scratch; restored to
`models/control-qwen35-08b-…`, SHA-256 verified vs mini), long_control/long_s32 (`--max-tokens 256` rejected
with ValueError by runner admission).
Google Chrome and the Hermes Chrome Beta job browser were closed on the M2 before control2 at the user's request.
