# Qwen3.8-27B width-3 verify kernel, v1 (2026-11-05)

Code: `spikes/031-qwen38-width3-kernel/`, a copy of 030 plus the files below. Run snapshots are
`spikes/031-qwen38-width3-kernel-s{1,2}{a..f}/`. Each is mirrored to the mini with SHA-256
readback (`*.deploy.json`).

Setup:
- Launcher: `run.sh <run> 0.5`.
- Rank0 = M2 Pro, layers 36..63 plus head, verifier `native`.
- Rank1 = M4 mini, layers 0..35, new verifier `slot-qmv-tuned-m4`.
- group32, prefix rollback, target `10c35caa…`, sidecar `b643c01b…`.
- Bridge: 169.254.201.199 ↔ 169.254.130.250. Routes were checked to use bridge0 in both
  directions.

## Stage 1: profile (PASS)

### Synthetic Q4 weights at real 27B shapes

Timing is synchronized, with warm-up and interleaved runs. Sources: `profile_width.py`,
`op_screen.py`, `tune.py`.

**M2, 8 layers** (`profile-m2-8layer*.json`), native:

| width | time |
|---:|---:|
| W1 | 13.4–15.4 ms |
| W2 | 19.8–21.4 ms |
| W3 | 29.0–30.8 ms |

- Width 3 adds about 1.15 ms per layer, which is about 32 ms across 28 layers.
- Quantized matmuls account for about 85% of that. They run about 1.5× W2 and are ALU/issue-bound,
  not bandwidth-bound.
- GDN recurrence costs 0.3–0.45 ms per call and is flat across widths.
- lm_head at W3 costs 10.7 ms vs 7.5 ms at W2.

**M4, 8 layers** (`profile-m4-8layer.json`):

| verifier | W2 | W3 |
|---|---:|---:|
| native | 21.5 ms | 23.6 ms |
| 030 slot-qmv | 22.5 ms | 31.0 ms |

- Stock QMM on M4 is bandwidth-bound and flat from W1 to W3.
- Almost all of the M4 width-3 increase comes from the 030 slot-QMV launch. Its 4-rows × 2-simdgroups
  geometry runs about 33% slower at 3 slots.

## Stage 2: kernel (PASS)

### What was built

- `slot_qmv_t.py` with `slot_qmv_header_t.metal` / `slot_qmv_body_t.metal`:
  - templated rows per simdgroup and simdgroups per threadgroup;
  - MODE 1: decode each 4-weight chunk once and apply it to all slots;
  - MODE 2: slot-outer loop.
- Per-output K partition and arithmetic are the pinned stock-qmv ones.
- `tune.py` admits a geometry only if it is bitwise equal to stock singleton QMV on 6 inputs per
  shape. It then picks the fastest.
- `geometry_table.json` records the choices, with per-host defaults (m2 native, m4 030 slot-qmv).
- `verifier.py` adds the opt-in modes `slot-qmv-tuned-m2` and `slot-qmv-tuned-m4`, applied at width 3
  only. Kernel exceptions propagate; there is no hidden fallback.
- All new files are in `remote_actor.SOURCES`.
- CPU tests: 283 + 4 new (`test_tuned.py`) pass.

### Results per host

**M4**: the tuned kernel reached about stock speed at W3. Examples:
- mlp 0.80 → 0.60 ms;
- qkv 0.47 → 0.36 ms.

Op-level evidence: `tune-m4-w23.json`, `small-shapes-exact-m4.json`. The 0.8B-control shapes use
fixed 2x2 MODE 1, so the small gate also exercises the new kernel.

**M2**: exact geometries won only about 4% per op (`tune-m2-w23.json`) and nothing at layer level
(32.0 vs 30.8 ms native). Rank0 therefore stays native.

Tried and rejected:
- Word-load variant: a type-punning bug gave inexact output.
- MODE 2: slower.
- lm_head slot kernel: slower than native.

### Prediction and admission

- Predicted from 030 calibration rounds (`tau-prediction.json`, MODELLED): adaptive vs depth-1 is
  1.14× at tau 0.5, or 1.11× with only 75% of the saving.
- Physical admission passed in all 12 processes, each in the same process at atol 1e-6 / rtol 1e-5:
  - small and full checkpoints;
  - width 2 and 3 vs sequential width 1: full logits and every live cache, on both ranks;
  - prefix keep 0..3 with continuation.
- `strict_numerical_admission=true` in every `run-*/result.json`. Rank1 mode is recorded as
  `slot-qmv-tuned-m4`.

## Stage 3: paired benchmark (`summary-final.json`, `analysis-final.json`)

Design:
- Broad-v1, 18 cases × 2 samples, in 12 processes (`run-s1a..f`, `run-s2a..f`).
- Each process runs native / depth-1 MTP / adaptive (tau 0.5) triples, with rotating
  counterbalanced orders and one code-short warm-up triple.
- At most 2 long cases (6 long requests) per process. All runs were under the log cap.

Status:
- All 12 supervisors reported `success`: rank actions agree, listeners closed, owned actors cleaned.
- All 72 accelerated outputs match native token IDs and stop reasons.

### Per-round cost (rank0 wall, medians)

| | 030 | 031 |
|---|---:|---:|
| width-2 verify | 163.3 ms | 162.3 ms (n=4046) |
| width-3 verify | 228.3 ms | **194.4 ms** (n=2438) |
| width-3 round period | 250.5 ms | 217.0 ms |

### Adaptive (031) vs depth-1

Equal-case geomean of per-case sample geomeans.

| scope | warm post-first decode | whole request |
|---|---:|---:|
| all 18 cases | **1.105×** | **1.059×** |
| code | 1.164 | 1.081 |
| math | 1.210 | 1.120 |
| structured | 1.185 | 1.115 |
| prose (explanation) | 1.033 | 1.002 |
| Spanish | 1.003 | 1.007 |
| Chinese | 1.053 | 1.036 |

vs native: depth-1 gives 1.416× decode / 1.119× whole; adaptive gives 1.565× / 1.185×.

On 1536-token prompts, whole-request time is below native for both methods (0.93 / 0.98), because
prefill is serialized. Weak cases:
- explanation-512: 0.971 / 0.941;
- spanish-1536: 0.969 decode.

Math-1536 at 1.416 decode is an outlier driven by two samples.

## Caveats

- tau 0.5 was chosen from a model of 030 calibration rounds (6 cases), not tuned on this suite.
- The gain is acceptance-dependent: about 0% to +5% on prose, Spanish and Chinese.
- M2 width-3 compute is unchanged. The remaining width-3 premium (~32 ms) is the M2 native QMM plus
  lm_head.
- Rank1 `rank1.stdout` copies are capped at 288 KiB in the run directory. The supervisor's own
  verification is authoritative.
