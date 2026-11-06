# Qwen3.8-27B adaptive depth-1/2 MTP drafting, v1 (2026-11-05)

Two-Mac runtime: M2 Pro rank0 (final 28 layers + MTP head, **native** verifier), M4 mini rank1
(first 36 layers, **slot-QMV** verifier), group32, prefix rollback, target rev `10c35caa…`,
sidecar `b643c01b…`, bridge0 169.254.201.199 ↔ 169.254.130.250 (routes checked both ways).
Code: `spikes/030-qwen38-adaptive-mtp/` (copy of the 029/023 schedule runner). Each run's
snapshot is `spikes/030-qwen38-adaptive-mtp-<run>/`, mirrored to the mini under
`~/Projects/paretoquant-dual-host/spikes/` with full-file SHA-256 readback (`*.deploy.json`).
Launcher: `run.sh <run> <tau>`.

**Result:** adaptive depth is correct and admitted, but its speed gain is small: about
+1.5% decode over depth-1 across all cases, and roughly −1% to +1% on code. A width-3
verify costs about 2.3× more than the earlier model assumed.

## A. Admitted (physical, same-process, strict atol 1e-6 / rtol 1e-5, unchanged)
Every run (cal, bench-a, a1, a2, b) first passed these gates on both GPUs:
- Small 0.8B checkpoint and the loaded 27B shards.
- Width-2 and **width-3** verify vs. sequential width-1: full logits, argmax, and every live
  cache on both ranks.
- Prefix capture gates: keep 0/1/2 at width 2 and **keep 0/1/2/3 at width 3**, each with a
  continuation.

M4 slot-QMV at 3 slots and M2 native at width 3 were both **bit-exact** (max_abs 0.0), so the
singleton-QMM fallback was not needed. The supervisor checks the width-3 evidence for both
ranks.

Local checks (M2 only):
- The small gate passed for native, slot-qmv and singleton-qmm.
- A mutation test (sabotaged width-3 keep-2 commit) was caught.
- CPU unit tests: 283 passed. This includes 183 symbolic adaptive control-flow cases (EOS,
  caps, tau 0/0.5/1, cache convention) and rank-command bounds.

## B. Measured
### Per-round cost (rank0 snapshot-ACK → verify-ACK wall, medians, `analysis-final.json`)

| | verify | round period |
|---|---:|---:|
| width 2 (depth-1) | 163.3 ms (n=1129) | 177.7 ms |
| width 3 (adaptive) | 228.3 ms (n=517) | 250.5 ms |

The width-3 step adds about 65 ms. Earlier work assumed 14–28 ms. The extra cost is
superlinear and roughly split between the hosts:
- M2 block: 63.5 → 92.3 ms.
- M4 compute: about 91 → 119 ms (from `run-cal`).

### Calibration (`run-cal`, tau=0, v1 rule: always verify d2)
Six cases (code-short/512, structured-short, math-short, explanation-short, chinese-short),
1 warm-up triple. Uniform depth-2 was **0.968× depth-1 decode** overall; code was 1.028×;
prose and Chinese were 0.85–0.88×.

Acceptance by head confidence, P(d2 accepted | d1 accepted):

| p2 | rate | n |
|---|---:|---:|
| ≥ 0.99 | 1.00 | 164 |
| 0.95–0.99 | 0.97 | 32 |
| 0.8–0.95 | 0.68 | 37 |
| 0.5–0.8 | 0.59 | 39 |
| < 0.5 | 0.23 | 26 |

### Paired benchmark (`run-bench-a1`, `-a2`, `-b`)
- Same-process **native / depth-1 MTP / adaptive (tau=0.9)** triples, method order
  counterbalanced (rotating permutations), one code-short warm-up triple per process.
- 11 measured cases: all code, structured and math cases, plus explanation-512 and
  spanish-short. One sample per method per case.
- **All 33 accelerated outputs match native token IDs and stop reasons.** All supervisors
  reported `success`: rank actions agree, listeners closed, owned groups gone.

Warm post-first decode is `(N-1)/(wall-first)`. Whole-request is the native/candidate wall
ratio.

| case | native tok/s | depth1 | adaptive | adaptive/depth1 decode | adaptive/depth1 whole | width-3 rounds |
|---|---:|---:|---:|---:|---:|---:|
| code-short | 7.50 | 11.03 | 10.53 | 0.955 | 0.955 | 32/51 |
| code-512 | 7.22 | 10.69 | 10.84 | 1.014 | 1.080 | 29/52 |
| code-1536 | 6.58 | 10.59 | 10.56 | 0.997 | 1.035 | 61/106 |
| structured-short | 7.43 | 11.21 | 11.77 | 1.050 | 1.044 | 39/44 |
| structured-512 | 7.40 | 11.19 | 11.61 | 1.038 | 0.966 | 39/45 |
| structured-1536 | 7.01 | 11.03 | 11.16 | 1.012 | 0.975 | 69/95 |
| math-short | 7.45 | 10.66 | 10.65 | 0.999 | 0.998 | 34/53 |
| math-512 | 7.38 | 10.52 | 10.65 | 1.012 | 1.058 | 28/55 |
| math-1536 | 5.89 | 10.61 | 10.94 | 1.031 | 1.010 | 63/103 |
| explanation-512 | 7.25 | 10.18 | 10.17 | 0.999 | 1.082 | 9/66 |
| spanish-short | 7.39 | 9.93 | 10.54 | 1.061 | 1.055 | 18/61 |

Equal-case geometric means:

| | vs native decode | vs native whole | adaptive / depth1 decode | adaptive / depth1 whole |
|---|---:|---:|---:|---:|
| all 11 cases, depth1 | 1.501× | 1.154× | | |
| all 11 cases, adaptive | 1.523× | 1.180× | **1.015×** | 1.023× |
| code (3 cases), depth1 | 1.519× | 1.137× | | |
| code (3 cases), adaptive | 1.501× | 1.163× | **0.989×** | 1.022× |

Notes:
- Run-to-run noise with one sample is about ±3–5%. For example, code-short adaptive/depth1
  was 1.016 in the failed bench-a trial versus 0.955 here.
- Whole-request ratios include prefill variance.
- Native long-prompt decode was unusually slow in `-a1`/`-a2` (5.9–6.6 tok/s). This inflates
  the native-relative numbers there; adaptive/depth1 is unaffected.
- The difference between adaptive and depth-1 is **not resolved from noise**.

## C. Modelled (not measured), `tau-selection.json`, `analysis-cal.json`
Inputs: in-sample on the 6 calibration cases, with measured costs held fixed at round
boundaries.

| rule | modelled vs depth1 |
|---|---:|
| p2-only gate | ≤ 1.014× |
| p1 prefilter + p2 gate (both = tau) | 1.03× at 0.8–0.99, a flat optimum |

tau = 0.9 was chosen. The benchmark measured 1.015×, below the 1.03× model.

## Failed or preserved trials
- `failed-launch-missing-runsh.log`: the launcher was missing (bash 127). Nothing ran.
- `run-bench-a` (5 cases incl. 3×1536/256-token): all 18 requests completed and outputs
  matched native.
  - Rank1's stderr reached the 64 MiB actor log cap (67,110,876 B). Rank1 was terminated
    (−15) and the supervisor marked the run `failed`.
  - Its numbers are kept only as `analysis-bench-a-FAILED-TRIAL-unverified.json` and are
    not used above.
  - The rerun split it into a1/a2.

## Caveats
- One sample per method per case, greedy decoding, caps 128/256. This is descriptive, with
  no significance claim.
- tau was chosen in-sample (code-short/512, structured-short and math-short are calibration
  cases). The two prose cases are held out.
- `run-cal` used rule v1 (d2 always computed). The bench runs use v2 (p1 prefilter).
- The depth-1 control is the 021 draft path (`head.forward`, no chain). It shares the 030
  verifier/rank code, whose width-2 behaviour is unchanged.

## Recommendation
Keep adaptive depth opt-in and do not adopt it as the default. It is correct, but the gain is
within noise, including on code. The bottleneck is width-3 verify compute (+65 ms), not draft
acceptance. The next step is to make width 3 cheaper on M2 (e.g. slot-QMV or singleton at
width 3 on rank0) and on M4, then rerun this paired suite.

## Reproduce
    spikes/030-qwen38-adaptive-mtp/make_run.py <suffix> results/qwen38-adaptive-mtp-v1/broad-v1-cases.json <cases> --warmup code-short --order-offset N
    results/qwen38-adaptive-mtp-v1/run.sh <suffix> <tau>
    .venv/bin/python -B spikes/030-qwen38-adaptive-mtp/analyze.py out.json results/qwen38-adaptive-mtp-v1/run-bench-a1 …-a2 …-b
