# TP all-reduce feasibility screen v1 — 2026-11-05

**Verdict: NO-GO for 2-way tensor parallelism of Qwen3.8-27B over the current
Thunderbolt TCP ring.** Measured dependent all-reduce cost (~0.9–1.3 ms) is
~2.5–3× the break-even (~0.41–0.43 ms with a bandwidth-proportional split,
~0.29–0.30 ms with an even split). Modeled TP pass ≈ 157–213 ms vs the current
~96–100 ms pipeline pass (≈1.6–2.2× slower). Independently, spike 013's native
BF16 TP numerical gate already failed (treated as rejection).

## What ran

- `spikes/028-tp-allreduce-screen/{bench.py,launch.py,model.py}`; MLX 0.32.3 both hosts.
- Two real ranks, `mx.distributed.init(strict=True, backend="ring")`, hostfile
  `[[169.254.201.199:p0],[169.254.130.250:p1]]` (rank0 = M2 Pro local, rank1 = M4 mini),
  same convention as the 021 supervisor. `route -n get` showed `bridge0` both ways.
- Each step: `quantized_matmul` (random 4-bit [D,D], group 64) on the previous
  step's reduced BF16 `[1,D]` output → `all_sum` → `rms_norm`. So every all-reduce
  input is a fresh dependent GPU product. 300 warm-up steps + 3 warm-up tokens per phase.
- Phases per width (D=5120 "1x", 10240 "2x"): per-call (eval each step, 1024 samples)
  and per-token lazy chain (128 dependent steps, one eval, 32 tokens), each with and
  without all_sum. Overhead = comm − no-comm (no-comm = slower rank, the M4).
- Runs: `gpu-a`, `gpu-b` (default/GPU stream all_sum), `cpu-a` (`stream=mx.cpu`).
  No model weights; peak MLX allocation 688 MB per rank. Ranks' final hashes agreed.
- ICMP RTT over the bridge (context): avg 0.65–0.74 ms, min 0.46–0.52 ms.

## Measured (per all-reduce, ms)

| width | run | in-chain overhead (token p50) | per-call comm p50 (r0/r1) | per-call overhead p50 |
|---|---|---:|---|---:|
| 1x 5120 (10 KB) | gpu-a | 1.085 | 1.52 / 1.43 | 1.14 |
| 1x | gpu-b | 1.050 | 1.39 / 1.36 | 0.97 |
| 1x | cpu-a | 1.015 | 1.42 / 1.35 | 1.04 |
| 2x 10240 (20 KB) | gpu-a | 1.311 | 2.82 / 2.41 | 1.93 |
| 2x | gpu-b | 1.159 | 1.92 / 1.85 | 1.02 |
| 2x | cpu-a | 0.894 | 1.68 / 1.62 | 0.79 |

Per-call p90 ≈ 2.7–5.0 ms (heavy tail). Token-chain comm p50 at 1x: 150–159 ms per
128 calls vs 20 ms compute-only on the M4. Full distributions in `runs/*.rank*.json`.

## Model (`analysis.json`)

Per-layer times derived from the current 28/36 pass and the measured M4/M2 qmm time
ratio 1.75: M2 ≈1.06–1.10 ms/layer, M4 ≈1.85–1.92 ms/layer.
TP compute per pass: even split 59–62 ms; bandwidth-proportional split 43–45 ms.
TP pass = compute + 128 × all-reduce → 157–213 ms (proportional) / 174–229 ms (even).
Break-even all-reduce: (pass − TP compute)/128 = **0.41–0.43 ms** (proportional),
**0.29–0.30 ms** (even). The bridge's bare ICMP RTT already exceeds this.
Model is optimistic for TP (ignores the pipeline hop cost already inside 96–100 ms,
embed/head, MTP verify width-2 traffic, non-ideal half-layer kernels).

## Cleanup

All launcher receipts (`runs/*.launcher.json`) show exit 0 both ranks, no bench
PIDs left on either host, both ring ports refused connections afterwards, and rank1
JSON SHA-256 matched after fetch. Remote copies (bench.py, hostfiles, rank1 JSONs)
remain in `paretoquant-dual-host/spikes/028-tp-allreduce-screen/` on the mini.
