# Qwen3.8-27B on the MSI gaming laptop (RTX 4070 Laptop 8 GB, Ryzen 7 8845HS, 32 GB) — 2026-11-05

- Model: `unsloth/Qwen3.8-27B-GGUF` rev `4ca720788d1e01f1bff70c033e0d0028fd02e502`, `Qwen3.8-27B-UD-Q4_K_M.gguf`,
  16,464,440,224 bytes, SHA-256 `322e194f…3482` (matches HF LFS; re-verified after copy into Docker volume `pq-27b`).
- Engine: llama.cpp b11968 (`ac8e1694d`), CUDA, the model-lab build, one-off container `-m 14g`, `-fa 1 -t 8 -b 2048 -ub 512`.
- Bench: `llama-bench -p 512 -n 128 -r 3 -d 0,4096`. Raw JSON in `c1/`.

| GPU layers (of 64) | prefill t/s (d0) | decode t/s (d0) | prefill t/s (d4096) | decode t/s (d4096) |
|---|---|---|---|---|
| 20 | 245.9 | 2.86 | 305.3 | 2.67 |
| 24 | 320.7 | 3.09 | 319.1 | 2.89 |
| 28 | 320.6 | 3.33 | 335.0 | 3.09 |
| 31 | 65.6* | 3.53 | 327.7 | 3.03 |

*first test of that process; likely a warm-up outlier, not re-run.

ngl 34 ran but its output was lost when the controlling SSH session was killed; not reported.

## Invalid attempt (kept for honesty)
`sweep1/` — native Windows llama.cpp: pp512 19 t/s because WSL (vmmemWSL 16.4 GB) left Windows 0.9 GB free and the
CPU-resident weights paged from disk. Aborted; do not use.

## Comparison context (not same-day, different runtime)
Two 16 GB Macs (M2 Pro + M4 mini, Thunderbolt), MLX 4-bit same model family: 9.49–11.34 tok/s warm decode with MTP
(`results/qwen38-broad-v1`). Different quant format and engine; a rough hardware comparison, not a controlled one.

## Tuning session (2026-11-05 17:06–17:58 EST), same model/engine, container `-m 14g`
llama-bench `-p 512 -n 128 -r 3 -d 0`, raw in `tune1/ tune2/ tune3/raw.txt` (parse with `parse_bench.py`).

| Config | prefill t/s | decode t/s |
|---|---|---|
| ngl 28, t8 (control, 3 repeats across sessions) | 262–339 | 3.31–3.33 |
| ngl 32, q8_0 KV | 52 | 3.23 |
| ngl 34, q8_0 KV | 59 | 3.29 |
| ngl 36, q8_0 KV | 56 | 3.02 |
| all attention on GPU, FFN of layers 22–63 on CPU | 64 | 3.10 |
| all attention on GPU, FFN of layers 26–63 on CPU | 64 | 2.69 |
| ngl 28, 6 threads | 306 | 3.23 |
| ngl 28, 16 threads | 261 | 2.23 |
| ngl 28, no mmap (`-lm none`) | 363 | 3.29 |
| ngl 32, no mmap | 69 | 3.58 |

Layer placement/threads/KV/mmap tuning does not move decode beyond ~3.3–3.6 t/s: the CPU-resident half of the weights
is memory-bandwidth bound. Pushing past ~28 GPU layers collapses prefill ~5× (VRAM near full).

## Native MTP speculative decoding (llama-server, `--spec-type draft-mtp`, unsloth `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`,
SHA-256 `50d9ce5a…1c6e`, n-max 3, head on GPU), ngl 24, greedy, 4 prompts × 256 tokens, off / MTP / off bracketing.
Raw in `mtp1/raw.txt`.

| Prompt | off decode | MTP decode | MTP accept rate |
|---|---|---|---|
| merge sorted lists (code) | 3.07 / 3.07 | 5.09 | 96% |
| turbopump explanation | 3.07 / 3.04 | 3.27 | 50% |
| first 40 primes | 2.76 / 3.07 | 4.67 | 85% |
| 2008 crisis essay | 3.05 / 3.03 | 3.40 | 53% |

Geometric-mean decode speedup ≈1.33×; total wall for the 4 requests 268.5 s vs 351.4 / 343.5 s off (≈1.29×).
First 40–80 output characters match off-mode; full-output identity was not checked. Single repetition, n=4 prompts:
a pilot, not a stable claim.
