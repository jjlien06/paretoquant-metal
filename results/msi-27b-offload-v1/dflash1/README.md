# MSI Qwen3.8-27B DFlash2 vs MTP vs Off (llama.cpp b11968, 2026-11-06)

Target: Qwen3.8-27B-UD-Q4_K_M (pq-27b volume), `-ngl 22 -fa on -t 8 -c 4096 -np 1`, one-off model-lab:latest container (-m 14g).
Drafter: z-lab/Qwen3.8-27B-DFlash2-GGUF rev 2d9571f8ce46e151f61c6499c99dee6079e1d610, Q4_K_M,
SHA-256 1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd (matches HF LFS), `-ngld 99`.
Harness: same 4 prompts x 256 tokens greedy, enable_thinking false (mtp1/runs.tmpl.sh), 1 rep. Order D1..D5 (Off brackets).
Greedy outputs of DFlash2 identical to Off (head check). Q8_0 not downloaded (paretoquant disk budget headroom 1.28 GB).

tg tok/s from server timings; τ = llama.cpp "mean len" (accepted tokens per verify step).

| config | p0 code | p1 turbopump | p2 primes | p3 essay | geomean tg speedup | geomean wall speedup |
|---|---|---|---|---|---|---|
| D1_off | 2.92 | 2.89 | 2.95 | 2.92 | 1.00x | 1.00x |
| D2_dflash n-max 7 | 7.52 (τ 7.29) | 3.32 (τ 3.15) | 6.78 (τ 6.51) | 2.90 (τ 2.77) | 1.61x | 1.53x |
| D3_dflash n-max 15 (clamped to 7, block_size 8) | 7.24 (τ 7.29) | 3.33 (τ 3.15) | 6.74 (τ 6.51) | 2.86 (τ 2.77) | 1.59x | 1.45x |
| D4_mtp n-max 3 | 4.84 (τ 3.79) | 3.46 (τ 2.63) | 3.65 (τ 3.53) | 3.26 (τ 2.58) | 1.28x | 1.11x |
| D5_off_repeat | 2.94 | 2.91 | 2.95 | 2.95 | 1.00x | 1.00x |

Draft counts (accepted/generated), dflash7: 220/240, 174/559, 215/272, 163/636.

Notes:
- Prose prompts: DFlash τ≈2.8-3.2 but each verify step is an 8-token batch; at partial offload (CPU-bandwidth bound) the
  wider verify costs ~as much as the tokens saved -> ~1.0-1.15x, essay slightly slower than MTP3.
- Structured prompts (code, primes): τ 6.5-7.3 -> 2.3-2.6x, far above MTP3.
- Verify cost per step ≈ τ/(tg/off_tg): p0 7.29/2.56≈2.8 Off-token-times, p3 2.77/0.99≈2.8 -> an 8-wide verify costs ~2.8x a single decode step here.
- MTP3 in this session (1.28x) reproduces mtp1 (~1.33x); p2 MTP lower than mtp1 (3.65 vs 4.67) with identical draft counts -> timing noise.
- Raw: raw.txt; scripts: runs.sh, go.sh/go2.sh (go.sh grant-grep bug held the lease ~20 min; fixed).
