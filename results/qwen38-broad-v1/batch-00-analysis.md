# Qwen3.8 workload evidence analysis

**Coverage: partial** — 14/84 requests; 12 measured, 2 explicit warmups; 24 requested cases.

Observed successful rows only; not a full-suite performance claim. Missing requests and failures have no imputed throughput and remain listed separately.

Statistics are descriptive: sample stdev is undefined for n<2. Lower-tail minima are not a confidence interval. Two repeats cannot establish reliable tail latency. Cases are never pooled as tokens/sec. Token events include any emitted EOS. Single-token outputs are excluded only from post-first decode.

No quality scoring or runtime verification is performed. Rank agreement, checkpoint identity, placement, eval32, gate receipts, cleanup, and resident-sidecar memory baselines must be established upstream.

EOS token identity checked using supplied metadata: True.

## Per-case measured statistics

| Case | Method | Metric | n | Median | Min | Max | Sample stdev |
|---|---|---|---:|---:|---:|---:|---:|
| b00-warmup | native | post_first_tps | 0 | — | — | — | — |
| b00-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b00-warmup | native | wall_seconds | 0 | — | — | — | — |
| b00-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b00-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b00-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b00-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b00-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| code-1536 | native | post_first_tps | 2 | 7.44567 | 7.38091 | 7.51044 | 0.0915941 |
| code-1536 | native | first_token_seconds | 2 | 15.081 | 14.832 | 15.33 | 0.35208 |
| code-1536 | native | wall_seconds | 2 | 49.3317 | 48.7848 | 49.8786 | 0.773419 |
| code-1536 | native | whole_request_tps | 2 | 5.19 | 5.13247 | 5.24754 | 0.0813686 |
| code-1536 | mtp | post_first_tps | 2 | 11.0361 | 10.9952 | 11.0771 | 0.0578958 |
| code-1536 | mtp | first_token_seconds | 2 | 25.2624 | 25.0446 | 25.4802 | 0.307976 |
| code-1536 | mtp | wall_seconds | 2 | 48.3686 | 48.2366 | 48.5007 | 0.186761 |
| code-1536 | mtp | whole_request_tps | 2 | 5.29273 | 5.27828 | 5.30718 | 0.0204362 |
| code-512 | native | post_first_tps | 2 | 7.59347 | 7.57417 | 7.61276 | 0.027289 |
| code-512 | native | first_token_seconds | 2 | 6.01348 | 6.01158 | 6.01537 | 0.00268044 |
| code-512 | native | wall_seconds | 2 | 22.7385 | 22.6941 | 22.7829 | 0.062786 |
| code-512 | native | whole_request_tps | 2 | 5.62924 | 5.61825 | 5.64023 | 0.0155436 |
| code-512 | mtp | post_first_tps | 2 | 11.2106 | 11.1951 | 11.2262 | 0.021996 |
| code-512 | mtp | first_token_seconds | 2 | 9.43746 | 8.49633 | 10.3786 | 1.33096 |
| code-512 | mtp | wall_seconds | 2 | 20.766 | 19.8092 | 21.7229 | 1.35319 |
| code-512 | mtp | whole_request_tps | 2 | 6.17703 | 5.89241 | 6.46165 | 0.402517 |
| code-short | native | post_first_tps | 2 | 7.57759 | 7.51598 | 7.6392 | 0.0871295 |
| code-short | native | first_token_seconds | 2 | 1.06607 | 1.0368 | 1.09533 | 0.0413868 |
| code-short | native | wall_seconds | 2 | 17.8271 | 17.6616 | 17.9927 | 0.234111 |
| code-short | native | whole_request_tps | 2 | 7.18069 | 7.11401 | 7.24737 | 0.0942988 |
| code-short | mtp | post_first_tps | 2 | 11.2351 | 11.2143 | 11.256 | 0.0294398 |
| code-short | mtp | first_token_seconds | 2 | 1.19885 | 1.19529 | 1.20242 | 0.00504409 |
| code-short | mtp | wall_seconds | 2 | 12.5027 | 12.4782 | 12.5272 | 0.0346639 |
| code-short | mtp | whole_request_tps | 2 | 10.2378 | 10.2178 | 10.2579 | 0.0283845 |
| b01-warmup | native | post_first_tps | 0 | — | — | — | — |
| b01-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b01-warmup | native | wall_seconds | 0 | — | — | — | — |
| b01-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b01-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b01-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b01-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b01-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b02-warmup | native | post_first_tps | 0 | — | — | — | — |
| b02-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b02-warmup | native | wall_seconds | 0 | — | — | — | — |
| b02-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b02-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b02-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b02-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b02-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b03-warmup | native | post_first_tps | 0 | — | — | — | — |
| b03-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b03-warmup | native | wall_seconds | 0 | — | — | — | — |
| b03-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b03-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b03-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b03-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b03-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b04-warmup | native | post_first_tps | 0 | — | — | — | — |
| b04-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b04-warmup | native | wall_seconds | 0 | — | — | — | — |
| b04-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b04-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b04-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b04-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b04-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b05-warmup | native | post_first_tps | 0 | — | — | — | — |
| b05-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b05-warmup | native | wall_seconds | 0 | — | — | — | — |
| b05-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b05-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b05-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b05-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b05-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| chinese-1536 | native | post_first_tps | 0 | — | — | — | — |
| chinese-1536 | native | first_token_seconds | 0 | — | — | — | — |
| chinese-1536 | native | wall_seconds | 0 | — | — | — | — |
| chinese-1536 | native | whole_request_tps | 0 | — | — | — | — |
| chinese-1536 | mtp | post_first_tps | 0 | — | — | — | — |
| chinese-1536 | mtp | first_token_seconds | 0 | — | — | — | — |
| chinese-1536 | mtp | wall_seconds | 0 | — | — | — | — |
| chinese-1536 | mtp | whole_request_tps | 0 | — | — | — | — |
| chinese-512 | native | post_first_tps | 0 | — | — | — | — |
| chinese-512 | native | first_token_seconds | 0 | — | — | — | — |
| chinese-512 | native | wall_seconds | 0 | — | — | — | — |
| chinese-512 | native | whole_request_tps | 0 | — | — | — | — |
| chinese-512 | mtp | post_first_tps | 0 | — | — | — | — |
| chinese-512 | mtp | first_token_seconds | 0 | — | — | — | — |
| chinese-512 | mtp | wall_seconds | 0 | — | — | — | — |
| chinese-512 | mtp | whole_request_tps | 0 | — | — | — | — |
| chinese-short | native | post_first_tps | 0 | — | — | — | — |
| chinese-short | native | first_token_seconds | 0 | — | — | — | — |
| chinese-short | native | wall_seconds | 0 | — | — | — | — |
| chinese-short | native | whole_request_tps | 0 | — | — | — | — |
| chinese-short | mtp | post_first_tps | 0 | — | — | — | — |
| chinese-short | mtp | first_token_seconds | 0 | — | — | — | — |
| chinese-short | mtp | wall_seconds | 0 | — | — | — | — |
| chinese-short | mtp | whole_request_tps | 0 | — | — | — | — |
| explanation-1536 | native | post_first_tps | 0 | — | — | — | — |
| explanation-1536 | native | first_token_seconds | 0 | — | — | — | — |
| explanation-1536 | native | wall_seconds | 0 | — | — | — | — |
| explanation-1536 | native | whole_request_tps | 0 | — | — | — | — |
| explanation-1536 | mtp | post_first_tps | 0 | — | — | — | — |
| explanation-1536 | mtp | first_token_seconds | 0 | — | — | — | — |
| explanation-1536 | mtp | wall_seconds | 0 | — | — | — | — |
| explanation-1536 | mtp | whole_request_tps | 0 | — | — | — | — |
| explanation-512 | native | post_first_tps | 0 | — | — | — | — |
| explanation-512 | native | first_token_seconds | 0 | — | — | — | — |
| explanation-512 | native | wall_seconds | 0 | — | — | — | — |
| explanation-512 | native | whole_request_tps | 0 | — | — | — | — |
| explanation-512 | mtp | post_first_tps | 0 | — | — | — | — |
| explanation-512 | mtp | first_token_seconds | 0 | — | — | — | — |
| explanation-512 | mtp | wall_seconds | 0 | — | — | — | — |
| explanation-512 | mtp | whole_request_tps | 0 | — | — | — | — |
| explanation-short | native | post_first_tps | 0 | — | — | — | — |
| explanation-short | native | first_token_seconds | 0 | — | — | — | — |
| explanation-short | native | wall_seconds | 0 | — | — | — | — |
| explanation-short | native | whole_request_tps | 0 | — | — | — | — |
| explanation-short | mtp | post_first_tps | 0 | — | — | — | — |
| explanation-short | mtp | first_token_seconds | 0 | — | — | — | — |
| explanation-short | mtp | wall_seconds | 0 | — | — | — | — |
| explanation-short | mtp | whole_request_tps | 0 | — | — | — | — |
| math-1536 | native | post_first_tps | 0 | — | — | — | — |
| math-1536 | native | first_token_seconds | 0 | — | — | — | — |
| math-1536 | native | wall_seconds | 0 | — | — | — | — |
| math-1536 | native | whole_request_tps | 0 | — | — | — | — |
| math-1536 | mtp | post_first_tps | 0 | — | — | — | — |
| math-1536 | mtp | first_token_seconds | 0 | — | — | — | — |
| math-1536 | mtp | wall_seconds | 0 | — | — | — | — |
| math-1536 | mtp | whole_request_tps | 0 | — | — | — | — |
| math-512 | native | post_first_tps | 0 | — | — | — | — |
| math-512 | native | first_token_seconds | 0 | — | — | — | — |
| math-512 | native | wall_seconds | 0 | — | — | — | — |
| math-512 | native | whole_request_tps | 0 | — | — | — | — |
| math-512 | mtp | post_first_tps | 0 | — | — | — | — |
| math-512 | mtp | first_token_seconds | 0 | — | — | — | — |
| math-512 | mtp | wall_seconds | 0 | — | — | — | — |
| math-512 | mtp | whole_request_tps | 0 | — | — | — | — |
| math-short | native | post_first_tps | 0 | — | — | — | — |
| math-short | native | first_token_seconds | 0 | — | — | — | — |
| math-short | native | wall_seconds | 0 | — | — | — | — |
| math-short | native | whole_request_tps | 0 | — | — | — | — |
| math-short | mtp | post_first_tps | 0 | — | — | — | — |
| math-short | mtp | first_token_seconds | 0 | — | — | — | — |
| math-short | mtp | wall_seconds | 0 | — | — | — | — |
| math-short | mtp | whole_request_tps | 0 | — | — | — | — |
| spanish-1536 | native | post_first_tps | 0 | — | — | — | — |
| spanish-1536 | native | first_token_seconds | 0 | — | — | — | — |
| spanish-1536 | native | wall_seconds | 0 | — | — | — | — |
| spanish-1536 | native | whole_request_tps | 0 | — | — | — | — |
| spanish-1536 | mtp | post_first_tps | 0 | — | — | — | — |
| spanish-1536 | mtp | first_token_seconds | 0 | — | — | — | — |
| spanish-1536 | mtp | wall_seconds | 0 | — | — | — | — |
| spanish-1536 | mtp | whole_request_tps | 0 | — | — | — | — |
| spanish-512 | native | post_first_tps | 0 | — | — | — | — |
| spanish-512 | native | first_token_seconds | 0 | — | — | — | — |
| spanish-512 | native | wall_seconds | 0 | — | — | — | — |
| spanish-512 | native | whole_request_tps | 0 | — | — | — | — |
| spanish-512 | mtp | post_first_tps | 0 | — | — | — | — |
| spanish-512 | mtp | first_token_seconds | 0 | — | — | — | — |
| spanish-512 | mtp | wall_seconds | 0 | — | — | — | — |
| spanish-512 | mtp | whole_request_tps | 0 | — | — | — | — |
| spanish-short | native | post_first_tps | 0 | — | — | — | — |
| spanish-short | native | first_token_seconds | 0 | — | — | — | — |
| spanish-short | native | wall_seconds | 0 | — | — | — | — |
| spanish-short | native | whole_request_tps | 0 | — | — | — | — |
| spanish-short | mtp | post_first_tps | 0 | — | — | — | — |
| spanish-short | mtp | first_token_seconds | 0 | — | — | — | — |
| spanish-short | mtp | wall_seconds | 0 | — | — | — | — |
| spanish-short | mtp | whole_request_tps | 0 | — | — | — | — |
| structured-1536 | native | post_first_tps | 0 | — | — | — | — |
| structured-1536 | native | first_token_seconds | 0 | — | — | — | — |
| structured-1536 | native | wall_seconds | 0 | — | — | — | — |
| structured-1536 | native | whole_request_tps | 0 | — | — | — | — |
| structured-1536 | mtp | post_first_tps | 0 | — | — | — | — |
| structured-1536 | mtp | first_token_seconds | 0 | — | — | — | — |
| structured-1536 | mtp | wall_seconds | 0 | — | — | — | — |
| structured-1536 | mtp | whole_request_tps | 0 | — | — | — | — |
| structured-512 | native | post_first_tps | 0 | — | — | — | — |
| structured-512 | native | first_token_seconds | 0 | — | — | — | — |
| structured-512 | native | wall_seconds | 0 | — | — | — | — |
| structured-512 | native | whole_request_tps | 0 | — | — | — | — |
| structured-512 | mtp | post_first_tps | 0 | — | — | — | — |
| structured-512 | mtp | first_token_seconds | 0 | — | — | — | — |
| structured-512 | mtp | wall_seconds | 0 | — | — | — | — |
| structured-512 | mtp | whole_request_tps | 0 | — | — | — | — |
| structured-short | native | post_first_tps | 0 | — | — | — | — |
| structured-short | native | first_token_seconds | 0 | — | — | — | — |
| structured-short | native | wall_seconds | 0 | — | — | — | — |
| structured-short | native | whole_request_tps | 0 | — | — | — | — |
| structured-short | mtp | post_first_tps | 0 | — | — | — | — |
| structured-short | mtp | first_token_seconds | 0 | — | — | — | — |
| structured-short | mtp | wall_seconds | 0 | — | — | — | — |
| structured-short | mtp | whole_request_tps | 0 | — | — | — | — |

## Coverage, acceptance, and memory

| Case | Method | Observed / expected measured | Decode excluded | Decode below 10 (%) | Acceptance accepted / rounds | Rank 0 peak bytes | Rank 1 peak bytes |
|---|---|---:|---:|---:|---|---:|---:|
| b00-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b00-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| code-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8289622558 | 9763237776 |
| code-1536 | mtp | 2 / 2 | 0 | 0 | 246 / 264 (rate 0.931818; 0 rows without counters) | 8424656571 | 9763258260 |
| code-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247913314 | 9702981528 |
| code-512 | mtp | 2 / 2 | 0 | 0 | 124 / 132 (rate 0.939394; 0 rows without counters) | 8378823166 | 9702982391 |
| code-short | native | 2 / 2 | 0 | 100 | not supplied | 7895893300 | 9380387754 |
| code-short | mtp | 2 / 2 | 0 | 0 | 122 / 132 (rate 0.924242; 0 rows without counters) | 7963401518 | 9519713228 |
| b01-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b01-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b02-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b02-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b03-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b03-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b04-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b04-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b05-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b05-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| chinese-1536 | native | 0 / 2 | 0 | — | not supplied | — | — |
| chinese-1536 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| chinese-512 | native | 0 / 2 | 0 | — | not supplied | — | — |
| chinese-512 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| chinese-short | native | 0 / 2 | 0 | — | not supplied | — | — |
| chinese-short | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| explanation-1536 | native | 0 / 2 | 0 | — | not supplied | — | — |
| explanation-1536 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| explanation-512 | native | 0 / 2 | 0 | — | not supplied | — | — |
| explanation-512 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| explanation-short | native | 0 / 2 | 0 | — | not supplied | — | — |
| explanation-short | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| math-1536 | native | 0 / 2 | 0 | — | not supplied | — | — |
| math-1536 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| math-512 | native | 0 / 2 | 0 | — | not supplied | — | — |
| math-512 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| math-short | native | 0 / 2 | 0 | — | not supplied | — | — |
| math-short | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| spanish-1536 | native | 0 / 2 | 0 | — | not supplied | — | — |
| spanish-1536 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| spanish-512 | native | 0 / 2 | 0 | — | not supplied | — | — |
| spanish-512 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| spanish-short | native | 0 / 2 | 0 | — | not supplied | — | — |
| spanish-short | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| structured-1536 | native | 0 / 2 | 0 | — | not supplied | — | — |
| structured-1536 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| structured-512 | native | 0 / 2 | 0 | — | not supplied | — | — |
| structured-512 | mtp | 0 / 2 | 0 | — | not supplied | — | — |
| structured-short | native | 0 / 2 | 0 | — | not supplied | — | — |
| structured-short | mtp | 0 / 2 | 0 | — | not supplied | — | — |

## Paired ratios (MTP / native throughput)

| Case | Repeat | Decode speedup | Whole-request speedup |
|---|---:|---:|---:|
| b00-warmup | median | — | — |
| code-1536 | 0 | 1.48968 | 1.03404 |
| code-1536 | 1 | 1.47489 | 1.00586 |
| code-1536 | median | 1.48229 | 1.01995 |
| code-512 | 0 | 1.47056 | 1.04471 |
| code-512 | 1 | 1.48216 | 1.15012 |
| code-512 | median | 1.47636 | 1.09741 |
| code-short | 0 | 1.49207 | 1.43629 |
| code-short | 1 | 1.47345 | 1.4154 |
| code-short | median | 1.48276 | 1.42584 |
| b01-warmup | median | — | — |
| b02-warmup | median | — | — |
| b03-warmup | median | — | — |
| b04-warmup | median | — | — |
| b05-warmup | median | — | — |
| chinese-1536 | median | — | — |
| chinese-512 | median | — | — |
| chinese-short | median | — | — |
| explanation-1536 | median | — | — |
| explanation-512 | median | — | — |
| explanation-short | median | — | — |
| math-1536 | median | — | — |
| math-512 | median | — | — |
| math-short | median | — | — |
| spanish-1536 | median | — | — |
| spanish-512 | median | — | — |
| spanish-short | median | — | — |
| structured-1536 | median | — | — |
| structured-512 | median | — | — |
| structured-short | median | — | — |

## Observed-only aggregate — NOT full suite

```json
{
  "interpretation": "Equal case weight: geometric mean of each case's median paired ratio. Lower-tail minimum is descriptive, not a confidence interval; tiny repeat counts do not establish tail reliability. No cross-case pooled tokens/sec; no quality scoring or runtime/numerical gate verification.",
  "decode": {
    "case_count": 3,
    "excluded_case_count": 21,
    "geometric_mean": 1.4804665119077638,
    "lower_tail_min": 1.4763644085823866,
    "case_speedup_statistics": {
      "count": 3,
      "median": 1.482286853580671,
      "min": 1.4763644085823866,
      "max": 1.4827568551789931,
      "stdev": 0.003562761797412135
    }
  },
  "whole_request": {
    "case_count": 3,
    "excluded_case_count": 21,
    "geometric_mean": 1.1686198437976696,
    "lower_tail_min": 1.0199491787934136,
    "case_speedup_statistics": {
      "count": 3,
      "median": 1.097413287579648,
      "min": 1.0199491787934136,
      "max": 1.4258407508313145,
      "stdev": 0.21548907745690526
    }
  }
}
```

## Missing request IDs

```json
[
  "b01-warmup-mtp",
  "b01-warmup-native",
  "b02-warmup-mtp",
  "b02-warmup-native",
  "b03-warmup-mtp",
  "b03-warmup-native",
  "b04-warmup-mtp",
  "b04-warmup-native",
  "b05-warmup-mtp",
  "b05-warmup-native",
  "chinese-1536-mtp-r0",
  "chinese-1536-mtp-r1",
  "chinese-1536-native-r0",
  "chinese-1536-native-r1",
  "chinese-512-mtp-r0",
  "chinese-512-mtp-r1",
  "chinese-512-native-r0",
  "chinese-512-native-r1",
  "chinese-short-mtp-r0",
  "chinese-short-mtp-r1",
  "chinese-short-native-r0",
  "chinese-short-native-r1",
  "explanation-1536-mtp-r0",
  "explanation-1536-mtp-r1",
  "explanation-1536-native-r0",
  "explanation-1536-native-r1",
  "explanation-512-mtp-r0",
  "explanation-512-mtp-r1",
  "explanation-512-native-r0",
  "explanation-512-native-r1",
  "explanation-short-mtp-r0",
  "explanation-short-mtp-r1",
  "explanation-short-native-r0",
  "explanation-short-native-r1",
  "math-1536-mtp-r0",
  "math-1536-mtp-r1",
  "math-1536-native-r0",
  "math-1536-native-r1",
  "math-512-mtp-r0",
  "math-512-mtp-r1",
  "math-512-native-r0",
  "math-512-native-r1",
  "math-short-mtp-r0",
  "math-short-mtp-r1",
  "math-short-native-r0",
  "math-short-native-r1",
  "spanish-1536-mtp-r0",
  "spanish-1536-mtp-r1",
  "spanish-1536-native-r0",
  "spanish-1536-native-r1",
  "spanish-512-mtp-r0",
  "spanish-512-mtp-r1",
  "spanish-512-native-r0",
  "spanish-512-native-r1",
  "spanish-short-mtp-r0",
  "spanish-short-mtp-r1",
  "spanish-short-native-r0",
  "spanish-short-native-r1",
  "structured-1536-mtp-r0",
  "structured-1536-mtp-r1",
  "structured-1536-native-r0",
  "structured-1536-native-r1",
  "structured-512-mtp-r0",
  "structured-512-mtp-r1",
  "structured-512-native-r0",
  "structured-512-native-r1",
  "structured-short-mtp-r0",
  "structured-short-mtp-r1",
  "structured-short-native-r0",
  "structured-short-native-r1"
]
```

## Failures (0)

```json
[]
```

## Explicit first-loaded warmups (excluded from measured statistics)

```json
[
  {
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "text": "A queue is a linear data structure that operates on a First-In, First-Out (FIFO) basis, where elements are added to the back and removed from",
    "wall_seconds": 4.568999500013888,
    "first_token_seconds": 0.3735162499942817,
    "wall_tokens_per_second": 7.003721493053946,
    "native_generation_tps": null,
    "prompt_tokens": 26,
    "finish_reason": "length",
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "id": "b00-warmup-native",
    "case_id": "b00-warmup",
    "method": "native",
    "warmup": true,
    "max_tokens": 32,
    "repeat": 0,
    "sample_repeat": 0,
    "eval_group_size": 32,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "prompt_token_ids": [
      248045,
      846,
      198,
      814,
      20139,
      303,
      1330,
      2716,
      22157,
      1092,
      264,
      6951,
      369,
      303,
      6165,
      7785,
      13,
      248046,
      198,
      248045,
      74455,
      198,
      248068,
      271,
      248069,
      271
    ],
    "local_rank": 0,
    "local_rank_wall_seconds": 4.570030124974437,
    "local_peak_mlx_bytes": 7865509692,
    "sidecar_resident_on_rank0": true,
    "numerical_admission": "same-process-small-and-full-strict-passed",
    "verifier_mode": "native",
    "post_first_tokens_per_second": 7.388898525540563,
    "timing_convention": "EOS included; fresh cache; gates/head load excluded; MTP finish ACK included; independent local_rank_wall includes final synchronization",
    "peak_mlx_bytes_by_rank": [
      7865509692,
      9320782699
    ],
    "output_tokens": 32,
    "post_first_tps": 7.388898525540563,
    "whole_request_tps": 7.003721493053946
  },
  {
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "verification_rounds": 17,
    "accepted_drafts": 15,
    "teacher_forwards": 17,
    "numerical_admission": "same-process-small-and-full-strict-passed",
    "wall_seconds": 3.776770124968607,
    "first_token_seconds": 0.7198357079760171,
    "prompt_tokens": 26,
    "text": "A queue is a linear data structure that operates on a First-In, First-Out (FIFO) basis, where elements are added to the back and removed from",
    "id": "b00-warmup-mtp",
    "case_id": "b00-warmup",
    "method": "mtp",
    "warmup": true,
    "max_tokens": 32,
    "repeat": 1,
    "sample_repeat": 0,
    "eval_group_size": 32,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "prompt_token_ids": [
      248045,
      846,
      198,
      814,
      20139,
      303,
      1330,
      2716,
      22157,
      1092,
      264,
      6951,
      369,
      303,
      6165,
      7785,
      13,
      248046,
      198,
      248045,
      74455,
      198,
      248068,
      271,
      248069,
      271
    ],
    "local_rank": 0,
    "local_rank_wall_seconds": 3.778642042016145,
    "local_peak_mlx_bytes": 7963462958,
    "sidecar_resident_on_rank0": true,
    "verifier_mode": "native",
    "wall_tokens_per_second": 8.472848211874158,
    "post_first_tokens_per_second": 10.140878334739606,
    "timing_convention": "EOS included; fresh cache; gates/head load excluded; MTP finish ACK included; independent local_rank_wall includes final synchronization",
    "peak_mlx_bytes_by_rank": [
      7963462958,
      9519553484
    ],
    "output_tokens": 32,
    "post_first_tps": 10.140878334739606,
    "whole_request_tps": 8.472848211874158
  }
]
```

## Provenance (as supplied, not verified here)

```json
{
  "eos_token_ids": [
    248044,
    248046
  ],
  "trial": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-00-v1",
  "coverage": "batch00 of six only",
  "native_sidecar_resident": true,
  "strict_numerical_admission": true,
  "source_hashes_match": true,
  "owned_cleanup": true
}
```
