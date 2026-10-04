# Same-weight fused/stock replay

| Context | Prompt tokens | Stock tok/s | Fused tok/s | Ratio | Bootstrap 95% interval |
|---|---:|---:|---:|---:|---|
| short | 35 | 212.32 | 220.18 | 1.037x | [1.022, 1.051] |
| medium | 109 | 210.27 | 218.99 | 1.041x | [1.000, 1.055] |
| long | 448 | 205.63 | 216.03 | 1.051x | [1.034, 1.066] |

## Limits

- Same saved precision map and weights; only gate/up execution backend differs.
- Teacher-forced wall-clock timing excludes token sampling and rendering.
- Bootstrap intervals cover sampled trials, not other models or devices.
- Background load, cache residency, and prior swap can influence latency.
- Three authored contexts are not a standard task-accuracy benchmark.
