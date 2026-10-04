# Same-weight fused/stock replay

| Context | Prompt tokens | Stock tok/s | Fused tok/s | Ratio | Bootstrap 95% interval |
|---|---:|---:|---:|---:|---|
| short | 35 | 234.84 | 236.86 | 1.009x | [0.988, 1.024] |
| medium | 109 | 228.82 | 230.98 | 1.009x | [0.981, 1.022] |
| long | 448 | 218.33 | 220.14 | 1.008x | [0.994, 1.019] |

## Limits

- Same saved precision map and weights; only gate/up execution backend differs.
- Teacher-forced wall-clock timing excludes token sampling and rendering.
- Bootstrap intervals cover sampled trials, not other models or devices.
- Background load, cache residency, and prior swap can influence latency.
- Three authored contexts are not a standard task-accuracy benchmark.
