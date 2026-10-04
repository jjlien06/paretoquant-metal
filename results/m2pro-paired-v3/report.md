# Same-weight fused/stock replay

| Context | Prompt tokens | Stock tok/s | Fused tok/s | Ratio | Bootstrap 95% interval |
|---|---:|---:|---:|---:|---|
| short | 35 | 212.74 | 221.73 | 1.042x | [1.036, 1.057] |
| medium | 109 | 209.88 | 220.22 | 1.049x | [1.039, 1.060] |
| long | 448 | 206.03 | 213.12 | 1.034x | [1.027, 1.048] |

## Limits

- Same saved precision map and weights; only gate/up execution backend differs.
- Teacher-forced wall-clock timing excludes token sampling and rendering.
- Bootstrap intervals describe sampled trials, not generalization to other models or devices.
- Background load, cache residency, and prior swap can influence latency.
- Three authored contexts are not a standard task-accuracy benchmark.
