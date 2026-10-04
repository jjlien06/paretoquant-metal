# ParetoQuant-Metal measured smoke experiment

Device: Apple M2 Pro
Source: qwen2.5-0.5b-instruct

These are actual measurements, not projected larger-model results.

| Variant | Parameter MiB | Smoke NLL | Decode tok/s | Ratio vs uniform4 | Fused pairs |
|---|---:|---:|---:|---:|---:|
| uniform4 | 265.12 | 4.2696 | 216.26 | 1.000x | 0 |
| sensitivity_only | 248.49 | 4.3412 | 207.31 | 0.959x | 0 |
| hardware_stock | 248.49 | 4.3536 | 205.99 | 0.953x | 0 |
| hardware_fused | 248.49 | 4.3412 | 213.35 | 0.987x | 19 |
| hardware_fused_stock_replay | 248.49 | 4.3412 | 204.50 | 0.946x | 0 |
| uniform4_forced_fused | 265.12 | 4.2696 | 221.24 | 1.023x | 24 |

## Limitations

- Authored smoke-text NLL is not a standard reasoning/coding accuracy benchmark.
- Only gate/up pairs vary; other eligible weights use 4-bit affine quantization.
- Parameter budget excludes KV cache, activations, allocator workspace, and macOS.
- Summed isolated gate/up latency is a proxy, not a full-model latency guarantee.
- Teacher-forced decode timing excludes token sampling and text rendering.
- Forced-fusion ablation bypasses automatic performance fallback on purpose.
- Calibration loads the unquantized reference; layer-sharded calibration is not implemented.
- Prior swap and background workloads can influence measurements.

## Actual mixed-precision model output

Binary search is a search algorithm that is used to find a specific element in a sorted list. It is a more efficient search algorithm than a linear search, but it has a time complexity of O(log n) in the average case and O
