# ParetoQuant-Metal measured smoke experiment

Device: Apple M2 Pro
Source: qwen2.5-0.5b-instruct

These are actual measurements, not projected larger-model results.

| Variant | Parameter MiB | Smoke NLL | Decode tok/s | Ratio vs uniform4 | Fused pairs |
|---|---:|---:|---:|---:|---:|
| uniform4 | 265.12 | 4.2696 | 242.82 | 1.000x | 0 |
| sensitivity_only | 265.12 | 4.2696 | 233.46 | 0.961x | 0 |
| hardware_stock | 265.12 | 4.2696 | 238.63 | 0.983x | 0 |
| hardware_fused | 265.12 | 4.2696 | 237.61 | 0.979x | 0 |
| hardware_fused_stock_replay | 265.12 | 4.2696 | 234.91 | 0.967x | 0 |
| uniform4_forced_fused | 265.12 | 4.2696 | 205.09 | 0.845x | 24 |

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

Binary search is a search algorithm that allows us to find a specific element in a sorted list of data. The algorithm works by repeatedly dividing the data into smaller groups
