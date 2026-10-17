# Actual checkpoint verification-width localization

Four real mixed-3/4-bit Qwen2.5-0.5B cases ran on the physical M4 GPU using the unchanged grouped decoder (group8), fresh native KV caches and identical forced continuation inputs. Prompt0 uses widths2,5,9; a second Moon prompt uses width5. These are forward differential diagnostics, not speculative-generation or speed benchmarks.

The first layer input RMSNorm is bit-exact across batch and serial modes in every case. The first nonexact recorded outputs are its quantized q/k/v projections, before attention. By layer1, live teacher KV exceeds the original atol=rtol=.002 gate in each case. Attention and later MLP differences may additionally contribute; this trace does not establish that projections are the sole cause or that serial-shaped attention alone would repair the full trajectory.

Instrumentation did not change same-mode results: each case ran uninstrumented and instrumented in both modes; every live KV prefix, every offset and all logits were bit-exact within mode. Intermediate hooks were restored after each run. Captured operators are compared over aligned positions, not unrelated free-generation inputs. Finite-value checks, raw widths, forced token IDs, elementwise failure counts, checkpoint inventory, installed source/version hashes and GPU/peak allocation are retained.

The actor completed in 2.843494 seconds with a 150-second deadline. Child and owned process group cleanup passed; the full32B guarded source bytes were unchanged. Peak MLX allocation was367,624,816 bytes; this is allocation, not residency proof. The checkpoint inventory was unchanged before/after. No32B run or performance gain is established here.

An isolated slot-shared quantized GEMV candidate is being built separately. Its intent is to retain single-token QMV arithmetic while sharing packed weight reads across verification positions. That intent is unproven until it clears real Metal compilation, per-position operator and full live-KV gates, an independent review and matched physical32B timings. Default decoder and numerical tolerances remain unchanged.

`evidence/` includes the exact executed observer/controller and decoder dependencies, actor/startup/process records, fetched raw JSON, checkpoint/source identities and pinned upstream MLX0.32.3 source/license. `verified-summary.json` is derived programmatically from the four reports. `integrity.json` hashes the archive excluding itself.
