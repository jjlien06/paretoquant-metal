# Held-equal real checkpoint attention replay

Four physical M4 GPU cases isolate the SDPA operation from earlier quantized projection drift. Genuine Q/K/V and the original batched attention output are captured from the unchanged mixed3/4bit Qwen2.5-0.5B decoder. Every single-query replay uses those exact evaluated arrays, the corresponding causal KV prefix, the same scale and a contiguous query layout matching width-one decode. No model cache is mutated by replay.

Prompt0 widths2,5,9 and a second Moon prompt width5 cover all24layers each (96 operator comparisons). Width2 is bit-exact in every layer. Both width5 cases independently exceed atol=rtol=.002 in some layers with Q/K/V held equal. Width9 differs bitwise but remains within the original tolerance in all24layers. This establishes a separate width-dependent attention arithmetic issue in these cases; fixing projections alone does not admit batching. It does not establish identical attention dispatch for the32B target, whose head geometry differs.

Capture did not change the model: instrumented/uninstrumented same-mode logits, offsets and every live KV prefix are bit-exact in all four cases. Source/config/checkpoint inventory and exact per-layer failures are bound in the raw report. The checkpoint inventory was unchanged before/after; the guarded full32B source was untouched. The owned actor completed in1.69890125seconds and its child/process group was verified gone, with peak MLX allocation315,687,144bytes. Allocation is not a residency assertion.

An isolated per-query causal attention component is being implemented separately with test-first metadata/prefix admission. It must pass actual GPU/full-cache gates and independent review. More attention calls may add overhead; this replay is not a speed benchmark or any32B speculation admission. Numerical tolerances and the working decoder remain unchanged.

`evidence/` contains exact executed sources and fetched job/startup/process/raw reports plus the reused controller; `verified-summary.json` is programmatically derived. The diagnostic operated on the existing checkpoint and downloaded no weights. `integrity.json` hashes the archive excluding itself.
