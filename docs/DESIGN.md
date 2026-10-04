# Design: precision allocation and Metal execution

## Problem and scope

Select a precision and execution backend for each coupled gate/up MLP pair under
parameter-storage and isolated-operation latency budgets. The current verified
architecture adapter is the pinned MLX-LM Qwen2 MLP implementation used by Qwen2.5.
Other activation/forward implementations must be rejected, not silently converted to SiLU.

This is an engineering co-design project, not a claim that mixed precision, affine
quantization, Pareto search, or fusion is new. Its contribution is the complete measured
pipeline and custom packed Metal implementation on a constrained consumer machine.

## Calibration

A temporary wrapper records inputs from an actual unquantized model forward pass.
All original modules are restored, including on exceptions. Calibration and held-out
smoke texts are disjoint. For each precision, compute relative squared error of the
joint gate/up activation on those captured inputs.

The proxy is local. Layer interactions, quantization of fixed attention/down weights,
and distribution shift are not captured by an additive sum of local errors. Full-model
held-out evaluation is mandatory; low proxy error does not establish reasoning accuracy.
The converter currently loads its reference in full, so it cannot claim large-model
calibration on a 16 GB machine until layer/shard streaming is implemented.

## Coupled decisions and actual storage

Gate and up projections form one allocation unit. Their precisions match so the fused
backend can consume them together. Each candidate records:

- Actual packed weight, scale, and bias bytes, rather than parameters × nominal bit width.
- Measured synchronized wall-clock latency at the real projection dimensions.
- Relative calibration error of the joint activation.
- Backend and launch configuration.

All other eligible weights use affine 4-bit quantization. Their fixed bytes are subtracted
from the total parameter budget before solving. This budget excludes KV cache, temporary
activations, calibration workspace, other applications, and macOS.

## Exact surrogate allocation

The allocator chooses one candidate per unit. Partial plans accumulate memory, latency,
and loss. Dominance pruning uses a loss-sorted sweep with a Fenwick prefix-min structure
over compressed latency ranks. Label tie handling preserves deterministic choices when
floating-point additions erase an earlier numerical difference.

The frontier can grow exponentially. A configured limit raises `FrontierOverflow`;
it never silently truncates the frontier while labeling the result exact. Infeasible
budgets raise `InfeasibleBudget`. Exactness refers only to the supplied additive surrogate
cost problem, not global task accuracy or real full-model latency.

## Packed Metal implementation

The scalar reference kernel decodes individual values across 32-bit word boundaries.
The optimized kernel consumes complete packs: eight 4-bit values from one word, eight
3-bit values from three bytes, or four 6-bit values from three bytes. Packs align within
the supported quantization group sizes.

Each SIMD group computes one output row. A lane accumulates gate/up integer-code dot
products and an input sum, then applies the group's scale and bias once per pack.
SIMD reductions combine lane contributions. Projection outputs are cast to their stock
output dtype before the fused SiLU/multiply expression, and the down projection remains
separate. The packed representation is never expanded into a full dense weight matrix.

Different floating-point accumulation orders can change logits slightly. Numerical
checks use explicit tolerances and actual cached-logit diagnostics; no bit-exactness or
identical future token sequence is claimed.

## Tuning and dispatch

Both isolated stock and fused paths are compiled. Candidate launch configurations are
numerically checked before timing. A tuning split picks a launch configuration, then
fresh interleaved validation measurements compare it with stock execution. Fused candidates
are admitted only if their median exceeds a 5% heuristic improvement margin. This margin
is not a significance test and is not an end-to-end guarantee.

Fusion must use an explicit verified architecture adapter. Matching attribute names and
weight shapes do not prove that an arbitrary MLP uses SiLU. The fallback must invoke the
verified original operation on the wrapper's current projection fields, not retain stale
copies of an old module. Unsupported activation/forward semantics fail closed.

Multi-token prefill uses the stock operation. Single-token compatible inputs can use
fused decode. Serialized parameters retain the original module paths; runtime counters
are excluded from model weights. Saved hardware/package versions govern dispatch reuse.

## Benchmark interpretation

End-to-end experiments retain all raw timing samples and fixed reference token schedules.
Stock and fused copies of the same saved mixed model are alternated, with fresh KV caches
for each trial. Synchronization and evaluation happen at timing boundaries.

The measured result includes host dispatch and GPU execution, but excludes sampling and
text rendering. Paired bootstrap intervals describe the sampled trials only. Additional
GPU-counter profiling is required before attributing gains exclusively to memory traffic.
Background workloads, prior swap, and cache residency are recorded limitations.

The authored smoke corpus is intentionally small. Its NLL is an integration check,
not a coding/reasoning score. The broader offline evaluator now scores a deterministic
16,384-target prefix of a pinned WikiText-2 raw test export on both 0.5B and 1.5B,
using exact target masks, bounded windows, and host-double aggregation. This is
explicitly noncanonical held-out perplexity, not task accuracy.

## Saved-model integrity and bounded-residency scaling

Schema-v2 execution manifests bind every saved weight shard, config.json, and shipped
Metal kernel by SHA-256 and retain original hardware/runtime profiling metadata.
Bindings detect changes relative to the manifest; they are not signatures or proof
of fresh benchmarking. Invalid/legacy manifests trigger explicit stock generation
fallback, while same-weight replay refuses them before model loading. The `seal`
command copies into a new directory, preserves the original manifest, and records
that profiling was not performed.

The scaling runner profiles one reference, then separately constructs/evaluates/releases
uniform4 and mixed variants. Byte accounting is checked against actual constructed
parameters. Profile reuse checks source weight/config and kernel hashes. Model inputs
for likelihood evaluation are bounded; the unquantized reference still must fit.

The 1.5B experiment selected zero fused pairs, correctly retaining stock execution.
A fresh same-weight 0.5B replay produced confidence intervals including parity,
so the earlier positive replay is historical evidence, not a robust speedup claim.
Broader task accuracy, 7B/14B scaling, and layer-sharded calibration remain unimplemented.
