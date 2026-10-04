# Resume wording and interview preparation

## Suggested project entry

**ParetoQuant-Metal — Hardware-Aware LLM Inference on Apple Silicon**
Python, Metal Shading Language, MLX, GPU Programming, Quantization, Optimization

- Built a hardware-aware mixed-precision quantizer and packed 3/4/6-bit Metal decode
  kernels; reduced Qwen2.5-0.5B/1.5B parameter storage by approximately 6.3% versus
  uniform affine4, including quantization metadata.
- Implemented exact Pareto-frontier precision/backend allocation under byte and
  measured-operation latency budgets; evaluated 60 budget/strategy combinations
  and quality on 16,384 held-out next-token targets per model variant.
- Engineered integrity-bound checkpoint manifests, mixed 3/6-bit serialization
  tests, conservative architecture guards, and paired same-weight replay with
  raw timing samples and bootstrap confidence intervals.

These are claims backed by the repository, not placeholders. The support for 6-bit
weights does not imply that the measured mixed model selected 6-bit: it selected
16 gate/up pairs at 3-bit and 8 at 4-bit.

## Performance claims: distinguish history from reproducibility

The original paired experiment measured 3.4–4.9% higher cached decode throughput
against stock execution of the same mixed weights. The October 4, 2026 release rerun
observed only approximately 0.8–0.9%, with every 95% paired bootstrap interval
including parity. Do not use the original fixed-dispatch result as a reproducible
speedup headline. Both runs are retained under `results/`.

Fresh saved-weight retuning subsequently selected 18 fused pairs and measured
stock/fused cached-decode latency ratios of 1.037006x, 1.041444x and 1.050541x.
Short and long 95% paired bootstrap intervals excluded parity; medium's lower
endpoint was 0.999557 and did not. Describe this as a workload-specific retuned
benefit, not a universal or repeatable fixed-dispatch guarantee. The weights and
precision map did not change, and two layers stayed stock after numerical failures.

The practical configured vLLM Metal server comparison observed 27–33% higher client
wall throughput on the 0.5B model, but includes HTTP/scheduling overhead, differing
versions, and sequential backend blocks. It is not an isolated custom-kernel gain,
a CUDA-vLLM result, a batched-serving comparison, or a larger-model claim.
TensorFlow required hybrid CPU cache updates and FP16 weights; avoid a generic
“faster than TensorFlow” bullet.

On the 1.5B shape, custom fusion did not pass the measured dispatch threshold and
stock execution was selected. Its 6.34% storage reduction is verified independently
of a fusion speedup. A negative scaling result is useful evidence of conservative
systems design, not something to hide.

On the explicitly noncanonical WikiText-2 raw test-prefix protocol, mixed PPL was
about 11.1% higher than uniform4 for 0.5B and 8.9% higher for 1.5B. Neither local
activation error nor perplexity establishes coding/reasoning accuracy. Explain the
memory–quality tradeoff rather than calling compression lossless.

## What to demonstrate

1. Explain the allocator's decision variables, constraints, Pareto pruning, and frontier cap.
2. Show packed 3-bit decoding across byte/word boundaries and scale/bias overhead.
3. Run a numerical GPU test and explain its tolerance and why bit-exact output is not promised.
4. Reload a saved mixed model and generate with both stock and fused execution.
5. Open the raw paired replay and distinguish kernel, backend, and full-model measurements.
6. Explain the activation-semantics review finding and the verified architecture guard.
7. State why the current calibration path does not yet support a reference too large for RAM.

## Questions you should be able to answer

- Why does a 3-bit format not necessarily run faster than a 4-bit format?
- Why measure packed bytes instead of estimating parameter count times bit width?
- Why couple the gate/up precision choices?
- What does “exact” mean for this allocator, and what does it not mean?
- Why can an isolated kernel win fail to improve full-model decoding?
- Why does the replay hold weights and token schedules constant?
- How do warmup, synchronization, interleaving, and background swap affect measurements?
- Why are local activation error and a tiny text NLL corpus insufficient evidence of reasoning quality?
- How would you implement layer-sharded calibration and validate a 7B/14B target next?

The value of this project is the systems work and your ability to explain its decisions.
Do not claim an implementation or evaluation step you have not personally understood.
