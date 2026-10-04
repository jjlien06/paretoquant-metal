# Measured local framework comparison

Qwen2.5-0.5B-Instruct on Apple M2 Pro, 16 GB unified memory.
Median client wall-clock tokens/second, including tokenization, prefill, generation and detokenization; model loading excluded.
Three identical raw prompts, 64 generated tokens, seven trials and two excluded warmups per case/backend.
All 84 measured generations and 24 warmups verified. Backend blocks ran sequentially, not cross-backend paired.

| Prompt tokens | ParetoQuant-Metal | Stock MLX | vLLM Metal | TF Metal hybrid graph |
|---|---:|---:|---:|---:|
| 48 | 295.78 | 261.98 | 232.46 | 10.81 |
| 99 | 278.04 | 257.28 | 217.55 | 10.67 |
| 209 | 257.71 | 236.12 | 194.52 | 10.52 |

## Interpretation

Configured same-weight, single-request vLLM Metal throughput was lower; project throughput was higher by 27.24%, 27.80%, 32.49% across the three cases.
This compares practical execution paths, including vLLM HTTP/scheduling overhead and differing library/interpreter versions. It does not isolate the custom Metal kernel contribution or establish a CUDA-vLLM or batched-serving win.

The TensorFlow baseline uses official KerasHub QwenCausalLM and audited source checkpoint conversion, native TF graph generation (`run_eagerly=False`, `jit_compile=False`), nonquantized FP16 weights, and CPU cache/sampler slice updates. Ordinary soft device placement is enabled. It is not a precision-matched comparison, an all-GPU baseline, or a basis for a generic "faster than TensorFlow" headline.
Latest-stack and native all-GPU failure evidence is retained under `../tensorflow-metal-probe-v1/`; working-package versions are recorded in the raw JSON.

No answer-quality or larger-model capability claim is supported.

## Reproduce

Start the vLLM Metal server using the exact command in `../vllm-metal-probe-v1/probe.json`, then run each backend with a new output path. Stop it before TensorFlow allocation.

```sh
.venv/bin/python spikes/006-framework-head-to-head/main.py --engine paretoquant_mlx --output artifacts/my-framework/paretoquant.json --repeats 7 --warmup 2 --max-tokens 64
.venv/bin/python spikes/006-framework-head-to-head/main.py --engine stock_mlx --output artifacts/my-framework/stock.json --repeats 7 --warmup 2 --max-tokens 64
.venv/bin/python spikes/006-framework-head-to-head/main.py --engine vllm_metal --output artifacts/my-framework/vllm.json --repeats 7 --warmup 2 --max-tokens 64
artifacts/tensorflow-metal-probe-py312/bin/python spikes/005-tensorflow-metal/probe.py --cpu-cache-updates --graph-generation --new-tokens 64 --benchmark-inputs results/ollama-head-to-head-v1/comparison.json --benchmark-repeats 7 --benchmark-warmup 2 --output artifacts/my-framework/tensorflow.json
```

## Evidence

Raw per-generation outputs, counts, timing samples, script/weight/fixture hashes and versions are in the adjacent JSON files. Consolidated data and raw-evidence hashes: `comparison.json`. Earlier Ollama results: `../ollama-head-to-head-v1/comparison.json`.

## Limitations

- Sequential backend blocks, not cross-backend paired trials; background state can drift.
- Project/stock MLX/vLLM Metal use identical saved mixed weights; serving overhead and MLX/Python versions differ.
- vLLM Metal is MLX-backed on Apple Silicon, not CUDA vLLM or a batching scalability comparison.
- TensorFlow is native KerasHub Qwen graph mode, FP16 weights, explicit CPU slice/cache updates and soft CPU placement; not precision-matched or all-GPU.
- No standard task-quality or larger-model benchmark.
- Earlier Ollama measurements are preserved separately; not contemporaneous with this block comparison.
