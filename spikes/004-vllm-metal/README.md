# 004: vLLM Metal on M2 Pro — READY

## Verdict: VALIDATED (startup and generation, not performance or answer quality)

The SAME existing mixed affine 3/4-bit Qwen2.5-0.5B-Instruct checkpoint loaded and generated exactly 8 tokens through vLLM Metal. No full-precision fallback, model conversion, new trained model download, project implementation changes, or main-environment changes were needed. One generation request was executed. An initial startup failed on the deliberately conservative memory budget; a second startup succeeded after adjusting only that budget. No comparative performance trials were run.

Actual continuation: `The sky is blue because it is composed`

Actual output token IDs: `[785, 12884, 374, 6303, 1576, 432, 374, 23415]`.

Actual usage: prompt 30 tokens, completion 8 tokens, finish reason `length`. This truncated output is not an assessment of model answer quality.

### Isolated installation

- Environment: `/Users/jeremylien/Projects/paretoquant-metal/artifacts/vllm-metal-venv`
- Python: 3.12.14, native arm64
- vLLM core: 0.30.0+cpu
- vLLM Metal: 0.30.0.dev20261003204550
- Release revision: `68e93fc95656e97a9c5df744261d644e0b6e5daf`
- MLX / mlx-metal: 0.32.1
- mlx-lm: 0.32.0, installed from revision `9e6acca691e64d6d8bb808c328fcdea459099cca`
- torch: 2.13.0; transformers: 5.18.0

The release API returned the development wheel published October 3, 2026, 20:46:00 UTC as the newest available wheel; the latest stable wheel was v0.30.0.[4] The inspected upstream installer defaults to and recommends the development channel and pins its compatible core wheel through release metadata.[3] Installation documentation requires Apple Silicon, macOS 15+, Python 3.12 and prebuilt wheels.[5] We installed those wheels manually with `uv` into a repo-local environment rather than execute the installer or create Homebrew taps/global tools.

Exact installation commands used:

```sh
uv venv --python 3.12 --managed-python /Users/jeremylien/Projects/paretoquant-metal/artifacts/vllm-metal-venv
uv pip install --python /Users/jeremylien/Projects/paretoquant-metal/artifacts/vllm-metal-venv/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl' \
  'https://github.com/vllm-project/vllm-metal/releases/download/v0.30.0.dev20261003204550/vllm_metal-0.30.0.dev20261003204550-cp312-cp312-macosx_15_0_arm64.whl'
```

### Exact installed CLI launch/probe commands

No new adapter was necessary. These are the installed CLI commands exercised, not an untested custom serving script. The current server has already been stopped.

Launch in one terminal (foreground; Ctrl-C shuts it down):

```sh
VLLM_MLX_DEVICE=gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS=metal \
/Users/jeremylien/Projects/paretoquant-metal/artifacts/vllm-metal-venv/bin/vllm serve \
  /Users/jeremylien/Projects/paretoquant-metal/artifacts/m2pro-mixed-v2/model \
  --served-model-name paretoquant-mixed \
  --host 127.0.0.1 --port 11436 \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.075 --kv-cache-memory-bytes 268435456 \
  --enforce-eager --no-enable-prefix-caching
```

Wait until application startup completes, then from another terminal:

```sh
curl -fsS http://127.0.0.1:11436/health
curl -fsS http://127.0.0.1:11436/v1/models
curl -fsS --max-time 120 http://127.0.0.1:11436/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/request.json
```

Do not overwrite the saved original response when running a later probe. Stop the foreground server with Ctrl-C after use; verify no listener remains with:

```sh
lsof -nP -iTCP:11436 -sTCP:LISTEN
```

The actual background server for this task was stopped with SIGTERM and all recorded API/worker PIDs were verified absent. `cleanup.json` records the exact process and listener readback.

### Same weights and GPU/backend verification

Model path: `/Users/jeremylien/Projects/paretoquant-metal/artifacts/m2pro-mixed-v2/model`.

Successful server log records:

- `MLX device set to: Device(gpu, 0)`
- `PyTorch device set to: mps`
- `MLX-LM model loaded ...` followed by that exact local path
- `Native paged-attention Metal kernels loaded`
- `Shared attention cache: 1236 blocks, 0.23 GiB across 24 Metal regions`

Independent MLX device inspection reports Apple M2 Pro, `applegpu_g14s`, Metal available, default MLX GPU, physical memory 17,179,869,184 bytes. The vLLM core wheel has `+cpu` in its name and logs `device_config=cpu` and a CPU KV-cache label; these labels are NOT evidence of CPU inference when the activated plugin's worker explicitly reports MLX GPU / MPS and loaded native Metal kernels.

The installed Metal loader delegates the text checkpoint to `mlx_lm.load`. Saved loader excerpts show that mlx-lm honors per-module `config.quantization` dictionaries. Model SHA-256 fingerprints, including safetensors/config/tokenizer, are in `model-fingerprints.json`. No packed weight bytes were rewritten. The project's execution manifest is not executed by this backend.

### Initial failure and memory handling

Exact initial error (same model, fraction 0.05):

```text
Paged attention: not enough Metal memory for KV cache. metal_limit=12.71GB, fraction=0.05, usable_metal=0.64GB, model_memory=0.26GB, overhead=0.49GB, kv_budget=-0.11GB. Mitigations: increase --gpu-memory-utilization (currently 0.05); use a smaller or more quantized model.
```

This was a cache-budget failure, NOT unsupported mixed quantization. `--kv-cache-memory-bytes` alone did not bypass the plugin's memory-fraction validation. Changing only fraction to 0.075 yielded a logged 0.24GB cache budget and 0.23GiB allocated cache; the explicit flag remained 256MiB. The cache was small, not the usual multi-GB default. Configuration docs specify that `gpu_memory_utilization` controls Metal KV budgeting.[6] Saved source excerpt explains the validation before upstream cache-layout allocation.

Nonfatal warnings: duplicate FFmpeg/OpenCV Objective-C classes, fd limit 2048, missing Triton and DeepSelect, model generation-config overrides. The saved request explicitly sets greedy sampling, no repetition/frequency/presence penalties, top_p=1, top_k=-1, seed=0 and ignore_eos=true, avoiding sampling-default ambiguity. Startup and the request succeeded despite those warnings.

### Sequential benchmark handoff

- Use the saved request's exact prompt or the exact same input token IDs across backends.
- Match output count, greedy selection, penalties, seed and EOS behavior. Payload uses raw already-chat-templated text via `/v1/completions`, avoiding another backend's automatic chat-template differences.
- Use the same mixed-weight directory and verify fingerprints; full-precision baselines must be separately labeled.
- Batch/concurrency 1, context limit 1024, max batched tokens 1024, prefix cache disabled, no speculative decoding.
- Run backends sequentially with no competing model/server workloads, and measure client/server overhead consistently.
- vLLM Metal itself uses MLX / MLX-LM; this tests vLLM scheduling/serving plus its Metal implementation, not CUDA vLLM or an independent TensorFlow kernel implementation.
- No speed claim can be drawn from this single untimed correctness probe.

### Absolute artifact paths

- Summary / benchmark handoff JSON: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/probe.json`
- Actual response: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/response.json`
- Exact request: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/request.json`
- Successful server log: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/mixed-server-ready.log`
- Initial error log: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/mixed-server.log`
- Versions / device: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/backend.json`
- Full installed dependencies: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/packages.txt`
- Installer output: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/install.log`
- Shutdown verification: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/cleanup.json`
- Original local model fingerprints: `/Users/jeremylien/Projects/paretoquant-metal/results/vllm-metal-probe-v1/model-fingerprints.json`
- Inspected primary docs, installer, release API, commit metadata and installed-loader evidence: `/Users/jeremylien/Projects/paretoquant-metal/spikes/004-vllm-metal/upstream/`

### What worked

Identical mixed checkpoint, isolated manual prebuilt wheel install, MLX GPU/MPS worker, native Metal attention, exactly one 8-token completion, verified shutdown.

### What did not / surprises

The 0.05 memory fraction was too low because startup profiling overhead was about 0.49GB. Merely specifying 256MiB KV bytes did not bypass plugin budget validation. This was resolved without changing model weights.

### Recommendation

Ready for the parent to run a later sequential same-weight benchmark using the installed command and explicit request parameters. Keep it separately labeled from full-precision or CUDA results.

## Sources

[3] https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh
[4] https://api.github.com/repos/vllm-project/vllm-metal/releases
[5] https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.30.0.dev20261003204550/docs/installation.md
[6] https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.30.0.dev20261003204550/docs/configuration.md
