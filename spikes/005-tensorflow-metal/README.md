# 005: TensorFlow Metal / same local Qwen2.5 checkpoint

## Verdict: PARTIAL — READY HYBRID; unmodified GPU cache generation BLOCKED

Native TensorFlow/Metal GPU execution works. The SAME local `models/qwen2.5-0.5b-instruct/model.safetensors` checkpoint loads through official `keras_hub.models.QwenCausalLM.from_preset(local_path, preprocessor=None)`. No decoder, attention implementation, trained-model substitution, random checkpoint, or generated dummy output was written.

The native KerasHub model and native greedy sampler successfully generated **8 new tokens** when native `keras.ops.slice_update` was explicitly placed on CPU. All native Qwen forward weights and forward logits are on Metal GPU. This is a **hybrid CPU-cache/Metal-forward baseline**, not an unmodified all-GPU baseline. No throughput benchmark was performed.

## Reusable commands

Working environment: `/Users/jeremylien/Projects/paretoquant-metal/artifacts/tensorflow-metal-probe-py312`.

```sh
# Actual strict GPU runtime check (CPU fallback disabled).
/Users/jeremylien/Projects/paretoquant-metal/artifacts/tensorflow-metal-probe-py312/bin/python \
  /Users/jeremylien/Projects/paretoquant-metal/spikes/005-tensorflow-metal/probe.py \
  --runtime-only \
  --output /Users/jeremylien/Projects/paretoquant-metal/results/tensorflow-metal-probe-v1/runtime-rerun.json

# Working same-checkpoint native generation, with explicit CPU cache updates.
/Users/jeremylien/Projects/paretoquant-metal/artifacts/tensorflow-metal-probe-py312/bin/python \
  /Users/jeremylien/Projects/paretoquant-metal/spikes/005-tensorflow-metal/probe.py \
  --cpu-cache-updates --new-tokens 8 \
  --output /Users/jeremylien/Projects/paretoquant-metal/results/tensorflow-metal-probe-v1/correctness8-rerun.json

# Remove --cpu-cache-updates to reproduce the unmodified native GPU-cache failure.
# --prompt accepts the exact raw prompt; no implicit chat formatting is applied
# to a caller-supplied prompt. The default is an explicit Qwen chat prompt.
```

The CLI defaults to the existing local checkpoint, sets `HF_HUB_OFFLINE=1`, uses its exact tokenizer JSON through the installed Hugging Face tokenizers library, and uses the official KerasHub model/weight converter. It accepts `--checkpoint`, `--prompt`, and `--new-tokens`. Its checkpoint audit, loading, and correctness forward pass are not a generation speed metric.

## Actual results

- `runtime.json` and `runtime.log`: real Metal GPU matmul + activation, strict device placement, GPU result `[[11, 0], [25, 0]]`.
- `correctness8-hybrid.json` and `.log`: actual 8-token native model generation. Continuation: `The sky is blue because it reflects sunlight`.
- Token IDs: `[785, 12884, 374, 6303, 1576, 432, 25963, 39020]`.
- Source checkpoint SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- All **290** checkpoint tensors covered exactly by the official converter; every imported value checked against the converter's prescribed transpose/reshape and float16 cast.
- Model parameters: **494032768**. Embedding tensor and finite full forward logits on `/device:GPU:0`.
- CPU fallback audit: **440** native slice-update calls, all on `/device:CPU:0`. These include KV cache and sampler token-array updates. Other unsupported operations may also use ordinary TensorFlow soft placement; no all-GPU claim is made.
- `tests-hybrid-green.log`: **2 tests passed** (strict GPU runtime and same-checkpoint 8-token generation acceptance).
- Dependency check: all installed packages compatible. Fully frozen versions in `requirements-frozen.txt`.

“Correctness” here means integration correctness, finite logits, prompt preservation, exact token count, and audited checkpoint loading. The continuation is the real model's output, not a claim that its scientific explanation is correct. Independent cross-framework logit/token agreement is NOT yet checked.

## Package versions

Python **3.12.14**, native macOS ARM64; TensorFlow **2.18.1**; tensorflow-metal **1.2.0**; Keras **3.15.1**; KerasHub **0.21.1**; tensorflow-text **2.18.1**; NumPy **2.0.2**; safetensors **0.8.0**; tokenizers **0.23.2**. Environment occupies approximately **1.3 GiB**.

The source checkpoint is BF16. This baseline casts weights/compute to **nonquantized FP16**, versus the parent's mixed affine 3/4-bit MLX weights. It is not a precision-matched comparison. Native generation is eager, greedy, with `jit_compile=False`; early stopping disabled only to obtain exactly eight new tokens. It uses KerasHub's own KV cache and sampler. The placement shim only calls the original native slice-update operation inside a CPU device scope; it does not implement a decoder or change native model source.

## Exact blockers and alternatives attempted

1. Current packages: TensorFlow **2.21.0**, tensorflow-metal **1.2.0**, KerasHub **0.32.0**, Keras **3.15.1**, tensorflow-text **2.21.1**, NumPy **2.5.3**. TensorFlow import FAILED loading Metal plugin: `Library not loaded: @rpath/_pywrap_tensorflow_internal.so`. Exact traceback: `runtime-latest.log`. This observed dynamic-library incompatibility prevents a latest-stack Metal baseline.
2. Compatible alternative: TensorFlow **2.18.1** + tensorflow-text **2.18.1** + KerasHub **0.21.1**. Imports, strict GPU execution, official local checkpoint conversion, and full native Qwen forward pass succeed.
3. Native cached generation on GPU then FAILED in `QwenAttention.call()` at `keras.ops.slice_update`, which dispatches TensorFlow `XlaDynamicUpdateSlice`: `could not find registered platform with id` on GPU. Exact checkpoint/traceback: `native-gpu-cache-blocked.json` and `.log`. `jit_compile=False` and `run_eagerly=True` do not remove that explicitly XLA-backed primitive.[4]
4. Original native slice-update succeeds when placed on CPU (`cpu-cache-op.log`). Explicit CPU placement then allows the SAME native Qwen model to complete real 8-token generation. A first successful-generation attempt exposed output serialization expecting Tensor instead of KerasHub's NumPy output; failure preserved in `hybrid-output-serialization-failure.json` and `.log`, then fixed and the full acceptance test passed.

## Primary-source support evidence

Apple documents installing TensorFlow plus its tensorflow-metal PluggableDevice plugin on Apple GPUs.[1] The official KerasHub model matrix includes Qwen2.5-0.5B and Qwen2.5-Instruct-0.5B, and documents native QwenCausalLM loading/generation.[2] Its release converter explicitly maps the Hugging Face `qwen2` config to QwenBackbone and imports this architecture's weight tensors.[3] Therefore **checkpoint architecture support is present**, not absent.

The tensorflow-metal release's ARM64 wheel tags cover Python 3.9–3.12, not the parent project's Python 3.13.[5] KerasHub 0.21.1's requirements do not force a newer tensorflow-text, whereas 0.32.0 requires tensorflow-text >=2.20.1 and Keras >=3.15; this is why the compatible alternative uses the earlier KerasHub release, without bypassing dependency constraints.[6][7]

The inspected current Hugging Face Transformers repository tree has no native TensorFlow implementation under its `models/qwen2` directory; it is not the route used here.[8] Primary source snapshots, repository trees, package metadata, and package-release source code are retained in the result directory.

## Scope and remaining work

Created only this spike directory, `results/tensorflow-metal-probe-v1/`, and the new isolated environment under `artifacts/tensorflow-metal-probe-py312`. Main `.venv`, project sources/tests/docs/Git, the source checkpoint, and other children's files were not modified. All probe subprocesses ran foreground and exited. Parent owns any sequential performance comparison; it must label this baseline hybrid and FP16, or report unmodified native all-GPU generation blocked instead.

## Sources

[1] https://developer.apple.com/metal/tensorflow-plugin
[2] https://keras.io/keras_hub/api/models/qwen/qwen_causal_lm
[3] https://raw.githubusercontent.com/keras-team/keras-hub/v0.21.1/keras_hub/src/utils/transformers/convert_qwen.py
[4] https://raw.githubusercontent.com/keras-team/keras/v3.15.1/keras/src/backend/tensorflow/core.py
[5] https://pypi.org/pypi/tensorflow-metal/json
[6] https://pypi.org/pypi/keras-hub/0.21.1/json
[7] https://pypi.org/pypi/keras-hub/json
[8] https://api.github.com/repos/huggingface/transformers/git/trees/main?recursive=1
