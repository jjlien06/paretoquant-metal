"""Isolated TensorFlow/Metal runtime and native KerasHub Qwen probe."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--cpu-cache-updates", action="store_true", help="Explicit hybrid fallback: native Keras slice_update on CPU; native Qwen matmuls on Metal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[2] / "models/qwen2.5-0.5b-instruct")
    parser.add_argument("--prompt", default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nIn one sentence, explain why the sky is blue.<|im_end|>\n<|im_start|>assistant\n")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--graph-generation", action="store_true")
    parser.add_argument("--benchmark-inputs", type=Path)
    parser.add_argument("--benchmark-repeats", type=int, default=7)
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing evidence")
    if args.benchmark_inputs and (
        args.benchmark_repeats < 2 or args.benchmark_warmup < 1
        or not args.cpu_cache_updates or args.runtime_only
    ):
        parser.error("benchmark requires native model, CPU cache updates, two repeats and one warmup")
    result = {
        "status": "blocked",
        "python": sys.version,
        "architecture": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "tensorflow", "tensorflow-metal", "keras", "keras-hub",
            "tensorflow-text", "numpy", "safetensors",
        )},
    }
    code = 1
    try:
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices("GPU")
        if not gpu_devices:
            raise RuntimeError("No physical TensorFlow GPU found")
        tf.config.set_soft_device_placement(False)
        with tf.device("/GPU:0"):
            values = tf.nn.relu(tf.matmul(
                tf.constant([[1., 2.], [3., 4.]]),
                tf.constant([[3., -3.], [4., -4.]]),
            ))
        result["runtime"] = {
            "physical_gpus": [str(d) for d in gpu_devices],
            "tensor_device": values.device,
            "values": values.numpy().tolist(),
            "soft_device_placement": tf.config.get_soft_device_placement(),
        }
        if "GPU:0" not in values.device:
            raise RuntimeError("Runtime result was not executed on GPU")
        result["status"] = "runtime_ready"
        if not args.runtime_only:
            import hashlib
            import numpy as np
            import keras
            import keras_hub
            import safetensors
            from tokenizers import Tokenizer
            from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

            checkpoint = args.checkpoint.resolve()
            config = json.loads((checkpoint / "config.json").read_text())
            if config.get("model_type") != "qwen2":
                raise ValueError("Only the same Qwen2 checkpoint is accepted")
            if args.new_tokens < 1:
                raise ValueError("--new-tokens must be positive")
            sha = hashlib.sha256()
            with (checkpoint / "model.safetensors").open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(block)
            result["checkpoint"] = {"path": str(checkpoint), "sha256": sha.hexdigest(), "original_dtype": "bfloat16", "keras_weight_dtype": "float16", "config": config}
            result["status"] = "blocked"
            print("PHASE: loading same local checkpoint with native KerasHub QwenCausalLM", flush=True)
            keras.mixed_precision.set_global_policy("float16")
            tf.config.set_soft_device_placement(True)
            loaded_weights = []
            original_port_weight = SafetensorLoader.port_weight

            def audited_port_weight(loader, keras_variable, hf_weight_key, hook_fn=None):
                original_port_weight(loader, keras_variable, hf_weight_key, hook_fn=hook_fn)
                expected = loader.get_tensor(hf_weight_key)
                if hook_fn:
                    expected = hook_fn(expected, list(keras_variable.shape))
                expected = np.asarray(expected, dtype=np.float16)
                actual = keras_variable.numpy()
                equal = np.array_equal(actual, expected)
                if not equal:
                    raise ValueError(f"Converted weight mismatch: {hf_weight_key}")
                loaded_weights.append({"key": hf_weight_key, "shape": list(actual.shape), "exact_after_float16_cast": equal})

            SafetensorLoader.port_weight = audited_port_weight
            try:
                with tf.device("/GPU:0"):
                    model = keras_hub.models.QwenCausalLM.from_preset(str(checkpoint), preprocessor=None)
            finally:
                SafetensorLoader.port_weight = original_port_weight
            with safetensors.safe_open(str(checkpoint / "model.safetensors"), framework="np") as weights:
                source_keys = set(weights.keys())
            if source_keys != {w["key"] for w in loaded_weights}:
                raise ValueError("Native conversion did not cover exactly all checkpoint tensor keys")
            result["loaded_weights"] = loaded_weights
            tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
            total_length = len(prompt_ids) + args.new_tokens
            token_ids = np.zeros((1, total_length), dtype=np.int32)
            padding_mask = np.zeros((1, total_length), dtype=bool)
            token_ids[0, :len(prompt_ids)] = prompt_ids
            padding_mask[0, :len(prompt_ids)] = True
            model.compile(sampler="greedy", run_eagerly=not args.graph_generation, jit_compile=False)
            print("PHASE: native Qwen forward correctness and greedy cached generation", flush=True)
            with tf.device("/GPU:0"):
                logits = model({"token_ids": tf.constant([prompt_ids], dtype=tf.int32), "padding_mask": tf.ones((1, len(prompt_ids)), dtype=tf.bool)}, training=False)
                finite_logits = bool(tf.reduce_all(tf.math.is_finite(logits)).numpy())
                forward_device = logits.device
                if not finite_logits or "GPU:0" not in forward_device:
                    raise RuntimeError("Forward logits were not finite GPU output")
                result["forward_correctness"] = {"finite_logits": finite_logits, "logits_device": forward_device}
                cache_update_devices = []
                original_slice_update = keras.ops.slice_update

                def cpu_slice_update(inputs, start_indices, updates):
                    with tf.device("/CPU:0"):
                        updated = original_slice_update(inputs, start_indices, updates)
                    cache_update_devices.append(updated.device)
                    return updated

                if args.cpu_cache_updates:
                    keras.ops.slice_update = cpu_slice_update
                try:
                    generated = model.generate({"token_ids": tf.constant(token_ids), "padding_mask": tf.constant(padding_mask)}, max_length=total_length, stop_token_ids=None)
                finally:
                    keras.ops.slice_update = original_slice_update
            output_ids = np.asarray(generated["token_ids"])[0].tolist()
            if output_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("Native generation changed the prompt prefix")
            new_ids = output_ids[len(prompt_ids):]
            if len(new_ids) != args.new_tokens:
                raise ValueError("Native generation returned wrong token count")
            result["generation"] = {
                "native_model_class": str(type(model)), "keras_backend": keras.backend.backend(),
                "parameter_count": model.count_params(), "all_weights_verified": True,
                "loaded_tensor_count": len(loaded_weights), "precision": "float16 nonquantized",
                "embedding_device": model.backbone.token_embedding.embeddings.value.device,
                "forward_logits_device": forward_device, "finite_forward_logits": finite_logits,
                "generation_output_location": getattr(generated["token_ids"], "device", "host NumPy array returned by native KerasHub.generate"),
                "soft_device_placement": tf.config.get_soft_device_placement(),
                "sampler": "greedy", "run_eagerly": not args.graph_generation, "jit_compile": False,
                "cpu_cache_updates": args.cpu_cache_updates,
                "cache_update_call_count": len(cache_update_devices),
                "cache_update_devices": sorted(set(cache_update_devices)),
                "mode": "native_keras_qwen_hybrid_cpu_cache_metal_forward" if args.cpu_cache_updates else "native_keras_qwen_metal",
                "prompt": args.prompt, "prompt_token_ids": prompt_ids,
                "new_token_ids": new_ids, "new_token_count": len(new_ids),
                "completion": tokenizer.decode(new_ids, skip_special_tokens=False),
                "full_text": tokenizer.decode(output_ids, skip_special_tokens=False),
                "stop_token_ids": None,
            }
            if args.benchmark_inputs:
                import statistics
                from time import perf_counter

                fixture = json.loads(args.benchmark_inputs.read_text())
                prompts = fixture["raw_prompts"]
                if len(prompts) != 3:
                    raise ValueError("expected three shared raw prompts")
                benchmark_trials = []
                benchmark_warmups = []
                before_updates = len(cache_update_devices)
                progress_path = args.output.with_suffix(".samples.jsonl")
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                keras.ops.slice_update = cpu_slice_update
                try:
                    with progress_path.open("x") as progress:
                        for trial in range(-args.benchmark_warmup, args.benchmark_repeats):
                            for case, raw in enumerate(prompts):
                                started = perf_counter()
                                ids = tokenizer.encode(raw, add_special_tokens=False).ids
                                length = len(ids) + args.new_tokens
                                if length > 1024:
                                    raise ValueError("workload exceeds shared 1024-token limit")
                                tokens = np.zeros((1, length), dtype=np.int32)
                                mask = np.zeros((1, length), dtype=bool)
                                tokens[0, :len(ids)] = ids
                                mask[0, :len(ids)] = True
                                with tf.device("/GPU:0"):
                                    batch = model.generate(
                                        {"token_ids": tf.constant(tokens),
                                         "padding_mask": tf.constant(mask)},
                                        max_length=length, stop_token_ids=None,
                                    )
                                actual_ids = np.asarray(batch["token_ids"])[0].tolist()
                                if actual_ids[:len(ids)] != ids:
                                    raise ValueError("generation changed the prompt prefix")
                                emitted = actual_ids[len(ids):]
                                text = tokenizer.decode(emitted, skip_special_tokens=False)
                                elapsed = perf_counter() - started
                                if len(emitted) != args.new_tokens or not text or elapsed <= 0:
                                    raise ValueError("invalid actual fixed-token generation")
                                row = {
                                    "engine": "tensorflow_metal_hybrid", "trial": trial,
                                    "case": case, "prompt_tokens": len(ids),
                                    "generated_tokens": len(emitted), "token_ids": emitted,
                                    "response": text, "wall_seconds": elapsed,
                                    "wall_tokens_per_second": len(emitted) / elapsed,
                                }
                                target = benchmark_trials if trial >= 0 else benchmark_warmups
                                target.append(row)
                                progress.write(json.dumps(row) + "\n")
                                progress.flush()
                                print(f"tensorflow trial={trial} case={case} "
                                      f"tokens={len(emitted)} wall_tps={len(emitted)/elapsed:.2f}",
                                      flush=True)
                finally:
                    keras.ops.slice_update = original_slice_update
                summaries = []
                for case in range(len(prompts)):
                    rows = [row for row in benchmark_trials if row["case"] == case]
                    if len(rows) != args.benchmark_repeats:
                        raise ValueError("actual trial count differs from requested repeats")
                    summaries.append({
                        "case": case, "prompt_tokens": rows[0]["prompt_tokens"],
                        "repeats": len(rows),
                        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
                        "median_wall_tokens_per_second": statistics.median(
                            row["wall_tokens_per_second"] for row in rows),
                    })
                result["benchmark"] = {
                    "engine": "tensorflow_metal_hybrid",
                    "mode": "native_keras_qwen_hybrid_cpu_cache_metal_forward",
                    "max_generated_tokens": args.new_tokens,
                    "warmup_per_case": args.benchmark_warmup,
                    "run_eagerly": not args.graph_generation,
                    "fixture_sha256": hashlib.sha256(args.benchmark_inputs.read_bytes()).hexdigest(),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "raw_prompts": prompts, "trials": benchmark_trials,
                    "warmups": benchmark_warmups, "summaries": summaries,
                    "cpu_cache_update_calls": len(cache_update_devices) - before_updates,
                    "cpu_cache_update_devices": sorted(set(cache_update_devices)),
                    "limitations": [
                        "Native KerasHub Qwen, FP16 unquantized, not precision-matched to affine3/4.",
                        "Explicit CPU cache/sampler slice updates; ordinary soft CPU placement also enabled.",
                        "Native eager or TF graph generation, jit_compile=False; not an all-GPU engine.",
                        "Wall time includes tokenization, prefill, generation and detokenization, not model load.",
                        "Sequential backend blocks, not paired cross-backend trials; background state may drift.",
                    ],
                }
            result["status"] = "ready"
        code = 0
    except Exception as error:
        result["status"] = "blocked"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    sys.exit(code)
