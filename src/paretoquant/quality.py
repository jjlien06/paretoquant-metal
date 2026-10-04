"""Offline, bounded-context held-out likelihood with explicit target accounting.

CPU accounting has no MLX dependency. The CLI loads only one local model at a
 time; no generation, KV reuse, fused dispatch, or remote model code is used.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Corpus:
    text: str
    metadata: dict


def load_corpus(path, *, label, separator="\n\n"):
    """Join local JSON records verbatim, preserving order and blank records."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        texts = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("corpus must be a UTF-8 JSON list of texts") from exc
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("corpus must be a nonempty JSON list of strings")
    if not any(t.strip() for t in texts):
        raise ValueError("corpus must contain nonblank text")
    text = separator.join(texts)
    return Corpus(
        text,
        {
            "path": str(path),
            "label": label,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "joined_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_record_count": len(texts),
            "blank_record_count": sum(not t.strip() for t in texts),
            "sequence_count": 1,
            "separator": separator,
            "sequence_policy": "verbatim_records_joined_then_tokenized_once",
            "special_tokens_added": False,
            "canonical_benchmark": False,
            "designation": "noncanonical_local_held_out_corpus_or_subsample",
            "held_out_status": "user_asserted_not_independently_verified",
        },
    )


def tokenize_corpus(corpus, tokenizer, *, max_target_tokens=None):
    tokens = list(tokenizer.encode(corpus.text, add_special_tokens=False))
    _validate_tokens(tokens)
    if max_target_tokens is not None and (
        type(max_target_tokens) is not int or max_target_tokens < 1
    ):
        raise ValueError("max_target_tokens must be positive")
    evaluated_count = (
        len(tokens) if max_target_tokens is None else min(len(tokens), max_target_tokens + 1)
    )
    encoded = np.asarray(tokens, dtype="<u8").tobytes()
    return tokens, {
        "stream_token_count": len(tokens),
        "evaluated_stream_token_count": evaluated_count,
        "stream_sha256": hashlib.sha256(encoded).hexdigest(),
        "evaluated_stream_sha256": hashlib.sha256(encoded[: evaluated_count * 8]).hexdigest(),
        "hash_encoding": "unsigned_64bit_little_endian_token_ids",
    }


def local_model_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"model must be an existing local directory with config.json: {path}")
    return path


def fingerprint_model(path):
    """SHA-256 of sorted non-hidden file paths, sizes, and content hashes.

    Symlinked HF snapshot files are dereferenced, supporting offline snapshots.
    Hidden cache metadata is excluded. The result contains no model tensors.
    """
    path = local_model_path(path)
    files = []
    for file in sorted(path.rglob("*")):
        relative = file.relative_to(path)
        if not file.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": file.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(manifest).hexdigest(),
        "hash_policy": "sorted_nonhidden_files_content_manifest_v1",
        "files": files,
    }


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    target_start: int
    target_end: int


def iter_windows(token_count, *, window_length, stride, max_target_tokens=None):
    """Partition target positions 1..N-1, retaining bounded preceding context.

    Offsets are global, zero-based, end-exclusive. ``window_length`` counts the
    full token slice including its final target, so model input is at most L-1.
    """
    validate_settings(window_length, stride, max_target_tokens)
    if type(token_count) is not int or token_count < 2:
        raise ValueError("token stream must contain at least two tokens")
    stop = token_count if max_target_tokens is None else min(token_count, max_target_tokens + 1)
    for target_start in range(1, stop, stride):
        end = min(target_start + stride, stop)
        yield Window(max(0, end - window_length), end, target_start, end)


def validate_settings(window_length, stride, max_target_tokens):
    if type(window_length) is not int or window_length < 2:
        raise ValueError("window_length must be an integer >= 2")
    if type(stride) is not int or not 1 <= stride < window_length:
        raise ValueError("stride must satisfy 1 <= stride < window_length")
    if max_target_tokens is not None and (
        type(max_target_tokens) is not int or max_target_tokens < 1
    ):
        raise ValueError("max_target_tokens must be a positive integer or None")


def _validate_tokens(tokens, *, minimum=2):
    if len(tokens) < minimum or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError(f"token stream requires >= {minimum} nonnegative integer token ids")


def _validate_logits_shape(shape, targets, score_start):
    _validate_tokens(targets, minimum=1)
    if len(shape) != 3 or shape[0] != 1 or shape[2] < 1:
        raise ValueError("logits must have shape [1, sequence_length, vocabulary_size]")
    if type(score_start) is not int or not 0 <= score_start <= shape[1] - len(targets):
        raise ValueError("scored logit positions exceed the input sequence")
    if max(targets) >= shape[2]:
        raise ValueError("target token exceeds the model vocabulary")


def numpy_logit_losses(logits, targets, score_start):
    """CPU oracle: stable log-softmax evaluated in host float64."""
    logits = np.asarray(logits, dtype=np.float64)
    _validate_logits_shape(logits.shape, targets, score_start)
    if not np.isfinite(logits).all():
        raise ValueError("model logits must be finite")
    selected = logits[0, score_start : score_start + len(targets)]
    shifted = selected - np.max(selected, axis=-1, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=-1))
    return log_z - shifted[np.arange(len(targets)), targets]


def _finite_sum(values):
    try:
        return math.fsum(values)
    except OverflowError as exc:
        raise ValueError("total NLL must be finite") from exc


def evaluate_tokens(tokens, loss_fn, *, window_length=512, stride=256, max_target_tokens=None):
    """Score exact disjoint targets; ``loss_fn`` returns one NLL per target.

    The callback receives input token ids, target ids, and the first scored
    logit position within input. Context-only logits must never enter the sum.
    """
    validate_settings(window_length, stride, max_target_tokens)
    _validate_tokens(tokens)
    totals = []
    audits = []
    count = 0
    for window in iter_windows(
        len(tokens), window_length=window_length, stride=stride, max_target_tokens=max_target_tokens
    ):
        inputs = list(tokens[window.start : window.end - 1])
        targets = list(tokens[window.target_start : window.target_end])
        losses = np.asarray(
            loss_fn(inputs, targets, window.target_start - window.start - 1), dtype=np.float64
        )
        if losses.shape != (len(targets),):
            raise ValueError("scorer must return exactly one NLL per target")
        if not np.isfinite(losses).all() or np.any(losses < 0):
            raise ValueError("per-target NLL must be finite and nonnegative")
        subtotal = _finite_sum(float(loss) for loss in losses)
        totals.append(subtotal)
        count += len(targets)
        audits.append(
            {
                **window.__dict__,
                "target_token_count": len(targets),
                "total_nll": subtotal,
                "input_token_count": len(inputs),
                "scored_logit_start": window.target_start - window.start - 1,
                "scored_logit_end": len(inputs),
            }
        )
    total = _finite_sum(totals)
    mean = total / count
    try:
        perplexity = math.exp(mean)
    except OverflowError as exc:
        raise ValueError("perplexity is not finite") from exc
    if not all(math.isfinite(x) for x in (total, mean, perplexity)):
        raise ValueError("evaluation statistics must be finite")
    return {
        "total_nll": total,
        "mean_nll": mean,
        "perplexity": perplexity,
        "target_token_count": count,
        "available_target_token_count": len(tokens) - 1,
        "stream_token_count": len(tokens),
        "unscored_initial_token_count": 1,
        "unscored_tail_target_token_count": len(tokens) - 1 - count,
        "scored_sequence_count": 1,
        "window_count": len(audits),
        "window_length": window_length,
        "stride": stride,
        "max_target_tokens": max_target_tokens,
        "truncated": count < len(tokens) - 1,
        "windows": audits,
    }


@dataclass(frozen=True)
class Variant:
    name: str
    path: Path
    uniform_bits: int | None = None
    dtype: str = "native"


def mlx_model_losses(model, inputs, targets, score_start):
    """Compute device FP32 NLL, transferring only scored losses to the host."""
    import mlx.core as mx

    logits = model(mx.array([inputs])).astype(mx.float32)
    _validate_logits_shape(logits.shape, targets, score_start)
    finite = mx.all(mx.isfinite(logits))
    selected = logits[0, score_start : score_start + len(targets)]
    shifted = selected - mx.max(selected, axis=-1, keepdims=True)
    selected_targets = mx.take_along_axis(shifted, mx.array(targets)[:, None], axis=-1)[:, 0]
    losses = mx.logsumexp(shifted, axis=-1) - selected_targets
    mx.eval(finite, losses)
    if not finite.item():
        raise ValueError("model logits must be finite")
    return np.asarray(losses, dtype=np.float64)


def _preflight_variant(variant):
    path = local_model_path(variant.path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model config must be an object")
    configs = [config, config.get("text_config", {})]
    tokenizer_config = path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        configs.append(json.loads(tokenizer_config.read_text(encoding="utf-8")))
    if not all(isinstance(c, dict) for c in configs):
        raise ValueError("model text_config and tokenizer config must be objects")
    if any(c.get("model_file") or c.get("auto_map") for c in configs):
        raise ValueError("custom model/tokenizer code is forbidden in offline evaluation")
    if (variant.dtype == "float16" or variant.uniform_bits) and any(
        c.get("quantization") or c.get("quantization_config") for c in configs[:2]
    ):
        raise ValueError("FP16 reference and uniform4 require an unquantized local source")
    return path, config


def _model_metadata(model):
    from mlx.utils import tree_flatten

    parameters = tree_flatten(model.parameters())
    quantized = [
        {
            "path": path,
            "bits": int(module.bits),
            "group_size": int(module.group_size),
            "mode": module.mode,
        }
        for path, module in model.named_modules()
        if hasattr(module, "bits") and hasattr(module, "group_size")
    ]
    return {
        "weight_dtype_counts": dict(sorted(Counter(str(p.dtype) for _, p in parameters).items())),
        "resident_parameter_bytes": sum(p.nbytes for _, p in parameters),
        "quantized_modules": quantized,
    }


def evaluate_local_model(
    variant, corpus, *, window_length=512, stride=256, max_target_tokens=4096, group_size=64
):
    """Load, score, and release one local variant, including device caches."""
    validate_settings(window_length, stride, max_target_tokens)
    if variant.dtype not in ("native", "float16") or variant.uniform_bits not in (None, 4):
        raise ValueError("unsupported variant dtype or uniform bit width")
    if group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    path, source_config = _preflight_variant(variant)
    configured_context = source_config.get("max_position_embeddings") or source_config.get(
        "text_config", {}
    ).get("max_position_embeddings")
    if configured_context is not None and (
        type(configured_context) is not int or configured_context < 1
    ):
        raise ValueError("max_position_embeddings must be a positive integer")
    if configured_context and window_length - 1 > configured_context:
        raise ValueError("window input length exceeds model max_position_embeddings")
    fingerprint = fingerprint_model(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_lm.utils import load_model, load_tokenizer, quantize_model

    model = tokenizer = scorer = None
    try:
        # load_model accepts a Path and cannot resolve a Hub repository.
        model, config = load_model(path, lazy=False, strict=True, trust_remote_code=False)
        tokenizer = load_tokenizer(
            path,
            {"local_files_only": True, "trust_remote_code": False},
            eos_token_ids=config.get("eos_token_id"),
        )
        if variant.dtype == "float16" or variant.uniform_bits:
            if any(hasattr(module, "bits") for _, module in model.named_modules()):
                raise ValueError("reference source has quantized modules")
            model.set_dtype(mx.float16)
            mx.eval(model.parameters())
        if variant.uniform_bits:
            with redirect_stdout(sys.stderr):
                model, config = quantize_model(
                    model, config, group_size, variant.uniform_bits, mode="affine"
                )
            mx.eval(model.parameters())
        metadata = _model_metadata(model)
        if variant.uniform_bits and not metadata["quantized_modules"]:
            raise ValueError("uniform quantization converted no eligible modules")
        tokens, tokenization = tokenize_corpus(
            corpus, tokenizer, max_target_tokens=max_target_tokens
        )
        scorer = partial(mlx_model_losses, model)
        metrics = evaluate_tokens(
            tokens,
            scorer,
            window_length=window_length,
            stride=stride,
            max_target_tokens=max_target_tokens,
        )
        return {
            "name": variant.name,
            "model": fingerprint,
            "transformation": {
                "dtype": "float16" if variant.uniform_bits else variant.dtype,
                "uniform_bits": variant.uniform_bits,
                "group_size": group_size if variant.uniform_bits else None,
                "mode": "affine" if variant.uniform_bits else None,
                "quantization_scope": "mlx_lm_eligible_modules_not_all_weights"
                if variant.uniform_bits
                else "loaded_artifact",
            },
            "effective_model_config": config,
            "device": str(mx.default_device()),
            "tokenization": tokenization,
            "metrics": metrics,
            **metadata,
        }
    finally:
        # No model, tokenizer, callback, logits, or parameter tree escapes.
        mx.synchronize()
        scorer = model = tokenizer = None
        gc.collect()
        mx.clear_cache()


def _environment():
    packages = {}
    for package in ("mlx", "mlx-lm", "numpy", "transformers", "tokenizers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def evaluate_variants(
    variants,
    corpus,
    *,
    window_length=512,
    stride=256,
    max_target_tokens=4096,
    group_size=64,
    runner=None,
):
    """Compare only identical token streams; never retain models in the report."""
    validate_settings(window_length, stride, max_target_tokens)
    if not variants or len({v.name for v in variants}) != len(variants):
        raise ValueError("model variants must have distinct nonempty names")
    if any(
        not v.name or v.dtype not in ("native", "float16") or v.uniform_bits not in (None, 4)
        for v in variants
    ):
        raise ValueError("invalid model variant")
    if group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    runner = runner or evaluate_local_model
    results = []
    for variant in variants:
        result = runner(
            variant,
            corpus,
            window_length=window_length,
            stride=stride,
            max_target_tokens=max_target_tokens,
            group_size=group_size,
        )
        if results:
            first = results[0]
            if (
                any(
                    result["tokenization"][key] != first["tokenization"][key]
                    for key in ("stream_sha256", "evaluated_stream_sha256")
                )
                or result["metrics"]["target_token_count"] != first["metrics"]["target_token_count"]
            ):
                raise ValueError("model token streams or target counts differ; comparison refused")
        baseline = results[0] if results else result
        result["delta_mean_nll_from_first"] = (
            result["metrics"]["mean_nll"] - baseline["metrics"]["mean_nll"]
        )
        result["perplexity_ratio_from_first"] = (
            result["metrics"]["perplexity"] / baseline["metrics"]["perplexity"]
        )
        results.append(result)
    return {
        "schema_version": 1,
        "scope": "noncanonical_held_out_perplexity_not_task_accuracy",
        "canonical_benchmark": False,
        "corpus": corpus.metadata,
        "execution": "sequential_one_model_resident_at_a_time",
        "method": {
            "window_length": window_length,
            "stride": stride,
            "max_target_tokens": max_target_tokens,
            "target_policy": "score_prefix_positions_1_through_cap_once",
            "window_length_includes_final_target": True,
            "context_policy": "reset_each_window_no_kv_reuse",
            "per_target_nll_precision": "mlx_float32_stable_logsumexp",
            "aggregation": "host_float64_math_fsum_per_window_then_global",
            "inference_backend": "stock_mlx_no_custom_fusion",
        },
        "environment": _environment(),
        "results": results,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Offline noncanonical held-out perplexity; not task accuracy or canonical WikiText."
        )
    )
    parser.add_argument("--corpus", required=True, type=Path, help="local JSON list of texts")
    parser.add_argument(
        "--corpus-name", required=True, help="provenance label, e.g. wikitext-2-raw-v1:test:prefix"
    )
    parser.add_argument("--separator", default="\n\n", help="literal record join separator")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=LOCAL_PATH",
        help="named local models, evaluated in order; repeatable",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="unquantized local model: creates FP16 and fresh uniform4 variants",
    )
    parser.add_argument(
        "--mixed", type=Path, help="local mixed artifact, stock MLX; requires --reference"
    )
    parser.add_argument(
        "--uniform-bits",
        type=int,
        choices=[4],
        help="add a fresh uniform4 companion for every --model (unquantized only)",
    )
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument(
        "--window-length",
        type=int,
        default=512,
        help="full window including final target (default: 512)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="new targets per window; must be < window length (default: 256)",
    )
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=4096,
        help="deterministic prefix target cap across corpus (default: 4096)",
    )
    parser.add_argument("--output", type=Path, help="JSON report path; otherwise stdout")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        validate_settings(args.window_length, args.stride, args.max_target_tokens)
        if args.reference and (args.model or args.uniform_bits):
            raise ValueError("--reference cannot be combined with --model or --uniform-bits")
        if args.mixed and not args.reference:
            raise ValueError("--mixed requires --reference")
        variants = []
        if args.reference:
            reference = local_model_path(args.reference)
            variants.extend(
                [
                    Variant("reference_fp16", reference, dtype="float16"),
                    Variant("uniform4", reference, uniform_bits=4, dtype="float16"),
                ]
            )
            if args.mixed:
                variants.append(Variant("mixed_stock", local_model_path(args.mixed)))
        else:
            for mapping in args.model:
                name, equals, path = mapping.partition("=")
                if not equals or not name.strip() or not path:
                    raise ValueError("--model requires NAME=LOCAL_PATH")
                path = local_model_path(path)
                variants.append(Variant(name, path))
                if args.uniform_bits:
                    variants.append(Variant(f"{name}:uniform4", path, 4, "float16"))
        if not variants or len({v.name for v in variants}) != len(variants):
            raise ValueError("at least one model with a distinct name is required")
        corpus = load_corpus(args.corpus, label=args.corpus_name, separator=args.separator)
        if args.output:
            output = args.output.expanduser().resolve()
            if output == args.corpus.expanduser().resolve() or any(
                output.is_relative_to(v.path) for v in variants
            ):
                raise ValueError(
                    "output must not overwrite corpus or files inside a model directory"
                )
            if output.exists():
                raise ValueError("output already exists; preserve prior evaluation evidence")
        report = evaluate_variants(
            variants,
            corpus,
            window_length=args.window_length,
            stride=args.stride,
            max_target_tokens=args.max_target_tokens,
            group_size=args.group_size,
        )
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
            print(f"Wrote {output}", file=sys.stderr)
        else:
            print(encoded, end="")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
