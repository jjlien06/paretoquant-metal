"""Archived-target-trace acceptance diagnostic; never verifies target logits."""

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path


def read_bytes(path, limit=1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Expected bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("File exceeds bound")
    return data


def read_json(path):
    try:
        return json.loads(read_bytes(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid local JSON: {path}") from error


def check_ids(ids, vocab):
    if not ids or any(type(t) is not int or not 0 <= t < vocab for t in ids):
        raise ValueError("Expected nonempty exact integer vocabulary IDs")


def read_reference(path, prompt, vocab, eos):
    record = read_json(path)
    try:
        request = record["requests"][0]
        tokens = request["token_ids"]
        check_ids(tokens, vocab)
        if (
            request["finish_reason"] != "stop"
            or tokens[-1] not in eos
            or any(t in eos for t in tokens[:-1])
        ):
            raise ValueError("Reference must finish at its first EOS, with stop reason")
        original = record.get("prompt")
        if original is not None and prompt is not None and original != prompt:
            raise ValueError("Explicit prompt disagrees with archived prompt")
        prompt = original if original is not None else prompt
        if not isinstance(prompt, str) or not 0 < len(prompt) <= 4096:
            raise ValueError("Original prompt or explicit bounded --prompt required")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Malformed reference") from error
    return record, prompt, tokens


TARGET = "mlx-community/Qwen2.5-32B-Instruct-4bit"
REVISION = "2938092373e5f97b95538884112085364c2da315"
CONFIG_SHA = "c027829d800805358d67ac87819a3754fd8240be973f7147840651310fd30ae3"


def prepare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft-model", "target-tokenizer", "reference-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-receipt", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-seconds", type=float, default=150)
    args = parser.parse_args(argv)
    if (
        not 1 <= len(args.block_sizes) <= 8
        or len(set(args.block_sizes)) != len(args.block_sizes)
        or any(not 1 <= k <= 16 for k in args.block_sizes)
    ):
        raise ValueError("Require unique block sizes in 1..16, at most eight")
    if not 0 < args.max_seconds <= 180:
        raise ValueError("max-seconds must be in (0,180]")
    if os.path.lexists(args.output) or not args.output.parent.is_dir():
        raise ValueError("Output must be NEW in an existing directory")
    if (
        not args.draft_model.is_dir()
        or not args.target_tokenizer.is_dir()
        or args.draft_model.resolve() == args.target_tokenizer.resolve()
    ):
        raise ValueError("Require distinct local draft and target metadata directories")
    if list(args.target_tokenizer.glob("*.safetensors")):
        raise ValueError("Target directory must contain metadata only")
    config_bytes = read_bytes(args.target_tokenizer / "config.json")
    config = json.loads(config_bytes)
    receipt_path = args.target_receipt or args.target_tokenizer / "target-receipt.json"
    receipt = read_json(receipt_path)
    digest = hashlib.sha256(config_bytes).hexdigest()
    if (
        receipt.get("model") != TARGET
        or receipt.get("revision") != REVISION
        or digest != CONFIG_SHA
        or receipt.get("config_sha256", digest) != digest
    ):
        raise ValueError("Pinned target revision/config receipt mismatch")
    eos = {config["eos_token_id"]}
    record, prompt, tokens = read_reference(
        args.reference_json, args.prompt, config["vocab_size"], eos
    )
    reference_sha = hashlib.sha256(read_bytes(args.reference_json)).hexdigest()
    if receipt.get("reference_sha256") != reference_sha:
        raise ValueError(
            "Receipt must explicitly bind reference_sha256 (not independent verification)"
        )
    if len(tokens) > 256:
        raise ValueError("Reference exceeds 256-token diagnostic bound")
    draft_config = read_json(args.draft_model / "config.json")
    if draft_config.get("model_type") != "qwen2" or any(
        type(draft_config.get(key)) is not int or not 0 < draft_config[key] <= maximum
        for key, maximum in [
            ("hidden_size", 1024),
            ("intermediate_size", 8192),
            ("num_hidden_layers", 24),
            ("vocab_size", config["vocab_size"]),
        ]
    ):
        raise ValueError("Draft must be a bounded small Qwen2 model, never the 32B target")
    return dict(
        args=args,
        config=config,
        draft_config=draft_config,
        record=record,
        prompt=prompt,
        tokens=tokens,
        eos=eos,
        reference_sha256=reference_sha,
        receipt_sha256=hashlib.sha256(read_bytes(receipt_path)).hexdigest(),
    )


def encode_prompt(record, prompt, tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    archived = record.get("formatted_prompt", rendered)
    if (
        not isinstance(archived, str)
        or len(archived) > 16384
        or archived.encode() != rendered.encode()
    ):
        raise ValueError(
            "Regenerated target template differs from authoritative archived formatted_prompt"
        )
    bos = getattr(tokenizer, "bos_token", None)
    ids = tokenizer.encode(archived, add_special_tokens=bos is None or not archived.startswith(bos))
    expected = record["requests"][0].get("prompt_tokens", len(ids))
    if not ids or len(ids) != expected:
        raise ValueError("Archived prompt token count differs")
    return archived, ids


def compatible(ids, target, draft, target_size, draft_size):
    check_ids(ids, min(target_size, draft_size))
    for token in ids:
        if token not in target or token not in draft or target[token] != draft[token]:
            raise ValueError(f"Incompatible/unmapped token ID {token}; no repair or retokenization")


def inventory(directory, limit=512 * 1024 * 1024):
    result, total = {}, 0
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise ValueError("Source inventory refuses symlinks")
        if path.is_dir():
            continue
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            total += info.st_size
            if not stat.S_ISREG(info.st_mode) or total > limit or len(result) >= 64:
                raise ValueError("Source inventory exceeds regular-file/size/count bounds")
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                if size > info.st_size:
                    raise ValueError("Source changed during hashing")
            result[path.name] = dict(bytes=size, sha256=digest.hexdigest())
    return result


def run(prepared):
    import platform
    import resource
    from importlib.metadata import version

    args = prepared["args"]
    deadline = time.monotonic() + args.max_seconds
    draft_files = inventory(args.draft_model)
    target_files = inventory(args.target_tokenizer, 32 * 1024 * 1024)
    if not any(name.endswith(".safetensors") for name in draft_files):
        raise ValueError("Local draft safetensors required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_tokenizer

    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = load_tokenizer(
        args.target_tokenizer, tokenizer_config_extra=options, eos_token_ids=list(prepared["eos"])
    )
    formatted, prompt_ids = encode_prompt(prepared["record"], prepared["prompt"], tokenizer)
    if len(prompt_ids) + len(prepared["tokens"]) > 512:
        raise ValueError("Prompt plus reference exceeds 512-token context bound")
    model, draft_tokenizer = load(
        str(args.draft_model), tokenizer_config=options, trust_remote_code=False, lazy=False
    )
    target_map = {i: t for t, i in tokenizer.get_vocab().items()}
    draft_map = {i: t for t, i in draft_tokenizer.get_vocab().items()}
    sizes = prepared["config"]["vocab_size"], prepared["draft_config"]["vocab_size"]
    compatible(prompt_ids + prepared["tokens"], target_map, draft_map, *sizes)
    common = sorted(set(target_map) & set(draft_map))
    compatible(common, target_map, draft_map, *sizes)
    parameters = [
        dict(name=name, shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes)
        for name, value in tree_flatten(model.parameters())
    ]
    parameter_bytes = sum(p["bytes"] for p in parameters)
    if not 0 < parameter_bytes <= 512 * 1024 * 1024:
        raise ValueError("Loaded draft parameter bytes exceed small-model bound")

    def propose(prefix, count):
        proposal, elapsed = greedy(
            mx,
            model,
            make_prompt_cache,
            prefix,
            count,
            set(draft_tokenizer.eos_token_ids) | prepared["eos"],
            deadline,
        )
        compatible(proposal, target_map, draft_map, *sizes)
        return proposal, elapsed

    screens = [
        screen_trace(prompt_ids, prepared["tokens"], k, prepared["eos"], propose)
        for k in args.block_sizes
    ]
    if (
        inventory(args.draft_model) != draft_files
        or inventory(args.target_tokenizer, 32 * 1024 * 1024) != target_files
    ):
        raise ValueError("Source files changed during diagnostic")
    source_hashes = {Path(__file__).name: hashlib.sha256(read_bytes(__file__)).hexdigest()}
    draft_sha = draft_files["config.json"]["sha256"]
    kind = (
        "mixedprecision 3/4bit (parent-pinned source)"
        if draft_sha == "43a1c2f3b692e8cca5974f4753514e94ee4c969f8853b6d02acc16c7e64805da"
        else "source config; not presumed uniform4"
    )
    return dict(
        diagnostic="archived-target-trace acceptance",
        actual_target_verification=False,
        prefix_recomputed_each_round=True,
        limitations=[
            "No target load or logits verification",
            "Acceptance alone predicts no speedup, performance or quality",
            "Draft times include fresh-cache prefix recomputation",
            "Receipt/reference binding is not independent target provenance verification",
            "MLX memory is not OS residency",
        ],
        target_model=TARGET,
        target_revision=REVISION,
        reference_sha256=prepared["reference_sha256"],
        target_receipt_sha256=prepared["receipt_sha256"],
        target_metadata_inventory=target_files,
        draft_source_inventory=draft_files,
        draft_source_binding=(
            "stock mlx_lm.load, local files hashed before/after; "
            "not independent packed-array extraction"
        ),
        draft_kind=kind,
        draft_config=prepared["draft_config"],
        draft_loaded_parameters=parameters,
        draft_loaded_parameter_bytes=parameter_bytes,
        source_hashes=source_hashes,
        prompt=prepared["prompt"],
        formatted_prompt=formatted,
        prompt_ids=prompt_ids,
        reference_tokens=prepared["tokens"],
        shared_token_ids_checked=len(common),
        screens=screens,
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            mlx=version("mlx"),
            mlx_lm=version("mlx-lm"),
            transformers=version("transformers"),
            device=str(mx.default_device()),
            device_info=mx.metal.device_info(),
            pid=os.getpid(),
        ),
        memory=dict(
            active_mlx_bytes=mx.get_active_memory(),
            peak_mlx_bytes=mx.get_peak_memory(),
            cache_mlx_bytes=mx.get_cache_memory(),
            max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )


def greedy(mx, model, make_cache, prefix, k, eos, deadline):
    mx.synchronize()
    started = time.perf_counter()
    cache = make_cache(model)
    current, proposal = mx.array([prefix], dtype=mx.int32), []
    for _ in range(k):
        if time.monotonic() >= deadline:
            raise TimeoutError("Draft diagnostic deadline reached")
        scores = model(current, cache=cache)
        selected = mx.argmax(scores[:, -1, :], axis=-1)
        mx.eval(selected)
        token = int(selected.item())
        proposal.append(token)
        if token in eos:
            break
        current = mx.array([[token]], dtype=mx.int32)
    mx.synchronize()
    return proposal, time.perf_counter() - started


def screen_trace(prompt_ids, tokens, k, eos, propose):
    cursor, rounds = 0, []
    while cursor < len(tokens):
        count = min(k, len(tokens) - cursor)
        proposal, elapsed = propose(prompt_ids + tokens[:cursor], count)
        if not proposal or len(proposal) > count:
            raise ValueError("Draft must return 1..K proposals")
        window = tokens[cursor : cursor + count + 1]
        matched, emitted = acceptance(proposal, window, eos)
        rounds.append(
            dict(
                cursor=cursor,
                proposal=proposal,
                reference_window=window,
                accepted_prefix=proposal[:matched],
                accepted=matched,
                emitted=emitted,
                draft_seconds=elapsed,
                prefix_tokens=len(prompt_ids) + cursor,
            )
        )
        cursor += emitted
    return dict(
        block_size=k,
        rounds=rounds,
        round_count=len(rounds),
        emitted_tokens=cursor,
        proposed_tokens=sum(len(r["proposal"]) for r in rounds),
        accepted_tokens=sum(r["accepted"] for r in rounds),
        draft_seconds=sum(r["draft_seconds"] for r in rounds),
    )


def acceptance(proposal, remaining, eos):
    matched = 0
    for draft, target in zip(proposal, remaining):
        if draft != target:
            break
        matched += 1
        if target in eos:
            return matched, matched
    return matched, min(matched + 1, len(remaining))


def main(argv=None):
    prepared = prepare(argv)
    report = run(prepared)
    with prepared["args"].output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    main()
