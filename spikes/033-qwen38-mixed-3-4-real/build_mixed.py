"""Assemble a text-only mixed affine checkpoint with bounded raw CPU I/O.

Installed mlx_lm/utils.py load_model._quantize.class_predicate returns
config['quantization'][module_path] for per-module overrides. Paths retain
language_model.*, as used by mlx_lm/models/qwen3_5.py Model.
No MLX imports, tensor decoding, mmap, or shard-sized reads are used.
"""
import argparse
import hashlib
import json
import math
import shutil
import struct
from collections import defaultdict
from pathlib import Path

from allocate_real import (EMBED, HEAD, NORM, PARTS, PFX, decoder_modules,
                           layer_of, nbytes, text_tensor)

CHUNK = 8 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def safe_path(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes checkpoint directory: {name}")
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path):
    """Validate a safetensors shard and return absolute payload byte ranges."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length > min(size - 8, CHUNK):
            raise ValueError(f"Invalid or >8MiB safetensors header: {path}")
        header = json.loads(handle.read(length), object_pairs_hook=unique_object)
    base = 8 + length
    result, ranges = {}, []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape, dtype = info["shape"], info["dtype"]
        offsets = info["data_offsets"]
        if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                or dtype not in DTYPE_BYTES or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError(f"Invalid shape/dtype/offsets: {path}:{name}")
        start, end = offsets
        if start < 0 or end < start or base + end > size:
            raise ValueError(f"Out-of-bounds tensor: {path}:{name}")
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor byte length disagrees with shape/dtype: {name}")
        result[name] = {"file": str(path), "start": base + start, "end": base + end,
                        "shape": shape, "dtype": dtype}
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"Overlapping tensors or payload gaps: {path}")
        cursor = end
    if base + cursor != size:
        raise ValueError(f"Unindexed payload bytes: {path}")
    return result


def scan_checkpoint(root):
    index = root / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
    files = sorted(set(weight_map.values())) if weight_map is not None else sorted(
        p.name for p in root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors shards in {root}")
    tensors = {}
    for name in files:
        shard = read_header(safe_path(root, name))
        if tensors.keys() & shard.keys():
            raise ValueError("Duplicate tensors across shards")
        if weight_map is not None:
            for tensor in shard:
                if weight_map.get(tensor) != name:
                    raise ValueError(f"Index/header mismatch: {tensor}")
        tensors.update(shard)
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError("Index references missing tensors")
    return tensors


def hash_range(info):
    digest = hashlib.sha256()
    left = nbytes(info)
    with Path(info["file"]).open("rb") as handle:
        handle.seek(info["start"])
        while left:
            chunk = handle.read(min(CHUNK, left))
            if not chunk:
                raise ValueError("Source tensor truncated during hashing")
            digest.update(chunk)
            left -= len(chunk)
    return digest.hexdigest()


def write_shard(path, tensors):
    offset = 0
    header: dict[str, dict] = {"__metadata__": {"format": "mlx"}}
    for name, info in sorted(tensors.items()):
        end = offset + nbytes(info)
        header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                        "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    if len(raw) > CHUNK:
        raise ValueError("Output header exceeds bounded header size")
    hashes = {}
    with path.open("xb") as dest:
        dest.write(struct.pack("<Q", len(raw)))
        dest.write(raw)
        for name, info in sorted(tensors.items()):
            digest, left = hashlib.sha256(), nbytes(info)
            with Path(info["file"]).open("rb") as source:
                source.seek(info["start"])
                while left:
                    chunk = source.read(min(CHUNK, left))
                    if not chunk:
                        raise ValueError(f"Source tensor truncated while copying: {name}")
                    dest.write(chunk)
                    digest.update(chunk)
                    left -= len(chunk)
            hashes[name] = digest.hexdigest()
    # Read the exact output back independently; compare bytes and full metadata.
    actual = read_header(path)
    if set(actual) != set(tensors):
        raise ValueError("Output header tensor set changed")
    for name, info in actual.items():
        if any(info[k] != tensors[name][k] for k in ("shape", "dtype")):
            raise ValueError(f"Output tensor metadata mismatch: {name}")
        if nbytes(info) != nbytes(tensors[name]) or hash_range(info) != hashes[name]:
            raise ValueError(f"Output tensor hash/length mismatch: {name}")
    return actual, hashes


def quant_config(config, module, bits):
    q = config.get("quantization") or config.get("text_config", {}).get("quantization")
    if not isinstance(q, dict):
        raise ValueError("Source requires affine quantization config")
    effective = q.get(module, q)
    if not isinstance(effective, dict):
        raise ValueError(f"Quantized module disabled in source config: {module}")
    result = {"bits": effective.get("bits", q.get("bits")),
              "group_size": effective.get("group_size", q.get("group_size")),
              "mode": effective.get("mode", q.get("mode", "affine"))}
    if result["bits"] != bits or result["mode"] != "affine" or type(result["group_size"]) is not int or result["group_size"] <= 0:
        raise ValueError(f"Expected affine q{bits} config for {module}: {result}")
    return result


def copy_assets(source, output):
    """Copy discovered non-weight assets recursively, including tokenizer subdirs."""
    copied = []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        if (rel.as_posix() == "config.json" or path.name.endswith(".safetensors.index.json")
                or path.suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf")):
            continue
        target = safe_path(output, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
        if hash_range({"file": str(path), "start": 0, "end": path.stat().st_size}) != hash_range(
                {"file": str(target), "start": 0, "end": target.stat().st_size}):
            raise ValueError(f"Asset copy verification failed: {rel}")
        copied.append(rel.as_posix())
    return copied


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def build(q4_dir, q3_dir, plan_document, plan_index, output):
    q4_dir, q3_dir, output = Path(q4_dir), Path(q3_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    if any(output.resolve().is_relative_to(source.resolve()) for source in (q4_dir, q3_dir)):
        raise ValueError("Output cannot be nested inside a source checkpoint")
    if plan_index < 0 or plan_index >= len(plan_document["plans"]):
        raise ValueError("plan-index out of range")
    plan = plan_document["plans"][plan_index]
    if "error" in plan:
        raise ValueError(f"Cannot build infeasible plan: {plan['error']}")
    chosen = plan["q3_layers"]
    if not isinstance(chosen, list) or any(type(i) is not int for i in chosen) or len(set(chosen)) != len(chosen):
        raise ValueError("q3_layers must be a list of unique integer layer IDs")
    q4, q3 = scan_checkpoint(q4_dir), scan_checkpoint(q3_dir)
    layers = {layer_of(n) for n in q4 if n.startswith(PFX)}
    if not layers or layers != set(range(len(layers))) or not set(chosen) <= layers:
        raise ValueError("Noncontiguous decoder layers or selected layer out of range")
    if plan_document.get("n_units", len(layers)) != len(layers):
        raise ValueError("Plan layer count disagrees with actual headers")
    mods4, mods3 = decoder_modules(q4), decoder_modules(q3)
    if mods4 != mods3 or {layer_of(m) for m in mods4} != layers:
        raise ValueError("q3/q4 decoder module sets differ or a layer lacks linears")
    cfg4 = json.loads((q4_dir / "config.json").read_text())
    cfg3 = json.loads((q3_dir / "config.json").read_text())
    if cfg4.get("text_config") and cfg3.get("text_config"):
        # Only quantization metadata may differ between source text architectures.
        strip = lambda c: {k: v for k, v in c.items() if k not in ("quantization", "quantization_config", "torch_dtype")}
        if strip(cfg4["text_config"]) != strip(cfg3["text_config"]):
            raise ValueError("q3/q4 text model configurations disagree")
    selected_names = {f"{m}.{p}" for m in mods4 if layer_of(m) in chosen for p in PARTS}
    tensors = {n: dict(q3[n] if n in selected_names else v) for n, v in q4.items() if text_tensor(n)}
    labels = {n: "q3" if n in selected_names else "q4" for n in tensors}
    for prefix in (EMBED, HEAD, NORM):
        if not any(n.startswith(prefix) for n in tensors):
            raise ValueError(f"Missing fixed text tensors: {prefix}")
    # Validate selected quant modules have matching logical affine dimensions.
    for m in mods4:
        for part in ("scales", "biases"):
            a, b = q4[m + "." + part], q3[m + "." + part]
            if a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
                raise ValueError(f"Incompatible affine metadata: {m}.{part}")
        if q4[m + ".weight"]["shape"][0] != q3[m + ".weight"]["shape"][0]:
            raise ValueError(f"Incompatible output dimension: {m}")
    all_modules = sorted(n[:-7] for n in tensors if n.endswith(".scales"))
    quant = {"bits": 4, "group_size": quant_config(cfg4, all_modules[0], 4)["group_size"], "mode": "affine"}
    for module in all_modules:
        use3 = layer_of(module) in chosen
        quant[module] = quant_config(cfg3 if use3 else cfg4, module, 3 if use3 else 4)
    total = sum(nbytes(v) for v in tensors.values())
    baseline = sum(nbytes(v) for n, v in q4.items() if text_tensor(n))
    for key, actual in (("text_stored_bytes", total), ("actual_saved_bytes", baseline - total),
                        ("saved_bytes_vs_uniform_q4", baseline - total)):
        if key in plan and plan[key] != actual:
            raise ValueError(f"Plan {key} disagrees with actual source headers: {plan[key]} != {actual}")
    groups = defaultdict(dict)
    for name, info in tensors.items():
        layer = layer_of(name)
        group = f"layer-{layer:02d}" if layer is not None else (
            "embed" if name.startswith(EMBED) else "head" if name.startswith(HEAD) else "norm")
        groups[group][name] = info
    # mkdir is exclusive; partial builds remain conspicuously incomplete on error.
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "BUILD_INCOMPLETE.json", {"plan_index": plan_index})
    manifest, weight_map = {}, {}
    for group, members in sorted(groups.items()):
        filename = f"{group}.safetensors"
        actual, hashes = write_shard(output / filename, members)
        for name, info in actual.items():
            weight_map[name] = filename
            manifest[name] = {"source": labels[name], "source_file": tensors[name]["file"],
                              "source_start": tensors[name]["start"], "source_end": tensors[name]["end"],
                              "sha256": hashes[name], "file": filename,
                              "start": info["start"], "end": info["end"],
                              "shape": info["shape"], "dtype": info["dtype"]}
    assets = copy_assets(q4_dir, output)
    config = dict(cfg4)
    config.pop("vision_config", None)
    config.pop("quantization_config", None)
    config["quantization"] = quant
    if "text_config" in config:
        config["text_config"] = dict(config["text_config"])
        config["text_config"].pop("quantization_config", None)
        config["text_config"]["quantization"] = quant
    dump(output / "config.json", config)
    dump(output / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})
    # Validate index/header agreement through the same public reader.
    if set(scan_checkpoint(output)) != set(tensors):
        raise ValueError("Final checkpoint index verification failed")
    report = {"schema_version": 1, "complete": True, "text_only": True,
              "q4": str(q4_dir.resolve()), "q3": str(q3_dir.resolve()),
              "plan_index": plan_index, "plan": plan, "q3_layers": sorted(chosen),
              "text_stored_bytes": total, "actual_saved_bytes": baseline - total,
              "max_copy_chunk_bytes": CHUNK, "all_tensor_hashes_verified": True,
              "assets": assets, "tensors": manifest}
    dump(output / "mixed-manifest.json", report)
    (output / "BUILD_INCOMPLETE.json").unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("q4", "q3", "plan", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--plan-index", type=int, required=True)
    args = parser.parse_args()
    try:
        report = build(args.q4, args.q3, json.loads(args.plan.read_text()), args.plan_index, args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Mixed build failed; no output overwritten: {exc}\n")
    print(json.dumps({"output": str(args.output), "text_stored_bytes": report["text_stored_bytes"],
                      "tensors_verified": len(report["tensors"])}))


if __name__ == "__main__":
    main()
