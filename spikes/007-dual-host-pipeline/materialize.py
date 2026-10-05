"""Stream selected packed tensors into a new per-rank checkpoint, without MLX."""

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

from shard_plan import merge_safetensors_headers, read_safetensors_header, select_pipeline_rank

CHUNK_BYTES = 1024**2


def _check_symlinks(path):
    for candidate in [path, *path.parents]:
        if candidate.is_symlink():
            raise ValueError(f"Symlink paths are not admitted: {candidate}")


def materialize_rank(source, destination, *, split, rank, reserve_bytes=6 * 1024**3):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    _check_symlinks(source)
    _check_symlinks(destination)
    source, destination = source.resolve(strict=True), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source checkpoint")
    if destination.exists():
        raise FileExistsError(destination)
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Nonnegative integer disk reserve required")
    config_bytes = (source / "config.json").read_bytes()
    config = json.loads(config_bytes)
    sources, provenance, headers = {}, {}, []
    for path in sorted(source.glob("model*.safetensors")):
        _check_symlinks(path)
        header = read_safetensors_header(path)
        headers.append(header)
        stat = path.stat()
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            encoded = handle.read(length)
        sources[path.name] = {
            "path": path,
            "data_start": 8 + length,
            "stat": (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
            "header_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        for name in header:
            if name in provenance:
                raise ValueError(f"Duplicate tensor {name}")
            provenance[name] = path.name
    tensors = merge_safetensors_headers(headers)
    plan = select_pipeline_rank(
        tensors,
        model_type=config["model_type"],
        num_hidden_layers=config["num_hidden_layers"],
        split=split,
        rank=rank,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )
    selected = plan["selected_keys"]
    new_header, offset = {}, 0
    for name in selected:
        descriptor = tensors[name]
        count = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        new_header[name] = {
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "data_offsets": [offset, offset + count],
        }
        offset += count
    encoded = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    assets, asset_stats = [], {}
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if any(p.startswith(".") for p in rel.parts):
            continue
        if not path.is_file() or path.suffix not in {
            ".json",
            ".jinja",
            ".txt",
            ".model",
            ".tiktoken",
        }:
            continue
        if path.name in {"model.safetensors.index.json", "manifest.json", "rank_manifest.json"}:
            continue
        _check_symlinks(path)
        assets.append((path, rel))
        stat = path.stat()
        asset_stats[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = offset + len(encoded) + 8 + sum(p.stat().st_size for p, _ in assets)
    free = shutil.disk_usage(destination.parent).free
    if free < required + reserve_bytes:
        raise ValueError(f"Disk preflight failed: {free} free, {required + reserve_bytes} required")
    # Exclusive directory reservation; incomplete failures have no completion manifest.
    destination.mkdir(mode=0o700)
    handles, tensor_hashes, asset_hashes = {}, {}, {}
    try:
        with (destination / "model.safetensors").open("xb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            since_check = 0
            for name in selected:
                descriptor = tensors[name]
                original = sources[provenance[name]]
                if original["path"] not in handles:
                    handles[original["path"]] = original["path"].open("rb")
                stream = handles[original["path"]]
                lo, hi = descriptor["data_offsets"]
                stream.seek(original["data_start"] + lo)
                remaining = hi - lo
                digest = hashlib.sha256()
                while remaining:
                    chunk = stream.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError(f"Truncated source tensor {name}")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                    since_check += len(chunk)
                    if since_check >= 64 * 1024**2:
                        if shutil.disk_usage(destination).free < reserve_bytes:
                            raise ValueError("Disk reserve exhausted while copying")
                        since_check = 0
                tensor_hashes[name] = digest.hexdigest()
            output.flush()
            os.fsync(output.fileno())
        for original in sources.values():
            stat = original["path"].stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != original["stat"]:
                raise ValueError("Source checkpoint changed during materialization")
        if (source / "config.json").read_bytes() != config_bytes:
            raise ValueError("Source config changed during materialization")
        for path, rel in assets:
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with path.open("rb") as src, target.open("xb") as dst:
                while chunk := src.read(CHUNK_BYTES):
                    dst.write(chunk)
                    digest.update(chunk)
                    since_check += len(chunk)
                    if since_check >= 64 * 1024**2:
                        if shutil.disk_usage(destination).free < reserve_bytes:
                            raise ValueError("Disk reserve exhausted while copying")
                        since_check = 0
            asset_hashes[rel] = digest.hexdigest()
        index = {
            "metadata": {"total_size": offset},
            "weight_map": {name: "model.safetensors" for name in selected},
        }
        with (destination / "model.safetensors.index.json").open("x") as handle:
            json.dump(index, handle, indent=2)
        copied = read_safetensors_header(destination / "model.safetensors")
        if copied != new_header:
            raise ValueError("Final safetensors descriptors differ from plan")
        digest = hashlib.sha256()
        with (destination / "model.safetensors").open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
        record = {
            "complete": True,
            "source": str(source),
            "destination": str(destination),
            "rank": rank,
            "split": split,
            "parameter_bytes": offset,
            "layer_start": plan["start_idx"],
            "layer_end": plan["end_idx"],
            "selected_keys": selected,
            "replicated_keys": plan["replicated_keys"],
            "weight_file_sha256": digest.hexdigest(),
            "tensor_sha256": tensor_hashes,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_headers_sha256": {k: v["header_sha256"] for k, v in sources.items()},
            "source_file_by_tensor": {k: provenance[k] for k in selected},
            "reserve_bytes": reserve_bytes,
        }
        if (destination / "config.json").read_bytes() != config_bytes:
            raise ValueError("Copied config changed from admitted configuration")
        for path, rel in assets:
            stat = path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != asset_stats[path]:
                raise ValueError(f"Source asset changed during materialization: {path}")
            digest = hashlib.sha256()
            with (destination / rel).open("rb") as handle:
                while chunk := handle.read(CHUNK_BYTES):
                    digest.update(chunk)
            if digest.hexdigest() != asset_hashes[rel]:
                raise ValueError(f"Copied asset changed during materialization: {rel}")
        if shutil.disk_usage(destination).free < reserve_bytes:
            raise ValueError("Disk reserve exhausted before completion")
        with (destination / "rank_manifest.json").open("x") as handle:
            json.dump(record, handle, indent=2)
        return record
    finally:
        for handle in handles.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--split", type=int, nargs="+", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--reserve-bytes", type=int, default=6 * 1024**3)
    args = parser.parse_args()
    result = materialize_rank(
        args.source,
        args.destination,
        split=args.split,
        rank=args.rank,
        reserve_bytes=args.reserve_bytes,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"tensor_sha256", "selected_keys", "source_file_by_tensor"}
            }
        )
    )


if __name__ == "__main__":
    main()
