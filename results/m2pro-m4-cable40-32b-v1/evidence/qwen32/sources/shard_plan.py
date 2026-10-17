"""Stdlib-only safetensors storage and Qwen2 pipeline planning.

This spike follows the supplied mlx-lm 0.32.0 PipelineMixin convention:
start_idx = sum(split[rank + 1:]); rank 0 keeps the last layers. All ranks
retain embedding, final norm and any untied output-head tensors, including
quantization scales/biases. Inputs use serialized safetensors storage dtypes,
not logical quantization bits; U32 packed weights are already packed.

The plan is a retained-parameter lower bound, not a safe peak-RAM budget.
It does not load/evaluate tensors, copy shards, contact hosts, or import MLX.
File-local offsets in merged maps cannot validate the original files: parse
individual headers against independently known payload sizes first. Layer
coverage checks require at least one tensor per layer, not full architectural
parameter completeness. Unknown model types/dtypes/naming schemes fail closed.
"""

import json
import math
import os
import re
import struct
from collections.abc import Mapping

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def tensor_storage_bytes(metadata):
    """Validate one tensor descriptor and return its packed storage byte count."""
    if not isinstance(metadata, dict):
        raise ValueError("tensor metadata must be an object")
    dtype = metadata.get("dtype")
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported storage dtype: {dtype!r}")
    shape = metadata.get("shape")
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("shape must contain nonnegative integers")
    offsets = metadata.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(n) is not int or n < 0 for n in offsets)
    ):
        raise ValueError("data_offsets must contain two nonnegative integers")
    size = math.prod(shape) * _DTYPE_BYTES[dtype]
    if offsets[1] - offsets[0] != size:
        raise ValueError("dtype/shape byte count disagrees with data_offsets")
    return size


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def parse_safetensors_header(header_bytes, *, data_size):
    """Validate UTF-8 header JSON against the known file payload size.

    Offsets are relative to the payload, not the file. The tensor ranges must
    cover the complete payload without overlaps or holes. __metadata__ is
    validated, then omitted from the returned weight-name -> descriptor map.
    """
    if type(data_size) is not int or data_size < 0:
        raise ValueError("data_size must be a nonnegative integer")
    if not isinstance(header_bytes, bytes) or not header_bytes.startswith(b"{"):
        raise ValueError("header must be UTF-8 JSON bytes starting with '{'")
    try:
        header = json.loads(
            header_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid header JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("header must be a JSON object")
    if "__metadata__" in header:
        metadata = header.pop("__metadata__")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(not isinstance(value, str) for value in metadata.values())
        ):
            raise ValueError("__metadata__ must be null or map strings to strings")
    intervals = []
    for metadata in header.values():
        tensor_storage_bytes(metadata)
        intervals.append(tuple(metadata["data_offsets"]))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor or end > data_size:
            raise ValueError("payload ranges overlap, have gaps, or extend beyond file")
        cursor = end
    if cursor != data_size:
        raise ValueError("payload has unindexed trailing bytes")
    return header


def read_safetensors_header(source, *, max_header_bytes=100_000_000):
    """Read a local path or seekable binary stream without reading its payload.

    Paths are opened read-only and closed; borrowed streams stay open and their
    original positions are restored, even if validation fails.
    """
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if isinstance(source, (str, bytes, os.PathLike)):
        with open(source, "rb") as stream:
            return read_safetensors_header(stream, max_header_bytes=max_header_bytes)
    position = source.tell()
    try:
        source.seek(0, 2)
        file_size = source.tell()
        source.seek(0)
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= max_header_bytes or length > file_size - 8:
            raise ValueError("invalid, oversized, or truncated safetensors header length")
        header_bytes = source.read(length)
        if len(header_bytes) != length:
            raise ValueError("truncated safetensors header")
        return parse_safetensors_header(header_bytes, data_size=file_size - 8 - length)
    finally:
        source.seek(position)


def merge_safetensors_headers(headers):
    """Merge parsed weight-name -> metadata maps, rejecting duplicate tensor names.

    Each descriptor is revalidated. Offsets stay relative to the original file;
    cross-file offsets may overlap. Call parse_safetensors_header on each original
    header with its known payload size first for complete file-layout validation.
    This helper expects maps without the reserved __metadata__ entry.
    """
    merged = {}
    for header in headers:
        if not isinstance(header, Mapping):
            raise ValueError("each header must be a tensor metadata mapping")
        for key, metadata in header.items():
            if not isinstance(key, str) or key == "__metadata__":
                raise ValueError("expected string tensor names, without __metadata__")
            if key in merged:
                raise ValueError(f"duplicate tensor across headers: {key}")
            tensor_storage_bytes(metadata)
            merged[key] = metadata
    return merged


def _validate_configuration(model_type, num_hidden_layers, split, tie_word_embeddings):
    if model_type != "qwen2":
        raise ValueError("only model_type='qwen2' is supported")
    if type(num_hidden_layers) is not int or num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")
    if (
        not isinstance(split, (list, tuple))
        or not split
        or any(type(n) is not int or n <= 0 for n in split)
    ):
        raise ValueError("split must be a nonempty list/tuple of positive integers")
    if sum(split) != num_hidden_layers:
        raise ValueError("split must sum to num_hidden_layers")
    if type(tie_word_embeddings) is not bool:
        raise ValueError("tie_word_embeddings must be a boolean")


def _classify_tensors(tensors, num_hidden_layers, tie_word_embeddings):
    if not isinstance(tensors, Mapping):
        raise ValueError("tensors must be a weight-name -> metadata mapping")
    layers = {index: [] for index in range(num_hidden_layers)}
    replicated = []
    prefixes = ("model.embed_tokens.", "model.norm.", "lm_head.")
    for key, metadata in tensors.items():
        if not isinstance(key, str):
            raise ValueError("tensor keys must be strings")
        tensor_storage_bytes(metadata)
        match = re.fullmatch(r"model\.layers\.(0|[1-9][0-9]*)\.(.+)", key)
        if match:
            index = int(match[1])
            if index not in layers:
                raise ValueError(f"layer outside configured range: {key}")
            layers[index].append(key)
        elif any(key.startswith(prefix) and len(key) > len(prefix) for prefix in prefixes):
            if tie_word_embeddings and key.startswith("lm_head."):
                raise ValueError("tied embeddings must not contain a separate lm_head")
            replicated.append(key)
        else:
            raise ValueError(f"unrecognized Qwen2 parameter name: {key!r}")
    missing = [index for index, keys in layers.items() if not keys]
    if missing:
        raise ValueError(f"incomplete layer coverage; missing layers: {missing}")
    required = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tie_word_embeddings:
        required.append("lm_head.weight")
    for key in required:
        if key not in tensors:
            raise ValueError(f"missing required replicated parameter: {key}")
    return layers, sorted(replicated)


def select_pipeline_rank(
    tensors, *, model_type, num_hidden_layers, split, rank, tie_word_embeddings=False
):
    """Select retained parameter keys/packed bytes with rank 0 owning last layers."""
    _validate_configuration(model_type, num_hidden_layers, split, tie_word_embeddings)
    if type(rank) is not int or not 0 <= rank < len(split):
        raise ValueError("rank must be an integer within split")
    start = sum(split[rank + 1 :])
    end = start + split[rank]
    all_layers, replicated = _classify_tensors(tensors, num_hidden_layers, tie_word_embeddings)
    layers = sorted(key for index in range(start, end) for key in all_layers[index])
    replicated_bytes = sum(tensor_storage_bytes(tensors[key]) for key in replicated)
    layer_bytes = sum(tensor_storage_bytes(tensors[key]) for key in layers)
    return {
        "rank": rank,
        "start_idx": start,
        "end_idx": end,
        "layer_indices": list(range(start, end)),
        "layer_keys": layers,
        "replicated_keys": replicated,
        "selected_keys": sorted(replicated + layers),
        "replicated_bytes": replicated_bytes,
        "layer_bytes": layer_bytes,
        "selected_bytes": replicated_bytes + layer_bytes,
    }


def plan_qwen2_pipeline(
    tensors, *, model_type, num_hidden_layers, split, tie_word_embeddings=False
):
    """Return a JSON-serializable retained-parameter plan, not a peak-memory estimate."""
    _validate_configuration(model_type, num_hidden_layers, split, tie_word_embeddings)
    ranks = [
        select_pipeline_rank(
            tensors,
            model_type=model_type,
            num_hidden_layers=num_hidden_layers,
            split=split,
            rank=rank,
            tie_word_embeddings=tie_word_embeddings,
        )
        for rank in range(len(split))
    ]
    coverage = [layer for rank in ranks for layer in rank["layer_indices"]]
    disjoint = len(coverage) == len(set(coverage))
    exhaustive = sorted(coverage) == list(range(num_hidden_layers))
    if not disjoint or not exhaustive:
        raise ValueError("rank layer coverage must be disjoint and exhaustive")
    layer_owners = [None] * num_hidden_layers
    for rank in ranks:
        for layer in rank["layer_indices"]:
            layer_owners[layer] = rank["rank"]
    replicated = ranks[0]["replicated_keys"]
    replicated_bytes = ranks[0]["replicated_bytes"]
    return {
        "model_type": model_type,
        "num_hidden_layers": num_hidden_layers,
        "world_size": len(split),
        "split": list(split),
        "tie_word_embeddings": tie_word_embeddings,
        "ranks": ranks,
        "coverage": {
            "disjoint": disjoint,
            "exhaustive": exhaustive,
            "layer_owner_by_index": layer_owners,
        },
        "replicated_keys": replicated,
        "replicated_bytes_per_rank": replicated_bytes,
        "replicated_parameters": {
            "embedding": [key for key in replicated if key.startswith("model.embed_tokens.")],
            "norm": [key for key in replicated if key.startswith("model.norm.")],
            "untied_head": [key for key in replicated if key.startswith("lm_head.")],
        },
        "tensor_count": len(tensors),
        "unique_tensor_bytes": sum(tensor_storage_bytes(meta) for meta in tensors.values()),
        "aggregate_selected_bytes": sum(rank["selected_bytes"] for rank in ranks),
        "replication_overhead_bytes": (len(split) - 1) * replicated_bytes,
        "storage_note": (
            "Bytes use actual storage dtype/shape: U32 packed quantized words occupy "
            "4 bytes each, even with 4-bit logical weights; do not divide by bits again. "
            "Scales and biases are counted separately. Excludes file headers, runtime "
            "objects, KV cache, activations, allocator overhead and load-time copies. "
            "Selected tensor bytes are not whole-file download/storage or peak RAM."
        ),
    }
