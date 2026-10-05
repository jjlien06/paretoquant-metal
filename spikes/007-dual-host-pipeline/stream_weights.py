"""Disk-free, fail-closed Qwen2 pipeline weight loading from bounded tensor records."""

import hashlib
import json
import struct
from pathlib import Path

from shard_plan import merge_safetensors_headers, read_safetensors_header, tensor_storage_bytes

MAX_TENSOR_BYTES = 512 * 1024**2


def checkpoint_inventory(source):
    source = Path(source).resolve(strict=True)
    headers, provenance = [], {}
    for path in sorted(source.glob("model*.safetensors")):
        if path.is_symlink():
            raise ValueError("Weight symlinks are not admitted")
        header = read_safetensors_header(path)
        with path.open("rb") as handle:
            base = 8 + struct.unpack("<Q", handle.read(8))[0]
        headers.append(header)
        for name, metadata in header.items():
            if name in provenance:
                raise ValueError("Duplicate tensor")
            if tensor_storage_bytes(metadata) > MAX_TENSOR_BYTES:
                raise ValueError("Tensor exceeds bounded loading buffer")
            provenance[name] = {"path": str(path), "data_start": base}
    return merge_safetensors_headers(headers), provenance


def tensor_records(tensors, provenance):
    for name in sorted(tensors):
        descriptor = tensors[name]
        origin = provenance[name]
        lo, hi = descriptor["data_offsets"]
        with Path(origin["path"]).open("rb") as handle:
            handle.seek(origin["data_start"] + lo)
            payload = handle.read(hi - lo)
        if len(payload) != tensor_storage_bytes(descriptor):
            raise ValueError("Source payload truncated")
        yield {
            "name": name,
            "descriptor": descriptor,
            "payload": payload,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def load_from_tensors(config, group, split, tensors, records, *, budget_bytes, progress=None):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs

    if config.get("model_type") != "qwen2" or config.get("model_file"):
        raise ValueError("Only built-in Qwen2, without remote code, is admitted")
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("A positive parameter budget is required")
    total = sum(tensor_storage_bytes(x) for x in tensors.values())
    if total > budget_bytes:
        raise ValueError(f"Stored parameter bytes {total} exceed budget {budget_bytes}")
    model = Model(ModelArgs.from_dict(config))
    model.model.pipeline(group, split=split)
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Explicit affine quantization config required")

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or path + ".scales" not in tensors:
            return False
        scales = tensors[path + ".scales"]
        weight = tensors[path + ".weight"]
        if scales["dtype"] not in {"F16", "BF16", "F32"}:
            raise ValueError("Only affine quantization storage is admitted")
        dim = module.weight.shape[-1]
        if dim % scales["shape"][-1] or weight["shape"][-1] * 32 % dim:
            raise ValueError("Invalid quantization packing dimensions")
        return {
            "group_size": dim // scales["shape"][-1],
            "bits": weight["shape"][-1] * 32 // dim,
            "mode": "affine",
        }

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        class_predicate=predicate,
    )
    expected = dict(tree_flatten(model.parameters()))
    if set(expected) != set(tensors):
        missing, extra = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
        raise ValueError(f"Incomplete architectural parameters: missing={missing}, extra={extra}")
    for name, initial in expected.items():
        if list(initial.shape) != tensors[name]["shape"]:
            raise ValueError(f"Architectural shape mismatch: {name}")
        if initial.dtype == mx.uint32 and tensors[name]["dtype"] != "U32":
            raise ValueError(f"Packed weight storage must be U32: {name}")
        if mx.issubdtype(initial.dtype, mx.floating) and tensors[name]["dtype"] not in {
            "F16",
            "BF16",
            "F32",
        }:
            raise ValueError(f"Floating parameter storage must be F16, BF16 or F32: {name}")
    del expected
    dtype_map = {
        "F16": np.float16,
        "F32": np.float32,
        "U32": np.uint32,
        "I32": np.int32,
        "BF16": np.uint16,
    }
    emit = progress or (lambda event: None)
    emit({"phase": "tensor_loading_before", "tensor_count": len(tensors), "parameter_bytes": total})
    loaded, hashes = set(), {}
    for record in records:
        name = record["name"]
        if name not in tensors or name in loaded:
            raise ValueError("Unexpected or duplicate streamed tensor")
        if record["descriptor"] != tensors[name]:
            raise ValueError("Stream descriptor differs from admitted metadata")
        emit({"phase": "tensor_before", "tensor_index": len(loaded), "tensor_name": name})
        payload, descriptor = record["payload"], tensors[name]
        if len(payload) != tensor_storage_bytes(descriptor):
            raise ValueError("Stream payload length mismatch")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise ValueError("Tensor digest mismatch")
        if descriptor["dtype"] not in dtype_map:
            raise ValueError("Unsupported runtime storage dtype")
        view = np.frombuffer(payload, dtype=dtype_map[descriptor["dtype"]]).reshape(
            descriptor["shape"]
        )
        array = mx.array(view)
        if descriptor["dtype"] == "BF16":
            array = array.view(mx.bfloat16)
        mx.eval(array)
        model.load_weights([(name, array)], strict=False)
        loaded.add(name)
        hashes[name] = digest
        del payload, view, array, record
        mx.clear_cache()
        emit({"phase": "tensor_done", "tensor_index": len(loaded), "tensor_count": len(tensors)})
    if loaded != set(tensors):
        raise ValueError("Stream ended before all retained tensors were loaded")
    mx.eval(model.parameters())
    actual_bytes = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    if actual_bytes != total:
        raise ValueError("Loaded parameter bytes differ from serialized storage")
    emit(
        {
            "phase": "tensor_loading_done",
            "tensor_count": len(loaded),
            "parameter_bytes": actual_bytes,
        }
    )
    return model, {
        "parameter_bytes": total,
        "tensor_count": len(loaded),
        "all_retained_tensors_loaded": True,
        "tensor_sha256": hashes,
        "loading": "one packed tensor at a time; no checkpoint file on receiver",
    }


def send_json(sock, metadata):
    raw = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > 2 * 1024**2:
        raise ValueError("Metadata frame too large")
    sock.sendall(struct.pack("!I", len(raw)))
    sock.sendall(raw)


def exact(sock, count):
    chunks = bytearray(count)
    view = memoryview(chunks)
    offset = 0
    while offset < count:
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise ValueError("Stream truncated")
        offset += n
    return chunks


def receive_json(sock):
    length = struct.unpack("!I", exact(sock, 4))[0]
    if not 0 < length <= 2 * 1024**2:
        raise ValueError("Invalid metadata frame length")
    metadata = json.loads(exact(sock, length))
    if not isinstance(metadata, dict):
        raise ValueError("Metadata frame must be an object")
    return metadata


def send_records(sock, records):
    for record in records:
        metadata = {k: v for k, v in record.items() if k != "payload"}
        send_json(sock, metadata)
        sock.sendall(record["payload"])
    send_json(sock, {"end": True})


def receive_records(sock, tensors):
    while True:
        metadata = receive_json(sock)
        if metadata == {"end": True}:
            return
        name = metadata.get("name")
        if name not in tensors or metadata.get("descriptor") != tensors[name]:
            raise ValueError("Unexpected network tensor metadata")
        count = tensor_storage_bytes(tensors[name])
        if count > MAX_TENSOR_BYTES:
            raise ValueError("Network tensor exceeds bounded loading buffer")
        yield {**metadata, "payload": exact(sock, count)}
