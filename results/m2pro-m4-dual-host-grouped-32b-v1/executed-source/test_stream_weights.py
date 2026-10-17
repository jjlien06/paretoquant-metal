"""Native small-model correctness checks for loading one tensor at a time."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen2 import Model, ModelArgs


class SingleGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


@pytest.fixture(params=["F32", "BF16"])
def tiny_checkpoint(tmp_path, request):
    config = {
        "model_type": "qwen2",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "tie_word_embeddings": False,
        "quantization": {"group_size": 32, "bits": 4},
    }
    mx.random.seed(0)
    original = Model(ModelArgs.from_dict(config))
    nn.quantize(original, group_size=32, bits=4)
    if request.param == "BF16":
        original.apply(
            lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x
        )
    mx.eval(original.parameters())
    path = tmp_path / "tiny"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(original.parameters())))
    return path, config, original


def test_streamed_loader_preserves_real_quantized_logits(tiny_checkpoint):
    assert (Path(__file__).parent / "stream_weights.py").exists(), "stream loader not implemented"
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, original = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    loaded, record = load_from_tensors(
        config,
        SingleGroup(),
        [2],
        tensors,
        tensor_records(tensors, provenance),
        budget_bytes=1024**2,
    )
    x = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected, actual = original(x), loaded(x)
    mx.eval(expected, actual)
    assert mx.max(mx.abs(expected - actual)).item() < 1e-6
    assert record["parameter_bytes"] == sum(
        a.nbytes for _, a in tree_flatten(original.parameters())
    )
    assert record["tensor_count"] == len(tensors)
    assert record["all_retained_tensors_loaded"] is True


def test_streamed_loader_rejects_missing_architectural_weight(tiny_checkpoint):
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    tensors.pop("model.layers.0.self_attn.q_proj.bias")
    with pytest.raises(ValueError, match="architectural"):
        load_from_tensors(
            config,
            SingleGroup(),
            [2],
            tensors,
            tensor_records(tensors, provenance),
            budget_bytes=1024**2,
        )


@pytest.mark.parametrize("tiny_checkpoint", ["F32"], indirect=True)
@pytest.mark.parametrize("storage_dtype", ["F32", "I32"])
def test_streamed_loader_rejects_non_u32_packed_words(tiny_checkpoint, storage_dtype):
    from shard_plan import tensor_storage_bytes
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    name = "model.layers.0.self_attn.q_proj.weight"
    original = tensors[name]
    assert original["dtype"] == "U32"
    tensors[name] = {**original, "dtype": storage_dtype}
    assert tensors[name]["shape"] == original["shape"]
    assert tensor_storage_bytes(tensors[name]) == tensor_storage_bytes(original)
    with pytest.raises(ValueError, match="Packed weight storage must be U32"):
        load_from_tensors(
            config,
            SingleGroup(),
            [2],
            tensors,
            tensor_records(tensors, provenance),
            budget_bytes=1024**2,
        )


@pytest.mark.parametrize("tiny_checkpoint", ["F32"], indirect=True)
@pytest.mark.parametrize("storage_dtype", ["I32", "U32"])
def test_streamed_loader_rejects_integer_affine_biases(tiny_checkpoint, storage_dtype):
    from shard_plan import tensor_storage_bytes
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    name = "model.layers.0.self_attn.q_proj.biases"
    original = tensors[name]
    tensors[name] = {**original, "dtype": storage_dtype}
    assert tensors[name]["shape"] == original["shape"]
    assert tensor_storage_bytes(tensors[name]) == tensor_storage_bytes(original)
    with pytest.raises(ValueError, match="Floating parameter storage must be F16, BF16 or F32"):
        load_from_tensors(
            config,
            SingleGroup(),
            [2],
            tensors,
            tensor_records(tensors, provenance),
            budget_bytes=1024**2,
        )


@pytest.mark.parametrize("tiny_checkpoint", ["F32"], indirect=True)
@pytest.mark.parametrize("storage_dtype", ["I32", "U32"])
@pytest.mark.parametrize(
    "name",
    ["model.layers.0.self_attn.q_proj.bias", "model.layers.0.input_layernorm.weight"],
)
def test_streamed_loader_rejects_integer_floating_parameters(tiny_checkpoint, storage_dtype, name):
    from shard_plan import tensor_storage_bytes
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    original = tensors[name]
    tensors[name] = {**original, "dtype": storage_dtype}
    assert tensors[name]["shape"] == original["shape"]
    assert tensor_storage_bytes(tensors[name]) == tensor_storage_bytes(original)
    with pytest.raises(ValueError, match="Floating parameter storage must be F16, BF16 or F32"):
        load_from_tensors(
            config,
            SingleGroup(),
            [2],
            tensors,
            tensor_records(tensors, provenance),
            budget_bytes=1024**2,
        )


@pytest.mark.parametrize("tiny_checkpoint", ["F32"], indirect=True)
@pytest.mark.parametrize("storage_dtype", ["I32", "U32"])
def test_streamed_loader_rejects_integer_affine_scales(tiny_checkpoint, storage_dtype):
    from shard_plan import tensor_storage_bytes
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    name = "model.layers.0.self_attn.q_proj.scales"
    original = tensors[name]
    tensors[name] = {**original, "dtype": storage_dtype}
    assert tensors[name]["shape"] == original["shape"]
    assert tensor_storage_bytes(tensors[name]) == tensor_storage_bytes(original)
    with pytest.raises(ValueError, match="Only affine quantization storage is admitted"):
        load_from_tensors(
            config,
            SingleGroup(),
            [2],
            tensors,
            tensor_records(tensors, provenance),
            budget_bytes=1024**2,
        )


@pytest.mark.parametrize("tiny_checkpoint", ["F32"], indirect=True)
@pytest.mark.parametrize(
    "storage_dtype", [mx.float16, mx.bfloat16, mx.float32], ids=["F16", "BF16", "F32"]
)
def test_streamed_loader_preserves_compatible_floating_storage(tiny_checkpoint, storage_dtype):
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, original = tiny_checkpoint
    parameters = {
        name: array.astype(storage_dtype) if mx.issubdtype(array.dtype, mx.floating) else array
        for name, array in tree_flatten(original.parameters())
    }
    mx.save_safetensors(str(path / "model.safetensors"), parameters)
    tensors, provenance = checkpoint_inventory(path)
    loaded, record = load_from_tensors(
        config,
        SingleGroup(),
        [2],
        tensors,
        tensor_records(tensors, provenance),
        budget_bytes=1024**2,
    )
    actual = dict(tree_flatten(loaded.parameters()))
    assert set(actual) == set(parameters) == set(tensors)
    for name, expected in parameters.items():
        assert actual[name].dtype == expected.dtype
        assert actual[name].shape == expected.shape
        if expected.dtype == mx.uint32:
            assert mx.array_equal(actual[name], expected).item()
        else:
            assert mx.array_equal(
                actual[name].astype(mx.float32), expected.astype(mx.float32)
            ).item()
    assert record["parameter_bytes"] == sum(array.nbytes for array in parameters.values())
    assert record["tensor_count"] == len(parameters)
    assert record["all_retained_tensors_loaded"] is True


def test_streamed_loader_rejects_payload_corruption(tiny_checkpoint):
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = tiny_checkpoint
    tensors, provenance = checkpoint_inventory(path)
    records = list(tensor_records(tensors, provenance))
    first = records[0]
    records[0] = {**first, "payload": b"\x00" * len(first["payload"])}
    with pytest.raises(ValueError, match="digest"):
        load_from_tensors(config, SingleGroup(), [2], tensors, iter(records), budget_bytes=1024**2)
