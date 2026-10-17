import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs


class Group:
    def __init__(self, rank=0, size=1):
        self.r, self.n = rank, size

    def rank(self):
        return self.r

    def size(self):
        return self.n


def tiny_config():
    return dict(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        rope_parameters=dict(type="default", partial_rotary_factor=0.5, rope_theta=10000),
        quantization=dict(group_size=32, bits=4),
    )


@pytest.fixture(params=["F32", "BF16"])
def checkpoint(tmp_path, request):
    mx.set_default_device(mx.gpu)
    mx.random.seed(731)
    config = tiny_config()
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    if request.param == "BF16":
        model.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    model.eval()
    mx.eval(model.parameters())
    path = tmp_path / "tiny"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), params)
    return path, config, model


def test_streamed_hybrid_exact_serialization_and_text_plan(checkpoint):
    from shard_plan import select_pipeline_rank
    from stream_weights import (
        MAX_TENSOR_BYTES,
        checkpoint_inventory,
        load_from_tensors,
        tensor_records,
    )

    path, config, native = checkpoint
    tensors, provenance = checkpoint_inventory(path)
    augmented = {
        **tensors,
        "vision_tower.fake.weight": dict(dtype="F32", shape=[1], data_offsets=[0, 4]),
    }
    plans = [
        select_pipeline_rank(
            augmented, model_type="qwen3_5", num_hidden_layers=5, split=[2, 3], rank=r
        )
        for r in (0, 1)
    ]
    assert plans[0]["layer_indices"] == [3, 4]
    assert plans[1]["layer_indices"] == [0, 1, 2]
    assert not any("vision" in k for p in plans for k in p["selected_keys"])
    assert MAX_TENSOR_BYTES == 768 * 1024**2
    loaded, receipt = load_from_tensors(
        config, Group(), [5], tensors, tensor_records(tensors, provenance), budget_bytes=8 * 1024**2
    )
    expected, actual = (
        dict(tree_flatten(native.parameters())),
        dict(tree_flatten(loaded.parameters())),
    )
    assert expected.keys() == actual.keys() == tensors.keys()
    for name, a in expected.items():
        b = actual[name]
        assert a.shape == b.shape and a.dtype == b.dtype
        assert mx.array_equal(a, b).item(), name
    for record in tensor_records(tensors, provenance):
        assert (
            receipt["tensor_sha256"][record["name"]]
            == hashlib.sha256(record["payload"]).hexdigest()
        )
    conv = [v for k, v in tensors.items() if k.endswith("conv1d.weight")]
    assert conv and all(v["shape"][-2:] == [4, 1] for v in conv)
    for rank in (0, 1):
        selected = {k: tensors[k] for k in plans[rank]["selected_keys"]}
        shard, _ = load_from_tensors(
            config,
            Group(rank, 2),
            [2, 3],
            selected,
            tensor_records(selected, provenance),
            budget_bytes=8 * 1024**2,
        )
        assert shard.model.start_idx == plans[rank]["start_idx"]
        assert all(x is None for x in shard.model.layers[: shard.model.start_idx])
        assert set(dict(tree_flatten(shard.parameters()))) == set(selected)


def assert_caches(left, right):
    from mlx_lm.models.cache import ArraysCache, KVCache

    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert type(a) is type(b)
        if isinstance(a, KVCache):
            assert a.offset == b.offset
            pairs = [
                (a.keys[..., : a.offset, :], b.keys[..., : b.offset, :]),
                (a.values[..., : a.offset, :], b.values[..., : b.offset, :]),
            ]
        else:
            assert isinstance(a, ArraysCache)
            assert a.left_padding is b.left_padding is None
            assert a.lengths is b.lengths is None
            pairs = list(zip(a.cache, b.cache))
        for x, y in pairs:
            assert x.shape == y.shape and x.dtype == y.dtype
            assert mx.allclose(x, y, atol=1e-6, rtol=1e-5).item()


@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_gpu_phased_hybrid_chunked_prefill_decode_full_state(checkpoint, group_size):
    from phased_decode import phased_forward, phased_generate

    path, config, model = checkpoint
    native_cache, phased_cache = model.make_cache(), model.make_cache()
    events = []
    for ids in [[1, 7, 3], [4, 8], [9], [11], [12]]:
        x = mx.array([ids], mx.int32)
        expected = model(x, cache=native_cache)
        mx.eval(expected, [c.state for c in native_cache])
        actual = phased_forward(
            model,
            x,
            phased_cache,
            eval_group_size=group_size,
            all_logits=True,
            progress=events.append,
        )
        assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
        assert mx.array_equal(mx.argmax(expected, -1), mx.argmax(actual, -1)).item()
        assert_caches(native_cache, phased_cache)
    assert {e["layer_index"] for e in events if e["phase"] == "layer_done"} == set(range(5))

    class Tokenizer:
        eos_token_ids = []

        def decode(self, ids, **kwargs):
            return str(ids)

    prompt = [1, 7, 3, 4, 8, 9]
    cache = model.make_cache()
    for ids in [prompt[:2], prompt[2:4], prompt[4:5]]:
        mx.eval(model(mx.array([ids], mx.int32), cache=cache), [c.state for c in cache])
    current = mx.array([[prompt[-1]]], mx.int32)
    tokens = []
    for _ in range(4):
        scores = model(current, cache=cache)
        mx.eval(scores, [c.state for c in cache])
        token = int(mx.argmax(scores[:, -1, :], -1).item())
        tokens.append(token)
        current = mx.array([[token]], mx.int32)
    receipt = phased_generate(
        model, Tokenizer(), prompt, max_tokens=4, prefill_step_size=2, eval_group_size=group_size
    )
    assert receipt["token_ids"] == tokens


def test_supervisor_binds_canonical_text_inventory_and_sources(checkpoint):
    from remote_actor import SOURCES, source_hashes
    from shard_plan import select_pipeline_rank
    from stream_weights import checkpoint_inventory
    from supervise import expected_inventory, parser, ssh_argv

    path, config, _ = checkpoint
    tensors, _ = checkpoint_inventory(path)
    names = set(tensors) | {"vision_tower.foo.weight"}
    actual = expected_inventory(config, names, [2, 3])
    for r in (0, 1):
        plan = select_pipeline_rank(
            tensors, model_type="qwen3_5", num_hidden_layers=5, split=[2, 3], rank=r
        )
        assert actual[r] == set(plan["selected_keys"])
    nested = dict(model_type="qwen3_5", text_config=config)
    assert expected_inventory(nested, names, [2, 3]) == actual
    with pytest.raises(ValueError):
        expected_inventory(config, names | {"unknown.weight"}, [2, 3])
    root = Path(__file__).parent
    assert set(source_hashes(root)) == set(SOURCES)
    assert {
        "probe.py",
        "stream_weights.py",
        "stream_server.py",
        "shard_plan.py",
        "phased_decode.py",
        "cpu_communication.py",
        "remote_actor.py",
    } <= set(SOURCES)
    args = parser().parse_args(
        [
            "--remote-root",
            "/a/b/spikes/011-qwen38-dual-host",
            "--remote-source",
            "/model",
            "--local-metadata",
            "/metadata",
            "--output",
            "/out",
            "--ssh-key",
            "/key",
            "--split",
            "2",
            "3",
            "--ssh-host-key-alias",
            "old-trusted-host",
        ]
    )
    assert "HostKeyAlias=old-trusted-host" in ssh_argv(args, [])


def test_probe_formats_non_thinking_and_raw_tokens():
    import probe

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return "formatted"

    assert probe.prepare_prompt(Tokenizer(), "hi") == "formatted"
    assert probe.prepare_prompt(Tokenizer(), "tokens:[1,2,3]") == [1, 2, 3]


@pytest.mark.parametrize("split", [[2, 3], [1, 4], [4, 1]])
def test_gpu_ragged_shards_all_logits_and_states(checkpoint, monkeypatch, split):
    from phased_decode import phased_forward
    from shard_plan import select_pipeline_rank
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, native = checkpoint
    tensors, provenance = checkpoint_inventory(path)
    shards = []
    for r in (0, 1):
        plan = select_pipeline_rank(
            tensors, model_type="qwen3_5", num_hidden_layers=5, split=split, rank=r
        )
        selected = {k: tensors[k] for k in plan["selected_keys"]}
        model, _ = load_from_tensors(
            config,
            Group(r, 2),
            split,
            selected,
            tensor_records(selected, provenance),
            budget_bytes=8 * 1024**2,
        )
        shards.append(model)
    slot = []

    def send(array, rank, **kwargs):
        assert rank == 0
        slot.append(array)
        return array

    def recv(shape, dtype, rank, **kwargs):
        assert rank == 1 and len(slot) == 1
        array = slot.pop()
        assert array.shape == shape and array.dtype == dtype
        return array

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv", recv)
    caches = [s.make_cache() for s in shards]
    ref_cache = native.make_cache()
    for ids in [[1, 2, 3], [4, 5], [6], [7]]:
        x = mx.array([ids], mx.int32)
        expected = native(x, cache=ref_cache)
        mx.eval(expected, [c.state for c in ref_cache])
        assert phased_forward(shards[1], x, caches[1], group=Group(1, 2), eval_group_size=4) is None
        actual = phased_forward(
            shards[0], x, caches[0], group=Group(0, 2), eval_group_size=4, all_logits=True
        )
        assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
        assert mx.array_equal(mx.argmax(expected, -1), mx.argmax(actual, -1)).item()
        assert_caches(ref_cache, caches[1] + caches[0])


def test_fixture_creation_cli(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "fixture"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fixture.py")), "--output", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((path / "config.json").read_text())
    assert config["model_type"] == "qwen3_5"
    from stream_weights import checkpoint_inventory

    tensors, _ = checkpoint_inventory(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert set(tensors) == set(index["weight_map"])
    assert not any("vision" in k for k in tensors)


@pytest.mark.parametrize("bad", ["missing", "dtype", "conv", "digest", "vision", "overcap"])
def test_fail_closed_stream_admission(checkpoint, bad):
    import copy

    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

    path, config, _ = checkpoint
    tensors, origins = checkpoint_inventory(path)
    tensors = copy.deepcopy(tensors)
    records = tensor_records(tensors, origins)
    if bad == "missing":
        tensors.pop("language_model.model.layers.0.linear_attn.dt_bias")
    if bad == "dtype":
        tensors["language_model.lm_head.weight"]["dtype"] = "I32"
    if bad == "conv":
        value = tensors["language_model.model.layers.0.linear_attn.conv1d.weight"]
        value["shape"][-2:] = [1, 4]
    if bad == "digest":
        rows = list(records)
        rows[0]["sha256"] = "0" * 64
        records = iter(rows)
    if bad == "vision":
        tensors["vision_tower.fake.weight"] = dict(dtype="F32", shape=[1], data_offsets=[0, 4])
    if bad == "overcap":
        tensors["language_model.lm_head.weight"] = dict(
            dtype="U32", shape=[201326593], data_offsets=[0, 805306372]
        )
    with pytest.raises(ValueError):
        load_from_tensors(config, Group(), [5], tensors, records, budget_bytes=2 * 1024**3)
