"""Compiled clipped GDN runs in retained original-layer intervals."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from test_qwen38 import Group, assert_caches, tiny_config


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("cut", [1, 2, 5, 7])
def test_clipped_gdn_shards(dtype, cut, monkeypatch):
    import copy

    from compiler import GDNCompiler
    from phased_decode import phased_forward

    config = tiny_config()
    config.update(num_hidden_layers=9, full_attention_interval=4)
    mx.random.seed(731)
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    model.apply(lambda x: x.astype(dtype) if mx.issubdtype(x.dtype, mx.floating) else x)
    model.eval()
    mx.eval(model.parameters())
    # Preserve real native cache/type factories and layer ownership; no model pipeline surgery.
    shards = [copy.deepcopy(model), copy.deepcopy(model)]
    for rank, shard in enumerate(shards):
        shard.model.pipeline(Group(rank, 2), split=[9 - cut, cut])
    compilers = [GDNCompiler(s) for s in shards]
    caches = [s.make_cache() for s in shards]
    native_cache = model.make_cache()
    slot = []

    def send(array, rank, **kwargs):
        slot.append(array)
        return array

    def recv(shape, dtype, rank, **kwargs):
        array = slot.pop()
        assert array.shape == shape and array.dtype == dtype
        return array

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv", recv)
    for ids in [[1, 2, 3], [4, 5], [6], [7], [8]]:
        x = mx.array([ids], mx.int32)
        expected = model(x, cache=native_cache)
        mx.eval(expected, [c.state for c in native_cache])
        events = [[], []]
        assert (
            phased_forward(
                shards[1],
                x,
                caches[1],
                compiler=compilers[1],
                group=Group(1, 2),
                eval_group_size=4,
                progress=events[1].append,
            )
            is None
        )
        actual = phased_forward(
            shards[0],
            x,
            caches[0],
            compiler=compilers[0],
            group=Group(0, 2),
            eval_group_size=4,
            all_logits=True,
            progress=events[0].append,
        )
        assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
        assert mx.array_equal(mx.argmax(expected, -1), mx.argmax(actual, -1)).item()
        assert_caches(native_cache, caches[1] + caches[0])
        for rank, (a, b) in enumerate([(cut, 9), (0, cut)]):
            assert [e["layer_index"] for e in events[rank] if e["phase"] == "layer_done"] == list(
                range(a, b)
            )
    assert all(c.trace_count > 0 and c.compiled_calls > 0 for c in compilers)
