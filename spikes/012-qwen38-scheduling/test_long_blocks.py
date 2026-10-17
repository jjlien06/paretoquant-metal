"""Exercise real complete and ragged 16/32-layer hybrid GPU blocks."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from phased_decode import phased_forward
from test_qwen38 import assert_caches, tiny_config


@pytest.mark.parametrize("group_size", [16, 32])
def test_long_hybrid_blocks_preserve_logits_and_every_state(group_size):
    mx.set_default_device(mx.gpu)
    mx.random.seed(739)
    config = tiny_config()
    config["num_hidden_layers"] = 35
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    model.apply(lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a)
    model.eval()
    mx.eval(model.parameters())
    native_cache, candidate_cache = model.make_cache(), model.make_cache()
    for ids in ([1, 7, 3], [4, 8], [9], [11]):
        events = []
        inputs = mx.array([ids], mx.int32)
        expected = model(inputs, cache=native_cache)
        mx.eval(expected, [c.state for c in native_cache])
        actual = phased_forward(
            model,
            inputs,
            candidate_cache,
            eval_group_size=group_size,
            all_logits=True,
            progress=events.append,
        )
        assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
        assert mx.array_equal(mx.argmax(expected, -1), mx.argmax(actual, -1)).item()
        assert_caches(native_cache, candidate_cache)
        blocks = [(e["start_idx"], e["end_idx"]) for e in events if e["phase"] == "block_done"]
        assert blocks == [
            (start, min(start + group_size, 35)) for start in range(0, 35, group_size)
        ]
        assert [e["layer_index"] for e in events if e["phase"] == "layer_done"] == list(range(35))
