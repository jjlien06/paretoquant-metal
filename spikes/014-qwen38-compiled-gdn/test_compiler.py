"""Strict existing 011 tolerances: atol=1e-6, rtol=1e-5, including BF16."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from test_qwen38 import assert_caches, tiny_config


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_compiled_runs_full_state_reset_and_trace_reuse(dtype):
    from compiler import GDNCompiler
    from phased_decode import phased_forward

    mx.set_default_device(mx.gpu)
    mx.random.seed(731)
    config = tiny_config()
    config.update(num_hidden_layers=9, full_attention_interval=4)
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    model.apply(lambda x: x.astype(dtype) if mx.issubdtype(x.dtype, mx.floating) else x)
    model.eval()
    mx.eval(model.parameters())
    compiler = GDNCompiler(model)
    weights = [
        (k, id(v))
        for k, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            model.parameters()
        )
    ]
    for prompt in [[1, 7, 3, 4, 8], [2, 3], [1, 7, 3, 4, 8]]:
        left, right = model.make_cache(), model.make_cache()
        current = prompt
        for step in range(5):
            x = mx.array([current], mx.int32)
            expected = model(x, cache=left)
            mx.eval(expected, [c.state for c in left])
            events = []
            actual = phased_forward(
                model,
                x,
                right,
                eval_group_size=4,
                all_logits=True,
                compiler=compiler,
                progress=events.append,
            )
            assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
            assert_caches(left, right)
            a, b = mx.argmax(expected[:, -1, :], -1), mx.argmax(actual[:, -1, :], -1)
            assert mx.array_equal(a, b).item()
            assert [e["layer_index"] for e in events if e["phase"] == "layer_done"] == list(
                range(9)
            )
            current = [int(a.item())]
        assert compiler.trace_count == 3  # [0:3], [4:7], ragged [8:9]
    assert compiler.compiled_calls == 36
    print(
        {
            "dtype": str(dtype),
            "compiler": compiler.receipt(),
            "peak_mlx_bytes": mx.get_peak_memory(),
        }
    )
    assert weights == [
        (k, id(v))
        for k, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            model.parameters()
        )
    ]
