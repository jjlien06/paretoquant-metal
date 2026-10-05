"""Native GPU admission: real packed 4-bit Qwen logits and live KV state."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model, ModelArgs
from phased_decode import phased_forward


def test_cpu_parity_fixture_restores_default_device():
    import test_phased_decode

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        test_phased_decode.test_prefix_without_head_advances_cache()
        assert mx.default_device() == mx.gpu
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("tied", [False, True])
def test_real_quantized_gpu_prefill_and_cached_decode_parity(tied):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(49)
        model = Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=64,
                num_hidden_layers=3,
                intermediate_size=128,
                num_attention_heads=4,
                rms_norm_eps=1e-6,
                vocab_size=128,
                num_key_value_heads=2,
                tie_word_embeddings=tied,
            )
        )
        model.set_dtype(mx.float16)
        nn.quantize(model, group_size=32, bits=4)
        mx.eval(model.parameters())
        assert model.model.embed_tokens.weight.dtype == mx.uint32
        assert model.model.embed_tokens.scales.dtype == mx.float16
        assert mx.default_device() == mx.gpu
        native_cache = make_prompt_cache(model)
        explicit_cache = make_prompt_cache(model)
        phases = []
        for ids in [[[1, 2, 3]], [[4]], [[5, 6]], [[7]]]:
            inputs = mx.array(ids, dtype=mx.int32)
            expected = model(inputs, cache=native_cache)[:, -1:, :]
            actual = phased_forward(model, inputs, explicit_cache, progress=phases.append)
            mx.eval(expected, actual)
            np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0.002, atol=0.002)
            assert mx.argmax(expected).item() == mx.argmax(actual).item()
            for left, right in zip(native_cache, explicit_cache):
                assert left.offset == right.offset
                for a, b in zip(left.state, right.state):
                    assert mx.array_equal(a, b).item()
        assert [p["layer_index"] for p in phases if p["phase"] == "layer_done"] == [
            0,
            1,
            2,
        ] * 4
    finally:
        mx.synchronize()
        mx.set_default_device(previous)
