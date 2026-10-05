"""Explicit layer scheduling must preserve native Qwen logits and cache state."""

from functools import wraps

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model, ModelArgs


def preserve_device(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = mx.default_device()
        try:
            return function(*args, **kwargs)
        finally:
            mx.set_default_device(previous)

    return wrapped


@preserve_device
def test_phased_forward_matches_native_prefill_and_cached_logits():
    from phased_decode import phased_forward

    mx.set_default_device(mx.cpu)
    mx.random.seed(4)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    native_cache, phased_cache = make_prompt_cache(model), make_prompt_cache(model)
    phases = []
    for ids in [[[1, 2, 3]], [[4]], [[5, 6]]]:
        inputs = mx.array(ids)
        expected = model(inputs, cache=native_cache)[:, -1:, :]
        actual = phased_forward(model, inputs, phased_cache, progress=phases.append)
        mx.eval(expected, actual)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
        for left, right in zip(native_cache, phased_cache):
            assert left.offset == right.offset
            for a, b in zip(left.state, right.state):
                assert mx.array_equal(a, b).item()
    assert [p["layer_index"] for p in phases if p["phase"] == "layer_done"] == [0, 1] * 3


@preserve_device
def test_prefix_without_head_advances_cache():
    mx.set_default_device(mx.cpu)
    from phased_decode import phased_forward

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    cache = make_prompt_cache(model)
    assert phased_forward(model, mx.array([[1, 2]]), cache, logits=False) is None
    assert all(c.offset == 2 for c in cache)
    expected = model(mx.array([[1, 2, 3]]))[:, -1:, :]
    actual = phased_forward(model, mx.array([[3]]), cache)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)


@preserve_device
def test_greedy_generation_includes_eos_and_fresh_cache():
    mx.set_default_device(mx.cpu)
    from phased_decode import phased_generate

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)

    class Tokenizer:
        bos_token = None
        eos_token_ids = {0}

        def encode(self, prompt, add_special_tokens=True):
            return [1, 2, 3]

        def decode(self, tokens, skip_special_tokens=False):
            assert skip_special_tokens
            return "" if tokens == [0] else "unexpected"

    events = []
    first = phased_generate(
        model, Tokenizer(), "fixture", max_tokens=4, prefill_step_size=1, progress=events.append
    )
    second = phased_generate(model, Tokenizer(), "fixture", max_tokens=4)
    assert first["token_ids"] == second["token_ids"] == [0]
    assert first["finish_reason"] == "stop"
    assert first["native_generation_tps"] is None
    assert first["prompt_tokens"] == 3
    assert first["wall_seconds"] > 0
    assert [e["token_id"] for e in events if e["phase"] == "token_done"] == [0]


@preserve_device
def test_malformed_cache_and_group_are_rejected():
    mx.set_default_device(mx.cpu)
    import pytest
    from phased_decode import phased_forward

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    with pytest.raises(ValueError, match="Cache"):
        phased_forward(model, mx.array([[1]]), [])
    with pytest.raises(ValueError, match="inputs"):
        phased_forward(model, mx.array([1]), make_prompt_cache(model))
    cache = make_prompt_cache(model)
    cache[0].offset = 1
    with pytest.raises(ValueError, match="offset"):
        phased_forward(model, mx.array([[1]]), cache)


@preserve_device
def test_optional_parent_cpu_float16_parity():
    import os

    import pytest
    from phased_decode import phased_forward

    if os.environ.get("PARETOQUANT_PARENT_F16") != "1":
        pytest.skip("Parent opt-in CPU F16 gate; never run by the sandbox worker")
    mx.set_default_device(mx.cpu)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    model.set_dtype(mx.float16)
    native_cache, explicit_cache = make_prompt_cache(model), make_prompt_cache(model)
    for ids in ([[1, 2, 3]], [[4]]):
        inputs = mx.array(ids)
        try:
            expected = model(inputs, cache=native_cache)[:, -1:, :]
            mx.eval(expected)
        except RuntimeError as exc:
            if "float16" in str(exc) and "CPU" in str(exc):
                pytest.skip("CPU F16 unsupported; parent must perform native dtype gate")
            raise
        actual = phased_forward(model, inputs, explicit_cache)
        mx.eval(actual)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-3, atol=2e-3)
        for left, right in zip(native_cache, explicit_cache):
            assert left.offset == right.offset
            for a, b in zip(left.state, right.state):
                assert mx.array_equal(a, b).item()
