"""Explicit layer scheduling must preserve native Qwen logits and cache state."""

from functools import wraps

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model, ModelArgs


@pytest.fixture
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("entrypoint", ["phased_forward", "phased_generate"])
@pytest.mark.parametrize("invalid", [0, -1, 3, 16, True, False, 1.0, None, "2"])
def test_eval_group_size_rejected_before_any_work(cpu_device, monkeypatch, entrypoint, invalid):
    import inspect

    import phased_decode

    function = getattr(phased_decode, entrypoint)
    assert inspect.signature(function).parameters["eval_group_size"].default == 1

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid eval_group_size must fail before ownership or evaluation")

    monkeypatch.setattr(phased_decode, "ownership", forbidden)
    monkeypatch.setattr(mx, "eval", forbidden)
    with pytest.raises(ValueError, match="eval_group_size"):
        function(None, None, None, eval_group_size=invalid)


def tiny_model():
    return Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=5,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )


@pytest.mark.parametrize("invalid", [0, 1, None, "false"])
def test_all_logits_requires_exact_bool_before_ownership(cpu_device, monkeypatch, invalid):
    import phased_decode

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid all_logits must fail before ownership or evaluation")

    monkeypatch.setattr(phased_decode, "ownership", forbidden)
    monkeypatch.setattr(mx, "eval", forbidden)
    with pytest.raises(ValueError, match="all_logits"):
        phased_decode.phased_forward(None, None, None, all_logits=invalid)


@pytest.mark.parametrize("eval_group_size", [1, 2, 4, 8])
def test_all_logits_matches_native_verification_batch(cpu_device, eval_group_size):
    from phased_decode import phased_forward

    mx.random.seed(37)
    model = tiny_model()
    mx.eval(model.parameters())
    native_cache, verification_cache = make_prompt_cache(model), make_prompt_cache(model)
    for ids in ([[1, 2, 3]], [[4, 5]], [[6]]):
        inputs = mx.array(ids)
        expected = model(inputs, cache=native_cache)
        actual = phased_forward(
            model,
            inputs,
            verification_cache,
            eval_group_size=eval_group_size,
            all_logits=True,
        )
        mx.eval(expected, actual)
        assert actual.shape == expected.shape == (1, len(ids[0]), model.args.vocab_size)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
        for left, right in zip(native_cache, verification_cache):
            assert left.offset == right.offset
            for a, b in zip(left.state, right.state):
                assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("eval_group_size", [1, 2, 4, 8])
def test_grouped_blocks_match_native_logits_and_live_cache(
    cpu_device, monkeypatch, eval_group_size
):
    from phased_decode import phased_forward

    mx.random.seed(7)
    model = tiny_model()
    mx.eval(model.parameters())
    native_cache, grouped_cache = make_prompt_cache(model), make_prompt_cache(model)
    evaluate = mx.eval

    def leaves(value):
        if isinstance(value, (list, tuple)):
            return [leaf for item in value for leaf in leaves(item)]
        return [value]

    for ids in ([[1, 2, 3]], [[4]], [[5, 6]]):
        inputs = mx.array(ids)
        expected = model(inputs, cache=native_cache)[:, -1:, :]
        evaluate(expected)
        trace = []

        def record_eval(*args):
            evaluate(*args)
            trace.append({"phase": "evaluated", "arrays": leaves(args)})

        with monkeypatch.context() as patch:
            patch.setattr(mx, "eval", record_eval)
            actual = phased_forward(
                model, inputs, grouped_cache, eval_group_size=eval_group_size, progress=trace.append
            )
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
        for left, right in zip(native_cache, grouped_cache):
            assert left.offset == right.offset
            for a, b in zip(left.state, right.state):
                assert mx.array_equal(a, b).item()

        cursor = 3  # embedding_before, evaluated, embedding_done
        assert [event["phase"] for event in trace[:3]] == [
            "embedding_before",
            "evaluated",
            "embedding_done",
        ]
        for start in range(0, 5, eval_group_size):
            end = min(start + eval_group_size, 5)
            if eval_group_size == 1:
                assert trace[cursor] == {"phase": "layer_before", "layer_index": start}
            else:
                assert trace[cursor] == {
                    "phase": "block_before",
                    "start_idx": start,
                    "end_idx": end,
                }
            evaluated = trace[cursor + 1]
            assert evaluated["phase"] == "evaluated"
            live_states = [array for state in grouped_cache[start:end] for array in state.state]
            assert len(evaluated["arrays"]) == 1 + len(live_states)
            assert all(a is b for a, b in zip(evaluated["arrays"][1:], live_states))
            cursor += 2
            if eval_group_size > 1:
                assert trace[cursor] == {"phase": "block_done", "start_idx": start, "end_idx": end}
                cursor += 1
            for index in range(start, end):
                assert trace[cursor] == {"phase": "layer_done", "layer_index": index}
                cursor += 1
        assert [event["phase"] for event in trace[cursor:]] == [
            "head_before",
            "evaluated",
            "head_done",
        ]


@pytest.mark.parametrize("eval_group_size", [2, 4, 8])
def test_grouped_generate_threads_every_forward_and_keeps_fresh_cache(
    cpu_device, monkeypatch, eval_group_size
):
    import phased_decode

    model = tiny_model()

    class Tokenizer:
        eos_token_ids = set()

        def decode(self, tokens, skip_special_tokens=True):
            return str(tokens)

    options = {"max_tokens": 3, "prefill_step_size": 2}
    prompt = [1, 2, 3, 4, 5, 6]
    expected = phased_decode.phased_generate(model, Tokenizer(), prompt, **options)
    forward = phased_decode.phased_forward
    calls = []

    def record_forward(model, inputs, cache, **kwargs):
        calls.append((kwargs.get("eval_group_size", 1), cache, [state.offset for state in cache]))
        return forward(model, inputs, cache, **kwargs)

    monkeypatch.setattr(phased_decode, "phased_forward", record_forward)
    for _ in range(2):
        actual = phased_decode.phased_generate(
            model, Tokenizer(), prompt, eval_group_size=eval_group_size, **options
        )
        assert actual["token_ids"] == expected["token_ids"]
    assert len(calls) == 12
    assert all(size == eval_group_size for size, _, _ in calls)
    assert calls[0][1] is not calls[6][1]
    for request in (calls[:6], calls[6:]):
        assert all(cache is request[0][1] for _, cache, _ in request)
        assert [offsets for _, _, offsets in request] == [[n] * 5 for n in (0, 2, 4, 5, 6, 7)]
        assert [state.offset for state in request[0][1]] == [8] * 5


@pytest.mark.parametrize("eval_group_size", [2, 4, 8])
def test_failed_block_eval_never_reports_its_layers_done(cpu_device, monkeypatch, eval_group_size):
    from phased_decode import phased_forward

    model = tiny_model()
    cache = make_prompt_cache(model)
    events = []
    evaluate = mx.eval
    failed_start = 4 if eval_group_size < 8 else 0

    def fail_block(*args):
        if events[-1]["phase"] == "block_before" and events[-1]["start_idx"] == failed_start:
            raise RuntimeError("block evaluation failed")
        evaluate(*args)

    monkeypatch.setattr(mx, "eval", fail_block)
    with pytest.raises(RuntimeError, match="block evaluation failed"):
        phased_forward(
            model, mx.array([[1]]), cache, eval_group_size=eval_group_size, progress=events.append
        )
    assert [e["layer_index"] for e in events if e["phase"] == "layer_done"] == list(
        range(failed_start)
    )
    assert events[-1] == {"phase": "block_before", "start_idx": failed_start, "end_idx": 5}
    assert not any(
        e["phase"].startswith("handoff_send") or e["phase"] == "head_before" for e in events
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("eval_group_size", [2, 4, 8])
def test_cpu_handoff_boundaries_enclose_only_completed_local_blocks(
    cpu_device, monkeypatch, rank, eval_group_size
):
    from phased_decode import phased_forward

    model = tiny_model()
    cache = make_prompt_cache(model)
    core = model.model
    core.pipeline_rank, core.pipeline_size = rank, 2
    core.start_idx, core.end_idx = (2, 5) if rank == 0 else (0, 2)
    cache = cache[core.start_idx : core.end_idx]
    trace = []
    evaluate = mx.eval
    mx.eval(model.parameters())

    class Group:
        def rank(self):
            return rank

        def size(self):
            return 2

    for index in range(core.start_idx, core.end_idx):
        layer = core.layers[index]

        def record_layer(h, mask, state, layer=layer, index=index):
            trace.append({"phase": "layer_constructed", "layer_index": index})
            return layer(h, mask, state)

        core.layers[index] = record_layer

    def record_eval(*args):
        evaluate(*args)
        trace.append({"phase": "evaluated"})

    def recv(shape, dtype, peer, *, group, stream):
        assert rank == 0 and peer == 1 and stream == mx.cpu
        assert trace == [{"phase": "handoff_receive_before", "tokens": 1}]
        trace.append({"phase": "recv_called"})
        return mx.zeros(shape, dtype=dtype)

    def send(h, peer, *, group, stream):
        assert rank == 1 and peer == 0 and stream == mx.cpu
        assert [e["layer_index"] for e in trace if e["phase"] == "layer_done"] == list(
            range(core.start_idx, core.end_idx)
        )
        trace.append({"phase": "send_called"})
        return h

    def forbidden(*args, **kwargs):
        pytest.fail("Communication must never be compiled into a GPU graph")

    monkeypatch.setattr(mx, "eval", record_eval)
    monkeypatch.setattr(mx, "compile", forbidden)
    monkeypatch.setattr(mx.distributed, "recv", recv)
    monkeypatch.setattr(mx.distributed, "send", send)
    phased_forward(
        model,
        mx.array([[1]]),
        cache,
        group=Group(),
        eval_group_size=eval_group_size,
        progress=trace.append,
    )
    assert all(state.offset == 1 for state in cache)
    if rank == 0:
        assert [e["phase"] for e in trace[:5]] == [
            "handoff_receive_before",
            "recv_called",
            "evaluated",
            "handoff_receive_done",
            "block_before",
        ]
    else:
        assert [e["phase"] for e in trace[-5:]] == [
            "evaluated",
            "handoff_send_before",
            "send_called",
            "evaluated",
            "handoff_send_done",
        ]
    blocks = [e for e in trace if e["phase"] == "block_done"]
    assert blocks == [
        {
            "phase": "block_done",
            "start_idx": start,
            "end_idx": min(start + eval_group_size, core.end_idx),
        }
        for start in range(core.start_idx, core.end_idx, eval_group_size)
    ]
    for position, event in enumerate(trace):
        if event["phase"] == "block_done":
            assert trace[position - 1]["phase"] == "evaluated"
    assert [e["layer_index"] for e in trace if e["phase"] == "layer_constructed"] == list(
        range(core.start_idx, core.end_idx)
    )


@pytest.mark.parametrize("eval_group_size", [2, 4, 8])
def test_grouped_layers_reject_intermediate_dtype_change(cpu_device, eval_group_size):
    from phased_decode import phased_forward

    model = tiny_model()
    cache = make_prompt_cache(model)
    original = model.model.layers[0]
    calls = []

    def changed_dtype(h, mask, state):
        calls.append(0)
        return original(h, mask, state).astype(mx.float16)

    def restore_dtype(h, mask, state):
        calls.append(1)
        return h.astype(mx.float32)

    model.model.layers[:2] = [changed_dtype, restore_dtype]
    with pytest.raises(ValueError, match="Layer changed hidden state dtype"):
        phased_forward(model, mx.array([[1]]), cache, eval_group_size=eval_group_size)
    assert calls == [0]  # inspecting only the block's final dtype would miss the change


@pytest.mark.parametrize("eval_group_size", [2, 4, 8])
@pytest.mark.parametrize("violation", ["offset", "reverse_ownership"])
def test_grouped_admission_rejects_invalid_cache_or_ownership_before_eval(
    cpu_device, monkeypatch, eval_group_size, violation
):
    from phased_decode import phased_forward

    model = tiny_model()
    cache = make_prompt_cache(model)
    group = None
    message = "offset"
    if violation == "offset":
        cache[0].offset = 1
    else:

        class Group:
            def rank(self):
                return 0

            def size(self):
                return 2

        group = Group()
        model.model.pipeline_size = 2
        model.model.end_idx = 3  # rank 0 incorrectly owns the early interval
        cache = cache[:3]
        message = "reverse contiguous"

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid admission must fail before evaluation or communication")

    monkeypatch.setattr(mx, "eval", forbidden)
    monkeypatch.setattr(mx.distributed, "recv", forbidden)
    monkeypatch.setattr(mx.distributed, "send", forbidden)
    with pytest.raises(ValueError, match=message):
        phased_forward(model, mx.array([[1]]), cache, group=group, eval_group_size=eval_group_size)


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
