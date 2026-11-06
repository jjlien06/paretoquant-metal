"""Explicit width2 vs sequential-native-width1 gate. Single-host fixture helper.

Parent must repeat equivalent checks on both physical GPU ranks and exact target;
this CPU entry does not load weights or select a device. Exceptions retain failure.
"""

import mlx.core as mx
from cache_state import assert_cache_equal
from numerical_gate import ATOL, RTOL, assert_logits_equal
from phased_decode import phased_forward


def compare_width2(model, prompt, pair, *, prefill_step_size=256):
    if len(pair) != 2 or not prompt or not 1 <= prefill_step_size <= 256:
        raise ValueError("Require prompt, two forced tokens, bounded prefill chunk")
    native, batched, phased = [model.make_cache() for _ in range(3)]
    for start in range(0, len(prompt) - 1, prefill_step_size):
        ids = prompt[start : min(start + prefill_step_size, len(prompt) - 1)]
        x = mx.array([ids], mx.int32)
        for cache in (native, batched, phased):
            mx.eval(model(x, cache=cache), [c.state for c in cache])
    last = mx.array([[prompt[-1]]], mx.int32)
    for cache in (native, batched, phased):
        mx.eval(model(last, cache=cache), [c.state for c in cache])
    rows = []
    for token in pair:
        out = model(mx.array([[token]], mx.int32), cache=native)
        mx.eval(out, [c.state for c in native])
        rows.append(out)
    expected = mx.concatenate(rows, axis=1)
    native_width2 = model(mx.array([pair], mx.int32), cache=batched)
    actual = phased_forward(model, mx.array([pair], mx.int32), phased, all_logits=True)
    assert_logits_equal(expected, native_width2)
    assert_logits_equal(expected, actual)
    assert_cache_equal(native, batched)
    assert_cache_equal(native, phased)
    return {
        "widths": [1, 2],
        "atol": ATOL,
        "rtol": RTOL,
        "logits_pass": True,
        "cache_pass": True,
        "device_scope": "caller-selected; not real-target admission",
    }
