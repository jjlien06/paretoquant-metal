"""Strict CPU tracer bullets for the opt-in target hidden-only prefill."""

import mlx.core as mx
from test_phased_cpu import tiny_model

mx.set_default_device(mx.cpu)


def test_hidden_only_returns_every_normalized_position_without_projection(monkeypatch):
    from phased_decode import phased_forward
    from verification import cache_metrics, metric

    model = tiny_model()
    left, right = model.make_cache(), model.make_cache()
    inputs = mx.array([[1, 2, 3, 4, 5]], mx.int32)
    scores, expected = phased_forward(model, inputs, left, all_logits=True, return_hidden=True)
    projection = type(model.language_model.lm_head)
    original = projection.__call__
    calls = []

    def observe(self, x):
        if self is model.language_model.lm_head:
            calls.append(x.shape)
        return original(self, x)

    monkeypatch.setattr(projection, "__call__", observe)
    result = phased_forward(model, inputs, right, logits=False, return_hidden=True)
    assert result is not None, "hidden-only must normalize and return every position"
    omitted, hidden = result
    assert omitted is None and calls == []
    assert hidden.shape == (1, 5, 64)
    assert metric(expected, hidden)["strict_pass"]
    assert all(p["strict_pass"] for p in cache_metrics(left, right))
    # Existing cache-only native prefill contract is unchanged.
    assert phased_forward(model, mx.array([[6]], mx.int32), left, logits=False) is None
    final_scores, final_hidden = phased_forward(
        model, mx.array([[6]], mx.int32), right, return_hidden=True
    )
    assert calls == [(1, 1, 64)]
    assert final_scores.shape == (1, 1, scores.shape[-1])
    assert final_hidden.shape == (1, 1, 64)
    assert all(p["strict_pass"] for p in cache_metrics(left, right))
