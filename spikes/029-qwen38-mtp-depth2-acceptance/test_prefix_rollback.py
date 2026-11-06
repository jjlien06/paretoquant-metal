"""CPU only; physical GPU admission remains mandatory."""

import mlx.core as mx
import pytest
from cache_state import assert_cache_equal, restore, snapshot
from mlx_lm.models import qwen3_5
from numerical_gate import assert_logits_equal
from test_phased_cpu import tiny_model
from verifier import forward

mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("keep", [0, 1, 2])
def test_native_capture_preserves_same_width_then_each_prefix(keep):
    import prefix_rollback as pr

    model = tiny_model()
    original = qwen3_5.GatedDeltaNet.__call__, qwen3_5.gated_delta_update
    left, right = model.make_cache(), model.make_cache()
    for cache in (left, right):
        forward(model, mx.array([[1, 2, 3]]), cache)
    saved = snapshot(right)
    expected = forward(model, mx.array([[4, 5]]), left, all_logits=True, return_hidden=True)
    cap = pr.Capture(model, right)
    with cap:
        actual = forward(model, mx.array([[4, 5]]), right, all_logits=True, return_hidden=True)
    assert (qwen3_5.GatedDeltaNet.__call__, qwen3_5.gated_delta_update) == original
    for a, b in zip(expected, actual):
        assert_logits_equal(a, b)
    assert_cache_equal(left, right)
    assert len(cap.records) == 2
    for record in cap.records:
        assert record["args"][7] is record["prestate"]
        assert record["args"][0].shape[1] == 2
        assert record["conv_input"].shape[1] == 5
        assert record["use_kernel"] is False
    restore(left, saved)
    serial = []
    for token in [4, 5][:keep]:
        serial.append(
            forward(model, mx.array([[token]]), left, all_logits=True, return_hidden=True)
        )
    cap.commit(keep)
    assert_cache_equal(left, right)
    if keep:
        assert_logits_equal(serial[-1][0], actual[0][:, keep - 1 : keep])
        assert_logits_equal(serial[-1][1], actual[1][:, keep - 1 : keep])
    with pytest.raises(ValueError):
        cap.commit(keep)


def test_capture_rejects_unsupported_scope_and_restores_hooks_on_exception():
    import prefix_rollback as pr

    model = tiny_model()
    caches = model.make_cache()
    original = qwen3_5.GatedDeltaNet.__call__
    with pytest.raises(ValueError):
        with pr.Capture(model, caches):
            forward(model, mx.array([[1]]), caches)
    assert qwen3_5.GatedDeltaNet.__call__ is original
    caches[0].lengths = mx.array([4])
    with pytest.raises(ValueError):
        pr.Capture(model, caches)
