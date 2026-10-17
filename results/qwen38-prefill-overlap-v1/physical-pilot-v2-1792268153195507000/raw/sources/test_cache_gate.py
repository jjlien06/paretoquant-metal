import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

mx.set_default_device(mx.cpu)


def test_every_hybrid_cache_and_metadata_survives_snapshot_mutations():
    from cache_state import assert_cache_equal, restore, snapshot

    kv = KVCache()
    x = mx.ones((1, 2, 3, 4))
    kv.update_and_fetch(x, x * 2)
    ssm = ArraysCache(2)
    ssm[0] = mx.ones((1, 3, 4))
    ssm[1] = mx.ones((1, 2, 4, 4))
    ssm.left_padding = mx.array([2])
    ssm.lengths = mx.array([9])
    caches = [kv, ssm]
    saved = snapshot(caches)
    kv.keys[:] = 77
    kv.offset = 99
    ssm[0][:] = 88
    ssm[1] = mx.zeros_like(ssm[1])
    ssm.advance(4)
    restore(caches, saved)
    assert kv.offset == 3 and mx.all(kv.keys[..., : kv.offset, :] == 1).item()
    assert mx.all(ssm[0] == 1).item() and mx.all(ssm[1] == 1).item()
    assert ssm.left_padding.item() == 2 and ssm.lengths.item() == 9
    other = [KVCache(), ArraysCache(2)]
    restore(other, saved)
    assert_cache_equal(caches, other)
    other[1][1][:] = 1.01
    with pytest.raises(AssertionError):
        assert_cache_equal(caches, other)
    restore(other, saved)
    other[0].offset = 2
    with pytest.raises(AssertionError):
        assert_cache_equal(caches, other)


def test_fixed_original_numerical_gate_rejects_width_drift_even_if_argmax_same():
    from numerical_gate import ATOL, RTOL, assert_logits_equal

    assert (ATOL, RTOL) == (1e-6, 1e-5)
    native = mx.array([[[0.0, 1.0, 3.0]]])
    wider = mx.array([[[0.0, 1.01, 3.0]]])
    assert mx.array_equal(mx.argmax(native, -1), mx.argmax(wider, -1)).item()
    with pytest.raises(AssertionError):
        assert_logits_equal(native, wider)
    assert_logits_equal(native, native)
