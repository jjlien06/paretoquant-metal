"""Batch-copy synchronization must preserve every actual native cache leaf."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_snapshot_and_restore_evaluate_all_copies_once(monkeypatch, device):
    from cache_state import restore, snapshot

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    arrays = ArraysCache(size=2)
    arrays[0] = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)
    arrays[1] = mx.ones((1, 2, 4, 4), dtype=mx.float32)
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)) * 3)
    mx.eval(arrays.state, kv.state)
    original_eval = mx.eval
    calls = []

    def observe(*args, **kwargs):
        calls.append(args)
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", observe)
    saved = snapshot([arrays, kv])
    assert len(calls) == 1, "One completed batch must replace per-array eval barriers"
    arrays[0][:] = -1
    arrays[1][:] = -2
    kv.keys[:] = -3
    kv.values[:] = -4
    kv.offset = 2
    original_eval(arrays.state, kv.state)
    calls.clear()
    restore([arrays, kv], saved)
    assert len(calls) == 1
    assert mx.array_equal(arrays[0], mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)).item()
    assert mx.array_equal(arrays[1], mx.ones((1, 2, 4, 4))).item()
    assert kv.offset == 5
    assert mx.array_equal(kv.keys[:, :, :5], mx.ones((1, 2, 5, 4))).item()
    assert mx.array_equal(kv.values[:, :, :5], mx.ones((1, 2, 5, 4)) * 3).item()
