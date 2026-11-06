"""Evaluated snapshot/restore of native hybrid caches (same semantics as 021 cache_state)."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


def _copy(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(x) for x in value)
    if value is None or type(value) is int:
        return value
    raise ValueError("Unsupported cache state leaf")


def snapshot(caches):
    if any(type(c) not in (ArraysCache, KVCache) for c in caches):
        raise ValueError("Only native ArraysCache/KVCache")
    saved = [(type(c), _copy(c.state)) for c in caches]
    mx.eval([s for _, s in saved])
    return saved


def restore(caches, saved):
    if len(caches) != len(saved) or any(type(c) is not k for c, (k, _) in zip(caches, saved)):
        raise ValueError("Snapshot coverage/type mismatch")
    states = [_copy(s) for _, s in saved]
    mx.eval(states)
    for c, s in zip(caches, states):
        c.state = s
