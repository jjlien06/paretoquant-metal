"""Exact snapshots of pinned MLX hybrid caches, including recurrent metadata."""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache,KVCache
from numerical_gate import ATOL,RTOL


def _copy(value):
    if isinstance(value,mx.array):
        result=mx.array(value)
        mx.eval(result)
        return result
    if isinstance(value,list): return [_copy(x) for x in value]
    if isinstance(value,tuple): return tuple(_copy(x) for x in value)
    if value is None or type(value) is int: return value
    raise ValueError('Unsupported cache state leaf')


def snapshot(caches):
    if any(type(c) not in (ArraysCache,KVCache) for c in caches):
        raise ValueError('Only pinned native ArraysCache/KVCache are admitted')
    return [(type(c),_copy(c.state)) for c in caches]


def restore(caches,saved):
    if len(caches)!=len(saved) or any(type(c) is not kind for c,(kind,_) in zip(caches,saved)):
        raise ValueError('Snapshot cache coverage/type mismatch')
    for cache,(_,state) in zip(caches,saved): cache.state=_copy(state)


def _equal(a,b):
    if isinstance(a,mx.array):
        assert isinstance(b,mx.array) and a.shape==b.shape and a.dtype==b.dtype, 'cache metadata'
        assert mx.allclose(a,b,atol=ATOL,rtol=RTOL).item(), 'cache numerical gate'
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b), 'cache tree'
        for x,y in zip(a,b): _equal(x,y)
    else:
        assert type(a) is type(b) and a==b, 'cache scalar metadata'


def assert_cache_equal(native,candidate):
    assert len(native)==len(candidate), 'cache coverage'
    for left,right in zip(native,candidate):
        assert type(left) is type(right), 'cache types'
        if isinstance(left,KVCache):
            assert left.offset==right.offset, 'KV offset'
            if left.keys is None or right.keys is None:
                _equal(left.state,right.state)
            else:
                _equal(left.keys[...,:left.offset,:],right.keys[...,:right.offset,:])
                _equal(left.values[...,:left.offset,:],right.values[...,:right.offset,:])
        elif isinstance(left,ArraysCache): _equal(left.state,right.state)
        else: raise ValueError('Unsupported cache')
