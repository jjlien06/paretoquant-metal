"""029: chained diagnostic draft must not change d1 or the committed head KV."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from test_head import args

mx.set_default_device(mx.cpu)


class _Backend:
    from mlx_backend import MTPBackend

    draft = MTPBackend.draft
    CHAIN_EXTRA = MTPBackend.CHAIN_EXTRA


def test_chain_is_cache_neutral_and_matches_forward():
    from mtp_head import BoundHead, MTPHead

    mx.random.seed(7)
    module = MTPHead(args())
    bound = BoundHead(module, nn.Embedding(128, 64), nn.Linear(64, 128, bias=False))
    prefix_hidden = mx.random.normal((1, 5, 64))
    prefix_tokens = mx.array([[3, 9, 1, 4, 7]])
    ref, cand = KVCache(), KVCache()
    for c in (ref, cand):
        bound.forward(prefix_hidden, prefix_tokens, c, position=0, logits=False)
    events = []
    b = _Backend()
    b.head, b.mtp_cache, b.progress = bound, cand, events.append
    hidden = mx.random.normal((1, 1, 64))
    for step in range(4):
        y = 11 + step
        expected = int(mx.argmax(bound.forward(hidden, mx.array([[y]]), ref, position=ref.offset)[:, -1], -1).item())
        assert b.draft(hidden, y) == expected
        assert cand.offset == ref.offset
        for x, z in zip(cand.state, ref.state):
            assert mx.array_equal(x, z).item()
        # verified catch-up pair, as advance_mtp does
        nxt = mx.random.normal((1, 1, 64))
        for c in (ref, cand):
            bound.forward(nxt, mx.array([[expected]]), c, position=c.offset, logits=False)
        hidden = mx.random.normal((1, 1, 64))
    assert len(events) == 4 and all(len(e["chain"]) == 2 for e in events)
    # chain d2 equals an explicit head(norm(h1), d1) evaluation
    e = events[-1]
    assert e["phase"] == "depth_chain"
