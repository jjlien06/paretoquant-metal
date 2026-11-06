"""030: adaptive chained draft must not change d1 or the committed head KV."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from test_head import args

mx.set_default_device(mx.cpu)


class _Backend:
    from mlx_backend import MTPBackend

    draft_adaptive = MTPBackend.draft_adaptive


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
        d1, p1, d2, p2 = b.draft_adaptive(hidden, y, allow_second=step % 2 == 0)
        assert d1 == expected and 0 < p1 <= 1
        assert (d2 is None) == (step % 2 == 1) and (p2 is None or 0 < p2 <= 1)
        assert cand.offset == ref.offset
        for x, z in zip(cand.state, ref.state):
            assert mx.array_equal(x, z).item()
        # verified catch-up pair, as advance_mtp does
        nxt = mx.random.normal((1, 1, 64))
        for c in (ref, cand):
            bound.forward(nxt, mx.array([[expected]]), c, position=c.offset, logits=False)
        hidden = mx.random.normal((1, 1, 64))
