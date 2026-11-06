"""Scoped native-width2 observation and recurrence-only prefix commit.

No module replacement, parameter registration or forward reimplementation. Hooks
are process-global only during actor verification; context-local selection keeps
unrelated calls untouched. Captured arrays may change lazy fusion: fresh physical
same-width AND singleton-prefix gates are mandatory, never inherited from 019.
"""

from contextvars import ContextVar
from threading import RLock

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5 as native
from mlx_lm.models.cache import ArraysCache, KVCache

_active = ContextVar("prefix_capture", default=None)
_lock = RLock()
_users = 0
_original = None


def _gdn(module, inputs, mask=None, cache=None):
    cap = _active.get()
    if cap is None or id(module) not in cap.owned:
        return _original[0](module, inputs, mask, cache)
    if inputs.shape[:2] != (1, 2) or mask is not None or cache is not cap.owned[id(module)]:
        raise ValueError("Capture requires owned unpadded batch1 width2 verify")
    if cap.current is not None:
        raise ValueError("Nested GDN capture")
    rec = {
        "cache": cache,
        "kernel": module.conv_kernel_size,
        "prestate": cache[1],
        "module": module,
        "conv_input": None,
        "args": None,
    }
    cap.current = rec
    try:
        result = _original[0](module, inputs, mask, cache)
    finally:
        cap.current = None
    if rec["conv_input"] is None or rec["args"] is None:
        raise ValueError("Native GDN capture incomplete")
    cap.records.append(rec)
    return result


def _conv(module, inputs):
    cap = _active.get()
    rec = None if cap is None else cap.current
    if rec is not None and module is rec["module"].conv1d:
        rec["conv_input"] = inputs
    return _original[1](module, inputs)


def _delta(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    cap = _active.get()
    rec = None if cap is None else cap.current
    if rec is not None:
        if rec["args"] is not None or state is not rec["prestate"] or mask is not None:
            raise ValueError("Unexpected native recurrence contract")
        rec["args"] = (q, k, v, a, b, A_log, dt_bias, state, mask)
        rec["use_kernel"] = use_kernel
        rec["recurrence"] = _original[2]
    return _original[2](q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel)


class Capture:
    def __init__(self, model, caches):
        layers = model.model.pipeline_layers
        if len(layers) != len(caches):
            raise ValueError("Owned cache coverage")
        self.owned = {}
        for layer, cache in zip(layers, caches):
            if layer.is_linear:
                if (
                    type(cache) is not ArraysCache
                    or cache.lengths is not None
                    or cache.left_padding is not None
                    or len(cache.cache) != 2
                ):
                    raise ValueError("Only unpadded native ArraysCache admitted")
                self.owned[id(layer.linear_attn)] = cache
            elif type(cache) is not KVCache:
                raise ValueError("Only native KVCache admitted")
        self.caches, self.records, self.current = caches, [], None
        self.pre = [
            (list(c.cache), c.left_padding, c.lengths) if type(c) is ArraysCache else c.offset
            for c in caches
        ]
        self.used = False
        self.complete = False

    def __enter__(self):
        global _users, _original
        if _active.get() is not None or self.records or self.used:
            raise ValueError("Capture is single use, non-nested")
        with _lock:
            if _users == 0:
                _original = (
                    native.GatedDeltaNet.__call__,
                    nn.Conv1d.__call__,
                    native.gated_delta_update,
                )
                native.GatedDeltaNet.__call__, nn.Conv1d.__call__, native.gated_delta_update = (
                    _gdn,
                    _conv,
                    _delta,
                )
            _users += 1
        self.token = _active.set(self)
        return self

    def __exit__(self, kind, value, tb):
        global _users
        _active.reset(self.token)
        with _lock:
            _users -= 1
            if _users == 0:
                native.GatedDeltaNet.__call__, nn.Conv1d.__call__, native.gated_delta_update = (
                    _original
                )
        if kind is None:
            if len(self.records) != len(self.owned) or len(
                {id(r["cache"]) for r in self.records}
            ) != len(self.owned):
                raise ValueError("Missing/duplicate owned GDN capture")
            self.complete = True

    def commit(self, keep):
        if type(keep) is not int or keep not in (0, 1, 2) or not self.complete or self.used:
            raise ValueError("Require complete single-use prefix0/1/2 capture")
        self.used = True
        # Validate everything before touching a cache. Abort remains full restore.
        for c, pre in zip(self.caches, self.pre):
            if type(c) is KVCache and c.offset != pre + 2:
                raise ValueError("Stale capture KV offset")
        if keep == 0:
            for c, pre in zip(self.caches, self.pre):
                if type(c) is ArraysCache:
                    c.state = (list(pre[0]), pre[1], pre[2])
                else:
                    c.trim(2)
        elif keep == 1:
            for r in self.records:
                args = r["args"]
                prefix = tuple(x[:, :keep] for x in args[:5]) + args[5:]
                _, state = r["recurrence"](*prefix, use_kernel=r["use_kernel"])
                r["cache"][0] = mx.contiguous(r["conv_input"][:, keep : keep + r["kernel"] - 1, :])
                r["cache"][1] = state
            for c in self.caches:
                if type(c) is KVCache:
                    c.trim(1)
        # One evaluation boundary, no full trunk/head work, all live caches.
        mx.eval([c.state for c in self.caches])
