"""Opt-in decode-only GDN graphs; no model/cache objects captured as state.

One compiler belongs to one evaluated, immutable-weight model. Reuse it across
fresh-cache requests. Full attention and every synchronization remain native.
Compilation/GPU exceptions propagate, never silently switch implementations.
"""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache


class GDNCompiler:
    def __init__(self, model):
        self.model = model
        self.functions = {}
        self.trace_count = 0
        self.compiled_calls = 0

    def eligible(self, h, layers, caches, mask):
        return (
            h.ndim == 3
            and h.shape[:2] == (1, 1)
            and mask is None
            and bool(layers)
            and len(layers) == len(caches)
            and all(layer.is_linear and not layer.training for layer in layers)
            and all(
                isinstance(c, ArraysCache)
                and c.left_padding is None
                and c.lengths is None
                and len(c.cache) == 2
                and all(isinstance(a, mx.array) and a.shape[0] == 1 for a in c.cache)
                for c in caches
            )
        )

    def run(self, h, layers, caches, mask=None):
        """Return hidden state, publishing explicit returned arrays to native caches."""
        if not self.eligible(h, layers, caches, mask):
            for layer, cache in zip(layers, caches):
                h = layer(h, mask, cache)
            return h
        key = tuple(id(layer) for layer in layers)
        if key not in self.functions:
            owned_layers = tuple(layers)

            def forward(hidden, states):
                self.trace_count += 1  # Python tracing only, not GPU executions
                updated = []
                for layer, arrays in zip(owned_layers, states):
                    local = ArraysCache(size=2)
                    local.cache = list(arrays)
                    hidden = layer(hidden, None, local)
                    updated.append(tuple(local.cache))
                return hidden, tuple(updated)

            self.functions[key] = mx.compile(forward)
        h, states = self.functions[key](h, tuple(tuple(c.cache) for c in caches))
        for cache, arrays in zip(caches, states):
            cache.cache = list(arrays)
        self.compiled_calls += 1
        return h

    def receipt(self):
        return dict(
            trace_count=self.trace_count,
            compiled_calls=self.compiled_calls,
            compiled_functions=len(self.functions),
        )
