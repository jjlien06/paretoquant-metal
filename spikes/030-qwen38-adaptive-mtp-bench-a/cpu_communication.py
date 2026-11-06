"""Keep blocking ring waits off Metal command buffers in the standalone spike.

This changes scheduling, not model arithmetic: each outgoing GPU value is fully
materialized before a CPU-stream collective, and the result is materialized
before downstream GPU work. Scope is process-global and single-threaded.
"""

from contextlib import contextmanager

import mlx.core as mx


@contextmanager
def synchronous_cpu_pipeline(group):
    names = ("send", "recv_like", "all_gather")
    originals = {name: getattr(mx.distributed, name) for name in names}
    counts = {name: 0 for name in names}

    def wrapper(name):
        def run(*args, **kwargs):
            if name != "recv_like":
                mx.eval(args[0])
            kwargs["stream"] = mx.cpu
            kwargs.setdefault("group", group)
            result = originals[name](*args, **kwargs)
            mx.eval(result)
            counts[name] += 1
            return result

        return run

    try:
        for name in names:
            setattr(mx.distributed, name, wrapper(name))
        yield counts
    finally:
        for name, function in originals.items():
            setattr(mx.distributed, name, function)
