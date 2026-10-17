"""Synchronous pipeline boundaries must leave upstream MLX untouched on exit."""

import mlx.core as mx
import pytest


def test_cpu_pipeline_context_restores_native_functions_on_exception():
    from cpu_communication import synchronous_cpu_pipeline

    names = ("send", "recv_like", "all_gather")
    original = {name: getattr(mx.distributed, name) for name in names}
    with pytest.raises(RuntimeError, match="probe exception"):
        with synchronous_cpu_pipeline(None) as counts:
            assert all(getattr(mx.distributed, name) is not original[name] for name in names)
            assert counts == {name: 0 for name in names}
            raise RuntimeError("probe exception")
    assert all(getattr(mx.distributed, name) is original[name] for name in names)
