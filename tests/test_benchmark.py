import pytest

mx = pytest.importorskip("mlx.core")
from paretoquant.benchmark import benchmark_functions, environment  # noqa: E402


def test_benchmark_records_raw_samples_and_metadata():
    x = mx.ones((8,))
    result = benchmark_functions({"one": lambda: x + 1, "two": lambda: x * 2}, warmup=1, repeats=3)
    assert list(result) == ["one", "two"]
    for entry in result.values():
        assert len(entry["samples_ms"]) == 3
        assert entry["median_ms"] > 0
        assert entry["measurement"] == "synchronized_wall_clock"
    assert environment()["device"]["device_name"].startswith("Apple")


@pytest.mark.parametrize("warmup,repeats", [(-1, 2), (1, 0), (1, -1)])
def test_invalid_benchmark_counts(warmup, repeats):
    with pytest.raises(ValueError):
        benchmark_functions({"one": lambda: mx.ones((1,))}, warmup=warmup, repeats=repeats)


def test_empty_benchmark_rejected():
    with pytest.raises(ValueError):
        benchmark_functions({})
