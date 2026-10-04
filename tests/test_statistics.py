"""Deterministic fixture tests, not measured performance data."""

import pytest

from paretoquant.statistics import paired_latency_ratio


def test_constant_paired_speedup_has_exact_interval():
    result = paired_latency_ratio([2, 4, 6, 8], [1, 2, 3, 4], resamples=100)
    assert result["estimate"] == result["ci_low"] == result["ci_high"] == 2.0
    assert result["sample_count"] == 4
    assert result["paired"] is True


def test_seed_makes_bootstrap_reproducible():
    args = ([2, 3, 5, 7], [1, 2, 4, 6])
    assert paired_latency_ratio(*args, seed=3) == paired_latency_ratio(*args, seed=3)


@pytest.mark.parametrize(
    "baseline,optimized",
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1]),
        ([0, 1], [1, 1]),
        ([1, 2], [-1, 2]),
        ([float("nan"), 1], [1, 1]),
    ],
)
def test_invalid_paired_timings_rejected(baseline, optimized):
    with pytest.raises(ValueError):
        paired_latency_ratio(baseline, optimized)
