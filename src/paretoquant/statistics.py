"""Paired bootstrap intervals for measured latency ratios; no significance claims."""

import numpy as np


def paired_latency_ratio(baseline_ms, optimized_ms, *, resamples=5000, seed=0):
    baseline = np.asarray(baseline_ms, dtype=np.float64)
    optimized = np.asarray(optimized_ms, dtype=np.float64)
    if (
        baseline.ndim != 1
        or optimized.ndim != 1
        or baseline.shape != optimized.shape
        or baseline.size < 2
        or not np.isfinite(baseline).all()
        or not np.isfinite(optimized).all()
        or (baseline <= 0).any()
        or (optimized <= 0).any()
    ):
        raise ValueError(
            "paired timings must have matching lengths >= 2 and finite positive values"
        )
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, (resamples, baseline.size))
    ratios = np.median(baseline[indices], axis=1) / np.median(optimized[indices], axis=1)
    if not np.isfinite(ratios).all():
        raise ValueError("latency ratios must be finite")
    low, high = np.percentile(ratios, [2.5, 97.5])
    return {
        "estimate": float(np.median(baseline) / np.median(optimized)),
        "ci_low": float(low),
        "ci_high": float(high),
        "sample_count": baseline.size,
        "resamples": resamples,
        "seed": seed,
        "paired": True,
        "method": "paired_bootstrap_of_median_latency_ratio",
        "scope_note": "Interval covers sampled trials, not other devices/models/workloads",
    }
