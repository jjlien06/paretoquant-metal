"""Synthetic TEST timing rows. These are NOT model results or receipts."""

import pytest

from .test_harness import api, full_schedule


def timing_rows():
    rows = []
    for batch in full_schedule()["batches"]:
        for spec in batch["requests"]:
            baseline = (1.0, 10.0)[spec["repeat"]]
            candidate = (0.5, 2.0)[spec["repeat"]]
            value = baseline if spec["prefill_mode"] == "serial" else candidate
            rows.append(
                {
                    **spec,
                    "status": "success",
                    "first_token_seconds": value,
                    "wall_seconds": value * 2,
                }
            )
    return rows


def test_descriptive_median_of_repeat_paired_ratios():
    h = api()
    assert hasattr(h, "analyze"), "Descriptive paired analyzer missing"
    report = h.analyze(timing_rows(), full_schedule())
    assert report["coverage_complete"] is True
    assert report["statistical_targets_met"] is True
    assert report["accepted"] is False
    assert report["evidence_validation"] == "not_evaluated"
    assert len(report["cases"]) == 18
    for case in report["cases"]:
        assert case["first_token_seconds"]["paired_ratios"] == [0.5, 0.2]
        assert case["first_token_seconds"]["median_ratio"] == 0.35
        assert case["wall_seconds"]["reduction"] == 0.65
    assert report["regressions"] == []


def test_incomplete_and_failed_attempts_are_retained():
    rows = timing_rows()
    failed = rows.pop()
    failed["status"] = "failed"
    rows.append(failed)
    rows.pop(0)
    report = api().analyze(rows, full_schedule())
    assert report["coverage_complete"] is False
    assert report["statistical_targets_met"] is False
    assert report["attempted_rows"] == rows
    assert failed["id"] in [r["id"] for r in report["failed_rows"]]
    assert len(report["cases"]) == 18


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan"), 10**1000])
def test_invalid_metrics_never_complete(bad):
    rows = timing_rows()
    rows[2]["first_token_seconds"] = bad
    report = api().analyze(rows, full_schedule())
    assert report["coverage_complete"] is False
    assert report["statistical_targets_met"] is False


def test_one_long_failure_has_no_pooled_override_and_all_regressions_reported():
    rows = timing_rows()
    for row in rows:
        if row["prefill_mode"] == "overlap" and row["case_id"] == "structured-1536":
            baseline = (1.0, 10.0)[row["repeat"]]
            row["first_token_seconds"] = baseline * 0.71
            row["wall_seconds"] = baseline * 2 * 0.86
        elif row["prefill_mode"] == "overlap" and not row["case_id"].endswith("-1536"):
            baseline = (1.0, 10.0)[row["repeat"]]
            row["first_token_seconds"] = baseline * 1.1
            row["wall_seconds"] = baseline * 2 * 1.1
    report = api().analyze(rows, full_schedule())
    assert report["coverage_complete"] is True
    assert report["statistical_targets_met"] is False
    assert len({r["case_id"] for r in report["regressions"]}) == 12
    assert len(report["regressions"]) == 24


def test_unrounded_threshold_equality_passes():
    rows = timing_rows()
    for row in rows:
        baseline = (1.0, 10.0)[row["repeat"]]
        if row["prefill_mode"] == "overlap":
            row["first_token_seconds"] = baseline * 0.7
            row["wall_seconds"] = baseline * 2 * 0.85
    assert api().analyze(rows, full_schedule())["statistical_targets_met"] is True
