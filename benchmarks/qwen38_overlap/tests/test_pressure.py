"""TEST sampled pressure receipt, not live host measurements."""

from copy import deepcopy

import pytest

from .test_harness import api


def guard_fixture():
    return {
        "require_normal_pressure": True,
        "guard_source_sha256": "a" * 64,
        "interval_seconds": 0.5,
        "spawn_seconds": 1.0,
        "execution_end_seconds": 2.0,
        "samples": [
            {"monotonic_seconds": t, "pressure": 1, "read_error": None}
            for t in (0.5, 1.0, 1.5, 2.0)
        ],
    }


def test_sampled_pressure_guard_covers_execution():
    h = api()
    assert hasattr(h, "audit_pressure"), "Mandatory sampled pressure guard validator missing"
    result = h.audit_pressure(guard_fixture(), "a" * 64, max_gap=0.75)
    assert result["scope"] == "sampled_not_continuous"
    assert result["sample_count"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "disabled",
        "missing",
        "warning",
        "critical",
        "readerror",
        "uncovered_start",
        "uncovered_end",
        "gap",
        "nan",
        "source",
        "bool",
        "duplicate",
    ],
)
def test_pressure_guard_mutations(mutation):
    h = api()
    assert hasattr(h, "audit_pressure"), "Mandatory sampled pressure guard validator missing"
    guard = deepcopy(guard_fixture())
    if mutation == "disabled":
        guard["require_normal_pressure"] = False
    elif mutation == "missing":
        guard["samples"] = []
    elif mutation in ("warning", "critical", "bool"):
        guard["samples"][1]["pressure"] = {"warning": 2, "critical": 4, "bool": True}[mutation]
    elif mutation == "readerror":
        guard["samples"][1]["read_error"] = "failed"
    elif mutation == "uncovered_start":
        guard["spawn_seconds"] = 0.1
    elif mutation == "uncovered_end":
        guard["execution_end_seconds"] = 2.1
    elif mutation == "gap":
        guard["samples"].pop(2)
    elif mutation == "nan":
        guard["samples"][1]["monotonic_seconds"] = float("nan")
    elif mutation == "source":
        guard["guard_source_sha256"] = "b" * 64
    elif mutation == "duplicate":
        guard["samples"][1] = guard["samples"][0]
    with pytest.raises(ValueError):
        h.audit_pressure(guard, "a" * 64, max_gap=0.75)
