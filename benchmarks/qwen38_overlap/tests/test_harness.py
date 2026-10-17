"""CPU-only TEST fixtures; never physical/model benchmark results."""

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def api():
    try:
        return importlib.import_module("benchmarks.qwen38_overlap.harness")
    except ModuleNotFoundError:
        return None


def historical():
    return json.loads((ROOT / "results/qwen38-broad-v1/workloads/manifest.json").read_text())


def full_schedule():
    return api().derive_schedule(ROOT / "results/qwen38-broad-v1/workloads/manifest.json")


def test_separate_six_request_pilot():
    h = api()
    assert hasattr(h, "derive_pilot"), "Separate pilot producer is missing"
    pilot = h.derive_pilot(full_schedule())
    rows = pilot["batches"][0]["requests"]
    assert pilot["population"] == "pilot6"
    assert len(rows) == 6
    assert [r["warmup"] for r in rows] == [True, True, False, False, False, False]
    assert [r["prefill_mode"] for r in rows] == [
        "serial",
        "overlap",
        "serial",
        "overlap",
        "overlap",
        "serial",
    ]
    assert [r["case_id"] for r in rows[2:]] == ["code-1536"] * 2 + ["structured-1536"] * 2
    assert {r["repeat"] for r in rows} == {0}
    for row in rows:
        source = next(
            r
            for b in full_schedule()["batches"]
            for r in b["requests"]
            if r["source_request_id"] == row["source_request_id"]
        )
        assert {k: v for k, v in row.items() if k != "id"} == {
            k: v for k, v in source.items() if k != "id"
        }
        assert row["id"].startswith("pilot-v1-")


def test_frozen_same_loaded_schedule():
    h = api()
    assert h is not None, "Versioned CPU schedule producer is missing"
    schedule = h.derive_schedule(ROOT / "results/qwen38-broad-v1/workloads/manifest.json")
    assert schedule == h.derive_schedule(ROOT / "results/qwen38-broad-v1/workloads/manifest.json")
    assert schedule["schema"] == "qwen38-prefill-overlap-schedule-v1"
    assert len(schedule["batches"]) == 6
    flat = [r for b in schedule["batches"] for r in b["requests"]]
    old = [r for b in historical()["batches"] for r in b["requests"]]
    assert len(flat) == len({r["id"] for r in flat}) == 84
    assert sum(not r["warmup"] for r in flat) == 72
    for row, original in zip(flat, old, strict=True):
        assert row["source_request_id"] == original["id"]
        assert row["id"] != original["id"]
        assert row["prefill_mode"] == {"native": "serial", "mtp": "overlap"}[original["method"]]
        assert row["method"] == "mtp"
        for key in original.keys() - {"id", "method"}:
            assert row[key] == original[key]
    assert (
        next(c for c in schedule["cases"] if c["id"] == "structured-1536")[
            "formatted_prompt_tokens"
        ]
        == 1535
    )
