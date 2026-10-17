"""Published contract consistency; schema alone never grants admission."""

import json

from .test_harness import ROOT, full_schedule


def test_published_structural_schema_matches_frozen_row_fields():
    path = ROOT / "benchmarks/qwen38_overlap/schema.json"
    assert path.exists(), "Versioned structural contract schema missing"
    schema = json.loads(path.read_text())
    assert schema["x-integration-status"] == "pending_runtime025_adapter_and_independent_review"
    row = schema["$defs"]["request"]
    assert row["additionalProperties"] is False
    assert set(row["required"]) == set(full_schedule()["batches"][0]["requests"][0])
    assert row["properties"]["method"]["const"] == "mtp"
    assert row["properties"]["prefill_mode"]["enum"] == ["serial", "overlap"]
    assert "require_normal_pressure" in schema["$defs"]["configuration"]["required"]
