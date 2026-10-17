"""Raw025 adapter/collector fail-closed vertical boundary tests."""

import pytest

from benchmarks.qwen38_overlap import harness as h

from .test_integration025 import ROOT, metadata
from .test_supervisor025 import SMALL


def test_raw_adapter_refuses_preserved024_as025_and_retains_failed_rows(tmp_path):
    from benchmarks.qwen38_overlap import integration025 as i
    from benchmarks.qwen38_overlap import supervisor025 as s

    freeze = tmp_path / "freeze"
    manifest = s.freeze(
        freeze, metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"), metadata(SMALL)
    )
    original = ROOT / "results/qwen38-broad-v1/prefill-experiment/candidate-v1"
    # Exact old receipts remain old; no rewrite of their flags or sources.
    trust = {
        "schema": "qwen38-overlap-raw025-trusted-v2",
        "artifact_kind": i.TEST_KIND,
        "phase": "pilot",
        "manifest_sha256": h.sha((freeze / "integration-freeze.json").read_bytes()),
        "raw_artifacts": {
            "rank0.receipt.json": h.sha(h.read_bytes(original / "rank0.receipt.json"))
        },
    }
    with pytest.raises(ValueError, match="raw artifact"):
        i.audit_raw_trial(original, trust, manifest)
    result = i.collect_raw_trials([(original, trust)], manifest)
    assert result["accepted"] is False
    assert result["evidence_validation"] == "invalid"
    assert result["evidence_errors"] and result["missing_ids"]
    assert not result["complete_suite"]
    assert result["attempted_rows"], "failed raw inputs must retain actual public attempted rows"
    assert all(v["status"] == "failed" for v in result["attempted_rows"])
    from benchmarks.qwen38_overlap.cli import encoded, main

    trust_file = tmp_path / "TEST-trust.json"
    trust_file.write_bytes(encoded(trust))
    assert (
        main(
            [
                "collect025",
                "--manifest",
                str(freeze / "integration-freeze.json"),
                "--manifest-sha256",
                trust["manifest_sha256"],
                "--raw-dir",
                str(original),
                "--trusted",
                str(trust_file),
                "--trusted-sha256",
                h.sha(trust_file.read_bytes()),
                "--out-dir",
                str(tmp_path / "report"),
            ]
        )
        == 2
    )
    published = h.decode_json((tmp_path / "report/report.json").read_bytes())
    assert published["attempted_rows"] and published["accepted"] is False


def test_test_fixture_cannot_be_promoted_by_claimed_approval():
    from benchmarks.qwen38_overlap import integration025 as i

    audit = {
        "artifact_kind": i.TEST_KIND,
        "canonical025_valid": True,
        "architecture_metrics_valid": True,
        "original_native_bound": True,
        "parent_review": {"approved": True},
        "phase": "pilot",
    }
    assert i.physical_acceptance(audit) is False
    audit["artifact_kind"] = i.PHYSICAL_KIND
    # No independent source/metadata/raw/approval bindings, even if booleans agree.
    assert i.physical_acceptance(audit) is False
