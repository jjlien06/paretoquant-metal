"""CPU subprocess exercise of immutable actor snapshots, no model loading."""

import hashlib
import sys
from pathlib import Path

import pytest


def test_owned_snapshot_includes_exact_schedule_and_preserves_cleanup(tmp_path):
    from remote_actor import SOURCES, run_owned, source_hashes

    root = Path(__file__).parent
    expected = source_hashes(root)
    job = tmp_path / "snapshot"
    result = run_owned(
        [sys.executable, str(root / "probe.py"), "--help"],
        job,
        deadline=900,
        source_root=root,
        expected_hashes=expected,
    )
    assert result["reason"] == "completed" and result["log_evidence_complete"]
    assert result["cleanup"]["child_reaped"] and result["cleanup"]["group_gone"]
    assert source_hashes(job / "sources") == expected
    assert set(expected) == set(SOURCES)
    assert "--benchmark-schedule" in (job / "stdout").read_text()
    assert (
        hashlib.sha256((job / "sources/benchmark_schedule.json").read_bytes()).hexdigest()
        == expected["benchmark_schedule.json"]
    )


def test_serialized_retrieval_response_cap_is_not_silent(tmp_path, monkeypatch):
    import remote_actor as a

    root = tmp_path / ("a" * 32)
    root.mkdir()
    (root / "stdout").write_text("a" * 20)
    monkeypatch.setattr(a, "RETRIEVAL_CAP_BYTES", 32)
    with pytest.raises(ValueError, match="Retrieval response"):
        a.job_status(tmp_path, "a" * 32, records=True)
