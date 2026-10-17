"""Bounded watchdog CPU subprocess tests, no SSH/GPU."""

import sys

import pytest


def test_large_schedule_deadline_and_explicit_log_overflow(tmp_path, monkeypatch):
    import remote_actor as a

    assert a.LOG_CAP_BYTES == 64 * 1024**2
    monkeypatch.setattr(a, "LOG_CAP_BYTES", 1024)
    result = a.run_owned([sys.executable, "-c", "print('x'*2048)"], tmp_path / "job", deadline=900)
    assert result["reason"] == "log_cap"
    assert result["log_cap_bytes"] == 1024
    assert result["log_cap_exceeded"]
    assert result["cleanup"]["child_reaped"] and result["cleanup"]["group_gone"]
    with pytest.raises(ValueError):
        a.run_owned([], tmp_path / "invalid", deadline=901)


def test_log_retrieval_overflow_fails_closed(tmp_path, monkeypatch):
    import remote_actor as a

    assert a.RETRIEVAL_CAP_BYTES == 128 * 1024**2
    root = tmp_path / ("a" * 32)
    root.mkdir()
    (root / "stderr").write_bytes(b"x" * 65)
    monkeypatch.setattr(a, "LOG_CAP_BYTES", 64)
    with pytest.raises(ValueError, match="Log too large"):
        a.job_status(tmp_path, "a" * 32, records=True)
