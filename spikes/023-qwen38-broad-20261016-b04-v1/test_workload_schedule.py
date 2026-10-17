"""023 schedule admission: CPU/stdlib only."""

import hashlib
import json
from pathlib import Path

import pytest


def schedule():
    return {
        "schema": "qwen38-workload-schedule-v1",
        "requests": [
            {
                "id": method,
                "case_id": "a",
                "method": method,
                "prompt": "tokens:[1,2,3]",
                "max_tokens": 5,
                "warmup": False,
                "repeat": 0,
            }
            for method in ("native", "mtp")
        ],
    }


def test_schedule_admission_binds_exact_bytes_and_pairs(tmp_path):
    import workload_schedule as w

    data = json.dumps(schedule()).encode()
    path = tmp_path / "benchmark_schedule.json"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert w.load_schedule(path, digest) == schedule()["requests"]
    with pytest.raises(ValueError, match="SHA"):
        w.load_schedule(path, "0" * 64)
    for change in ("count", "method", "bool", "pair", "duplicate", "unknown"):
        bad = schedule()
        if change == "count":
            bad["requests"] *= 17
        elif change == "method":
            bad["requests"][0]["method"] = "fake"
        elif change == "bool":
            bad["requests"][0]["max_tokens"] = True
        elif change == "pair":
            bad["requests"][1]["prompt"] += " "
        elif change == "duplicate":
            bad["requests"][1]["id"] = "native"
        else:
            bad["ignored"] = True
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            w.load_schedule(path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_text('{"schema":"qwen38-workload-schedule-v1","requests":[]}')
    with pytest.raises(ValueError):
        w.load_schedule(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_registered_schedule_and_cli_are_explicit():
    import remote_actor
    import supervise

    assert {"workload_schedule.py", "benchmark_schedule.json"} <= set(remote_actor.SOURCES)
    assert "--benchmark-schedule-sha256" in supervise.parser().format_help()
    assert "--benchmark-schedule" in supervise.parser().format_help()
    assert Path(remote_actor.__file__).parent.joinpath("benchmark_schedule.json").exists()
