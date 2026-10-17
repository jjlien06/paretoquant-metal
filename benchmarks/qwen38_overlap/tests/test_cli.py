"""CLI TESTS: no subprocess/SSH/socket/model activity in harness commands."""

import hashlib
import importlib
import io
import json
from contextlib import redirect_stdout

import pytest

from .test_analysis import timing_rows
from .test_collect import record_fixture
from .test_harness import api, full_schedule


def cli_api():
    try:
        return importlib.import_module("benchmarks.qwen38_overlap.cli")
    except ModuleNotFoundError:
        return None


def invoke(args, **kwargs):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = cli_api().main(args, **kwargs)
    return code, stdout.getvalue()


def test_plan_default_is_pure_and_explicit_freeze_is_versioned(tmp_path, monkeypatch):
    cli = cli_api()
    assert cli is not None, "Standalone plan CLI missing"
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.rglob("*"))
    code, text = invoke(["plan"])
    assert code == 0
    plan = json.loads(text)
    assert plan["requests"] == 84
    assert plan["measured"] == 72
    assert plan["pilot_requests"] == 6
    assert "--require-normal-pressure" in plan["mandatory_supervisor_flags"]
    assert "--hidden-only-prefill" in plan["mandatory_supervisor_flags"]
    assert "--overlapped-prefill" in plan["mandatory_supervisor_flags"]
    assert plan["execution_permitted"] is False
    assert list(tmp_path.rglob("*")) == before
    output = tmp_path / "freeze"
    assert invoke(["plan", "--out-dir", str(output)])[0] == 0
    frozen = json.loads((output / "schedule.json").read_text())
    assert frozen == full_schedule()
    for name, digest in json.loads((output / "freeze.json").read_text())["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    # Exclusive artifacts: a second freeze cannot overwrite.
    assert invoke(["plan", "--out-dir", str(output)])[0] == 2


def test_collect_and_audit_cli_bind_external_inputs(tmp_path):
    cli = cli_api()
    assert cli is not None, "Standalone collect/audit CLI missing"
    _, schedule, trust, record = record_fixture(tmp_path)
    paths = {}
    for name, value in (("schedule", schedule), ("record", record)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        paths[name] = path

    def digest(name):
        return hashlib.sha256(paths[name].read_bytes()).hexdigest()

    trust["schedule_sha256"] = digest("schedule")
    paths["trusted"] = tmp_path / "trusted.json"
    paths["trusted"].write_text(json.dumps(trust))
    args = [
        "--schedule",
        str(paths["schedule"]),
        "--schedule-sha256",
        digest("schedule"),
        "--trusted",
        str(paths["trusted"]),
        "--trusted-sha256",
        digest("trusted"),
        "--record",
        str(paths["record"]),
        "--record-sha256",
        digest("record"),
    ]
    for command in ("collect", "audit"):
        code, text = invoke([command, *args])
        assert code == 2  # Valid TEST contract is not full physical acceptance.
        report = json.loads(text)
        assert report["evidence_validation"] == "contract_valid_integration_pending"
        assert report["accepted"] is False
        assert len(report["attempted_rows"]) == 14
        assert len(report["missing_ids"]) == 70
    paths["trusted"].write_text("{}")
    assert invoke(["collect", *args])[0] == 2


def test_analyze_json_csv_md_and_formula_escaping(tmp_path):
    cli = cli_api()
    assert cli is not None, "Standalone analysis/report CLI missing"
    h = api()
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(full_schedule()))
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps({"artifact_kind": "TEST_FIXTURE_NOT_MODEL_RESULTS", "rows": timing_rows()})
    )
    out = tmp_path / "report"
    code, text = invoke(
        [
            "analyze",
            "--schedule",
            str(schedule_path),
            "--schedule-sha256",
            h.sha(schedule_path.read_bytes()),
            "--input",
            str(rows_path),
            "--input-sha256",
            h.sha(rows_path.read_bytes()),
            "--out-dir",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(text)["statistical_targets_met"] is True
    assert (out / "report.csv").exists() and (out / "report.md").exists()
    assert "not confidence" in (out / "report.md").read_text()
    assert cli.csv_safe(" =SUM(A1:A9)").startswith("'")
    assert cli.csv_safe("\t+cmd").startswith("'")
    assert cli.csv_safe("ordinary") == "ordinary"


def test_execute_requires_parent_guarded_supervisor(tmp_path):
    cli = cli_api()
    assert cli is not None, "Guarded supervisor boundary missing"
    code, text = invoke(["plan", "--execute"])
    assert code == 2
    assert "parent_only_guarded_supervisor_unavailable" in text


def test_frozen_schedule_rejects_bool_as_repeat_even_when_hash_pinned(tmp_path):
    schedule = full_schedule()
    schedule["batches"][0]["requests"][0]["repeat"] = False
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(schedule))
    with pytest.raises(ValueError):
        cli_api().bound_schedule(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_nondict_surplus_rows_are_not_silently_discarded(tmp_path):
    h = api()
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": [*timing_rows(), "SURPLUS"]}))
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps(full_schedule()))
    code, result = invoke(
        [
            "analyze",
            "--input",
            str(path),
            "--input-sha256",
            h.sha(path.read_bytes()),
            "--schedule",
            str(schedule),
            "--schedule-sha256",
            h.sha(schedule.read_bytes()),
        ]
    )
    assert code == 2
    assert json.loads(result)["accepted"] is False
