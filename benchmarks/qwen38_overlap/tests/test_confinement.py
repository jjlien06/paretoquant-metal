"""CPU-only confinement regression probes; no hardware or network activity."""

import builtins
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

from .test_cli import invoke
from .test_collect import record_fixture


def test_plan_collect_audit_never_spawn_connect_import_mlx_or_write(tmp_path, monkeypatch):
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
    arguments = [
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
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()
    }
    original_import, original_open = builtins.__import__, Path.open

    def denied(*args, **kwargs):
        raise AssertionError("Forbidden model/network/process side effect")

    def cpu_import(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "mlx_lm", "mlx_vlm", "vlm"}:
            return denied()
        return original_import(name, *args, **kwargs)

    def readonly_open(self, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            return denied()
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_import)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(Path, "open", readonly_open)
    assert invoke(["plan"])[0] == 0
    assert invoke(["collect", *arguments])[0] == 2
    assert invoke(["audit", *arguments])[0] == 2
    after = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()
    }
    assert after == before
    assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)


def test_execute_boundary_calls_only_injected_parent_guard():
    calls = []

    def parent_guard(plan):
        calls.append(plan)
        return {"accepted": False, "execution_permitted": False, "test_supervisor_called": True}

    code, text = invoke(["plan", "--execute"], supervisor=parent_guard)
    assert code == 0
    assert json.loads(text)["execution_permitted"] is False
    assert len(calls) == 1
    assert {"--hidden-only-prefill", "--overlapped-prefill", "--require-normal-pressure"} <= set(
        calls[0]["mandatory_supervisor_flags"]
    )
