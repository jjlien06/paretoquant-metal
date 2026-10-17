"""Pressure watchdog CPU fixtures: no sysctl, GPU child or process signals."""

import os

import pytest


def test_initial_unhealthy_actor_never_spawns(monkeypatch, tmp_path):
    import remote_actor as actor

    spawned = []
    monkeypatch.setattr(actor.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    result = actor.run_owned(
        ["GPU-MUST-NOT-SPAWN"],
        tmp_path / "job",
        deadline=2,
        require_normal_pressure=True,
        pressure_sampler=lambda: 2,
    )
    assert not spawned
    assert result["reason"] == "pressure"
    assert result["child_pid"] is None
    assert result["pressure"]["samples"][0]["level"] == 2


def test_later_pressure_stops_only_owned_unreaped_child(monkeypatch, tmp_path):
    import remote_actor as actor

    ends = []

    class Child:
        pid = 314159
        returncode = None

        def __init__(self):
            for name in ("stdout", "stderr"):
                read, write = os.pipe()
                ends.append(write)
                setattr(self, name, os.fdopen(read, "rb", buffering=0))

        def poll(self):
            return self.returncode

    child = Child()
    killed = []

    def popen(*a, **kwargs):
        # The real bootstrap owns an inherited read end until exec.
        ends.append(os.dup(kwargs["pass_fds"][0]))
        return child

    monkeypatch.setattr(actor.subprocess, "Popen", popen)

    def terminate(owned):
        assert owned is child and owned.poll() is None
        killed.append(owned)
        owned.returncode = -15
        return {"child_reaped": True, "group_gone": True, "pending_os_gpu_termination": False}

    monkeypatch.setattr(actor, "terminate", terminate)
    values = iter((1, 4))
    try:
        result = actor.run_owned(
            ["CPU-fixture"],
            tmp_path / "job",
            deadline=2,
            require_normal_pressure=True,
            pressure_sampler=lambda: next(values),
        )
    finally:
        for fd in ends:
            os.close(fd)
    assert killed == [child]
    assert result["reason"] == "pressure"
    assert [s["level"] for s in result["pressure"]["samples"]] == [1, 4]
    assert result["cleanup"]["child_reaped"]


@pytest.mark.parametrize("value", [True, 0, "1", None, 2, 4])
def test_guard_exact_values_fail_closed(value):
    from pressure_guard import PressureFailure, PressureGuard

    guard = PressureGuard(True, sampler=lambda: value)
    with pytest.raises(PressureFailure):
        guard.sample("pre_spawn", force=True)
    assert guard.receipt()["scope"] == "sampled pressure only; no claim between samples"


def test_pressure_cli_descriptor_exact_types_and_both_host_job_wiring():
    from pathlib import Path

    import remote_actor
    import supervise

    assert "--require-normal-pressure" in supervise.parser().format_help()
    source = Path(remote_actor.__file__).read_text()
    assert "require_normal_pressure=job.get(" in source
    controller = Path(supervise.__file__).read_text()
    assert (
        '"require_normal_pressure": getattr(args, "require_normal_pressure", False)' in controller
    )
    job = dict(
        entrypoint="probe.py", argv=[], nonce="a" * 32, deadline=2, require_normal_pressure=1
    )
    with pytest.raises(ValueError, match="pressure"):
        remote_actor.validate_job(job, Path(remote_actor.__file__).parent)


def test_pressure_timeline_missing_periodic_samples_rejected():
    from pressure_guard import PressureGuard, verify_pressure
    from test_pressure_v2 import healthy

    _, binding, result = healthy()
    now = [100.0]
    guard = PressureGuard(True, sampler=lambda: 1, clock=lambda: now[0])
    now[0] = 100.05
    guard.sample("pre_spawn", force=True)
    now[0] = 101.0
    guard.sample("post_child", force=True)
    receipt = guard.receipt(binding=binding)
    verify_pressure(receipt, expected_binding=binding, actor_result=result)
    receipt["samples"][-1]["elapsed_seconds"] += 10
    with pytest.raises(ValueError):
        verify_pressure(receipt, expected_binding=binding, actor_result=result)


def test_live_guard_aborts_when_periodic_observation_is_late():
    from pressure_guard import PressureFailure, PressureGuard

    now = [0.0]
    guard = PressureGuard(True, sampler=lambda: 1, clock=lambda: now[0])
    guard.sample("pre_spawn", force=True)
    now[0] = 1.1
    with pytest.raises(PressureFailure):
        guard.sample("owned_child")


def test_gate_only_controller_requires_both_host_pressure_receipts(monkeypatch):
    from types import SimpleNamespace as S

    import supervise

    monkeypatch.setattr(supervise, "verify_trial", lambda *a, **k: {})
    monkeypatch.setattr(supervise, "verify_mtp_evidence", lambda *a, **k: {})
    args = S(
        require_normal_pressure=True,
        benchmark_schedule=False,
        repeats=1,
        eval_group_size=32,
        eval_group_schedule=None,
        execution="phased",
        verification_gate_only=True,
    )
    with pytest.raises(ValueError, match="pressure"):
        supervise.verify_completed_trial(
            ranks=[],
            server={},
            receipts=[{}] * 3,
            weight_files=[],
            args=args,
            phases={},
            source_sha256={},
            expected_inventory={},
        )


def test_sampler_timeout_and_flag_exact_types():
    from pressure_guard import PressureFailure, PressureGuard

    with pytest.raises(ValueError):
        PressureGuard(1)

    def timeout():
        raise TimeoutError("fixture")

    guard = PressureGuard(True, sampler=timeout)
    with pytest.raises(PressureFailure):
        guard.sample("pre_spawn", force=True)
    assert guard.receipt()["samples"][0]["error_type"] == "TimeoutError"
