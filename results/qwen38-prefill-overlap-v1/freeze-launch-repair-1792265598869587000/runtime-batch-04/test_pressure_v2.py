"""V2 CPU pressure regression tests; all children/signals/sysctl are mocked."""

from copy import deepcopy
from types import SimpleNamespace as S

import pytest
from pressure_guard import verify_pressure


def healthy():
    binding = dict(
        host_role="local", actor_role="rank0", nonce="a" * 32, actor_pid=123, child_pid=456
    )
    result = dict(
        reason="completed",
        require_normal_pressure=True,
        **binding,
        actor_started_monotonic=100.0,
        elapsed=1.0,
        execution=dict(child_start_seconds=0.1, gate_release_seconds=0.2, child_end_seconds=0.9),
    )
    receipt = dict(
        schema="qwen38-owned-pressure-v2",
        required=True,
        interval_seconds=0.5,
        sampler_timeout_seconds=0.25,
        sample_cap=2048,
        binding=deepcopy(binding),
        actor_started_monotonic=100.0,
        samples=[
            dict(stage="pre_spawn", elapsed_seconds=0.05, level=1),
            dict(stage="owned_child", elapsed_seconds=0.5, level=1),
            dict(stage="post_child", elapsed_seconds=1.0, level=1),
        ],
    )
    result["pressure"] = receipt
    return receipt, binding, result


def test_bound_receipt_covers_actual_execution_and_actor_duration():
    receipt, binding, result = healthy()
    assert verify_pressure(receipt, expected_binding=binding, actor_result=result)
    # Valid source-copy/setup time is not mistaken for child runtime.
    shifted = deepcopy(result)
    shifted["elapsed"] += 10
    for field in shifted["execution"]:
        shifted["execution"][field] += 10
    for sample in shifted["pressure"]["samples"]:
        sample["elapsed_seconds"] += 10
    assert verify_pressure(shifted["pressure"], expected_binding=binding, actor_result=shifted)
    for field, value in (
        ("elapsed", 900.0),
        (
            "execution",
            dict(child_start_seconds=0.1, gate_release_seconds=0.2, child_end_seconds=899.9),
        ),
    ):
        bad = deepcopy(result)
        bad[field] = value
        with pytest.raises(ValueError):
            verify_pressure(receipt, expected_binding=binding, actor_result=bad)


def test_historical_unbound_short_receipt_is_not_current_proof():
    old = dict(
        schema="qwen38-owned-pressure-v1",
        required=True,
        interval_seconds=0.5,
        sampler_timeout_seconds=0.25,
        sample_cap=2048,
        samples=[
            dict(stage="pre_spawn", elapsed_seconds=0.0, level=1),
            dict(stage="post_child", elapsed_seconds=0.01, level=1),
        ],
    )
    with pytest.raises(ValueError):
        verify_pressure(old)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_timeout",
        "wrong_timeout",
        "bool_cap",
        "unknown_stage",
        "duplicate_boundary",
        "negative_time",
        "zero_duration",
        "nan_time",
        "infinite_time",
        "bool_time",
        "non_dict_sample",
        "missing_binding",
        "replayed_nonce",
        "wrong_host",
        "wrong_role",
        "wrong_actor_pid",
        "wrong_child_pid",
        "bool_pid",
        "empty_binding",
        "zero_elapsed",
        "bad_execution",
        "bad_origin",
        "late_start",
        "gap",
        "duplicate_time",
    ],
)
def test_v2_grammar_identity_and_coverage_fail_closed(mutation):
    receipt, binding, result = healthy()
    if mutation == "missing_timeout":
        del receipt["sampler_timeout_seconds"]
    elif mutation == "wrong_timeout":
        receipt["sampler_timeout_seconds"] = 60
    elif mutation == "bool_cap":
        receipt["sample_cap"] = True
    elif mutation == "unknown_stage":
        receipt["samples"][1]["stage"] = "OTHER_JOB"
    elif mutation == "duplicate_boundary":
        receipt["samples"][1]["stage"] = "pre_spawn"
    elif mutation == "negative_time":
        receipt["samples"][0]["elapsed_seconds"] = -0.1
    elif mutation == "zero_duration":
        for sample in receipt["samples"]:
            sample["elapsed_seconds"] = 0.0
    elif mutation == "nan_time":
        receipt["samples"][1]["elapsed_seconds"] = float("nan")
    elif mutation == "infinite_time":
        receipt["samples"][1]["elapsed_seconds"] = float("inf")
    elif mutation == "bool_time":
        receipt["samples"][0]["elapsed_seconds"] = False
    elif mutation == "non_dict_sample":
        receipt["samples"][1] = []
    elif mutation == "missing_binding":
        del receipt["binding"]
    elif mutation == "replayed_nonce":
        receipt["binding"]["nonce"] = "b" * 32
    elif mutation == "wrong_host":
        receipt["binding"]["host_role"] = "remote"
    elif mutation == "wrong_role":
        receipt["binding"]["actor_role"] = "rank1"
    elif mutation == "wrong_actor_pid":
        receipt["binding"]["actor_pid"] += 1
    elif mutation == "wrong_child_pid":
        receipt["binding"]["child_pid"] += 1
    elif mutation == "bool_pid":
        binding["actor_pid"] = receipt["binding"]["actor_pid"] = result["actor_pid"] = True
    elif mutation == "empty_binding":
        binding.clear()
        receipt["binding"].clear()
    elif mutation == "zero_elapsed":
        result["elapsed"] = 0.0
    elif mutation == "bad_execution":
        result["execution"] = []
    elif mutation == "bad_origin":
        receipt["actor_started_monotonic"] = 99.0
    elif mutation == "late_start":
        receipt["samples"][0]["elapsed_seconds"] = 0.15
    elif mutation == "gap":
        receipt["samples"][-1]["elapsed_seconds"] = 2.0
    elif mutation == "duplicate_time":
        receipt["samples"][1]["elapsed_seconds"] = 0.05
    with pytest.raises(ValueError):
        verify_pressure(receipt, expected_binding=binding, actor_result=result)


@pytest.mark.parametrize("cause", ["stop", "deadline", "both"])
def test_prelaunch_sampler_cannot_spawn_expired_or_stopped_job(monkeypatch, tmp_path, cause):
    import remote_actor as actor

    now = [100.0]
    spawned = []
    monkeypatch.setattr(actor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(actor.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    def sampler():
        now[0] += 0.2 if cause != "stop" else 0.01
        if cause != "deadline":
            (tmp_path / "job" / "stop").write_text("{}")
        return 1

    result = actor.run_owned(
        ["NEVER"],
        tmp_path / "job",
        deadline=0.1,
        require_normal_pressure=True,
        pressure_sampler=sampler,
    )
    assert not spawned
    assert result["child_pid"] is None
    assert result["reason"] == ("deadline" if cause == "deadline" else "signal")


@pytest.mark.parametrize(
    "host_role,actor_role", [("local", "rank0"), ("remote", "rank1"), ("remote", "server")]
)
def test_actual_actor_produces_bound_execution_receipt(
    monkeypatch, tmp_path, host_role, actor_role
):
    import json
    import os

    import remote_actor as actor

    now = [100.0]
    monkeypatch.setattr(actor.time, "monotonic", lambda: now[0])
    ends = []

    class Child:
        pid = 456
        returncode = None

        def __init__(self):
            for name in ("stdout", "stderr"):
                read, write = os.pipe()
                os.close(write)
                setattr(self, name, os.fdopen(read, "rb", buffering=0))

        def poll(self):
            return self.returncode

    child = Child()

    def popen(*args, **kwargs):
        ends.append(os.dup(kwargs["pass_fds"][0]))
        now[0] += 0.02
        return child

    real_write = actor.write_json

    def write(path, value):
        real_write(path, value)
        if path.name == "startup.json":
            now[0] += 10.0  # legitimate source/setup delay
        else:
            now[0] += 0.01

    monkeypatch.setattr(actor, "write_json", write)
    monkeypatch.setattr(actor.subprocess, "Popen", popen)
    monkeypatch.setattr(actor, "terminate", lambda child: dict(child_reaped=True, group_gone=True))

    def sampler():
        now[0] += 0.01
        return 1

    def tick(child):
        now[0] += 0.1
        child.returncode = 0

    try:
        result = actor.run_owned(
            ["MOCK"],
            tmp_path / "job",
            deadline=20,
            tick=tick,
            nonce="a" * 32,
            host_role=host_role,
            actor_role=actor_role,
            require_normal_pressure=True,
            pressure_sampler=sampler,
        )
    finally:
        for fd in ends:
            os.close(fd)
    startup = json.loads((tmp_path / "job" / "startup.json").read_text())
    pids = json.loads((tmp_path / "job" / "pids.json").read_text())
    assert result["reason"] == "completed"
    expected = {k: pids[k] for k in ("host_role", "actor_role", "nonce", "actor_pid", "child_pid")}
    assert startup["actor_started_monotonic"] == result["actor_started_monotonic"] == 100.0
    assert pids["child_start_seconds"] == result["execution"]["child_start_seconds"]
    assert verify_pressure(result["pressure"], expected_binding=expected, actor_result=result)


def trusted_status(binding, result):
    return dict(
        nonce=binding["nonce"],
        result=deepcopy(result),
        startup={
            **binding,
            "require_normal_pressure": True,
            "actor_started_monotonic": result["actor_started_monotonic"],
        },
        pids={
            **binding,
            "actor_started_monotonic": result["actor_started_monotonic"],
            "child_start_seconds": result["execution"]["child_start_seconds"],
        },
    )


def test_expected_binding_comes_from_startup_pids_and_admitted_job():
    from pressure_guard import expected_pressure_binding

    _, binding, result = healthy()
    status = trusted_status(binding, result)
    expected = dict(nonce=binding["nonce"], host_role="local", actor_role="rank0")
    assert expected_pressure_binding(status, **expected) == binding
    for record, key, value in (
        ("startup", "nonce", "b" * 32),
        ("pids", "child_pid", 999),
        ("startup", "actor_pid", 999),
        ("pids", "actor_started_monotonic", 99.0),
        ("pids", "child_start_seconds", 0.3),
    ):
        bad = deepcopy(status)
        bad[record][key] = value
        with pytest.raises(ValueError):
            expected_pressure_binding(bad, **expected)
    with pytest.raises(ValueError):
        expected_pressure_binding(status, **{**expected, "nonce": "c" * 32})


def trial_receipts():
    receipts, bindings = [], []
    for i, (host, role) in enumerate(
        (("remote", "server"), ("remote", "rank1"), ("local", "rank0"))
    ):
        receipt, binding, result = healthy()
        binding.update(
            host_role=host,
            actor_role=role,
            nonce=str(i + 1) * 32,
            actor_pid=100 + i,
            child_pid=200 + i,
        )
        receipt["binding"] = deepcopy(binding)
        result.update(binding)
        receipts.append(result)
        bindings.append(binding)
    return receipts, bindings


def test_controller_rejects_replayed_other_job_and_short_coverage(monkeypatch):
    import supervise

    monkeypatch.setattr(supervise, "verify_trial", lambda *a, **kw: {})
    monkeypatch.setattr(supervise, "verify_mtp_evidence", lambda *a, **kw: {})
    args = S(
        require_normal_pressure=True,
        benchmark_schedule=False,
        repeats=1,
        eval_group_size=32,
        eval_group_schedule=None,
        execution="phased",
        verification_gate_only=True,
    )
    receipts, bindings = trial_receipts()
    common = dict(
        ranks=[],
        server={},
        weight_files=[],
        args=args,
        phases={},
        source_sha256={},
        expected_inventory={},
    )
    assert (
        supervise.verify_completed_trial(
            receipts=receipts, expected_pressure_bindings=bindings, **common
        )
        == {}
    )
    for variant in (receipts[:1] * 3, [dict(r, elapsed=900.0) for r in receipts]):
        with pytest.raises(ValueError):
            supervise.verify_completed_trial(
                receipts=variant, expected_pressure_bindings=bindings, **common
            )
    with pytest.raises(ValueError):
        supervise.verify_completed_trial(receipts=receipts, **common)


@pytest.mark.parametrize("mutation", ["nonce", "host", "actor_pid", "child_pid", "missing"])
def test_three_actor_pressure_layer_rejects_duplicate_and_missing_bindings(mutation):
    from pressure_guard import verify_actor_pressures

    receipts, bindings = trial_receipts()
    assert verify_actor_pressures(receipts, expected_bindings=bindings)
    if mutation == "missing":
        bindings.pop()
    elif mutation == "host":
        bindings[2]["host_role"] = "remote"
    else:
        bindings[1][mutation] = bindings[0][mutation]
        receipts[1][mutation] = bindings[0][mutation]
        receipts[1]["pressure"]["binding"][mutation] = bindings[0][mutation]
    with pytest.raises(ValueError):
        verify_actor_pressures(receipts, expected_bindings=bindings)


@pytest.mark.parametrize("cause", ["stop", "deadline"])
def test_gated_bootstrap_is_not_released_after_admission_expires(monkeypatch, tmp_path, cause):
    import os

    import remote_actor as actor

    now = [100.0]
    gate_released, ends = [], []
    monkeypatch.setattr(actor.time, "monotonic", lambda: now[0])

    class Child:
        pid = 456
        returncode = None

        def __init__(self):
            for name in ("stdout", "stderr"):
                r, w = os.pipe()
                os.close(w)
                setattr(self, name, os.fdopen(r, "rb", buffering=0))

        def poll(self):
            return self.returncode

    child = Child()

    def popen(*args, **kwargs):
        ends.append(os.dup(kwargs["pass_fds"][0]))
        return child

    real_write = actor.write_json

    def write(path, value):
        real_write(path, value)
        now[0] += 0.01
        if path.name == "pids.json":
            if cause == "stop":
                (tmp_path / "job" / "stop").write_text("{}")
            else:
                now[0] += 2.0

    monkeypatch.setattr(actor, "write_json", write)
    monkeypatch.setattr(actor.subprocess, "Popen", popen)
    monkeypatch.setattr(actor.os, "write", lambda fd, data: gate_released.append(data) or len(data))
    monkeypatch.setattr(actor, "terminate", lambda child: dict(child_reaped=True, group_gone=True))
    try:
        result = actor.run_owned(
            ["NEVER"],
            tmp_path / "job",
            deadline=1,
            require_normal_pressure=True,
            pressure_sampler=lambda: 1,
        )
    finally:
        for fd in ends:
            os.close(fd)
    assert gate_released == []
    assert result["reason"] == ("signal" if cause == "stop" else "deadline")
    assert "gate_release_seconds" not in result["execution"]


def test_pressure_job_admission_binds_exact_roles_and_preserves_optout():
    from pathlib import Path

    import remote_actor as actor

    root = Path(actor.__file__).parent
    base = dict(entrypoint="probe.py", argv=[], nonce="a" * 32, deadline=2)
    assert actor.validate_job(base, root) == base  # historical pressure opt-out
    good = {**base, "require_normal_pressure": True, "host_role": "local", "actor_role": "rank0"}
    assert actor.validate_job(good, root) == good
    for change in (
        {"host_role": "private-address"},
        {"actor_role": True},
        {"host_role": "remote"},
        {"actor_role": "rank1"},
        {"entrypoint": "stream_server.py"},
    ):
        with pytest.raises(ValueError):
            actor.validate_job({**good, **change}, root)
    with pytest.raises(ValueError):
        actor.validate_job({**base, "require_normal_pressure": True}, root)
    server = {
        **base,
        "entrypoint": "stream_server.py",
        "require_normal_pressure": True,
        "host_role": "remote",
        "actor_role": "server",
    }
    assert actor.validate_job(server, root) == server


def test_controller_run_sources_pressure_binding_from_job_startup_and_pids(monkeypatch, tmp_path):
    import hashlib
    import json
    from pathlib import Path

    import remote_actor
    import supervise
    from test_overlap_cli import argv

    args = supervise.parser().parse_args(argv())
    args.output = str(tmp_path / "controller")
    meta = tmp_path / "metadata"
    meta.mkdir()
    for name, data in (
        ("config.json", {"model_type": "qwen3_5", "num_hidden_layers": 64}),
        ("model.safetensors.index.json", {"weight_map": {"embed": "ignored"}}),
    ):
        (meta / name).write_text(json.dumps(data))
    args.local_metadata = str(meta)
    args.target_config_sha256 = hashlib.sha256((meta / "config.json").read_bytes()).hexdigest()
    args.target_index_sha256 = hashlib.sha256(
        (meta / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    captures, verified = {}, []

    class FakeController:
        def __init__(self, args):
            self.output = Path(args.output)
            self.output.mkdir()
            self.started = supervise.time.monotonic()
            self.children = {}
            self.control_cleanup = {}

        def check(self):
            pass

        def launch(self, name, command, job=None):
            captures[name] = job
            self.children[name] = S(poll=lambda: 0)

        def call(self, command, cleanup=False):
            hashes = json.loads((self.output / "source_sha256.json").read_text())
            if "--inspect" in command:
                return dict(source_sha256=hashes, available_port=23456)
            if "--status" in command:
                nonce = command[command.index("--status") + 1]
                name = next(k for k, job in captures.items() if job["nonce"] == nonce)
                return status(name)
            return dict(stop_requested=True)

        def close(self):
            return {}

    def status(name):
        job = captures[name]
        role = dict(
            server=("remote", "server"), rank1=("remote", "rank1"), rank0=("local", "rank0")
        )[name]
        assert job.get("host_role") == role[0] and job.get("actor_role") == role[1]
        _, binding, result = healthy()
        i = ("server", "rank1", "rank0").index(name)
        binding.update(
            host_role=role[0],
            actor_role=role[1],
            nonce=job["nonce"],
            actor_pid=100 + i,
            child_pid=200 + i,
        )
        result.update(binding)
        result["pressure"]["binding"] = deepcopy(binding)
        result["cleanup"] = dict(child_reaped=True, group_gone=True)
        out = trusted_status(binding, result)
        out.update(
            source_sha256=json.loads((Path(args.output) / "source_sha256.json").read_text()),
            stdout=json.dumps(dict(rank=i)),
            stderr="",
            ready=dict(address=[args.remote_address, 23457]),
        )
        return out

    def verify(**kwargs):
        from pressure_guard import verify_actor_pressures

        expected = kwargs.get("expected_pressure_bindings")
        assert verify_actor_pressures(kwargs["receipts"], expected_bindings=expected)
        verified.append(expected)
        return {}

    monkeypatch.setattr(supervise, "Controller", FakeController)
    monkeypatch.setattr(supervise, "expected_inventory", lambda *a: {0: [], 1: []})
    monkeypatch.setattr(supervise, "free_port", lambda *a: 23455)
    monkeypatch.setattr(supervise, "listener_closed", lambda *a: True)
    monkeypatch.setattr(remote_actor, "job_status", lambda jobs, nonce, **kw: status("rank0"))
    monkeypatch.setattr(supervise, "verify_completed_trial", verify)
    result = supervise.run(args, transport=lambda args, extra: list(extra))
    assert result["status"] == "success", result
    assert len(verified) == 1
    assert {b["nonce"] for b in verified[0]} == {job["nonce"] for job in captures.values()}


def test_pressure_v2_source_registry_contains_all_new_helpers():
    from remote_actor import SOURCES

    assert "test_pressure_v2.py" in SOURCES


def test_schedule_verifier_requires_and_receives_external_pressure_bindings(monkeypatch):
    import pressure_guard
    import workload_evidence
    from remote_actor import SOURCES
    from test_workload_schedule import schedule

    receipts, bindings = trial_receipts()
    args = S(benchmark_schedule_sha256="a" * 64, repeats=2, require_normal_pressure=True)
    common = dict(
        ranks=[dict(rank=0), dict(rank=1)],
        server={},
        receipts=receipts,
        weight_files=[],
        rows=schedule()["requests"],
        prompt_ids=[[1], [1]],
        args=args,
        phases={0: [], 1: []},
        source_sha256={name: "a" * 64 for name in SOURCES},
        expected_inventory={},
    )

    class Observed(Exception):
        pass

    def check(receipts, *, expected_bindings=None):
        assert expected_bindings == bindings
        raise Observed()

    monkeypatch.setattr(pressure_guard, "verify_actor_pressures", check)
    with pytest.raises(Observed):
        workload_evidence.verify_schedule_trial(**common, expected_pressure_bindings=bindings)
    # Controller must forward the same externally admitted bindings into schedule verification.
    import supervise

    args.benchmark_schedule = True
    monkeypatch.setattr(pressure_guard, "verify_actor_pressures", lambda *a, **kw: True)

    def schedule_check(**kwargs):
        assert kwargs.get("expected_pressure_bindings") == bindings
        raise Observed()

    monkeypatch.setattr(workload_evidence, "verify_schedule_trial", schedule_check)
    with pytest.raises(Observed):
        supervise.verify_completed_trial(**common, expected_pressure_bindings=bindings)


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("binding", "actor_pid", True),
        ("result", "actor_pid", True),
        ("result", "reason", "failure"),
        ("result", "require_normal_pressure", False),
    ],
)
def test_public_pressure_api_rejects_type_coercion_and_unsuccessful_actor(record, key, value):
    receipt, binding, result = healthy()
    binding["actor_pid"] = result["actor_pid"] = receipt["binding"]["actor_pid"] = 1
    target = receipt["binding"] if record == "binding" else result
    target[key] = value
    with pytest.raises(ValueError):
        verify_pressure(receipt, expected_binding=binding, actor_result=result)


def test_900_second_producer_and_exact_gap_limit():
    from pressure_guard import PressureFailure, PressureGuard

    receipt, binding, result = healthy()
    now = [100.0]
    guard = PressureGuard(True, sampler=lambda: 1, clock=lambda: now[0])
    guard.sample("pre_spawn", force=True)
    for i in range(1, 1801):
        now[0] = 100.0 + i * 0.5
        guard.sample("owned_child", force=True)
    now[0] += 0.001
    guard.sample("post_child", force=True)
    result.update(
        elapsed=900.001,
        execution=dict(
            child_start_seconds=0.01, gate_release_seconds=0.02, child_end_seconds=900.0
        ),
    )
    assert verify_pressure(
        guard.receipt(binding=binding), expected_binding=binding, actor_result=result
    )
    now[0] += 1.0
    guard.sample("owned_child", force=True)
    now[0] += 1.00001
    with pytest.raises(PressureFailure):
        guard.sample("owned_child", force=True)


def test_zero_child_interval_and_coerced_clock_origin_are_rejected():
    receipt, binding, result = healthy()
    result["execution"] = dict(
        child_start_seconds=0.1, gate_release_seconds=0.1, child_end_seconds=0.1
    )
    with pytest.raises(ValueError):
        verify_pressure(receipt, expected_binding=binding, actor_result=result)
    receipt, binding, result = healthy()
    result["actor_started_monotonic"] = 1.0
    receipt["actor_started_monotonic"] = True
    with pytest.raises(ValueError):
        verify_pressure(receipt, expected_binding=binding, actor_result=result)
