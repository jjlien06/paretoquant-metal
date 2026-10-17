"""CPU-only noncritical policy: no MLX, sysctl, network, child or signals."""

from copy import deepcopy
from types import SimpleNamespace as S

import pressure_guard as pg
import pytest
from test_pressure_v2 import healthy, trial_receipts, trusted_status


def noncritical():
    receipt, binding, result = healthy()
    receipt.update(schema="qwen38-owned-pressure-v3", policy="noncritical", allowed_levels=[1, 2])
    receipt["samples"][1]["level"] = 2
    result.update(pressure_policy="noncritical", require_normal_pressure=False)
    return receipt, binding, result


def test_optin_warning_is_sampled_and_externally_verified_without_relabeling():
    options = S(pressure_policy="noncritical", require_normal_pressure=False)
    assert pg.policy_name(options) == "noncritical"
    assert pg.pressure_required(options) is True
    assert pg.pressure_required(vars(options)) is True
    now = [100.0]
    values = iter([1, 2, 2])
    guard = pg.PressureGuard(
        False, policy=pg.policy_name(options), sampler=lambda: next(values), clock=lambda: now[0]
    )
    _, binding, result = noncritical()
    for stage, elapsed in [("pre_spawn", 0.05), ("owned_child", 0.5), ("post_child", 1.0)]:
        now[0] = 100.0 + elapsed
        guard.sample(stage, force=True)
    receipt = guard.receipt(binding=binding)
    assert receipt["schema"] == "qwen38-owned-pressure-v3"
    assert receipt["required"] is True
    assert receipt["policy"] == "noncritical" and receipt["allowed_levels"] == [1, 2]
    assert [s["level"] for s in receipt["samples"]] == [1, 2, 2]
    assert pg.verify_pressure(
        receipt, expected_binding=binding, actor_result=result, expected_policy="noncritical"
    )
    with pytest.raises(ValueError):
        pg.verify_pressure(receipt, expected_binding=binding, actor_result=result)


@pytest.mark.parametrize(
    "options", [{}, S(), {"require_normal_pressure": False}, S(require_normal_pressure=True)]
)
def test_legacy_helper_default_is_normal(options):
    assert pg.policy_name(options) == "normal"
    expected = (
        options.get("require_normal_pressure", False)
        if isinstance(options, dict)
        else getattr(options, "require_normal_pressure", False)
    )
    assert pg.pressure_required(options) is expected


@pytest.mark.parametrize(
    "options",
    [
        {"pressure_policy": "NONCRITICAL"},
        {"pressure_policy": None},
        {"pressure_policy": True},
        {"pressure_policy": "noncritical", "require_normal_pressure": True},
        {"pressure_policy": "noncritical", "require_normal_pressure": 0},
        {"require_normal_pressure": 1},
    ],
)
def test_helpers_reject_invalid_or_contradictory_policy(options):
    for value in (options, S(**options)):
        with pytest.raises(ValueError):
            pg.policy_name(value)
        with pytest.raises(ValueError):
            pg.pressure_required(value)


@pytest.mark.parametrize("level", [None, True, False, 0, 3, 4, "2", 2.0, [], {}])
def test_noncritical_guard_rejects_critical_and_malformed(level):
    guard = pg.PressureGuard(False, policy="noncritical", sampler=lambda: level)
    with pytest.raises(pg.PressureFailure):
        guard.sample("pre_spawn", force=True)
    assert len(guard.receipt()["samples"]) == 1


def test_noncritical_cannot_suppress_measurement_failure():
    def missing():
        raise TimeoutError("fixture")

    guard = pg.PressureGuard(False, policy="noncritical", sampler=missing)
    with pytest.raises(pg.PressureFailure):
        guard.sample("pre_spawn", force=True)
    assert guard.receipt()["samples"][0]["error_type"] == "TimeoutError"


@pytest.mark.parametrize(
    "mutation",
    [
        "normal_schema",
        "missing_policy",
        "normal_policy",
        "missing_allowed",
        "forged_allowed",
        "bool_allowed",
        "missing_actor_policy",
        "normal_actor_policy",
        "legacy_flag",
        "gap",
        "replay",
        "critical",
        "missing_level",
        "bool_level",
        "malformed_level",
        "missing_required",
        "missing_samples",
    ],
)
def test_noncritical_verifier_retains_policy_identity_and_sampling_gates(mutation):
    receipt, binding, result = noncritical()
    if mutation == "normal_schema":
        receipt["schema"] = pg.SCHEMA
    elif mutation == "missing_policy":
        del receipt["policy"]
    elif mutation == "normal_policy":
        receipt["policy"] = "normal"
    elif mutation == "missing_allowed":
        del receipt["allowed_levels"]
    elif mutation == "forged_allowed":
        receipt["allowed_levels"] = [1, 2, 4]
    elif mutation == "bool_allowed":
        receipt["allowed_levels"] = [True, 2]
    elif mutation == "missing_actor_policy":
        del result["pressure_policy"]
    elif mutation == "normal_actor_policy":
        result["pressure_policy"] = "normal"
    elif mutation == "legacy_flag":
        result["require_normal_pressure"] = True
    elif mutation == "gap":
        receipt["samples"][-1]["elapsed_seconds"] += 2
    elif mutation == "replay":
        receipt["binding"]["nonce"] = "b" * 32
    elif mutation == "critical":
        receipt["samples"][1]["level"] = 4
    elif mutation == "missing_level":
        del receipt["samples"][1]["level"]
    elif mutation == "bool_level":
        receipt["samples"][1]["level"] = True
    elif mutation == "malformed_level":
        receipt["samples"][1]["level"] = "2"
    elif mutation == "missing_required":
        del receipt["required"]
    elif mutation == "missing_samples":
        receipt["samples"] = []
    with pytest.raises(ValueError):
        pg.verify_pressure(
            receipt, expected_binding=binding, actor_result=result, expected_policy="noncritical"
        )


def test_normal_stays_v2_and_cannot_admit_forged_warning():
    receipt, binding, result = healthy()
    assert pg.verify_pressure(receipt, expected_binding=binding, actor_result=result)
    default = pg.PressureGuard(True, sampler=lambda: 2)
    with pytest.raises(pg.PressureFailure):
        default.sample("pre_spawn", force=True)
    assert default.receipt()["schema"] == pg.SCHEMA
    assert "policy" not in default.receipt() and "allowed_levels" not in default.receipt()
    receipt.update(policy="noncritical", allowed_levels=[1, 2])
    receipt["samples"][1]["level"] = 2
    with pytest.raises(ValueError):
        pg.verify_pressure(receipt, expected_binding=binding, actor_result=result)


def test_external_noncritical_policy_binds_startup_and_all_three_owners():
    results, bindings = trial_receipts()
    for result in results:
        result.update(pressure_policy="noncritical", require_normal_pressure=False)
        result["pressure"].update(
            schema=pg.NONCRITICAL_SCHEMA, policy="noncritical", allowed_levels=[1, 2]
        )
        result["pressure"]["samples"][1]["level"] = 2
    for result, binding in zip(results, bindings):
        status = trusted_status(binding, result)
        status["startup"].update(pressure_policy="noncritical", require_normal_pressure=False)
        kwargs = {k: binding[k] for k in ("nonce", "host_role", "actor_role")}
        assert (
            pg.expected_pressure_binding(status, **kwargs, expected_policy="noncritical") == binding
        )
        for record in ("startup", "result"):
            for key, value in (("pressure_policy", "normal"), ("require_normal_pressure", True)):
                bad = deepcopy(status)
                bad[record][key] = value
                with pytest.raises(ValueError):
                    pg.expected_pressure_binding(bad, **kwargs, expected_policy="noncritical")
    assert pg.verify_actor_pressures(
        results, expected_bindings=bindings, expected_policy="noncritical"
    )
    with pytest.raises(ValueError):
        pg.verify_actor_pressures(results, expected_bindings=bindings)
    with pytest.raises(ValueError):
        pg.verify_actor_pressures(
            [results[0]] * 3, expected_bindings=[bindings[0]] * 3, expected_policy="noncritical"
        )


@pytest.mark.parametrize(
    "host,role", [("local", "rank0"), ("remote", "rank1"), ("remote", "server")]
)
def test_actual_actor_noncritical_monitors_and_binds_truthful_policy(
    monkeypatch, tmp_path, host, role
):
    import json
    import os

    import remote_actor as actor

    now = [100.0]
    monkeypatch.setattr(actor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(actor.signal, "signal", lambda *a: None)
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

    def sampler():
        now[0] += 0.01
        return 2

    def tick(child):
        now[0] += 0.1
        child.returncode = 0

    monkeypatch.setattr(actor.subprocess, "Popen", popen)
    monkeypatch.setattr(actor, "terminate", lambda owned: dict(child_reaped=True, group_gone=True))
    try:
        result = actor.run_owned(
            ["MOCK"],
            tmp_path / "job",
            deadline=2,
            tick=tick,
            nonce="a" * 32,
            host_role=host,
            actor_role=role,
            pressure_policy="noncritical",
            require_normal_pressure=False,
            pressure_sampler=sampler,
        )
    finally:
        for fd in ends:
            os.close(fd)
    startup = json.loads((tmp_path / "job/startup.json").read_text())
    assert result["reason"] == "completed"
    for record in (startup, result):
        assert (
            record["pressure_policy"] == "noncritical"
            and record["require_normal_pressure"] is False
        )
    assert all(s["level"] == 2 for s in result["pressure"]["samples"])
    status = dict(
        nonce="a" * 32,
        startup=startup,
        pids=json.loads((tmp_path / "job/pids.json").read_text()),
        result=result,
    )
    pg.expected_pressure_binding(
        status, nonce="a" * 32, host_role=host, actor_role=role, expected_policy="noncritical"
    )


def test_actor_job_admission_carries_policy_and_cannot_disable_monitor():
    from pathlib import Path

    import remote_actor as actor

    root = Path(actor.__file__).parent
    base = dict(
        entrypoint="probe.py",
        argv=[],
        nonce="a" * 32,
        deadline=2,
        pressure_policy="noncritical",
        require_normal_pressure=False,
        host_role="local",
        actor_role="rank0",
    )
    assert actor.validate_job(base, root) == base
    for change in (
        {"require_normal_pressure": True},
        {"pressure_policy": "off"},
        {"actor_role": "server"},
        {"host_role": "remote"},
    ):
        with pytest.raises(ValueError):
            actor.validate_job({**base, **change}, root)
    for key in ("host_role", "actor_role"):
        bad = dict(base)
        del bad[key]
        with pytest.raises(ValueError):
            actor.validate_job(bad, root)


def test_actual_actor_main_forwards_noncritical_descriptor(monkeypatch, tmp_path):
    import sys
    from pathlib import Path

    import remote_actor as actor

    job = dict(
        entrypoint="stream_server.py",
        argv=[],
        nonce="a" * 32,
        deadline=2,
        pressure_policy="noncritical",
        require_normal_pressure=False,
        host_role="remote",
        actor_role="server",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["remote_actor.py", "--root", str(Path(actor.__file__).parent), "--jobs", str(tmp_path)],
    )
    monkeypatch.setattr(sys, "stdin", S(fileno=lambda: 0))
    monkeypatch.setattr(actor, "read_job", lambda fd: job)
    monkeypatch.setattr(actor.os, "set_blocking", lambda *a: None)
    seen = []

    def run(*a, **kw):
        seen.append(kw)
        return dict(reason="completed", cleanup=dict(group_gone=True))

    monkeypatch.setattr(actor, "run_owned", run)
    assert actor.main() == 0
    assert (
        seen[0]["pressure_policy"] == "noncritical" and seen[0]["require_normal_pressure"] is False
    )


def noncritical_argv():
    from test_overlap_cli import argv

    return [v for v in argv() if v != "--require-normal-pressure"] + [
        "--pressure-policy",
        "noncritical",
    ]


def test_supervisor_noncritical_cli_reaches_both_probe_options_and_dryrun(monkeypatch, capsys):
    import json
    import sys

    import supervise

    args = supervise.parser().parse_args(noncritical_argv())
    supervise.validate_args(args)
    assert args.require_normal_pressure is False
    for rank in (0, 1):
        opts = supervise.probe_options(args, rank=rank)
        assert opts[opts.index("--pressure-policy") + 1] == "noncritical"
        assert "--require-normal-pressure" not in opts
    monkeypatch.setattr(sys, "argv", ["supervise.py", *noncritical_argv()])
    monkeypatch.setattr(supervise, "run", lambda *a, **kw: pytest.fail("dryrun entered hardware"))
    assert supervise.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pressure_policy"] == "noncritical" and out["require_normal_pressure"] is False


def test_overlap_effective_guard_required_and_policy_conflict_rejected():
    from overlap_prefill import validate_modes

    good = S(
        overlapped_prefill=True,
        hidden_only_prefill=True,
        prefix_rollback=True,
        hidden_gate_chunk_size=256,
        pressure_policy="noncritical",
        require_normal_pressure=False,
    )
    validate_modes(good)
    good.require_normal_pressure = True
    with pytest.raises(ValueError, match="conflict"):
        validate_modes(good)
    good.overlapped_prefill = False
    with pytest.raises(ValueError, match="conflict"):
        validate_modes(good)
    good.pressure_policy = "normal"
    good.require_normal_pressure = False
    good.overlapped_prefill = True
    with pytest.raises(ValueError, match="watchdog"):
        validate_modes(good)


def test_probe_process_start_reports_noncritical_without_normal_claim(monkeypatch, capsys):
    import sys

    import probe

    reports = []
    monkeypatch.setattr(
        sys, "argv", ["probe.py", "--mode", "collective", "--pressure-policy", "noncritical"]
    )
    monkeypatch.setattr(probe, "phase_reporter", lambda **kw: reports.append)
    monkeypatch.setattr(probe, "collective", lambda: dict(mode="collective", mocked=True))
    probe.main()
    import json

    output = json.loads(capsys.readouterr().out)
    assert output["pressure_policy"] == "noncritical" and output["require_normal_pressure"] is False
    assert len(reports) == 1
    assert reports[0]["phase"] == "process_start"
    assert (
        reports[0]["pressure_policy"] == "noncritical"
        and reports[0]["require_normal_pressure"] is False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe.py",
            "--mode",
            "collective",
            "--pressure-policy",
            "noncritical",
            "--require-normal-pressure",
        ],
    )
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    assert len(reports) == 1


def test_probe_verification_producer_carries_policy_cpu_mocked_model_seams(monkeypatch):
    import sys
    import types

    import mtp_runtime

    for name, fields in {
        "mlx_backend": ("MTPBackend", "Rank1Target"),
        "mtp_head": ("load_trusted_head",),
    }.items():
        mod = types.ModuleType(name)
        for field in fields:
            setattr(mod, field, lambda *a, **kw: pytest.fail("model seam called"))
        monkeypatch.setitem(sys.modules, name, mod)
    events = []
    args = S(
        verifier_mode="native",
        verification_gate_only=True,
        target_revision="mock",
        pressure_policy="noncritical",
        require_normal_pressure=False,
    )
    result = mtp_runtime.run_loaded(
        None, None, [], args, group=None, small={}, eval_groups=[], report=events.append
    )
    for record in (result["verification"], events[0]):
        assert (
            record["pressure_policy"] == "noncritical"
            and record["require_normal_pressure"] is False
        )


def test_noncritical_completed_controller_enforces_external_policy_and_all_actors(monkeypatch):
    import supervise

    monkeypatch.setattr(supervise, "verify_trial", lambda *a, **kw: {})
    monkeypatch.setattr(supervise, "verify_mtp_evidence", lambda *a, **kw: {})
    args = S(
        pressure_policy="noncritical",
        require_normal_pressure=False,
        benchmark_schedule=False,
        repeats=1,
        eval_group_size=32,
        eval_group_schedule=None,
        execution="phased",
        verification_gate_only=True,
    )
    results, bindings = trial_receipts()
    common = dict(
        ranks=[],
        server={},
        weight_files=[],
        args=args,
        phases={},
        source_sha256={},
        expected_inventory={},
    )
    with pytest.raises(ValueError):
        supervise.verify_completed_trial(
            receipts=results, expected_pressure_bindings=bindings, **common
        )
    for result in results:
        result.update(pressure_policy="noncritical", require_normal_pressure=False)
        result["pressure"].update(
            schema=pg.NONCRITICAL_SCHEMA, policy="noncritical", allowed_levels=[1, 2]
        )
        result["pressure"]["samples"][1]["level"] = 2
    assert (
        supervise.verify_completed_trial(
            receipts=results, expected_pressure_bindings=bindings, **common
        )
        == {}
    )
    for idx in range(3):
        bad = deepcopy(results)
        bad[idx]["pressure"]["samples"][1]["level"] = 4
        with pytest.raises(ValueError):
            supervise.verify_completed_trial(
                receipts=bad, expected_pressure_bindings=bindings, **common
            )
    with pytest.raises(ValueError):
        supervise.verify_completed_trial(receipts=results, **common)


def test_noncritical_workload_verifier_uses_admitted_policy_and_bindings(monkeypatch):
    import workload_evidence
    from remote_actor import SOURCES
    from test_workload_schedule import schedule

    class Observed(Exception):
        pass

    receipts, bindings = trial_receipts()

    def check(receipts, *, expected_bindings=None, expected_policy="normal"):
        assert expected_bindings == bindings and expected_policy == "noncritical"
        raise Observed()

    monkeypatch.setattr(pg, "verify_actor_pressures", check)
    args = S(
        benchmark_schedule_sha256="a" * 64,
        repeats=2,
        pressure_policy="noncritical",
        require_normal_pressure=False,
    )
    with pytest.raises(Observed):
        workload_evidence.verify_schedule_trial(
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
            expected_pressure_bindings=bindings,
        )


def test_controller_run_noncritical_binds_both_hosts_including_server(monkeypatch, tmp_path):
    import hashlib
    import json
    from pathlib import Path

    import remote_actor
    import supervise

    args = supervise.parser().parse_args(noncritical_argv())
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
        result.update(pressure_policy="noncritical", require_normal_pressure=False)
        result["pressure"].update(
            schema=pg.NONCRITICAL_SCHEMA, policy="noncritical", allowed_levels=[1, 2]
        )
        result["pressure"]["samples"][1]["level"] = 2
        result["pressure"]["binding"] = deepcopy(binding)
        result["cleanup"] = dict(child_reaped=True, group_gone=True)
        assert job["pressure_policy"] == "noncritical" and job["require_normal_pressure"] is False
        out = trusted_status(binding, result)
        out["startup"].update(pressure_policy="noncritical", require_normal_pressure=False)
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
        assert verify_actor_pressures(
            kwargs["receipts"], expected_bindings=expected, expected_policy="noncritical"
        )
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
    assert result["pressure_policy"] == "noncritical" and result["require_normal_pressure"] is False
    startup = json.loads((Path(args.output) / "startup.json").read_text())
    assert (
        startup["pressure_policy"] == "noncritical" and startup["require_normal_pressure"] is False
    )
    assert len(verified) == 1
    assert {b["nonce"] for b in verified[0]} == {job["nonce"] for job in captures.values()}


def numerical_envelopes():
    """Synthetic dictionaries test verifier policy, NOT actual numerical execution."""
    args = S(
        pressure_policy="noncritical",
        require_normal_pressure=False,
        eval_group_size=1,
        eval_group_schedule=None,
        repeats=1,
        rank0_verifier="native",
        rank1_verifier="native",
        verification_gate_only=True,
        target_config_sha256="a" * 64,
        target_index_sha256="b" * 64,
        target_revision="fixture",
        prefix_rollback=False,
    )
    flags = dict(pressure_policy="noncritical", require_normal_pressure=False)
    ranks, phases = [], {}
    for rid in (0, 1):
        window = dict(
            argmax_equal=True,
            cache_pass=True,
            logits=dict(strict_pass=True),
            sequential_argmax=[1],
            batch_argmax=[1],
            caches=[dict(layer=0, strict_pass=True)],
        )
        numerical = dict(
            passed=True,
            mode="native",
            atol=1e-6,
            rtol=1e-5,
            windows=[deepcopy(window) for _ in range(3)],
            owned_layers=[0, 1],
        )
        gate = dict(
            passed=True,
            mode="native",
            same_loaded_shards=True,
            gate_only=True,
            groups={1: {"small": deepcopy(numerical), "full": deepcopy(numerical)}},
            **flags,
        )
        ranks.append(
            dict(
                rank=rid,
                verification=gate,
                target_binding={"config.json": "a" * 64, "model.safetensors.index.json": "b" * 64},
                target_revision="fixture",
                verifier_mode="native",
                layer_start=0,
                layer_end=1,
                **flags,
            )
        )
        phases[rid] = [
            dict(phase="process_start", **flags),
            dict(phase="numerical_admission_done", **flags),
        ]
    return ranks, phases, args


@pytest.mark.parametrize(
    "record", ["rank", "verification", "process_start", "numerical_admission_done"]
)
@pytest.mark.parametrize(
    "mutation", ["missing_policy", "wrong_policy", "legacy_true", "legacy_missing"]
)
def test_live_probe_verifier_rejects_unbound_or_conflicting_policy(record, mutation):
    import supervise

    ranks, phases, args = numerical_envelopes()
    assert supervise.verify_mtp_evidence(ranks, phases, args)["strict_numerical_admission"]
    for rid in (0, 1):
        bad_ranks, bad_phases = deepcopy(ranks), deepcopy(phases)
        target = (
            bad_ranks[rid]
            if record == "rank"
            else bad_ranks[rid]["verification"]
            if record == "verification"
            else next(e for e in bad_phases[rid] if e["phase"] == record)
        )
        if mutation == "missing_policy":
            del target["pressure_policy"]
        elif mutation == "wrong_policy":
            target["pressure_policy"] = "normal"
        elif mutation == "legacy_true":
            target["require_normal_pressure"] = True
        else:
            del target["require_normal_pressure"]
        with pytest.raises(ValueError, match="pressure"):
            supervise.verify_mtp_evidence(bad_ranks, bad_phases, args)


def test_noncritical_does_not_weaken_numerical_tolerances():
    import supervise

    ranks, phases, args = numerical_envelopes()
    for field, value in (("atol", 1e-5), ("rtol", 1e-4)):
        bad = deepcopy(ranks)
        bad[1]["verification"]["groups"][1]["full"][field] = value
        with pytest.raises(ValueError, match="strict gate"):
            supervise.verify_mtp_evidence(bad, phases, args)


@pytest.mark.parametrize("cause", ["stop", "deadline", "both"])
def test_noncritical_retains_pre_popen_stop_and_deadline(monkeypatch, tmp_path, cause):
    import remote_actor as actor
    from test_pressure_v2 import test_prelaunch_sampler_cannot_spawn_expired_or_stopped_job

    real_run = actor.run_owned

    def noncritical_run(*a, **kw):
        kw.update(pressure_policy="noncritical", require_normal_pressure=False)
        return real_run(*a, **kw)

    monkeypatch.setattr(actor, "run_owned", noncritical_run)
    monkeypatch.setattr(actor.signal, "signal", lambda *a: None)
    test_prelaunch_sampler_cannot_spawn_expired_or_stopped_job(monkeypatch, tmp_path, cause)


@pytest.mark.parametrize("cause", ["stop", "deadline"])
def test_noncritical_retains_bootstrap_gate_deadline(monkeypatch, tmp_path, cause):
    import remote_actor as actor
    from test_pressure_v2 import test_gated_bootstrap_is_not_released_after_admission_expires

    real_run = actor.run_owned

    def noncritical_run(*a, **kw):
        kw.update(pressure_policy="noncritical", require_normal_pressure=False)
        return real_run(*a, **kw)

    monkeypatch.setattr(actor, "run_owned", noncritical_run)
    monkeypatch.setattr(actor.signal, "signal", lambda *a: None)
    test_gated_bootstrap_is_not_released_after_admission_expires(monkeypatch, tmp_path, cause)


def test_noncritical_warning_then_critical_stops_only_owned_unreaped_child(monkeypatch, tmp_path):
    import os

    import remote_actor as actor

    monkeypatch.setattr(actor.signal, "signal", lambda *a: None)
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
    values = iter((2, 4))
    try:
        result = actor.run_owned(
            ["CPU-fixture"],
            tmp_path / "job",
            deadline=2,
            require_normal_pressure=False,
            pressure_policy="noncritical",
            pressure_sampler=lambda: next(values),
        )
    finally:
        for fd in ends:
            os.close(fd)
    assert killed == [child]
    assert result["reason"] == "pressure"
    assert [s["level"] for s in result["pressure"]["samples"]] == [2, 4]
    assert result["cleanup"]["child_reaped"]


def test_overlap_live_policy_is_noncritical_not_forged_normal(monkeypatch):
    """Policy-only envelope test: numerical producer seams are explicitly mocked."""
    import hidden_prefill_gate
    import overlap_gate
    import supervise

    ranks, phases, args = numerical_envelopes()
    args.overlapped_prefill = args.hidden_only_prefill = args.prefix_rollback = True
    args.hidden_gate_chunk_size = 256
    monkeypatch.setattr(overlap_gate, "metadata_layout", lambda *a, **kw: {})
    monkeypatch.setattr(overlap_gate, "verify_overlap_gate", lambda *a, **kw: None)
    monkeypatch.setattr(hidden_prefill_gate, "verify_prefill_evidence", lambda *a, **kw: None)
    monkeypatch.setattr(supervise, "verify_prefix_evidence", lambda *a, **kw: None)
    for rid, rank in enumerate(ranks):
        for record in (rank, rank["verification"], *phases[rid]):
            record.update(
                overlapped_prefill=True,
                hidden_only_prefill=True,
                prefix_rollback=True,
                hidden_gate_chunk_size=256,
                prefill_modes=["serial", "overlap"],
            )
        pair = rank["verification"]["groups"][1]
        pair["peer"] = {}
        for scope in ("small", "full"):
            hidden = dict(
                passed=True,
                same_process=True,
                schema="MOCKED_POLICY_ONLY",
                chunk_size=256,
                prompt_lengths=[513],
                owned_layers=[0, 1],
                rank=0 if scope == "small" else rid,
            )
            overlap = dict(
                schema="MOCKED_POLICY_ONLY",
                passed=True,
                same_process=True,
                physical_gpu=False,
                scope=scope,
                modes=["serial", "overlap"],
                chunk_size=256,
                prompt_lengths=[513],
                eval_group_size=1,
            )
            pair[scope].update(hidden_only_prefill=hidden, overlapped_prefill=overlap)
            pair["peer"][scope] = dict(
                hidden_only_prefill=deepcopy(hidden), overlapped_prefill=deepcopy(overlap)
            )
    assert supervise.verify_mtp_evidence(ranks, phases, args)["strict_numerical_admission"]
    args.pressure_policy = "normal"
    with pytest.raises(ValueError, match="watchdog"):
        supervise.verify_mtp_evidence(ranks, phases, args)
