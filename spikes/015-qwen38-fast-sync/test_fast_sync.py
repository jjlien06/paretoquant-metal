"""CPU-only fast-synchronization contracts; never GPU/performance evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_schedule import args_for


def test_fast_synch_cli_is_explicit_boolean_and_defaults_off():
    from supervise import validate_args

    assert args_for().fast_synch is False
    args = args_for("--fast-synch")
    assert args.fast_synch is True
    validate_args(args)
    for invalid in (None, 0, 1, "1", "false", []):
        args.fast_synch = invalid
        with pytest.raises(ValueError, match="fast_synch"):
            validate_args(args)


@pytest.mark.parametrize("enabled", [False, True])
def test_supervisor_binds_option_and_forwards_only_to_both_ranks(tmp_path, monkeypatch, enabled):
    """Stop at the launch seam; no rank code, SSH or MLX is executed."""
    import supervise
    from remote_actor import source_hashes
    from test_schedule import evidence

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _, _, _, options = evidence()
    names = options["expected_inventory"][0] | options["expected_inventory"][1]
    (metadata / "config.json").write_text(
        json.dumps(dict(model_type="qwen3_5", num_hidden_layers=2))
    )
    (metadata / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model.safetensors" for n in names}})
    )
    args = args_for()
    args.local_metadata, args.output = str(metadata), str(tmp_path / "trial")
    args.reserve_bytes, args.split, args.fast_synch = 0, [1, 1], enabled
    args.local_address = args.remote_address = "127.0.0.1"
    hashes, jobs = source_hashes(supervise.ROOT), {}

    def launch(self, name, argv, *, job=None):
        jobs[name] = job
        if name == "rank0":
            raise RuntimeError("CPU-only stop before any probe execution")
        return SimpleNamespace(poll=lambda: 0)

    def call(self, argv, **kwargs):
        if "--inspect" in argv:
            return {"source_sha256": hashes, "available_port": 12345}
        if "--status" in argv:
            return {"source_sha256": hashes, "ready": {"address": ["127.0.0.1", 12347]}}
        return {}

    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "1")
    monkeypatch.setattr(supervise.Controller, "launch", launch)
    monkeypatch.setattr(supervise.Controller, "call", call)
    monkeypatch.setattr(supervise, "free_port", lambda _: 12346)
    monkeypatch.setattr(supervise, "listener_closed", lambda *_: True)
    result = supervise.run(args, transport=lambda _, argv: argv)
    assert result["status"] == "failed"
    assert result["error_stage"] == "rank_execution"
    assert set(jobs) == {"server", "rank0", "rank1"}
    assert "MLX_METAL_FAST_SYNCH" not in jobs["server"]["env"]
    for rank in (0, 1):
        env = jobs[f"rank{rank}"]["env"]
        assert env["MLX_RANK"] == str(rank)
        assert env.get("MLX_METAL_FAST_SYNCH") == ("1" if enabled else None)
        saved = json.loads((tmp_path / f"trial/rank{rank}.job.json").read_text())
        assert saved["env"].get("MLX_METAL_FAST_SYNCH") == ("1" if enabled else None)
    startup = json.loads((tmp_path / "trial/startup.json").read_text())
    saved = json.loads((tmp_path / "trial/result.json").read_text())
    assert startup["args"]["fast_synch"] is enabled
    assert startup["fast_synch"] is enabled
    assert saved["fast_synch"] is result["fast_synch"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("observed_matches", [False, True])
def test_supervisor_success_gate_uses_trusted_option_not_rank_claims(
    tmp_path, monkeypatch, enabled, observed_matches
):
    """Synthetic complete transport/receipts exercise controller wiring, not hardware."""
    import remote_actor
    import supervise

    ranks, server, receipts, options = fast_evidence(enabled if observed_matches else not enabled)
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    names = options["expected_inventory"][0] | options["expected_inventory"][1]
    (metadata / "config.json").write_text(
        json.dumps(dict(model_type="qwen3_5", num_hidden_layers=2))
    )
    (metadata / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model.safetensors" for n in names}})
    )
    args = args_for()
    args.local_metadata, args.output = str(metadata), str(tmp_path / "trial")
    args.reserve_bytes, args.split, args.fast_synch = 0, [1, 1], enabled
    args.repeats, args.eval_group_size, args.eval_group_schedule = 2, 4, [1, 32]
    args.local_address = args.remote_address = "127.0.0.1"
    jobs = {}

    def launch(self, name, argv, *, job=None):
        jobs[job["nonce"]] = name
        child = SimpleNamespace(poll=lambda: 0)
        self.children[name] = child
        return child

    def status(nonce):
        name = jobs[nonce]
        rank = int(name[-1]) if name != "server" else None
        return dict(
            source_sha256=options["source_sha256"],
            ready={"address": ["127.0.0.1", 12347]},
            result=receipts[0],
            stdout=json.dumps(server if rank is None else ranks[rank]),
            stderr="" if rank is None else "\n".join(map(json.dumps, options["phases"][rank])),
        )

    def call(self, argv, **kwargs):
        if "--inspect" in argv:
            return {"source_sha256": options["source_sha256"], "available_port": 12345}
        if "--status" in argv:
            return status(argv[argv.index("--status") + 1])
        return {}

    monkeypatch.setattr(supervise.Controller, "launch", launch)
    monkeypatch.setattr(supervise.Controller, "call", call)
    monkeypatch.setattr(supervise, "free_port", lambda _: 12346)
    monkeypatch.setattr(supervise, "listener_closed", lambda *_: True)
    monkeypatch.setattr(supervise, "terminate", lambda _: receipts[0]["cleanup"])
    monkeypatch.setattr(remote_actor, "job_status", lambda _, nonce, **kw: status(nonce))
    result = supervise.run(args, transport=lambda _, argv: argv)
    assert result["status"] == ("success" if observed_matches else "failed")
    assert result["fast_synch"] is enabled
    if observed_matches:
        assert result["rank_environment_verified"] is True
    else:
        assert result["error_stage"] == "verification"


def rank_job(rank=0):
    return dict(
        entrypoint="probe.py",
        argv=[],
        env={"MLX_RANK": str(rank), "MLX_METAL_FAST_SYNCH": "1"},
        nonce="a" * 32,
        deadline=3,
        hostfile=[["127.0.0.1:12345"], ["127.0.0.1:12346"]],
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_actor_cli_delivers_exact_child_env_without_ambient_leaks(tmp_path, enabled, rank):
    """Real actor/exec subprocess, but a stdlib-only substitute probe, never MLX."""
    import remote_actor as actor

    root, jobs = tmp_path / "source", tmp_path / "jobs"
    root.mkdir()
    jobs.mkdir()
    for name in actor.SOURCES:
        (root / name).write_text("")
    (root / "probe.py").write_text(
        "import json,os\n"
        "print(json.dumps({k:os.environ.get(k) for k in "
        "('MLX_RANK','MLX_METAL_FAST_SYNCH','UNTRUSTED_ENV','MLX_HOSTFILE')}),flush=True)\n"
    )
    job = rank_job(rank)
    job["source_sha256"] = actor.source_hashes(root)
    if not enabled:
        job["env"].pop("MLX_METAL_FAST_SYNCH")
    result = subprocess.run(
        [sys.executable, actor.__file__, "--root", str(root), "--jobs", str(jobs)],
        input=json.dumps(job) + "\n",
        env={**os.environ, "MLX_METAL_FAST_SYNCH": "1", "UNTRUSTED_ENV": "no"},
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads((jobs / job["nonce"] / "stdout").read_text())
    assert actual["MLX_RANK"] == str(rank)
    assert actual["MLX_METAL_FAST_SYNCH"] == ("1" if enabled else None)
    assert actual["UNTRUSTED_ENV"] is None
    assert Path(actual["MLX_HOSTFILE"]).parent == jobs / job["nonce"]
    receipt = json.loads((jobs / job["nonce"] / "result.json").read_text())
    assert receipt["reason"] == "completed"
    assert receipt["cleanup"]["child_reaped"] and receipt["cleanup"]["group_gone"]


@pytest.mark.parametrize("value", ["0", "false", "true", "", "2", "01", "1\x00", False, 1, None])
def test_actor_rejects_false_unknown_or_malformed_fast_synch_values(value):
    import remote_actor as actor

    job = rank_job()
    job["env"]["MLX_METAL_FAST_SYNCH"] = value
    with pytest.raises(ValueError):
        actor.validate_job(job, Path(actor.__file__).parent)


@pytest.mark.parametrize("tamper", ["unknown", "server", "rankless", "hostless"])
def test_actor_keeps_environment_allowlist_and_rank_scope(tamper):
    import remote_actor as actor

    job = rank_job()
    if tamper == "unknown":
        job["env"]["MLX_METAL_FAST_SYNC"] = "1"
    elif tamper == "server":
        job["entrypoint"] = "stream_server.py"
    elif tamper == "rankless":
        job["env"].pop("MLX_RANK")
    else:
        job.pop("hostfile")
    with pytest.raises(ValueError):
        actor.validate_job(job, Path(actor.__file__).parent)


@pytest.mark.parametrize("value", [None, "1"])
def test_probe_reports_actual_start_and_completion_env_without_gpu(monkeypatch, capsys, value):
    import probe

    if value is None:
        monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)
    else:
        monkeypatch.setenv("MLX_METAL_FAST_SYNCH", value)
    monkeypatch.setattr(sys, "argv", ["probe.py", "--mode", "collective", "--repeats", "1"])
    # A result claiming the opposite setting must be overwritten by actual os.environ.
    monkeypatch.setattr(
        probe,
        "collective",
        lambda: {"unit_fixture": True, "fast_synch": value is None, "environment": {}},
    )
    probe.main()
    captured = capsys.readouterr()
    start, final = json.loads(captured.err), json.loads(captured.out)
    assert start["phase"] == "process_start"
    for record in (start, final):
        assert record["fast_synch"] is (value == "1")
        assert record["environment"] == {"MLX_METAL_FAST_SYNCH": value}


@pytest.mark.parametrize("value", ["0", "false", "true", "", "2", "01", "1 "])
def test_probe_rejects_unknown_environment_before_runtime(monkeypatch, value):
    import probe

    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", value)
    monkeypatch.setattr(sys, "argv", ["probe.py", "--mode", "collective", "--repeats", "1"])

    def forbidden():
        pytest.fail("Unadmitted environment reached runtime")

    monkeypatch.setattr(probe, "collective", forbidden)
    with pytest.raises(ValueError, match="MLX_METAL_FAST_SYNCH"):
        probe.main()


def fast_evidence(enabled):
    from test_schedule import evidence

    ranks, server, receipts, options = evidence()
    observed = {
        "fast_synch": enabled,
        "environment": {"MLX_METAL_FAST_SYNCH": "1" if enabled else None},
    }
    for rank in ranks:
        rank.update(observed)
        options["phases"][rank["rank"]][0].update(observed)
    return ranks, server, receipts, {**options, "fast_synch": enabled}


@pytest.mark.parametrize("enabled", [False, True])
def test_verifier_binds_both_rank_environments_to_trusted_option(enabled):
    from supervise import verify_trial

    ranks, server, receipts, options = fast_evidence(enabled)
    result = verify_trial(ranks, server, receipts, [], **options)
    assert result["fast_synch"] is enabled
    assert result["rank_environment_verified"] is True


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("location", ["start", "result"])
@pytest.mark.parametrize(
    "tamper", ["missing_env", "missing_bool", "opposite", "false", "unknown", "forged"]
)
def test_verifier_rejects_missing_unknown_or_forged_rank_env(enabled, rank, location, tamper):
    from supervise import verify_trial

    ranks, server, receipts, options = fast_evidence(enabled)
    record = ranks[rank] if location == "result" else options["phases"][rank][0]
    if tamper == "missing_env":
        record.pop("environment")
    elif tamper == "missing_bool":
        record.pop("fast_synch")
    elif tamper == "opposite":
        record["fast_synch"] = not enabled
        record["environment"] = {"MLX_METAL_FAST_SYNCH": None if enabled else "1"}
    elif tamper == "false":
        record["environment"] = {"MLX_METAL_FAST_SYNCH": "0"}
    elif tamper == "unknown":
        record["environment"] = {"MLX_METAL_FAST_SYNC": "1"}
    else:
        record["fast_synch"] = int(enabled)
    with pytest.raises(ValueError, match="synch|environment"):
        verify_trial(ranks, server, receipts, [], **options)


@pytest.mark.parametrize("enabled", [False, True])
def test_verifier_rejects_both_ranks_agreeing_on_untrusted_option(enabled):
    from supervise import verify_trial

    ranks, server, receipts, options = fast_evidence(not enabled)
    options["fast_synch"] = enabled
    with pytest.raises(ValueError, match="synch|environment"):
        verify_trial(ranks, server, receipts, [], **options)


@pytest.mark.parametrize("invalid", [None, 0, 1, "1", "false"])
def test_verifier_requires_strict_boolean_admission(invalid):
    from supervise import verify_trial

    ranks, server, receipts, options = fast_evidence(False)
    options["fast_synch"] = invalid
    with pytest.raises(ValueError, match="synch"):
        verify_trial(ranks, server, receipts, [], **options)


def test_actor_retains_eight_mib_log_retrieval_cap(tmp_path):
    import remote_actor as actor

    nonce = "b" * 32
    job = tmp_path / nonce
    job.mkdir()
    with (job / "stderr").open("wb") as log:
        log.truncate(8 * 1024**2 + 1)
    with pytest.raises(ValueError, match="Log too large"):
        actor.job_status(tmp_path, nonce, records=True)
    assert (job / "stderr").stat().st_size == 8 * 1024**2 + 1


def test_controller_retains_twenty_mib_response_cap(tmp_path):
    import supervise

    args = args_for()
    args.output, args.reserve_bytes = str(tmp_path / "output"), 0
    controller = supervise.Controller(args)
    try:
        with pytest.raises(ValueError, match="response too large"):
            controller.call(
                [sys.executable, "-c", "import sys; sys.stdout.write(' '*(20*1024**2+1))"]
            )
        assert controller.children == {}
    finally:
        controller.close()
