"""Tests for the bounded broad-benchmark operator."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("run_bench.py")
    assert path.exists(), "Benchmark operator has not been implemented"
    spec = importlib.util.spec_from_file_location("qwen38_run_bench", path)
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_commands_preserve_identity_and_bounded_schedule():
    m = module()
    profile = {
        "parameters": {
            "remote_source": "/approved/checkpoint",
            "local_metadata": "/approved/metadata",
            "ssh_key": "/outside/key",
            "split": [28, 36],
            "local_address": "169.254.47.213",
            "remote_address": "169.254.214.155",
            "ssh_target": "user@mini",
            "ssh_host_key_alias": "pinned-host",
            "eval_group_size": 32,
            "rank0_verifier": "native",
            "rank1_verifier": "slot-qmv",
            "target_revision": "pinned-rev",
            "target_config_sha256": "a" * 64,
            "target_index_sha256": "b" * 64,
            "sidecar": "/approved/head",
            "small_model_local": "/small/local",
            "small_model_remote": "/small/remote",
            "small_config_sha256": "c" * 64,
            "small_index_sha256": "d" * 64,
        }
    }
    batch = {"requests": [{"max_tokens": 128}, {"max_tokens": 256}]}
    cmd = m.build_supervisor_command(
        profile,
        Path("/new/local"),
        Path("/new/remote"),
        Path("/new/output"),
        batch,
        900,
        schedule_sha256="e" * 64,
    )
    assert "--prefix-rollback" in cmd
    assert "--benchmark-schedule" in cmd
    assert cmd[cmd.index("--benchmark-schedule-sha256") + 1] == "e" * 64
    for key, value in [
        ("split", ["28", "36"]),
        ("repeats", ["2"]),
        ("max-tokens", ["256"]),
        ("deadline", ["900"]),
        ("target-revision", ["pinned-rev"]),
        ("ssh-host-key-alias", ["pinned-host"]),
        ("eval-group-size", ["32"]),
    ]:
        i = cmd.index("--" + key)
        assert cmd[i + 1 : i + 1 + len(value)] == value
    assert not any("shell=True" in v for v in cmd)
    with pytest.raises(ValueError):
        m.build_supervisor_command(
            profile, Path("/new/local"), Path("/new/remote"), Path("/new/output"), batch, 901
        )


def test_preflight_requires_disk_pressure_and_verified_bridge():
    m = module()
    good = {
        "disk_free_bytes": 20 * 1024**3,
        "pressure_level": 1,
        "memory_bytes": 16 * 1024**3,
        "route": "interface: bridge0",
        "owned_inference_workers": [],
    }
    assert hasattr(m, "validate_health"), "Health admission has not been implemented"
    m.validate_health(good)
    for change in [
        {"disk_free_bytes": 1024},
        {"pressure_level": 4},
        {"route": "interface: en0"},
        {"owned_inference_workers": [1234]},
    ]:
        with pytest.raises(ValueError):
            m.validate_health({**good, **change})


def test_health_snapshot_is_real_local_machine_data():
    import platform

    if platform.system() != "Darwin":
        pytest.skip("Real macOS health collector smoke")
    m = module()
    assert hasattr(m, "collect_health"), "Real health collector has not been implemented"
    record = m.collect_health({"parameters": {"remote_address": "169.254.214.155"}})
    assert record["memory_bytes"] > 0
    assert record["disk_free_bytes"] > 0
    assert isinstance(record["pressure_level"], int)
    assert isinstance(record["owned_inference_workers"], list)
    assert record["route"]


@pytest.mark.parametrize(
    "prompt",
    ["Explain Newton's laws", 'Explain an "open quote', 'Explain "motion" and Newton\'s laws'],
)
@pytest.mark.parametrize("script", ["probe.py", "stream_server.py"])
@pytest.mark.parametrize("root", ["/repo/paretoquant-metal/spikes/023", "/job/sources"])
def test_health_detects_workers_with_flat_ps_prompt_quotes(
    monkeypatch, capsys, prompt, script, root
):
    import os
    import platform
    import subprocess
    import sys
    from types import SimpleNamespace

    m = module()
    command = f"/venv/bin/python3 -u {root}/{script} --prompt {prompt}"
    ps = "\n".join(
        [
            "999 " + command,
            "1000 /bin/echo " + command,
            "1001 /venv/bin/python /unrelated/probe.py --prompt " + prompt,
            "1002 /venv/bin/python /repo/paretoquant-metal/other.py",
        ]
    )

    def run(args, **kwargs):
        responses = {
            ("ps", "-axo", "pid=,command="): ps,
            ("sysctl", "-n", "hw.memsize"): str(16 * 1024**3),
            ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1",
            ("sysctl", "vm.swapusage"): "synthetic swap",
            ("route", "-n", "get", "169.254.1.2"): "interface: bridge0",
            ("pmset", "-g", "batt"): "synthetic power",
            ("pmset", "-g", "therm"): "synthetic thermal",
        }
        return SimpleNamespace(stdout=responses[tuple(args)])

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "statvfs", lambda _: SimpleNamespace(f_bavail=20 * 1024**3, f_frsize=1))
    monkeypatch.setattr(platform, "platform", lambda: "synthetic platform")
    monkeypatch.setattr(sys, "argv", ["health", "169.254.1.2"])
    exec(m.HEALTH_CODE, {})
    record = json.loads(capsys.readouterr().out)
    assert [worker["pid"] for worker in record["owned_inference_workers"]] == [999]
    with pytest.raises(ValueError, match="worker"):
        m.validate_health(record)


def test_frozen_snapshot_preserves_original_and_rejects_overwrite_and_symlinks(tmp_path):
    import hashlib
    import json

    m = module()
    assert hasattr(m, "freeze_snapshot"), "Immutable snapshot helper has not been implemented"
    source = tmp_path / "original"
    source.mkdir()
    for name in ("supervise.py", "remote_actor.py"):
        (source / name).write_text("# synthetic unit-test source\n")
    original = b'{"schema":"placeholder"}'
    (source / "benchmark_schedule.json").write_bytes(original)
    schedule = json.dumps(
        {"schema": "qwen38-workload-schedule-v1", "requests": [{"id": "test"}]}
    ).encode()
    dest = tmp_path / "frozen"
    hashes = m.freeze_snapshot(source, dest, schedule)
    assert (source / "benchmark_schedule.json").read_bytes() == original
    assert (dest / "benchmark_schedule.json").read_bytes() == schedule
    assert hashes["benchmark_schedule.json"] == hashlib.sha256(schedule).hexdigest()
    with pytest.raises(FileExistsError):
        m.freeze_snapshot(source, dest, schedule)
    (source / "linked.py").symlink_to(source / "supervise.py")
    with pytest.raises(ValueError):
        m.freeze_snapshot(source, tmp_path / "bad", schedule)


def test_deploy_payload_is_exact_and_confined(tmp_path):
    import base64
    import hashlib

    m = module()
    assert hasattr(m, "deployment_payload"), "Deployment payload has not been implemented"
    (tmp_path / "probe.py").write_bytes(b"# explicit synthetic unit test\n")
    data = m.deployment_payload(tmp_path, "023-qwen38-broad-unit-b00")
    assert base64.b64decode(data["files"]["probe.py"]) == (tmp_path / "probe.py").read_bytes()
    assert (
        data["hashes"]["probe.py"]
        == hashlib.sha256((tmp_path / "probe.py").read_bytes()).hexdigest()
    )
    for name in ("../other", "existing-project", "023-qwen38-broad-/escape", "-bad"):
        with pytest.raises(ValueError):
            m.deployment_payload(tmp_path, name)


def test_deploy_rejects_bad_destination_before_ssh(tmp_path):
    m = module()
    assert hasattr(m, "deploy_snapshot"), "Verified remote deployment has not been implemented"
    with pytest.raises(ValueError):
        m.deploy_snapshot({}, tmp_path, "../outside")


def operator_fixture(tmp_path):
    """Synthetic CPU-only paired schedules; never benchmark measurements."""
    root = tmp_path / "registration"
    workloads = root / "workloads"
    workloads.mkdir(parents=True)
    batches = []
    for i in range(6):
        requests = [
            dict(
                id=f"b{i}-{method}",
                case_id=f"case-{i}",
                method=method,
                warmup=False,
                repeat=0,
                prompt=f"Synthetic prompt {i}",
                max_tokens=2,
            )
            for method in ("native", "mtp")
        ]
        batches.append(dict(schema="qwen38-workload-schedule-v1", requests=requests))
        (workloads / f"schedule-{i:02d}.json").write_text(json.dumps(batches[-1]))
    profile = json.loads(
        (module().REPO / "results/qwen38-speed-v2/fast-profile-v2.json").read_text()
    )
    params = profile["parameters"]
    manifest = dict(
        schema="qwen38-workload-manifest-v1",
        model=profile["model"],
        revision=params["target_revision"],
        split=params["split"],
        eval_group_size=params["eval_group_size"],
        batches=batches,
    )
    (workloads / "manifest.json").write_text(json.dumps(manifest))
    files = {
        "workloads/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in workloads.iterdir()
    }
    (root / "preregistration.json").write_text(
        json.dumps(dict(schema="qwen38-broad-preregistration-v1", files=files))
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("supervise.py", "remote_actor.py"):
        (runtime / name).write_text("# synthetic, never executed\n")
    return profile_path, workloads, runtime, tmp_path / "experiment"


def test_plan_validates_frozen_schedules_without_execution_or_writes(tmp_path, monkeypatch, capsys):
    m = module()
    profile, workloads, runtime, output = operator_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("plan must not execute subprocess/SSH/health/deployment")

    monkeypatch.setattr(m.subprocess, "run", forbidden)
    assert hasattr(m, "main"), "Operator CLI missing"
    assert (
        m.main(
            [
                "--profile",
                str(profile),
                "--workloads",
                str(workloads),
                "--runtime",
                str(runtime),
                "--output",
                str(output),
                "--batches",
                "1",
                "2",
                "3",
                "4",
                "5",
                "--plan",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_batches"] == [1, 2, 3, 4, 5]
    assert plan["full_suite_selected"] is False
    assert len(plan["expected_requests"]) == 12
    assert len(plan["selected_request_ids"]) == 10
    assert plan["deadline"] == 900
    assert not output.exists()
    assert not list(runtime.parent.glob("023-qwen38-broad-*"))


@pytest.mark.parametrize(
    "bad",
    [
        "deadline-zero",
        "deadline-large",
        "batch-negative",
        "batch-large",
        "batch-duplicate",
        "overwrite",
        "source-child",
        "source-parent",
        "workload-child",
        "checkpoint-child",
        "sidecar-parent",
        "symlink-parent",
        "changed-manifest",
        "changed-schedule",
        "traversal-registration",
        "invalid-cap",
        "invalid-split",
    ],
)
def test_plan_rejects_unsafe_bounds_paths_and_frozen_changes(tmp_path, bad):
    m = module()
    profile_path, workloads, runtime, output = operator_fixture(tmp_path)
    batches, deadline = [1], 900
    if bad == "deadline-zero":
        deadline = 0
    if bad == "deadline-large":
        deadline = 901
    if bad == "batch-negative":
        batches = [-1]
    if bad == "batch-large":
        batches = [6]
    if bad == "batch-duplicate":
        batches = [1, 1]
    if bad == "overwrite":
        output.mkdir()
    if bad == "source-child":
        output = runtime / "child"
    if bad == "source-parent":
        output = runtime.parent
    if bad == "workload-child":
        output = workloads / "child"
    if bad == "checkpoint-child":
        p = json.loads(profile_path.read_text())
        p["parameters"]["local_metadata"] = str(tmp_path / "checkpoint")
        profile_path.write_text(json.dumps(p))
        output = tmp_path / "checkpoint/child"
    if bad == "sidecar-parent":
        p = json.loads(profile_path.read_text())
        p["parameters"]["sidecar"] = str(output / "head")
        profile_path.write_text(json.dumps(p))
    if bad == "symlink-parent":
        link = tmp_path / "link"
        link.symlink_to(runtime, target_is_directory=True)
        output = link / "new"
    if bad in ("changed-manifest", "invalid-cap", "invalid-split"):
        manifest_path = workloads / "manifest.json"
        data = json.loads(manifest_path.read_text())
        if bad == "changed-manifest":
            data["seed"] = 999
        if bad == "invalid-cap":
            data["batches"][0]["requests"][0]["max_tokens"] = 4097
        if bad == "invalid-split":
            data["split"] = [1, 63]
        manifest_path.write_text(json.dumps(data))
    if bad == "changed-schedule":
        (workloads / "schedule-05.json").write_text("{}")
    if bad == "traversal-registration":
        p = workloads.parent / "preregistration.json"
        data = json.loads(p.read_text())
        data["files"]["../outside"] = "0" * 64
        p.write_text(json.dumps(data))
    assert hasattr(m, "prepare_plan")
    with pytest.raises((ValueError, FileExistsError), match=".+"):
        m.prepare_plan(profile_path, workloads, output, runtime, batches, deadline)


def completed_trial_fixture(tmp_path):
    """Synthetic real-023-shaped disk receipts; no actor/GPU launched."""
    m = module()
    profile_path, workloads, runtime, output = operator_fixture(tmp_path)
    profile = json.loads(profile_path.read_text())
    p = profile["parameters"]
    metadata = tmp_path / "trusted-metadata"
    metadata.mkdir()
    p["local_metadata"] = str(metadata)
    config = dict(
        model_type="qwen3_5", text_config=dict(num_hidden_layers=64, tie_word_embeddings=False)
    )
    names = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
    names += [f"language_model.model.layers.{i % 64}.fixture_{i}.weight" for i in range(1844)]
    for name, value, key in [
        ("config.json", config, "target_config_sha256"),
        (
            "model.safetensors.index.json",
            {"weight_map": dict.fromkeys(names, "fixture.safetensors")},
            "target_index_sha256",
        ),
    ]:
        data = json.dumps(value).encode()
        (metadata / name).write_bytes(data)
        p[key] = hashlib.sha256(data).hexdigest()
    binding = {
        "config.json": p["target_config_sha256"],
        "model.safetensors.index.json": p["target_index_sha256"],
    }
    trial = tmp_path / "completed"
    trial.mkdir()
    sources = trial / "sources"
    sources.mkdir()
    actor_source = (m.REPO / "spikes/023-qwen38-workload-bench/remote_actor.py").read_bytes()
    import ast

    tree = ast.parse(actor_source)
    registry = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in n.targets)
    )
    schedule_bytes = (workloads / "schedule-00.json").read_bytes()
    hashes = {}
    for name in (*registry, "supervise.py"):
        data = (
            actor_source
            if name == "remote_actor.py"
            else schedule_bytes
            if name == "benchmark_schedule.json"
            else b"# synthetic fixture, never executed\n"
        )
        (sources / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    registered = {name: hashes[name] for name in registry}

    def write(name, value):
        (trial / name).write_text(json.dumps(value))

    batch = json.loads(schedule_bytes)
    write("source_sha256.json", registered)
    write("controller_sha256.json", {"supervise.py": hashes["supervise.py"]})
    args = dict(
        p,
        execution="phased",
        benchmark_schedule=True,
        benchmark_schedule_sha256=registered["benchmark_schedule.json"],
        prefix_rollback=True,
        verification_gate_only=False,
        eval_group_schedule=None,
        max_tokens=2,
        repeats=2,
    )
    inventory = {}
    union = {}
    for rank in (0, 1):
        lo, hi = (36, 64) if rank == 0 else (0, 36)
        selected = [n for n in names if ".layers." not in n or lo <= int(n.split(".")[3]) < hi]
        inventory[str(rank)] = sorted(selected)
        digests = dict.fromkeys(selected, "f" * 64)
        union.update(digests)
        requests = []
        for ordinal, spec in enumerate(batch["requests"]):
            request = {k: spec[k] for k in ("id", "case_id", "method", "warmup", "max_tokens")}
            request.update(
                sample_repeat=spec["repeat"],
                repeat=ordinal,
                eval_group_size=32,
                prompt_sha256=hashlib.sha256(spec["prompt"].encode()).hexdigest(),
                prompt_token_ids=[10, 20, 30],
                prompt_tokens=3,
                token_ids=[100, 101],
                finish_reason="length",
                wall_seconds=2.5 + rank,
                first_token_seconds=0.5 + rank,
                local_rank_wall_seconds=2.5 + rank,
                local_rank=rank,
                local_peak_mlx_bytes=1000 + rank * 777,
                sidecar_resident_on_rank0=True,
                accepted_drafts=0,
                verification_rounds=ordinal,
                teacher_forwards=ordinal,
            )
            requests.append(request)
        gate = dict(
            passed=True,
            mode=p[f"rank{rank}_verifier"],
            same_loaded_shards=True,
            prefix_rollback=True,
            gate_only=False,
        )
        result = dict(
            rank=rank,
            world_size=2,
            layer_start=lo,
            layer_end=hi,
            target_revision=p["target_revision"],
            target_binding=binding,
            execution="phased",
            eval_group_size=32,
            eval_group_schedule=[],
            verifier_mode=p[f"rank{rank}_verifier"],
            prefix_rollback=True,
            benchmark_schedule_sha256=registered["benchmark_schedule.json"],
            sidecar_resident_on_rank0=True,
            verification=gate,
            requests=requests,
            parameter_bytes=len(selected),
            parameter_layout=[dict(name=n, bytes=1) for n in selected],
            weight_loading=dict(
                tensor_sha256=digests,
                all_retained_tensors_loaded=True,
                tensor_count=len(selected),
                parameter_bytes=len(selected),
            ),
        )
        events = [dict(phase="process_start", rank=rank, source_sha256=registered)]
        actor = dict(
            source_sha256=registered,
            result=dict(
                returncode=0,
                reason="completed",
                log_evidence_complete=True,
                cleanup=dict(child_reaped=True, group_gone=True),
            ),
            stdout=json.dumps(result),
            stderr="\n".join(json.dumps(e) for e in events),
        )
        write(f"rank{rank}.receipt.json", actor)
    write(
        "server.receipt.json",
        dict(
            source_sha256=registered,
            result=dict(
                returncode=0,
                reason="completed",
                log_evidence_complete=True,
                cleanup=dict(child_reaped=True, group_gone=True),
            ),
            stdout=json.dumps(dict(completed=True)),
            stderr="",
        ),
    )
    write("startup.json", dict(args=args, metadata_sha256=binding, expected_tensor_names=inventory))
    write(
        "schedule_admission.json",
        dict(
            sha256=registered["benchmark_schedule.json"],
            requests=batch["requests"],
            prompt_token_ids=[[10, 20, 30]] * 2,
        ),
    )
    write(
        "listeners.cleanup.json",
        dict(ports=[["169.254.1.1", 1234]], closed=[True], services_closed_verified=True),
    )
    write(
        "result.json",
        dict(
            status="success",
            owned_actor_cleanup_verified=True,
            services_closed_verified=True,
            cleanup_errors=[],
            strict_numerical_admission=True,
            rank_actions_agree=True,
            metadata_sha256=binding,
            tensor_completeness=True,
            paired_greedy_tokens_and_stop_agree=True,
            verified_request_count=2,
            benchmark_schedule_sha256=registered["benchmark_schedule.json"],
        ),
    )
    profile["native_control_tensor_sha256"] = union
    return trial, batch, hashes, profile


def test_normalize_completed_trial_keeps_independent_rank_memory_and_rank0_times(tmp_path):
    m = module()
    trial, batch, hashes, profile = completed_trial_fixture(tmp_path)
    assert hasattr(m, "normalize_completed_trial"), "Real receipt normalizer missing"
    rows = m.normalize_completed_trial(trial, batch, hashes, profile)
    assert len(rows) == 2
    assert rows[0]["peak_mlx_bytes_by_rank"] == [1000, 1777]
    assert rows[0]["wall_seconds"] == 2.5
    assert rows[0]["first_token_seconds"] == 0.5
    assert rows[1]["teacher_forwards"] == 1
    analyzer = m.analyzer_module()
    assert analyzer.validate_pairs(rows, batch["requests"], eos_token_ids=[248044, 248046])[
        "complete"
    ]


@pytest.mark.parametrize(
    "tamper",
    [
        "status",
        "owned-cleanup",
        "services",
        "cleanup-errors",
        "strict-gate",
        "rank-actions",
        "actor-source",
        "actor-reap",
        "actor-group",
        "actor-return",
        "actor-reason",
        "actor-incomplete-log",
        "actual-start-source",
        "missing-start",
        "source-file",
        "source-manifest",
        "controller-hash",
        "listeners",
    ],
)
def test_normalize_rejects_unverified_completion_cleanup_sources(tmp_path, tamper):
    m = module()
    trial, batch, hashes, profile = completed_trial_fixture(tmp_path)
    path = trial / "result.json"
    data = json.loads(path.read_text())
    field = {
        "status": ("status", "failed"),
        "owned-cleanup": ("owned_actor_cleanup_verified", False),
        "services": ("services_closed_verified", False),
        "cleanup-errors": ("cleanup_errors", ["pending"]),
        "strict-gate": ("strict_numerical_admission", False),
        "rank-actions": ("rank_actions_agree", False),
    }
    if tamper in field:
        k, v = field[tamper]
        data[k] = v
    elif tamper.startswith("actor-") or tamper in ("actual-start-source", "missing-start"):
        path = trial / "rank1.receipt.json"
        data = json.loads(path.read_text())
        if tamper == "actor-source":
            data["source_sha256"]["probe.py"] = "0" * 64
        if tamper == "actor-reap":
            data["result"]["cleanup"]["child_reaped"] = False
        if tamper == "actor-group":
            data["result"]["cleanup"]["group_gone"] = False
        if tamper == "actor-return":
            data["result"]["returncode"] = 1
        if tamper == "actor-reason":
            data["result"]["reason"] = "deadline"
        if tamper == "actor-incomplete-log":
            data["result"]["log_evidence_complete"] = False
        if tamper == "actual-start-source":
            event = json.loads(data["stderr"])
            event["source_sha256"]["probe.py"] = "0" * 64
            data["stderr"] = json.dumps(event)
        if tamper == "missing-start":
            data["stderr"] = ""
    elif tamper == "source-file":
        (trial / "sources/probe.py").write_text("# tampered")
    elif tamper == "source-manifest":
        path = trial / "source_sha256.json"
        data = json.loads(path.read_text())
        data["probe.py"] = "0" * 64
    elif tamper == "controller-hash":
        path = trial / "controller_sha256.json"
        data = {"supervise.py": "0" * 64}
    elif tamper == "listeners":
        path = trial / "listeners.cleanup.json"
        data = json.loads(path.read_text())
        data["closed"] = [False]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=".+"):
        m.normalize_completed_trial(trial, batch, hashes, profile)


@pytest.mark.parametrize(
    "tamper",
    [
        "rank-id",
        "split",
        "eval",
        "revision",
        "target-binding",
        "inventory",
        "tensor-hash",
        "replica-hash",
        "tensor-count",
        "layout",
        "gate",
        "schedule-hash",
        "schedule-admission",
        "metadata-file",
        "startup-split",
        "request-id",
        "tokens",
        "stop",
        "prompt-ids",
        "prompt-count",
        "repeat",
        "local-rank",
        "memory-missing",
        "memory-zero",
        "counters",
        "extra-request",
        "eos",
        "control-hashes",
    ],
)
def test_normalize_rejects_runtime_or_rank_request_tampering(tmp_path, tamper):
    m = module()
    trial, batch, hashes, profile = completed_trial_fixture(tmp_path)
    path = trial / "rank1.receipt.json"
    actor = json.loads(path.read_text())
    rank = json.loads(actor["stdout"])
    r = rank["requests"][0]
    if tamper == "rank-id":
        rank["rank"] = 0
    if tamper == "split":
        rank["layer_end"] = 35
    if tamper == "eval":
        rank["eval_group_size"] = 1
    if tamper == "revision":
        rank["target_revision"] = "wrong"
    if tamper == "target-binding":
        rank["target_binding"] = {}
    if tamper == "inventory":
        rank["weight_loading"]["tensor_sha256"].pop(
            next(iter(rank["weight_loading"]["tensor_sha256"]))
        )
    if tamper == "tensor-hash":
        rank["weight_loading"]["tensor_sha256"][
            next(n for n in rank["weight_loading"]["tensor_sha256"] if ".layers." in n)
        ] = "0" * 64
    if tamper == "replica-hash":
        rank["weight_loading"]["tensor_sha256"]["language_model.model.norm.weight"] = "0" * 64
    if tamper == "tensor-count":
        rank["weight_loading"]["tensor_count"] = 1
    if tamper == "layout":
        rank["parameter_layout"][0]["bytes"] = 99
    if tamper == "gate":
        rank["verification"]["same_loaded_shards"] = False
    if tamper == "schedule-hash":
        rank["benchmark_schedule_sha256"] = "0" * 64
    if tamper == "request-id":
        r["id"] = "extra"
    if tamper == "tokens":
        r["token_ids"] = [100, 102]
    if tamper == "stop":
        r["finish_reason"] = "stop"
    if tamper == "prompt-ids":
        r["prompt_token_ids"] = [11, 20, 30]
    if tamper == "prompt-count":
        r["prompt_tokens"] = 999
    if tamper == "repeat":
        r["sample_repeat"] = 2
    if tamper == "local-rank":
        r["local_rank"] = 0
    if tamper == "memory-missing":
        r.pop("local_peak_mlx_bytes")
    if tamper == "memory-zero":
        r["local_peak_mlx_bytes"] = 0
    if tamper == "counters":
        r["accepted_drafts"] = 1
    if tamper == "extra-request":
        rank["requests"].append(deepcopy(r))
    if tamper == "eos":
        for q in rank["requests"]:
            q["token_ids"] = [248044, 101]
    actor["stdout"] = json.dumps(rank)
    path.write_text(json.dumps(actor))
    if tamper == "schedule-admission":
        path = trial / "schedule_admission.json"
        data = json.loads(path.read_text())
        data["prompt_token_ids"][0] = [9, 9, 9]
        path.write_text(json.dumps(data))
    if tamper == "metadata-file":
        (Path(profile["parameters"]["local_metadata"]) / "config.json").write_text("{}")
    if tamper == "startup-split":
        path = trial / "startup.json"
        data = json.loads(path.read_text())
        data["args"]["split"] = [1, 63]
        path.write_text(json.dumps(data))
    if tamper == "control-hashes":
        profile["native_control_tensor_sha256"] = {}
    with pytest.raises((ValueError, OSError), match=".+"):
        m.normalize_completed_trial(trial, batch, hashes, profile)


def serial_fixture(tmp_path, monkeypatch, *, fail_batch=None, bad_after=False):
    """Replace hardware boundaries only; real orchestration/files/normalizer."""
    import shutil

    m = module()
    template, batch, hashes, profile = completed_trial_fixture(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    workloads = tmp_path / "registration/workloads"
    repo = tmp_path / "isolated-repo"
    runtime = repo / "spikes/023-fixture"
    runtime.parent.mkdir(parents=True)
    shutil.copytree(template / "sources", runtime)
    monkeypatch.setattr(m, "REPO", repo)
    events = []
    runs = []
    good_health = dict(
        disk_free_bytes=20 * 1024**3,
        pressure_level=1,
        route="interface: bridge0",
        owned_inference_workers=[],
    )

    def health(profile, *, remote=False):
        events.append(("health", remote))
        if bad_after and runs:
            return dict(good_health, pressure_level=4)
        return good_health

    def deploy(profile, snapshot, name):
        events.append(("deploy", name))
        payload = m.deployment_payload(snapshot, name)
        return dict(
            remote_root="/Users/jeremylien/Projects/paretoquant-dual-host/spikes/" + name,
            source_sha256=payload["hashes"],
            readback_verified=True,
        )

    def supervise(command, log, deadline):
        events.append(("run", deadline))
        runs.append(command)
        dest = Path(command[command.index("--output") + 1])
        snapshot = Path(command[2]).parent
        shutil.copytree(template, dest)
        shutil.rmtree(dest / "sources")
        shutil.copytree(snapshot, dest / "sources")
        batch = json.loads((snapshot / "benchmark_schedule.json").read_text())
        registered = {
            n: hashlib.sha256((snapshot / n).read_bytes()).hexdigest()
            for n in m.source_registry(snapshot)
        }

        def write(name, data):
            (dest / name).write_text(json.dumps(data))

        write("source_sha256.json", registered)
        write(
            "controller_sha256.json",
            {"supervise.py": hashlib.sha256((snapshot / "supervise.py").read_bytes()).hexdigest()},
        )
        write(
            "schedule_admission.json",
            dict(
                sha256=registered["benchmark_schedule.json"],
                requests=batch["requests"],
                prompt_token_ids=[[10, 20, 30]] * len(batch["requests"]),
            ),
        )
        startup = json.loads((dest / "startup.json").read_text())
        startup["args"]["benchmark_schedule_sha256"] = registered["benchmark_schedule.json"]
        write("startup.json", startup)
        for rid in (0, 1):
            actor = json.loads((dest / f"rank{rid}.receipt.json").read_text())
            actor["source_sha256"] = registered
            actor["stderr"] = json.dumps(
                dict(phase="process_start", rank=rid, source_sha256=registered)
            )
            result = json.loads(actor["stdout"])
            result["benchmark_schedule_sha256"] = registered["benchmark_schedule.json"]
            for request, spec in zip(result["requests"], batch["requests"]):
                request.update(
                    {k: spec[k] for k in ("id", "case_id", "method", "warmup", "max_tokens")}
                )
                request["prompt_sha256"] = hashlib.sha256(spec["prompt"].encode()).hexdigest()
            actor["stdout"] = json.dumps(result)
            write(f"rank{rid}.receipt.json", actor)
        actor = json.loads((dest / "server.receipt.json").read_text())
        actor["source_sha256"] = registered
        write("server.receipt.json", actor)
        final = json.loads((dest / "result.json").read_text())
        final["benchmark_schedule_sha256"] = registered["benchmark_schedule.json"]
        if fail_batch is not None and len(runs) == fail_batch + 1:
            final["status"] = "failed"
            final["error_stage"] = "numerical_admission"
        write("result.json", final)
        log.write_text("Synthetic supervisor boundary receipt; no process launched\n")
        return 0 if final["status"] == "success" else 1

    monkeypatch.setattr(m, "collect_health", health)
    monkeypatch.setattr(m, "deploy_snapshot", deploy)
    monkeypatch.setattr(m, "run_supervisor", supervise, raising=False)
    plan = m.prepare_plan(profile_path, workloads, tmp_path / "serial-output", runtime, [0, 1], 17)
    return m, plan, events, runs


def test_serial_operator_writes_exclusive_evidence_and_reconciles_before_next_batch(
    tmp_path, monkeypatch
):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    assert hasattr(m, "run_operator"), "Serial operator is missing"
    assert m.run_operator(plan) == 0
    output = Path(plan["output"])
    assert len(runs) == 2
    assert [e[0] for e in events] == ["health", "health", "deploy", "run", "health", "health"] * 2
    snapshots = [Path(command[2]).parent for command in runs]
    assert len(set(snapshots)) == 2
    assert all(p.parent == m.REPO / "spikes" for p in snapshots)
    assert all(command[command.index("--deadline") + 1] == "17" for command in runs)
    for i in (0, 1):
        root = output / f"batch-{i:02d}"
        assert {
            "preflight.json",
            "deployment.json",
            "command.json",
            "supervisor.log",
            "after-health.json",
            "normalized.json",
        } <= {p.name for p in root.iterdir()}
    evidence = json.loads((output / "evidence.json").read_text())
    report = json.loads((output / "analysis.json").read_text())
    assert len(evidence["expected_requests"]) == 12
    assert len(evidence["rows"]) == 4
    assert evidence["provenance"]["selected_batches"] == [0, 1]
    assert evidence["provenance"]["eos_token_ids"] == [248044, 248046]
    assert report["full_suite_performance_claim"] is False
    assert report["coverage"]["observed_count"] == 4
    assert len(report["coverage"]["missing_ids"]) == 8
    assert not (m.REPO / ".qwen38-broad-operator.lock").exists()
    with pytest.raises(FileExistsError):
        m.run_operator(plan)


@pytest.mark.parametrize("fault", ["numerical", "after-pressure", "cleanup", "preflight"])
def test_serial_operator_stops_and_keeps_failure_missing_index(tmp_path, monkeypatch, fault):
    m, plan, events, runs = serial_fixture(
        tmp_path,
        monkeypatch,
        fail_batch=0 if fault == "numerical" else None,
        bad_after=fault == "after-pressure",
    )
    if fault == "cleanup":
        original = m.run_supervisor

        def runner(*args):
            code = original(*args)
            target = Path(args[0][args[0].index("--output") + 1]) / "result.json"
            result = json.loads(target.read_text())
            result["owned_actor_cleanup_verified"] = False
            target.write_text(json.dumps(result))
            return code

        monkeypatch.setattr(m, "run_supervisor", runner)
    if fault == "preflight":
        monkeypatch.setattr(
            m,
            "collect_health",
            lambda *a, **k: dict(
                disk_free_bytes=1,
                pressure_level=1,
                route="interface: bridge0",
                owned_inference_workers=[],
            ),
        )
    assert m.run_operator(plan) == 1
    assert len(runs) == (0 if fault == "preflight" else 1)
    output = Path(plan["output"])
    index = json.loads((output / "index.json").read_text())
    assert len(index["failures"]) == 1
    assert index["failures"][0]["batch_index"] == 0
    assert index["completed_batches"] == []
    assert len(index["missing_selected_ids"]) == 4
    assert not (output / "batch-01").exists()
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["rows"] == []
    assert evidence["failures"] == index["failures"]
    if fault != "preflight":
        assert (output / "batch-00/after-health.json").exists()
    assert not (m.REPO / ".qwen38-broad-operator.lock").exists()


def test_serial_operator_refuses_existing_lock_without_pid_cleanup(tmp_path, monkeypatch):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    lock = m.REPO / ".qwen38-broad-operator.lock"
    original = b'{"pid":999999,"status":"unknown-never-auto-break"}'
    lock.write_bytes(original)
    with pytest.raises(FileExistsError):
        m.run_operator(plan)
    assert lock.read_bytes() == original
    assert not Path(plan["output"]).exists()
    assert events == runs == []


def test_cli_dispatches_execution_not_just_a_plan(tmp_path, monkeypatch):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    assert (
        m.main(
            [
                "--profile",
                plan["profile_path"],
                "--workloads",
                plan["workloads"],
                "--runtime",
                plan["runtime"],
                "--output",
                plan["output"],
                "--batches",
                "0",
                "1",
                "--deadline",
                "17",
            ]
        )
        == 0
    )
    assert len(runs) == 2


def test_supervisor_runner_executes_cpu_child_and_preserves_exclusive_log(tmp_path):
    import sys

    m = module()
    assert hasattr(m, "run_supervisor"), "Bounded supervisor subprocess runner missing"
    log = tmp_path / "supervisor.log"
    assert (
        m.run_supervisor([sys.executable, "-c", "print('CPU-only subprocess smoke')"], log, 1) == 0
    )
    assert log.read_text().strip() == "CPU-only subprocess smoke"
    with pytest.raises(FileExistsError):
        m.run_supervisor([sys.executable, "-c", "raise SystemExit(99)"], log, 1)


def test_supervisor_timeout_allows_internal_cleanup_without_killing_process(tmp_path, monkeypatch):
    m = module()
    events = []

    class Child:
        pid = 12345

        def wait(self, timeout):
            events.append(("wait", timeout))
            raise m.subprocess.TimeoutExpired("synthetic", timeout)

        def kill(self):
            pytest.fail("outer timeout must not kill cleanup owner")

        def terminate(self):
            pytest.fail("outer timeout must not kill cleanup owner")

    def popen(*args, **kwargs):
        events.append(("spawn", kwargs))
        return Child()

    monkeypatch.setattr(m.subprocess, "Popen", popen)
    assert hasattr(m, "run_supervisor")
    with pytest.raises(m.subprocess.TimeoutExpired):
        m.run_supervisor(["synthetic-supervisor"], tmp_path / "log", 17)
    assert events[1] == ("wait", 137)
    assert events[0][1]["env"]["TMPDIR"] == "/Users/jeremylien/.hermes/cache/scratch"


@pytest.mark.parametrize("preserved_failure", [False, True])
def test_collect_only_loads_existing_trial_manifest_without_hardware(
    tmp_path, monkeypatch, preserved_failure
):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    assert m.run_operator(plan) == 0
    manifest = Path(plan["output"]) / "trial_manifest.json"
    if preserved_failure:
        data = json.loads(manifest.read_text())
        data["failures"] = [dict(batch_index=3, error_type="HistoricalNumericalFailure")]
        manifest.write_text(json.dumps(data))

    def forbidden(*a, **k):
        pytest.fail("collection must never execute health, deployment, or supervisor")

    monkeypatch.setattr(m, "collect_health", forbidden)
    monkeypatch.setattr(m, "deploy_snapshot", forbidden)
    monkeypatch.setattr(m, "run_supervisor", forbidden)
    output = tmp_path / "collected"
    code = m.main(
        [
            "--profile",
            plan["profile_path"],
            "--workloads",
            plan["workloads"],
            "--runtime",
            plan["runtime"],
            "--output",
            str(output),
            "--batches",
            "0",
            "1",
            "--collect-only",
            str(manifest),
        ]
    )
    assert code == int(preserved_failure)
    evidence = json.loads((output / "evidence.json").read_text())
    report = json.loads((output / "analysis.json").read_text())
    assert len(evidence["rows"]) == 4
    assert len(evidence["expected_requests"]) == 12
    assert evidence["provenance"]["collection_only"] is True
    assert len(evidence["failures"]) == int(preserved_failure)
    assert report["full_suite_performance_claim"] is False
    assert report["overall"] is None


def test_operator_rejects_runtime_drift_before_deployment(tmp_path, monkeypatch):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    (Path(plan["runtime"]) / "probe.py").write_text("# changed after plan admission\n")
    assert m.run_operator(plan) == 1
    assert runs == []
    assert not any(e[0] == "deploy" for e in events)
    assert len(json.loads((Path(plan["output"]) / "index.json").read_text())["failures"]) == 1


def test_health_collection_failure_retains_partial_receipt_and_stops(tmp_path, monkeypatch):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    original = m.collect_health

    def collector(profile, *, remote=False):
        if remote:
            raise OSError("Synthetic unavailable peer")
        return original(profile)

    monkeypatch.setattr(m, "collect_health", collector)
    assert m.run_operator(plan) == 1
    preflight = json.loads((Path(plan["output"]) / "batch-00/preflight.json").read_text())
    assert preflight["local"]["pressure_level"] == 1
    assert preflight["remote"]["error_type"] == "OSError"
    assert runs == []


@pytest.mark.parametrize("tokens", [0, 513, True, -1, 1.5])
def test_command_rejects_invalid_request_caps_before_execution(tokens):
    m = module()
    profile = json.loads((m.REPO / "results/qwen38-speed-v2/fast-profile-v2.json").read_text())
    with pytest.raises(ValueError):
        m.build_supervisor_command(
            profile,
            Path("/new/local"),
            Path("/new/remote"),
            Path("/new/output"),
            {"requests": [{"max_tokens": tokens}]},
            schedule_sha256="e" * 64,
        )


@pytest.mark.parametrize("bad", ["cap", "repeat", "too-many", "extra-field", "schema"])
def test_plan_rejects_rehashed_but_unbounded_schedule(tmp_path, bad):
    m = module()
    profile_path, workloads, runtime, output = operator_fixture(tmp_path)
    path = workloads / "manifest.json"
    manifest = json.loads(path.read_text())
    batch = manifest["batches"][0]
    if bad == "cap":
        for r in batch["requests"]:
            r["max_tokens"] = 513
    if bad == "repeat":
        for r in batch["requests"]:
            r["repeat"] = 32
    if bad == "too-many":
        batch["requests"] = [
            dict(r, id=f"r{i}-{r['method']}", repeat=i)
            for i in range(17)
            for r in batch["requests"]
        ]
    if bad == "extra-field":
        batch["requests"][0]["execute"] = "no"
    if bad == "schema":
        batch["schema"] = "unadmitted"
    path.write_text(json.dumps(manifest))
    (workloads / "schedule-00.json").write_text(json.dumps(batch))
    regpath = workloads.parent / "preregistration.json"
    reg = json.loads(regpath.read_text())
    reg["files"] = {
        "workloads/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in workloads.iterdir()
    }
    regpath.write_text(json.dumps(reg))
    with pytest.raises(ValueError):
        m.prepare_plan(profile_path, workloads, output, runtime, [0])


def test_workload_root_and_output_cannot_overlap_any_runtime_sources(tmp_path):
    m = module()
    profile, workloads, runtime, output = operator_fixture(tmp_path)
    nested_runtime = workloads / "runtime"
    nested_runtime.mkdir()
    for name in ("supervise.py", "remote_actor.py"):
        (nested_runtime / name).write_text("# synthetic\n")
    with pytest.raises(ValueError, match="disjoint"):
        m.prepare_plan(profile, workloads, output, nested_runtime, [0])
    data = json.loads(profile.read_text())
    data["local_runtime"] = str(output / "saved-runtime")
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint"):
        m.prepare_plan(profile, workloads, output, runtime, [0])


def test_all_six_selected_batches_are_full_selection_even_in_reverse_order(tmp_path):
    m = module()
    profile, workloads, runtime, output = operator_fixture(tmp_path)
    plan = m.prepare_plan(profile, workloads, output, runtime, [5, 4, 3, 2, 1, 0])
    assert plan["full_suite_selected"] is True


def test_collect_only_rejects_nonregistered_schedule_binding_before_output(tmp_path, monkeypatch):
    m, plan, events, runs = serial_fixture(tmp_path, monkeypatch)
    assert m.run_operator(plan) == 0
    manifest = Path(plan["output"]) / "trial_manifest.json"
    data = json.loads(manifest.read_text())
    data["trials"][0]["expected_sources"]["benchmark_schedule.json"] = "0" * 64
    manifest.write_text(json.dumps(data))
    plan["output"] = str(tmp_path / "bad-collect")
    with pytest.raises(ValueError, match="schedule"):
        m.collect_trials(plan, manifest)
    assert not Path(plan["output"]).exists()
