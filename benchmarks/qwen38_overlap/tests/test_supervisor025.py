"""Vertical safe plan -> exact freeze -> parent-owned mock supervisor tests."""

import json
from pathlib import Path

import pytest

from benchmarks.qwen38_overlap import harness as h
from benchmarks.qwen38_overlap.cli import main

from .test_integration025 import ROOT, metadata

SMALL = Path("/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control")


def test_freeze_and_safe_run_plan_never_load_private_profile(tmp_path, capsys):
    from benchmarks.qwen38_overlap import supervisor025 as s

    out = tmp_path / "freeze"
    target = metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only")
    small = metadata(SMALL)
    manifest = s.freeze(out, target, small)
    assert manifest["schema"] == "qwen38-overlap-integration-freeze-v2"
    assert list(manifest["phases"]) == ["pilot", *[f"batch-{i:02d}" for i in range(6)]]
    assert sum(manifest["phases"][f"batch-{i:02d}"]["requests"] for i in range(6)) == 84
    mpath = out / "integration-freeze.json"
    mdigest = h.sha(mpath.read_bytes())
    forbidden = tmp_path / "PRIVATE_MUST_NOT_BE_READ.json"
    # No execute means this deliberately nonexistent path must not be read.
    assert (
        main(
            [
                "run025",
                "--manifest",
                str(mpath),
                "--manifest-sha256",
                mdigest,
                "--phase",
                "pilot",
                "--out-dir",
                str(tmp_path / "trial"),
                "--profile",
                str(forbidden),
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value["hardware_executed"] is False
    assert not (tmp_path / "trial").exists()
    assert value["requests"] == 6 and value["deadline_seconds"] == 900
    assert {
        "--overlapped-prefill",
        "--require-normal-pressure",
        "--hidden-only-prefill",
        "--prefix-rollback",
        "--benchmark-schedule",
    } <= set(value["public_argv"])
    assert (
        main(
            [
                "run025",
                "--manifest",
                str(mpath),
                "--manifest-sha256",
                mdigest,
                "--phase",
                "pilot",
                "--out-dir",
                str(tmp_path / "trial"),
                "--profile",
                str(forbidden),
                "--execute",
            ]
        )
        == 2
    )
    assert "parent_only_guarded_supervisor_unavailable" in capsys.readouterr().out
    assert not (tmp_path / "trial").exists()
    with pytest.raises(FileExistsError):
        s.freeze(out, target, small)
    # An external digest is not permission to omit transitive harness code.
    from copy import deepcopy

    from benchmarks.qwen38_overlap.cli import encoded

    incomplete = deepcopy(manifest)
    incomplete["harness_sources"].pop("benchmarks/qwen38_overlap/numerical025.py")
    bad_manifest = tmp_path / "incomplete-freeze.json"
    bad_manifest.write_bytes(encoded(incomplete))
    with pytest.raises(ValueError, match="exact harness"):
        s.read_freeze(bad_manifest, h.sha(bad_manifest.read_bytes()))
    second = tmp_path / "cli-freeze"
    assert (
        main(
            [
                "freeze025",
                "--out-dir",
                str(second),
                "--target-config-sha256",
                target["config_sha256"],
                "--target-index-sha256",
                target["index_sha256"],
                "--small-metadata",
                small["path"],
                "--small-config-sha256",
                small["config_sha256"],
                "--small-index-sha256",
                small["index_sha256"],
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value["schema"] == manifest["schema"] and value["execution_permitted"] is False


def test_guarded_runner_mock_uses_exact_snapshot_and_never_publicizes_profile(tmp_path):
    from benchmarks.qwen38_overlap import supervisor025 as s
    from benchmarks.qwen38_overlap.cli import encoded

    out = tmp_path / "freeze"
    s.freeze(out, metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"), metadata(SMALL))
    mpath = out / "integration-freeze.json"
    digest = h.sha(mpath.read_bytes())
    profile = tmp_path / "private.json"
    profile.write_bytes(
        encoded(
            {
                "remote_source": "/TEST/checkpoints/pinned-27b",
                "ssh_key": "/TEST/private/key",
                "ssh_target": "TEST_PRIVATE_TARGET",
                "local_address": "169.254.1.1",
                "remote_address": "169.254.1.2",
                "ssh_host_key_alias": "TEST_ALIAS",
                "small_model_remote": "/TEST/small",
                "sidecar": "/TEST/sidecar",
                "remote_root_by_phase": {
                    p: "/TEST/sources/" + p
                    for p in ["pilot", *[f"batch-{i:02d}" for i in range(6)]]
                },
            }
        )
    )
    called = []

    def guard(plan):
        assert plan["manifest_sha256"] == digest
        called.append("guard")
        # The review callback receives an isolated plan, not mutable launch argv.
        plan["public_argv"][0] = "/TEST/private/not-python"
        return {
            "artifact_kind": "TEST_FIXTURE_NOT_MODEL_RESULTS",
            "manifest_sha256": digest,
            "phase": "pilot",
            "approved": True,
        }

    def run(argv, *, timeout, check, capture_output):
        called.append("runner")
        assert timeout == 930 and check is False and capture_output is True
        assert "TEST_PRIVATE_TARGET" in argv
        assert argv[0] == str(ROOT / ".venv/bin/python")
        source = Path(argv[1]).parent
        assert source != ROOT / "spikes/025-qwen38-overlapped-prefill"
        schedule = h.decode_json((source / "benchmark_schedule.json").read_bytes())
        assert len(schedule["requests"]) == 6
        assert argv[argv.index("--hidden-gate-chunk-size") + 1] == "256"
        assert argv[argv.index("--remote-root") + 1] == "/TEST/sources/pilot"
        assert argv[argv.index("--remote-source") + 1] == "/TEST/checkpoints/pinned-27b"
        # Exercise the actual producer's interpretation, not just our option labels.
        from benchmarks.qwen38_overlap.integration025 import runtime_api

        manifest = h.decode_json(mpath.read_bytes())
        with runtime_api(source, manifest["phases"]["pilot"]["sources"]) as api:
            producer = api("supervise")
            options = producer.parser().parse_args(argv[2:])
            producer.validate_args(options)
            ssh = producer.ssh_argv(options, ["--inspect"])
            assert "/TEST/sources/pilot/remote_actor.py" in ssh[-1]
            assert "/TEST/checkpoints/pinned-27b" not in ssh[-1]
        return type("Result", (), {"returncode": 1, "stdout": b"PRIVATE", "stderr": b"PRIVATE"})()

    report = s.guarded_run(
        mpath,
        digest,
        "pilot",
        tmp_path / "new-trial",
        execute=True,
        profile_path=profile,
        parent_guard=guard,
        runner=run,
    )
    assert called == ["guard", "runner"]
    assert report["accepted"] is False and report["returncode"] == 1
    assert "PRIVATE" not in json.dumps(report)
    with pytest.raises(ValueError, match="pilot"):
        s.guarded_run(
            mpath,
            digest,
            "batch-00",
            tmp_path / "forged-full",
            execute=True,
            profile_path=profile,
            parent_guard=guard,
            runner=run,
            pilot_admission={"passed": True, "physical_accepted": True},
        )
    with pytest.raises(ValueError, match="pilot"):
        s.guarded_run(
            mpath,
            digest,
            "batch-00",
            tmp_path / "full",
            execute=True,
            profile_path=profile,
            parent_guard=guard,
            runner=run,
        )
    private_bad = h.decode_json(profile.read_bytes())
    private_bad["local_address"] = "TEST_PRIVATE_INVALID_ADDRESS"
    invalid_profile = tmp_path / "private-invalid.json"
    invalid_profile.write_bytes(encoded(private_bad))
    with pytest.raises(ValueError) as caught:
        s.guarded_run(
            mpath,
            digest,
            "pilot",
            tmp_path / "invalid-private",
            execute=True,
            profile_path=invalid_profile,
            parent_guard=guard,
            runner=run,
        )
    assert "TEST_PRIVATE" not in str(caught.value)
    assert not (tmp_path / "invalid-private").exists()
    # Rehashing the manifest alone does not forgive an altered frozen source.
    source = out / "runtime-pilot" / "overlap_evidence.py"
    source.write_text("# TEST source mutation\n")
    with pytest.raises(ValueError, match="source"):
        s.guarded_run(
            mpath,
            digest,
            "pilot",
            tmp_path / "never",
            execute=True,
            profile_path=profile,
            parent_guard=guard,
            runner=run,
        )
