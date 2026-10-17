"""CPU-only amendment/policy tests; no live sampler, child, model or network."""

import json
from copy import deepcopy

import pytest

from benchmarks.qwen38_overlap import harness as h
from benchmarks.qwen38_overlap import supervisor025 as s
from benchmarks.qwen38_overlap.cli import encoded, parser

from .test_complete025 import complete_raw as _complete_raw
from .test_integration025 import ROOT, metadata
from .test_supervisor025 import SMALL

complete_raw = _complete_raw  # Register the unchanged legacy fixture for this module.

AMENDMENT = ROOT / "results/qwen38-prefill-overlap-v1/memory-acceptance-amendment-v1.json"
AMENDMENT_SHA = "a0341821118cfd83d45f0f8979152aaffae3f296b372f8dc29b58aacc35d409f"


def freeze_noncritical(path, **kwargs):
    return s.freeze(
        path,
        metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"),
        metadata(SMALL),
        pressure_policy="noncritical",
        memory_acceptance=AMENDMENT,
        memory_acceptance_sha256=AMENDMENT_SHA,
        **kwargs,
    )


def test_noncritical_freeze_exact_external_amendment_and_plan(tmp_path):
    out = tmp_path / "freeze"
    manifest = freeze_noncritical(out)
    assert manifest["pressure_policy"] == "noncritical"
    assert manifest["memory_acceptance"] == {
        "sha256": AMENDMENT_SHA,
        "bytes_utf8": AMENDMENT.read_text(),
    }
    path = out / "integration-freeze.json"
    digest = h.sha(path.read_bytes())
    assert s.read_freeze(path, digest) == manifest
    for phase in s.PHASES:
        plan = s.run_plan(path, digest, phase, tmp_path / "inert")
        argv = plan["public_argv"]
        assert argv[argv.index("--pressure-policy") + 1] == "noncritical"
        assert "--require-normal-pressure" not in argv
        assert plan["pressure_policy"] == "noncritical"
        from benchmarks.qwen38_overlap import integration025 as i

        item = manifest["phases"][phase]
        source = out / item["source_dir"]
        with i.runtime_api(source, item["sources"]) as api:
            options = (
                api("supervise")
                .parser()
                .parse_args(
                    argv[2:]
                    + [
                        "--remote-root",
                        "/TEST/root",
                        "--remote-source",
                        "/TEST/source",
                        "--ssh-key",
                        "/TEST/key",
                        "--sidecar",
                        "/TEST/sidecar",
                        "--small-model-remote",
                        "/TEST/small",
                        "--local-address",
                        "169.254.1.1",
                        "--remote-address",
                        "169.254.1.2",
                    ]
                )
            )
            api("supervise").validate_args(options)
            assert api("pressure_guard").policy_name(options) == "noncritical"
            assert api("pressure_guard").pressure_required(options) is True
            schedule = h.load_bound(source / "benchmark_schedule.json", item["schedule_sha256"])
            api("overlap_prefill").validate_modes(options, schedule["requests"])
    assert sum(manifest["phases"][f"batch-{n:02d}"]["requests"] for n in range(6)) == 84
    assert not (tmp_path / "inert").exists()


@pytest.mark.parametrize(
    "policy,path,digest",
    [
        ("noncritical", AMENDMENT, None),
        ("noncritical", None, AMENDMENT_SHA),
        ("normal", AMENDMENT, AMENDMENT_SHA),
        ("NONCRITICAL", AMENDMENT, AMENDMENT_SHA),
    ],
)
def test_freeze_requires_exact_policy_and_complete_external_pair(tmp_path, policy, path, digest):
    with pytest.raises(ValueError):
        s.freeze(
            tmp_path / "never",
            metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"),
            metadata(SMALL),
            pressure_policy=policy,
            memory_acceptance=path,
            memory_acceptance_sha256=digest,
        )
    assert not (tmp_path / "never").exists()


def test_freeze_cli_has_explicit_policy_and_external_pair():
    options = parser().parse_args(
        [
            "freeze025",
            "--out-dir",
            "TEST",
            "--target-config-sha256",
            "a" * 64,
            "--target-index-sha256",
            "b" * 64,
            "--small-metadata",
            "TEST",
            "--small-config-sha256",
            "c" * 64,
            "--small-index-sha256",
            "d" * 64,
            "--pressure-policy",
            "noncritical",
            "--memory-acceptance",
            str(AMENDMENT),
            "--memory-acceptance-sha256",
            AMENDMENT_SHA,
        ]
    )
    assert options.pressure_policy == "noncritical"
    assert options.memory_acceptance_sha256 == AMENDMENT_SHA


def test_explicit_normal_freeze_preserves_legacy_bytes_and_argv(tmp_path):
    args = (metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"), metadata(SMALL))
    default = s.freeze(tmp_path / "default", *args)
    normal = s.freeze(tmp_path / "normal", *args, pressure_policy="normal")
    assert encoded(normal) == encoded(default)
    assert "pressure_policy" not in default and "memory_acceptance" not in default


@pytest.mark.parametrize(
    "mutation", ["missing", "wrong_hash", "policy", "allowed", "abort", "original"]
)
def test_noncritical_amendment_rejected_before_output(tmp_path, mutation):
    amendment = h.decode_json(AMENDMENT.read_bytes())
    if mutation == "policy":
        amendment["pressure_policy"] = "normal"
    elif mutation == "allowed":
        amendment["allowed_pressure_levels"] = [1, 2, 4]
    elif mutation == "abort":
        amendment["abort_pressure_levels"] = []
    elif mutation == "original":
        amendment["original_acceptance_sha256"] = "f" * 64
    path = tmp_path / "amendment.json"
    path.write_bytes(encoded(amendment))
    digest = h.sha(path.read_bytes()) if mutation != "wrong_hash" else "0" * 64
    with pytest.raises(ValueError):
        s.freeze(
            tmp_path / "never",
            metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"),
            metadata(SMALL),
            pressure_policy="noncritical",
            memory_acceptance=None if mutation == "missing" else path,
            memory_acceptance_sha256=digest,
        )
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("mutation", ["bytes", "hash", "policy", "removed", "normal_escalation"])
def test_read_freeze_revalidates_amendment_even_resealed(tmp_path, mutation):
    manifest = freeze_noncritical(tmp_path / "freeze")
    bad = deepcopy(manifest)
    if mutation == "bytes":
        bad["memory_acceptance"]["bytes_utf8"] += " "
    elif mutation == "hash":
        bad["memory_acceptance"]["sha256"] = "0" * 64
    elif mutation == "policy":
        amendment = h.decode_json(bad["memory_acceptance"]["bytes_utf8"].encode())
        amendment["pressure_policy"] = "normal"
        data = encoded(amendment)
        bad["memory_acceptance"] = dict(sha256=h.sha(data), bytes_utf8=data.decode())
    elif mutation == "removed":
        bad.pop("memory_acceptance")
    else:
        bad.pop("pressure_policy")
    path = tmp_path / "bad.json"
    path.write_bytes(encoded(bad))
    with pytest.raises(ValueError):
        s.read_freeze(path, h.sha(path.read_bytes()))


@pytest.fixture
def noncritical_raw(complete_raw):
    """New TEST variant; leave all strict legacy fixtures and tests unchanged."""
    from benchmarks.qwen38_overlap import integration025 as i

    root, trust, manifest = complete_raw
    manifest = deepcopy(manifest)
    manifest.update(
        pressure_policy="noncritical",
        memory_acceptance={"sha256": AMENDMENT_SHA, "bytes_utf8": AMENDMENT.read_text()},
    )
    trust.update(manifest_sha256=h.sha(encoded(manifest)), pressure_policy="noncritical")
    options_doc = h.decode_json((root / "startup.json").read_bytes())
    options_doc["args"].update(pressure_policy="noncritical", require_normal_pressure=False)
    (root / "startup.json").write_bytes(encoded(options_doc))
    with i.runtime_api(root / "sources", manifest["phases"]["pilot"]["sources"]) as api:
        for role in ("rank0", "rank1", "server"):
            trust["actor_contexts"][role]["pressure_policy"] = "noncritical"
            path = root / (role + ".receipt.json")
            raw = h.decode_json(path.read_bytes())
            old = raw["result"]["pressure"]
            clock = [100.0]
            guard = api("pressure_guard").PressureGuard(
                False, policy="noncritical", sampler=lambda: 2, clock=lambda: clock[0]
            )
            for when, stage in (
                (100.1, "pre_spawn"),
                (100.5, "owned_child"),
                (100.9, "post_child"),
            ):
                clock[0] = when
                guard.sample(stage, force=True)
            raw["result"]["pressure"] = guard.receipt(binding=old["binding"])
            for section in ("startup", "result"):
                raw[section].update(pressure_policy="noncritical", require_normal_pressure=False)
            output = h.decode_json(raw["stdout"].splitlines()[-1].encode())
            output.update(pressure_policy="noncritical", require_normal_pressure=False)
            if role != "server":
                output["verification"].update(
                    pressure_policy="noncritical", require_normal_pressure=False
                )
            raw["stdout"] = json.dumps(output) + "\n"
            events = [h.decode_json(line.encode()) for line in raw["stderr"].splitlines()]
            for event in events:
                if event.get("phase") in ("process_start", "numerical_admission_done"):
                    event.update(pressure_policy="noncritical", require_normal_pressure=False)
            raw["stderr"] = "".join(json.dumps(e) + "\n" for e in events)
            path.write_bytes(encoded(raw))
            job_path = root / (role + ".job.json")
            job = h.decode_json(job_path.read_bytes())
            job.update(pressure_policy="noncritical", require_normal_pressure=False)
            job_path.write_bytes(encoded(job))
    trust["raw_artifacts"] = {name: h.sha((root / name).read_bytes()) for name in i.RAW_REQUIRED}
    (root.parent / "freeze/integration-freeze.json").write_bytes(encoded(manifest))
    return root, trust, manifest


def test_complete_warning2_requires_explicit_external_amendment(noncritical_raw, capsys):
    from benchmarks.qwen38_overlap import integration025 as i

    root, trust, manifest = noncritical_raw
    audit = i.audit_raw_trial(root, trust, manifest)
    assert audit["canonical025_valid"] is True
    assert audit["pressure_policy"] == "noncritical"
    assert len(audit["rows"]) == 6
    assert {r["prefill_mode"] for r in audit["rows"]} == {"serial", "overlap"}
    assert audit["pressure_observations"] == [
        {"host_role": host, "actor_role": role, "observed_levels": [2], "warning_samples": 3}
        for host, role in (("local", "rank0"), ("remote", "rank1"), ("remote", "server"))
    ]
    assert audit["physical_accepted"] is False
    report = i.collect_raw_trials([(root, trust)], manifest)
    assert report["pressure_policy"] == "noncritical"
    assert report["coverage_complete"] is True
    assert report["accepted"] is False
    from benchmarks.qwen38_overlap.cli import main

    trust_path = root / "TEST-noncritical-trust.json"
    trust_path.write_bytes(encoded(trust))
    assert (
        main(
            [
                "collect025",
                "--manifest",
                str(root.parent / "freeze/integration-freeze.json"),
                "--manifest-sha256",
                trust["manifest_sha256"],
                "--raw-dir",
                str(root),
                "--trusted",
                str(trust_path),
                "--trusted-sha256",
                h.sha(trust_path.read_bytes()),
                "--out-dir",
                str(root / "TEST-noncritical-report"),
            ]
        )
        == 2
    )
    published = h.decode_json(capsys.readouterr().out.encode())
    assert published["coverage_complete"] is True and published["accepted"] is False
    assert published["pressure_policy"] == "noncritical"
    assert published["pressure_observations"][0]["actors"] == audit["pressure_observations"]


@pytest.mark.parametrize(
    "mutation",
    [
        "no_amendment",
        "amendment_hash",
        "trust_policy",
        "actor_context_policy",
        "critical4",
        "unknown",
        "missing_sampler",
        "missing_samples",
        "coverage_gap",
        "job_guard",
        "rank_guard",
        "process_guard",
        "actor_guard",
        "cleanup",
        "identity",
    ],
)
def test_noncritical_resealed_evidence_still_fails_closed(noncritical_raw, mutation):
    from benchmarks.qwen38_overlap import integration025 as i

    root, trust, manifest = noncritical_raw
    if mutation == "no_amendment":
        manifest.pop("memory_acceptance")
    elif mutation == "amendment_hash":
        manifest["memory_acceptance"]["sha256"] = "0" * 64
    elif mutation == "trust_policy":
        trust["pressure_policy"] = "normal"
    elif mutation == "actor_context_policy":
        trust["actor_contexts"]["rank0"].pop("pressure_policy")
    else:
        name = "rank0.job.json" if mutation == "job_guard" else "rank0.receipt.json"
        raw = h.decode_json((root / name).read_bytes())
        if mutation == "job_guard":
            raw["require_normal_pressure"] = True
        else:
            pressure = raw["result"]["pressure"]
            if mutation in ("critical4", "unknown"):
                pressure["samples"][1]["level"] = 4 if mutation == "critical4" else 3
            elif mutation == "missing_sampler":
                pressure.pop("sampler_timeout_seconds")
            elif mutation == "missing_samples":
                pressure["samples"] = []
            elif mutation == "coverage_gap":
                pressure["samples"][-1]["elapsed_seconds"] = 0.6
            elif mutation == "actor_guard":
                raw["startup"]["pressure_policy"] = "normal"
            elif mutation == "cleanup":
                raw["result"]["cleanup"]["group_gone"] = False
            elif mutation == "identity":
                pressure["binding"]["nonce"] = "e" * 32
            elif mutation == "rank_guard":
                output = h.decode_json(raw["stdout"].splitlines()[-1].encode())
                output["pressure_policy"] = "normal"
                raw["stdout"] = json.dumps(output) + "\n"
            else:
                events = [h.decode_json(line.encode()) for line in raw["stderr"].splitlines()]
                events[0]["pressure_policy"] = "normal"
                raw["stderr"] = "".join(json.dumps(e) + "\n" for e in events)
        (root / name).write_bytes(encoded(raw))
        trust["raw_artifacts"][name] = h.sha((root / name).read_bytes())
    trust["manifest_sha256"] = h.sha(encoded(manifest))
    with pytest.raises(ValueError):
        i.audit_raw_trial(root, trust, manifest)


def test_normal_receipt_cannot_accept_warning2(complete_raw):
    from benchmarks.qwen38_overlap import integration025 as i

    root, trust, manifest = complete_raw
    path = root / "rank0.receipt.json"
    raw = h.decode_json(path.read_bytes())
    raw["result"]["pressure"]["samples"][1]["level"] = 2
    path.write_bytes(encoded(raw))
    trust["raw_artifacts"][path.name] = h.sha(path.read_bytes())
    with pytest.raises(ValueError):
        i.audit_raw_trial(root, trust, manifest)


@pytest.mark.parametrize("field", ["trusted_policy", "actor_context_policy"])
def test_physical_normal_context_must_explicitly_match_freeze(complete_raw, field):
    from benchmarks.qwen38_overlap import integration025 as i

    root, trust, manifest = complete_raw
    trust.update(artifact_kind=i.PHYSICAL_KIND, pressure_policy="normal")
    for context in trust["actor_contexts"].values():
        context["pressure_policy"] = "normal"
    if field == "trusted_policy":
        trust.pop("pressure_policy")
    else:
        trust["actor_contexts"]["rank0"].pop("pressure_policy")
    with pytest.raises(ValueError, match="Independent"):
        i.audit_raw_trial(root, trust, manifest)


def test_noncritical_public_markdown_labels_actual_warning_observations(tmp_path):
    from benchmarks.qwen38_overlap.cli import HISTORICAL, write_reports

    report = h.analyze([], h.derive_schedule(HISTORICAL))
    report.update(
        pressure_policy="noncritical",
        pressure_observations=[
            {
                "phase": "pilot",
                "actors": [
                    {
                        "host_role": "local",
                        "actor_role": "rank0",
                        "observed_levels": [1, 2],
                        "warning_samples": 2,
                    }
                ],
            }
        ],
    )
    output = tmp_path / "TEST-report"
    write_reports(report, output)
    text = (output / "report.md").read_text()
    assert "Pressure policy: **noncritical**" in text
    assert "warning_samples=2" in text and "observed_levels=[1, 2]" in text
    assert "not normal-pressure qualification" in text
