"""Complete raw025 TEST grammar, never physical or performance evidence."""

import json
import shutil
from copy import deepcopy

import pytest

from benchmarks.qwen38_overlap import harness as h
from benchmarks.qwen38_overlap import integration025 as i
from benchmarks.qwen38_overlap import supervisor025 as s
from benchmarks.qwen38_overlap.cli import encoded

from .fixtures025 import actor_documents, symbolic_gate, symbolic_legacy, symbolic_request
from .test_integration025 import ROOT, metadata
from .test_supervisor025 import SMALL


@pytest.fixture
def complete_raw(tmp_path):
    freeze_dir = tmp_path / "freeze"
    manifest = s.freeze(
        freeze_dir, metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"), metadata(SMALL)
    )
    digest = h.sha((freeze_dir / "integration-freeze.json").read_bytes())
    item = manifest["phases"]["pilot"]
    sources = item["sources"]
    root = tmp_path / "TEST_FIXTURE_NOT_MODEL_RESULTS"
    root.mkdir()
    source = root / "sources"
    shutil.copytree(freeze_dir / item["source_dir"], source)
    schedule = h.decode_json((source / "benchmark_schedule.json").read_bytes())
    rows = schedule["requests"]
    prompt_ids = [
        [1 + n % 100 for n in range(513 if "1536" in row["case_id"] else 55)] for row in rows
    ]
    # Reuse historical loading identity, not its numerical/admission/performance status.
    old = ROOT / "results/qwen38-broad-v1/prefill-experiment/candidate-v1"
    ranks = [
        h.decode_json(
            h.decode_json((old / f"rank{rid}.receipt.json").read_bytes())["stdout"]
            .splitlines()[-1]
            .encode()
        )
        for rid in (0, 1)
    ]
    trust = dict(
        schema="qwen38-overlap-raw025-trusted-v2",
        artifact_kind=i.TEST_KIND,
        phase="pilot",
        manifest_sha256=digest,
        raw_artifacts={},
        actor_contexts={},
        prompt_token_ids=prompt_ids,
        tensor_sha256_by_rank=[rank["weight_loading"]["tensor_sha256"] for rank in ranks],
        native_outputs=[dict(token_ids=[248044], finish_reason="stop") for _ in rows],
    )
    plan = s.run_plan(freeze_dir / "integration-freeze.json", digest, "pilot", tmp_path / "inert")
    with i.runtime_api(source, sources) as api:
        options = (
            api("supervise")
            .parser()
            .parse_args(
                plan["public_argv"][2:]
                + [
                    "--remote-root",
                    "/TEST/root",
                    "--remote-source",
                    "/TEST/remote-source",
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
        options.remote_source = "/TEST/remote-source"
        modes = [{k: row[k] for k in ("id", "prefill_mode", "source_request_id")} for row in rows]
        actors = {}
        for rid, rank in enumerate(ranks):
            mode = ("native", "slot-qmv")[rid]
            group = {}
            for scope in ("small", "full"):
                binding = manifest["small" if scope == "small" else "target"]
                pins = {
                    "config.json": binding["config_sha256"],
                    "model.safetensors.index.json": binding["index_sha256"],
                }
                entry = symbolic_legacy(
                    api, manifest["numerical_plan"][scope][str(rid)], mode, pins
                )
                expected = manifest["numerical_plan"][scope][str(rid)]
                entry["hidden_only_prefill"] = symbolic_gate(
                    api, expected["native_layout"], scope=scope, overlap=False
                )
                entry["overlapped_prefill"] = symbolic_gate(
                    api, expected["overlap_layout"], scope=scope, overlap=True
                )
                group[scope] = entry
            group["peer"] = {
                scope: {
                    "hidden_only_prefill": deepcopy(group[scope]["hidden_only_prefill"]),
                    "overlapped_prefill": deepcopy(group[scope]["overlapped_prefill"]),
                }
                for scope in ("small", "full")
            }
            rank.update(
                prompt="tokens:[1,2,3,4,5]",
                formatted_prompt=json.dumps(manifest["numerical_plan"]["width_prompt_ids"]),
                overlapped_prefill=True,
                require_normal_pressure=True,
                hidden_gate_chunk_size=256,
                benchmark_schedule_sha256=item["schedule_sha256"],
                requests=[],
            )
            rank["verification"].update(
                overlapped_prefill=True,
                prefill_modes=["serial", "overlap"],
                request_prefill_modes=modes,
                hidden_gate_chunk_size=256,
                groups={"32": group},
            )
            events = [
                dict(
                    phase="process_start",
                    rank=rid,
                    pid=(rid + 1) * 100 + 1,
                    source_sha256=sources,
                    overlapped_prefill=True,
                    require_normal_pressure=True,
                    hidden_only_prefill=True,
                    hidden_gate_chunk_size=256,
                ),
                dict(
                    phase="numerical_admission_done",
                    rank=rid,
                    overlapped_prefill=True,
                    prefill_modes=["serial", "overlap"],
                    request_prefill_modes=modes,
                    prefix_rollback=True,
                    hidden_only_prefill=True,
                    hidden_gate_chunk_size=256,
                ),
            ]
            for ordinal, (spec, prompt) in enumerate(zip(rows, prompt_ids, strict=True)):
                result, phases = symbolic_request(
                    api,
                    rid,
                    spec,
                    prompt,
                    manifest["numerical_plan"]["full"][str(rid)]["overlap_layout"],
                    ordinal,
                )
                rank["requests"].append(result)
                events.extend(phases)
            name = f"rank{rid}"
            nonce = str(rid + 1) * 32
            raw, context = actor_documents(
                api, root / nonce, sources, role=name, output=rank, events=events, nonce=nonce
            )
            (root / (name + ".receipt.json")).write_bytes(encoded(raw))
            trust["actor_contexts"][name] = context
            actors[name] = dict(nonce=nonce, remote=rid == 1)
            (root / (name + ".job.json")).write_bytes(
                encoded(
                    dict(
                        nonce=nonce,
                        source_sha256=sources,
                        require_normal_pressure=True,
                        host_role="local" if rid == 0 else "remote",
                        actor_role=name,
                    )
                )
            )
        server = h.decode_json(
            h.decode_json((old / "server.receipt.json").read_bytes())["stdout"]
            .splitlines()[-1]
            .encode()
        )
        raw, context = actor_documents(
            api, root / ("3" * 32), sources, role="server", output=server, nonce="3" * 32
        )
        (root / "server.receipt.json").write_bytes(encoded(raw))
        trust["actor_contexts"]["server"] = context
        actors["server"] = dict(nonce="3" * 32, remote=True)
        (root / "server.job.json").write_bytes(
            encoded(
                dict(
                    nonce="3" * 32,
                    source_sha256=sources,
                    require_normal_pressure=True,
                    host_role="remote",
                    actor_role="server",
                )
            )
        )
    pins = {
        "config.json": manifest["target"]["config_sha256"],
        "model.safetensors.index.json": manifest["target"]["index_sha256"],
    }
    trust["listener_endpoints"] = [
        ["169.254.1.1", 10001],
        ["169.254.1.2", 10002],
        ["169.254.1.2", 10003],
    ]
    docs = {
        "source_sha256.json": sources,
        "controller_sha256.json": {"supervise.py": sources["supervise.py"]},
        "peer_source.json": dict(source_sha256=sources),
        "actors.json": actors,
        "startup.json": dict(nonce="c" * 32, args=vars(options), metadata_sha256=pins),
        "result.json": dict(
            nonce="c" * 32,
            status="success",
            error_type=None,
            metadata_sha256=pins,
            owned_actor_cleanup_verified=True,
            services_closed_verified=True,
            cleanup_errors=[],
            transport_cleanup={
                name: dict(child_reaped=True, group_gone=True, pending_os_gpu_termination=False)
                for name in ("rank0", "rank1", "server")
            },
            control_transport_cleanup={
                "control1": dict(
                    child_reaped=True, group_gone=True, pending_os_gpu_termination=False
                )
            },
        ),
        "schedule_admission.json": dict(
            sha256=item["schedule_sha256"], requests=rows, prompt_token_ids=prompt_ids
        ),
        "listeners.cleanup.json": dict(
            services_closed_verified=True, ports=trust["listener_endpoints"], closed=[True] * 3
        ),
    }
    for name, value in docs.items():
        (root / name).write_bytes(encoded(value))
    trust["raw_artifacts"] = {name: h.sha((root / name).read_bytes()) for name in i.RAW_REQUIRED}
    return root, trust, manifest


def test_complete_actual025_raw_adapter_and_collector_test_only(complete_raw):
    root, trust, manifest = complete_raw
    audit = i.audit_raw_trial(root, trust, manifest)
    assert audit["canonical025_valid"] is True
    assert audit["architecture_metrics_valid"] is True
    assert len(audit["rows"]) == 6
    assert audit["physical_accepted"] is False
    report = i.collect_raw_trials([(root, trust)], manifest)
    assert report["evidence_validation"] == "source_bound_raw025_valid"
    assert report["coverage_complete"] is True
    assert report["accepted"] is False
    assert report["complete_suite"] is False
    assert report["record_audits"][0]["pilot_budget"]["passed"] is True
    assert "TEST/remote-source" not in json.dumps(report)
    from benchmarks.qwen38_overlap.cli import main

    trust_path = root / "TEST-trust.json"
    trust_path.write_bytes(encoded(trust))
    freeze_path = root.parent / "freeze/integration-freeze.json"
    assert (
        main(
            [
                "collect025",
                "--manifest",
                str(freeze_path),
                "--manifest-sha256",
                trust["manifest_sha256"],
                "--raw-dir",
                str(root),
                "--trusted",
                str(trust_path),
                "--trusted-sha256",
                h.sha(trust_path.read_bytes()),
                "--out-dir",
                str(root / "TEST-report"),
            ]
        )
        == 2
    )
    published = h.decode_json((root / "TEST-report/report.json").read_bytes())
    assert published["coverage_complete"] is True
    assert published["accepted"] is False
    assert len(published["rows"]) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_job",
        "short_pressure",
        "wrong_host",
        "stale_clock",
        "missing_kv",
        "numeric_shape",
        "wrong_process_source",
        "wrong_original_native",
        "changed_prompt",
        "changed_schedule",
    ],
)
def test_resealed_raw025_mutations_cannot_repair_semantics(complete_raw, mutation):
    root, trust, manifest = complete_raw
    name = "rank0.receipt.json"
    raw = h.decode_json((root / name).read_bytes())
    if mutation == "wrong_job":
        raw["result"]["pressure"]["binding"]["nonce"] = "e" * 32
    elif mutation == "short_pressure":
        raw["result"]["pressure"]["samples"][-1]["elapsed_seconds"] = 0.6
    elif mutation == "wrong_host":
        raw["result"]["pressure"]["binding"]["host_role"] = "remote"
    elif mutation == "stale_clock":
        raw["result"]["actor_started_monotonic"] = 99.0
    else:
        rank = h.decode_json(raw["stdout"].splitlines()[-1].encode())
        if mutation in ("missing_kv", "numeric_shape"):
            values = rank["verification"]["groups"]["32"]["full"]["windows"][0]["caches"]
            part = next(v for v in values if v["part"] == "values")
            if mutation == "missing_kv":
                values.remove(part)
            else:
                part["shape"][2] += 1
        elif mutation == "wrong_original_native":
            trust["native_outputs"][0]["token_ids"] = [248046]
        elif mutation == "changed_prompt":
            rank["requests"][0]["prompt_token_ids"][0] += 1
        elif mutation == "changed_schedule":
            rank["requests"][0]["source_request_id"] = "TEST-other-slot"
        else:
            phases = [h.decode_json(line.encode()) for line in raw["stderr"].splitlines()]
            phases[0]["source_sha256"]["overlap_gate.py"] = "0" * 64
            raw["stderr"] = "".join(json.dumps(e) + "\n" for e in phases)
        raw["stdout"] = json.dumps(rank) + "\n"
    (root / name).write_bytes(encoded(raw))
    trust["raw_artifacts"][name] = h.sha((root / name).read_bytes())
    with pytest.raises(ValueError):
        i.audit_raw_trial(root, trust, manifest)


def test_nonobject_actor_envelopes_are_invalid_inputs_not_collector_crashes(complete_raw):
    root, trust, manifest = complete_raw
    original = h.decode_json((root / "rank0.receipt.json").read_bytes())
    for key in ("startup", "pids", "result"):
        raw = deepcopy(original)
        raw[key] = []
        (root / "rank0.receipt.json").write_bytes(encoded(raw))
        trust["raw_artifacts"]["rank0.receipt.json"] = h.sha(
            (root / "rank0.receipt.json").read_bytes()
        )
        report = i.collect_raw_trials([(root, trust)], manifest)
        assert report["evidence_validation"] == "invalid"
        assert report["accepted"] is False


def test_raw_gate_prompt_is_bound_to_independent_five_token_plan(complete_raw):
    root, trust, manifest = complete_raw
    raw = h.decode_json((root / "rank0.receipt.json").read_bytes())
    rank = h.decode_json(raw["stdout"].splitlines()[-1].encode())
    rank["prompt"] = "tokens:[11,12,13,14,15]"
    rank["formatted_prompt"] = "[11, 12, 13, 14, 15]"
    raw["stdout"] = json.dumps(rank) + "\n"
    (root / "rank0.receipt.json").write_bytes(encoded(raw))
    trust["raw_artifacts"]["rank0.receipt.json"] = h.sha((root / "rank0.receipt.json").read_bytes())
    with pytest.raises(ValueError, match="gate prompt"):
        i.audit_raw_trial(root, trust, manifest)


@pytest.mark.parametrize("field", ["rank", "local_rank"])
def test_raw_rank_identity_cannot_use_bool_as_integer(complete_raw, field):
    root, trust, manifest = complete_raw
    raw = h.decode_json((root / "rank0.receipt.json").read_bytes())
    rank = h.decode_json(raw["stdout"].splitlines()[-1].encode())
    if field == "rank":
        rank["rank"] = False
    else:
        rank["requests"][0]["local_rank"] = False
    raw["stdout"] = json.dumps(rank) + "\n"
    (root / "rank0.receipt.json").write_bytes(encoded(raw))
    trust["raw_artifacts"]["rank0.receipt.json"] = h.sha((root / "rank0.receipt.json").read_bytes())
    with pytest.raises(ValueError, match="identity|local rank"):
        i.audit_raw_trial(root, trust, manifest)


def test_raw_adapter_reconciles_actual_job_role_with_external_owner(complete_raw):
    root, trust, manifest = complete_raw
    job = h.decode_json((root / "rank0.job.json").read_bytes())
    job["host_role"] = "remote"
    (root / "rank0.job.json").write_bytes(encoded(job))
    trust["raw_artifacts"]["rank0.job.json"] = h.sha((root / "rank0.job.json").read_bytes())
    with pytest.raises(ValueError, match="Actor job"):
        i.audit_raw_trial(root, trust, manifest)


def test_raw_adapter_rechecks_exact_listener_targets(complete_raw):
    root, trust, manifest = complete_raw
    listeners = h.decode_json((root / "listeners.cleanup.json").read_bytes())
    listeners["ports"][0][1] += 1
    (root / "listeners.cleanup.json").write_bytes(encoded(listeners))
    trust["raw_artifacts"]["listeners.cleanup.json"] = h.sha(
        (root / "listeners.cleanup.json").read_bytes()
    )
    with pytest.raises(ValueError, match="Listener"):
        i.audit_raw_trial(root, trust, manifest)


def test_raw_adapter_rechecks_controller_transport_cleanup_not_only_success(complete_raw):
    root, trust, manifest = complete_raw
    final = h.decode_json((root / "result.json").read_bytes())
    final["transport_cleanup"] = {"rank0": {"child_reaped": False, "group_gone": False}}
    (root / "result.json").write_bytes(encoded(final))
    trust["raw_artifacts"]["result.json"] = h.sha((root / "result.json").read_bytes())
    with pytest.raises(ValueError, match="transport cleanup"):
        i.audit_raw_trial(root, trust, manifest)
