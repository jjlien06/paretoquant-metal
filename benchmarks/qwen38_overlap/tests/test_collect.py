"""End-to-end proposed evidence TEST FIXTURE, never model/runtime evidence."""

import hashlib
from copy import deepcopy

import pytest

from .test_harness import api, full_schedule
from .test_numerics import inventory_fixture
from .test_pressure import guard_fixture
from .test_transcript import transcript_fixture


def record_fixture(tmp_path):
    h = api()
    schedule = full_schedule()
    batch = schedule["batches"][0]
    sources = {"runtime.py": hashlib.sha256(b"# TEST FIXTURE NOT RUNTIME\n").hexdigest()}
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/runtime.py").write_bytes(b"# TEST FIXTURE NOT RUNTIME\n")
    expected_parts, metrics = inventory_fixture()
    config = {
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "split": [28, 36],
        "eval_group_size": 32,
        "verifier_modes": ["native", "slot-qmv"],
        "precision": "Q4/BF16",
        "hidden_only_prefill": True,
        "overlapped_prefill": True,
        "prefix_rollback": True,
        "require_normal_pressure": True,
        "prefill_chunk_size": 256,
        "max_simultaneous_chunks": 2,
        "max_pending_handoffs": 1,
        "control_frame_bytes": 65536,
    }
    trust = {
        "schema": "qwen38-overlap-trusted-input-v1",
        "artifact_kind": "TEST_FIXTURE_NOT_MODEL_RESULTS",
        "sources": sources,
        "configuration": config,
        "metadata_sha256": {"config": "1" * 64, "index": "2" * 64, "sidecar": "3" * 64},
        "tensor_sha256_by_rank": [
            {"test.a": "4" * 64, "test.shared": "5" * 64},
            {"test.b": "6" * 64, "test.shared": "5" * 64},
        ],
        "guard_source_sha256": sources["runtime.py"],
        "max_pressure_gap_seconds": 0.75,
        "required_source_names": ["runtime.py"],
        "requests": {},
        "numerical_plan_by_rank": [{"TEST-gate": {"TEST-boundary": expected_parts}}] * 2,
        "raw_artifacts": {},
    }
    record = {
        "schema": "qwen38-overlap-normalized-record-v1",
        "artifact_kind": trust["artifact_kind"],
        "batch_index": 0,
        "configuration": config,
        "sources": sources,
        "metadata_sha256": trust["metadata_sha256"],
        "cleanup": {
            "owned_processes_exited": True,
            "owned_listeners_closed": True,
            "unrelated_processes_signalled": False,
        },
        "ranks": [],
        "request_evidence": [],
    }
    for rank in (0, 1):
        record["ranks"].append(
            {
                "rank": rank,
                "actor_nonce": ("c" if rank == 0 else "d") * 32,
                "source_sha256": sources,
                "configuration": config,
                "metadata_sha256": trust["metadata_sha256"],
                "verifier_mode": config["verifier_modes"][rank],
                "tensor_sha256": trust["tensor_sha256_by_rank"][rank],
                "parameter_bytes": 1024,
                "log_complete": True,
                "returncode": 0,
                "reason": "completed",
                "pressure_guard": {**guard_fixture(), "guard_source_sha256": sources["runtime.py"]},
                "execution_interval": {"spawn_seconds": 1.0, "end_seconds": 2.0},
                "reserve_bytes": 9 * 1024**3,
                "numerical_gates": {"TEST-gate": {"TEST-boundary": metrics}},
                "requests": [],
            }
        )
    case_lengths = {c["id"]: c["formatted_prompt_tokens"] for c in schedule["cases"]}
    for ordinal, spec in enumerate(batch["requests"]):
        binding, events = transcript_fixture(
            spec["prefill_mode"], case_lengths.get(spec["case_id"], 12)
        )
        binding.update(
            id=spec["id"],
            ordinal=ordinal,
            request_nonce=f"{ordinal + 1:032x}",
            cache_epoch=f"{ordinal + 100:032x}",
        )
        binding["layers"] = [list(range(36, 64)), list(range(36))]
        for rank in (0, 1):
            for event in events[rank]:
                event.update(
                    request_id=binding["id"],
                    ordinal=ordinal,
                    request_nonce=binding["request_nonce"],
                    cache_epoch=binding["cache_epoch"],
                )
                if event["phase"] == "target_done":
                    event["evaluated_layers"] = binding["layers"][rank]
        trust["requests"][spec["id"]] = {**binding, "token_ids": [248044], "finish_reason": "stop"}
        decode = []
        for rank in (0, 1):
            decode.append(
                [
                    {
                        "op": "finish",
                        "tokens": [],
                        "command_sequence": events[rank][-1]["command_sequence"] + 1,
                        "request_nonce": binding["request_nonce"],
                        "cache_epoch": binding["cache_epoch"],
                        "prefill_mode": spec["prefill_mode"],
                        "request_id": spec["id"],
                        "ordinal": ordinal,
                        "rank": rank,
                        "actor_nonce": binding["actor_nonces"][rank],
                    }
                ]
            )
            record["ranks"][rank]["requests"].append(
                {
                    **spec,
                    "status": "success",
                    "prompt_token_ids": binding["prompt_token_ids"],
                    "token_ids": [248044],
                    "finish_reason": "stop",
                    "first_token_seconds": 1.0,
                    "wall_seconds": 2.0,
                    "local_peak_mlx_bytes": 1024,
                    "verification_rounds": 0,
                    "accepted_drafts": 0,
                    "teacher_forwards": 0,
                }
            )
        record["request_evidence"].append(
            {
                "id": spec["id"],
                "binding": binding,
                "prefill_events": events,
                "decode_actions": decode,
            }
        )
    return h, schedule, trust, record


def test_public_audit_rows_do_not_dump_private_profile_or_argv(tmp_path):
    h, schedule, trust, record = record_fixture(tmp_path)
    record["ranks"][0]["requests"][0]["private_profile"] = {"ssh_key": "TEST_PRIVATE_SECRET"}
    record["ranks"][0]["requests"][0]["argv"] = ["TEST_PRIVATE_SECRET"]
    report = h.audit_record(record, trust, schedule, tmp_path)
    assert "TEST_PRIVATE_SECRET" not in str(report)


def test_external_trusted_record_collected_without_model_claim(tmp_path):
    h, schedule, trust, record = record_fixture(tmp_path)
    assert hasattr(h, "audit_record"), "Externally bound record collector missing"
    report = h.audit_record(record, trust, schedule, tmp_path)
    assert report["evidence_contract_valid"] is True
    assert report["accepted"] is False
    assert report["integration_status"] == "pending_runtime025_adapter_and_independent_review"
    assert len(report["rows"]) == 14


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_mode_both",
        "missing_rank",
        "extra_request",
        "missing_request",
        "duplicate_request",
        "source_coforge",
        "source_missing",
        "metadata",
        "tensor",
        "output_both",
        "prompt_both",
        "stop",
        "memory",
        "cleanup",
        "pressure",
        "guard_disabled",
        "numeric_missing",
        "events_missing",
        "unregistered_source",
        "bad_tensor_digest",
        "uncovered_actor_interval",
        "unbound_guard_source",
    ],
)
def test_external_record_mutations_rejected(tmp_path, mutation):
    h, schedule, trust, record = record_fixture(tmp_path)
    assert hasattr(h, "audit_record"), "Externally bound record collector missing"
    record = deepcopy(record)
    if mutation == "wrong_mode_both":
        for rank in record["ranks"]:
            rank["requests"][0]["prefill_mode"] = "overlap"
    elif mutation == "missing_rank":
        record["ranks"].pop()
    elif mutation in ("extra_request", "duplicate_request"):
        record["ranks"][0]["requests"].append(record["ranks"][0]["requests"][0])
    elif mutation == "missing_request":
        record["ranks"][1]["requests"].pop()
    elif mutation == "source_coforge":
        record["sources"]["runtime.py"] = "f" * 64
        (tmp_path / "sources/runtime.py").write_text("forged")
    elif mutation == "source_missing":
        (tmp_path / "sources/runtime.py").unlink()
    elif mutation == "metadata":
        record["ranks"][1]["metadata_sha256"] = {}
    elif mutation == "tensor":
        record["ranks"][1]["tensor_sha256"]["test.shared"] = "f" * 64
    elif mutation == "output_both":
        for rank in record["ranks"]:
            rank["requests"][0]["token_ids"] = [248046]
    elif mutation == "prompt_both":
        for rank in record["ranks"]:
            rank["requests"][0]["prompt_token_ids"] = [99]
    elif mutation == "stop":
        record["ranks"][0]["requests"][0]["finish_reason"] = "length"
    elif mutation == "memory":
        record["ranks"][1]["requests"][0].pop("local_peak_mlx_bytes")
    elif mutation == "cleanup":
        record["cleanup"]["owned_processes_exited"] = False
    elif mutation == "pressure":
        record["ranks"][1]["pressure_guard"]["samples"][1]["pressure"] = 2
    elif mutation == "guard_disabled":
        record["configuration"]["require_normal_pressure"] = False
    elif mutation == "numeric_missing":
        record["ranks"][0]["numerical_gates"]["TEST-gate"]["TEST-boundary"].pop()
    elif mutation == "events_missing":
        record["request_evidence"][0]["prefill_events"] = [[], []]
    elif mutation == "unregistered_source":
        (tmp_path / "sources/unregistered_helper.py").write_text("# unregistered")
    elif mutation == "unbound_guard_source":
        trust["guard_source_sha256"] = "f" * 64
        for rank in record["ranks"]:
            rank["pressure_guard"]["guard_source_sha256"] = "f" * 64
    elif mutation == "uncovered_actor_interval":
        record["ranks"][1]["execution_interval"]["end_seconds"] = 2.5
    elif mutation == "bad_tensor_digest":
        trust["tensor_sha256_by_rank"][0]["test.a"] = "INVALID"
        record["ranks"][0]["tensor_sha256"]["test.a"] = "INVALID"
    with pytest.raises((ValueError, OSError)):
        h.audit_record(record, trust, schedule, tmp_path)
