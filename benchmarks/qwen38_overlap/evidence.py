"""CPU evidence collector for a proposed normalized contract, not raw025 adapter.

External caller must pin the trusted manifest AND frozen schedule independently.
Receipt-selected registries never establish trust. Physical acceptance stays
blocked until a reviewed raw025 adapter and numerical-plan derivation exist.
"""

import re
from pathlib import Path

from . import harness as h
from .protocol import audit_decode, audit_transcript, same, validate_output

INTEGRATION_STATUS = "pending_runtime025_adapter_and_independent_review"
TEST_KIND = "TEST_FIXTURE_NOT_MODEL_RESULTS"
CONFIGURATION = {
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
MIN_RUNTIME_SOURCES = set(
    (
        "cache_state.py cpu_communication.py distributed_control.py hidden_prefill_gate.py "
        "mlx_backend.py mtp_audit.py mtp_head.py mtp_runtime.py numerical_gate.py phased_decode.py "
        "prefix_rollback.py probe.py rank_control.py remote_actor.py shard_plan.py "
        "sidecar_config.json "
        "slot_qmv.py slot_qmv_body.metal slot_qmv_header.metal speculative.py stream_server.py "
        "stream_weights.py supervise.py verification.py verifier.py width_gate.py "
        "workload_evidence.py workload_runtime.py workload_schedule.py benchmark_schedule.json"
    ).split()
)
REQUIRED_RAW = {
    "rank0.receipt.json",
    "rank1.receipt.json",
    "server.receipt.json",
    "startup.json",
    "final.json",
    "schedule_admission.json",
    "controller_sha256.json",
    "source_sha256.json",
}
REQUIRED_GATES = (
    {f"small-overlap-{n}-chunk256" for n in (1, 2, 256, 257, 258, 511, 512, 513)}
    | {f"full-overlap-{n}-chunk256" for n in (1, 258, 513)}
    | {f"{scope}-width2-prefix{p}" for scope in ("small", "full") for p in (0, 1, 2)}
    | {"small-hidden-only", "full-hidden-only-native256"}
)


def safe_relative(root, name):
    value = Path(name)
    h.require(
        not value.is_absolute() and value.parts and ".." not in value.parts, "Unsafe artifact path"
    )
    return Path(root) / value


def audit_record(record, trusted, schedule, record_root):
    h.require(
        trusted.get("schema") == "qwen38-overlap-trusted-input-v1",
        "External trusted manifest required",
    )
    h.require(
        record.get("schema") == "qwen38-overlap-normalized-record-v1",
        "Unsupported raw record: runtime adapter pending",
    )
    kind = trusted.get("artifact_kind")
    h.require(
        kind in (TEST_KIND, "PHYSICAL_RECEIPT") and record.get("artifact_kind") == kind,
        "Missing evidence provenance kind",
    )
    fixture = kind == TEST_KIND
    batch_index = record.get("batch_index")
    h.require(
        type(batch_index) is int and 0 <= batch_index < len(schedule["batches"]),
        "Batch index differs",
    )
    specs = schedule["batches"][batch_index]["requests"]
    h.require(
        same(trusted.get("configuration"), CONFIGURATION)
        and same(record.get("configuration"), CONFIGURATION),
        "Fixed arithmetic/placement/flags differ",
    )
    sources = trusted.get("sources")
    h.require(
        isinstance(sources, dict)
        and bool(sources)
        and set(sources) == set(trusted.get("required_source_names", [])),
        "External exact source registry required",
    )
    h.require(
        same(record.get("sources"), sources),
        "Receipt registry differs from external trusted registry",
    )
    h.require(
        trusted.get("guard_source_sha256") in sources.values(),
        "Pressure guard source lacks externally frozen source attestation",
    )
    if not fixture:
        h.require(
            trusted["guard_source_sha256"] == sources.get("remote_actor.py"),
            "Physical pressure watchdog must bind actual remote_actor source",
        )
        h.require(
            MIN_RUNTIME_SOURCES <= set(sources), "Runtime/helper/Metal/supervisor source omitted"
        )
        h.require(
            REQUIRED_RAW <= set(trusted.get("raw_artifacts", {})),
            "Raw receipts missing independent bindings",
        )
    source_root = Path(record_root) / "sources"
    actual_source_files = {
        p.relative_to(source_root).as_posix()
        for p in source_root.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    h.require(
        actual_source_files == set(sources),
        "Source snapshot set differs from external exact registry",
    )
    for name, digest in sources.items():
        data = h.read_bytes(safe_relative(source_root, name), 64 * 1024 * 1024)
        h.require(h.sha(data) == digest, "Executed source snapshot differs")
    for name, digest in trusted.get("raw_artifacts", {}).items():
        h.require(
            h.sha(h.read_bytes(safe_relative(record_root, name))) == digest, "Raw artifact differs"
        )
    metadata = trusted.get("metadata_sha256")
    h.require(
        isinstance(metadata, dict)
        and {"config", "index", "sidecar"} <= set(metadata)
        and same(record.get("metadata_sha256"), metadata),
        "Checkpoint metadata binding differs",
    )
    h.require(
        same(
            record.get("cleanup"),
            {
                "owned_processes_exited": True,
                "owned_listeners_closed": True,
                "unrelated_processes_signalled": False,
            },
        ),
        "Cleanup not proven",
    )
    ranks = record.get("ranks")
    h.require(isinstance(ranks, list) and len(ranks) == 2, "Both rank receipts required")
    tensors = trusted.get("tensor_sha256_by_rank")
    h.require(isinstance(tensors, list) and len(tensors) == 2, "Trusted tensor sets required")
    union = {}
    for rank, hashes in enumerate(tensors):
        h.require(isinstance(hashes, dict) and bool(hashes), "Empty rank tensor inventory")
        for name, digest in hashes.items():
            h.require(
                isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                "Malformed trusted tensor SHA256",
            )
            h.require(union.setdefault(name, digest) == digest, "Replicated tensor hash differs")
    if not fixture:
        h.require(len(union) == 1847, "Target union must contain exact 1847 tensors")
    numerical_plans = trusted.get("numerical_plan_by_rank")
    h.require(
        isinstance(numerical_plans, list) and len(numerical_plans) == 2,
        "Independent architecture numerical plans required",
    )
    for rid, rank in enumerate(ranks):
        h.require(
            type(rank.get("rank")) is int and rank["rank"] == rid,
            "Rank receipt order/identity differs",
        )
        for key, expected in (
            ("source_sha256", sources),
            ("configuration", CONFIGURATION),
            ("metadata_sha256", metadata),
            ("tensor_sha256", tensors[rid]),
        ):
            h.require(
                same(rank.get(key), expected), "Rank source/configuration/metadata/tensors differ"
            )
        h.require(
            rank.get("verifier_mode") == CONFIGURATION["verifier_modes"][rid],
            "Verifier mode differs",
        )
        h.require(
            type(rank.get("parameter_bytes")) is int and rank["parameter_bytes"] > 0,
            "Missing independent parameter bytes",
        )
        h.require(
            rank.get("log_complete") is True
            and type(rank.get("returncode")) is int
            and rank["returncode"] == 0
            and rank.get("reason") == "completed",
            "Failed/incomplete rank receipt",
        )
        guard = rank.get("pressure_guard", {})
        interval = rank.get("execution_interval", {})
        h.require(
            same(
                interval,
                {
                    "spawn_seconds": guard.get("spawn_seconds"),
                    "end_seconds": guard.get("execution_end_seconds"),
                },
            ),
            "Uncovered actor execution interval",
        )
        h.audit_pressure(
            guard, trusted["guard_source_sha256"], max_gap=trusted["max_pressure_gap_seconds"]
        )
        h.require(
            type(rank.get("reserve_bytes")) is int and rank["reserve_bytes"] >= 8 * 1024**3,
            "Missing 8GiB reserve",
        )
        plan, gates = numerical_plans[rid], rank.get("numerical_gates")
        h.require(
            isinstance(plan, dict)
            and bool(plan)
            and isinstance(gates, dict)
            and set(gates) == set(plan),
            "Missing/surplus mandatory numerical gates",
        )
        if not fixture:
            h.require(
                REQUIRED_GATES <= set(plan), "Missing full513/chunk256/small/rollback/hidden gates"
            )
        for gate, boundaries in plan.items():
            h.require(
                isinstance(boundaries, dict)
                and bool(boundaries)
                and set(gates[gate]) == set(boundaries),
                "Missing numerical boundary",
            )
            for boundary, inventory in boundaries.items():
                h.audit_numeric_inventory(gates[gate][boundary], inventory)
        h.require(
            isinstance(rank.get("requests"), list) and len(rank["requests"]) == len(specs),
            "Request set not exact",
        )
    expected_requests = trusted.get("requests")
    h.require(
        isinstance(expected_requests, dict) and set(expected_requests) == {s["id"] for s in specs},
        "External exact request/prompt/output set required",
    )
    evidence = record.get("request_evidence")
    h.require(
        isinstance(evidence, list) and len(evidence) == len(specs),
        "Missing/surplus action evidence",
    )
    seen_nonces, seen_epochs, rows = set(), set(), []
    for ordinal, spec in enumerate(specs):
        expected, item = expected_requests[spec["id"]], evidence[ordinal]
        h.require(item.get("id") == spec["id"], "Action request order differs")
        binding = {k: v for k, v in expected.items() if k not in ("token_ids", "finish_reason")}
        h.require(
            same(item.get("binding"), binding), "Action binding differs from external manifest"
        )
        h.require(
            binding["id"] == spec["id"]
            and type(binding["ordinal"]) is int
            and binding["ordinal"] == ordinal
            and binding["prefill_mode"] == spec["prefill_mode"],
            "Request mode/ordinal differs",
        )
        h.require(
            binding["actor_nonces"] == [r["actor_nonce"] for r in ranks], "Actor nonce differs"
        )
        h.require(
            binding["request_nonce"] not in seen_nonces
            and binding["cache_epoch"] not in seen_epochs,
            "Reused request nonce/cache epoch",
        )
        seen_nonces.add(binding["request_nonce"])
        seen_epochs.add(binding["cache_epoch"])
        prompt = binding["prompt_token_ids"]
        h.require(
            binding["layers"] == [list(range(36, 64)), list(range(36))], "Layer placement differs"
        )
        if not spec["warmup"]:
            case = next(c for c in schedule["cases"] if c["id"] == spec["case_id"])
            h.require(
                len(prompt) == case["formatted_prompt_tokens"], "Formatted prompt length differs"
            )
        h.require(len(prompt) + spec["max_tokens"] <= 4096, "Context cap differs")
        audit_transcript(item["prefill_events"], binding)
        left, right = (rank["requests"][ordinal] for rank in ranks)
        for rid, request in enumerate((left, right)):
            h.require(
                all(same(request.get(k), v) for k, v in spec.items()),
                "Actual row differs from scheduled request/mode",
            )
            h.require(
                request.get("status") == "success", "Failed request retained but not admitted"
            )
            h.require(
                same(request.get("prompt_token_ids"), prompt)
                and same(request.get("token_ids"), expected["token_ids"])
                and request.get("finish_reason") == expected["finish_reason"],
                "Prompt/output/stops differ from external reference",
            )
            validate_output(request)
            h.require(
                all(h.positive(request.get(k)) for k in h.METRICS)
                and request["wall_seconds"] >= request["first_token_seconds"],
                "Invalid elapsed metrics",
            )
            h.require(
                type(request.get("local_peak_mlx_bytes")) is int
                and request["local_peak_mlx_bytes"] > 0,
                "Missing independent rank peak memory",
            )
            start_sequence = item["prefill_events"][rid][-1]["command_sequence"] + 1
            audit_decode(
                item["decode_actions"][rid],
                request,
                binding,
                rank=rid,
                start_sequence=start_sequence,
            )
        compare = (
            "token_ids",
            "finish_reason",
            "prompt_token_ids",
            "accepted_drafts",
            "verification_rounds",
            "teacher_forwards",
        )
        h.require(
            all(same(left[k], right[k]) for k in compare), "Rank action/output progress differs"
        )
        public_fields = (
            set(spec) | set(compare) | set(h.METRICS) | {"status", "local_peak_mlx_bytes"}
        )
        rows.append(
            {
                **{k: v for k, v in left.items() if k in public_fields},
                "peak_mlx_bytes_by_rank": [
                    left["local_peak_mlx_bytes"],
                    right["local_peak_mlx_bytes"],
                ],
            }
        )
    return {
        "schema": "qwen38-overlap-record-audit-v1",
        "artifact_kind": kind,
        "batch_index": batch_index,
        "rows": rows,
        "evidence_contract_valid": True,
        "accepted": False,
        "integration_status": INTEGRATION_STATUS,
        "pressure_scope": "sampled_not_continuous",
        "physical_numerical_admission": "not_minted_by_CPU_audit",
    }
