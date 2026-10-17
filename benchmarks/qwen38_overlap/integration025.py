"""025 integration v2: preserve raw receipts and use pinned producer validators.

The v1 proposed-normalization API remains unchanged. This module never imports
MLX or invokes a producer at import time. Its runtime context is single-threaded
and restores Python's bare-module namespace after checking detached CPU evidence.
"""

import ast
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

from . import harness as h
from .numerical025 import expected_plan as expected_plan

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "spikes/025-qwen38-overlapped-prefill"
TEST_KIND = "TEST_FIXTURE_NOT_MODEL_RESULTS"
PHYSICAL_KIND = "PHYSICAL_RECEIPT"


def source_names(root):
    """Inspect the literal registry without executing unverified source."""
    tree = ast.parse(h.read_bytes(Path(root) / "remote_actor.py", 2 * 1024**2))
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "SOURCES" for t in node.targets)
    )
    h.require(
        isinstance(names, tuple)
        and len(set(names)) == len(names)
        and all(isinstance(n, str) and Path(n).name == n for n in names),
        "Invalid literal source registry",
    )
    return names


@contextmanager
def runtime_api(root, expected):
    """Load the real source-pinned CPU helpers; never stub an MLX dependency.

    This context is intentionally not thread-safe: source roots use bare imports.
    All possible local-module collisions are removed and restored, including lazy
    imports made by public verifier functions inside the context.
    """
    root = Path(root).absolute()
    h.require(set(expected) == set(source_names(root)), "External exact source set differs")
    for name, digest in expected.items():
        h.require(h.sha(h.read_bytes(root / name, 2 * 1024**2)) == digest, "Pinned source differs")
    names = {Path(n).stem for n in expected if n.endswith(".py")}
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))

    def api(name):
        h.require(name in names, "Unregistered verifier source")
        module = importlib.import_module(name)
        h.require(Path(module.__file__).absolute() == root / (name + ".py"), "Wrong source module")
        return module

    try:
        h.require(set(api("remote_actor").SOURCES) == set(expected), "Executed source set differs")
        yield api
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def require_pressure_policy(record, api, expected_policy):
    """Require the externally chosen policy AND an effective runtime guard."""
    guard = api("pressure_guard")
    h.require(
        type(record) is dict
        and record.get("require_normal_pressure") is (expected_policy == "normal")
        and guard.policy_name(record) == expected_policy
        and guard.pressure_required(record) is True,
        "Pressure policy/effective guard differs",
    )


def read_actor(path, digest, sources, api, *, expected_context, expected_policy="normal"):
    """Parse real job_status envelopes losslessly; no new producer field names."""
    import re

    raw = h.load_bound(path, digest)
    h.require(raw.get("source_sha256") == sources, "Actor source differs")
    nonce = raw.get("nonce")
    h.require(isinstance(nonce, str) and re.fullmatch("[0-9a-f]{32}", nonce), "Actor nonce differs")
    start, pids, result = (raw[k] for k in ("startup", "pids", "result"))
    h.require(all(type(v) is dict for v in (start, pids, result)), "Actor records must be objects")
    h.require(all(v.get("nonce") == nonce for v in (start, pids, result)), "Actor identity differs")
    h.require(
        type(start.get("actor_pid")) is int
        and start["actor_pid"] > 0
        and pids.get("actor_pid") == result.get("actor_pid") == start["actor_pid"]
        and type(pids.get("child_pid")) is int
        and pids["child_pid"] > 0
        and pids["child_pid"] == pids.get("pgid") == result.get("child_pid"),
        "Owned process identity differs",
    )
    require_pressure_policy(start, api, expected_policy)
    require_pressure_policy(result, api, expected_policy)
    h.require(
        result.get("reason") == "completed"
        and type(result.get("returncode")) is int
        and result["returncode"] == 0
        and result.get("error_type") is None
        and result.get("log_evidence_complete") is True
        and result.get("log_cap_exceeded") is False,
        "Incomplete actor result/log evidence",
    )
    cleanup = result.get("cleanup", {})
    h.require(
        cleanup.get("child_reaped") is True
        and cleanup.get("group_gone") is True
        and cleanup.get("pending_os_gpu_termination") is False,
        "Owned actor cleanup incomplete",
    )
    identity = {
        "nonce": nonce,
        "actor_pid": start["actor_pid"],
        "child_pid": pids["child_pid"],
        "host_role": start.get("host_role"),
        "actor_role": start.get("actor_role"),
    }
    h.require(
        type(expected_context) is dict
        and all(
            expected_context.get(k) == v and type(expected_context.get(k)) is type(v)
            for k, v in identity.items()
        ),
        "Independent actor identity differs",
    )
    for field in ("actor_started_monotonic", "execution", "elapsed"):
        h.require(
            expected_context.get(field) == result.get(field),
            "Independent actor execution interval differs",
        )
    h.require(
        start.get("actor_started_monotonic") == result.get("actor_started_monotonic"),
        "Actor clock origin differs",
    )
    binding = api("pressure_guard").expected_pressure_binding(
        raw,
        nonce=identity["nonce"],
        host_role=identity["host_role"],
        actor_role=identity["actor_role"],
        expected_policy=expected_policy,
    )
    h.require(
        result["pressure"].get("sampler_timeout_seconds") == 0.25, "Wrong pressure sampler timeout"
    )
    logs = {}
    for name in ("stdout", "stderr"):
        text = raw.get(name)
        h.require(
            type(text) is str and len(text.encode()) <= 64 * 1024**2, "Actor log budget exceeded"
        )
        logs[name] = [h.decode_json(line.encode()) for line in text.splitlines() if line.strip()]
        h.require(all(type(v) is dict for v in logs[name]), "Actor log record is not an object")
    h.require(logs["stdout"], "Missing actual actor output")
    return {
        "raw": raw,
        "input_sha256": digest,
        "actor_nonce": nonce,
        "pressure_binding": binding,
        "output": logs["stdout"][-1],
        "events": logs["stderr"],
    }


def audit_raw_request(spec, result, events, prompt, layout, rank, api):
    """Call actual025 grammar helpers on untouched producer events, never rename."""
    api("overlap_evidence").verify_request_modes(spec, result, events, enabled=True)
    h.require(
        result.get("local_rank") == rank and result.get("prompt_token_ids") == prompt,
        "Rank/prompt identity differs",
    )
    actions = [v for v in events if v.get("phase") == "rank_action"]
    h.require(
        actions
        and result.get("request_nonce") == actions[0].get("request")
        and result.get("cache_epoch") == result.get("request_nonce"),
        "Request/cache nonce differs",
    )
    api("mtp_audit").validate_mtp_actions(
        result,
        actions,
        prompt,
        expected_mode=("native", "slot-qmv")[rank],
        hidden_only_prefill=True,
        prefill_mode=spec["prefill_mode"],
    )
    compounds = [v for v in actions if v["op"] == "prefill_pipeline"]
    if spec["prefill_mode"] == "overlap" and len(prompt) > 1:
        h.require(len(compounds) == 1, "Compound count differs")
        value = compounds[0]
        command = api("rank_control").Command(
            value["request"], value["sequence"], value["op"], value["tokens"]
        )
        summary = api("overlap_evidence").verify_overlap_stream(
            events,
            command,
            rank,
            owned_layers=range(*layout["owned_layers"]),
            require_native=True,
            hidden_size=layout["hidden_size"],
            dtype=layout["dtype"],
            eval_group_size=32,
        )
        api("overlap_evidence").verify_compound_drain(events, command, rank)
        return summary
    h.require(
        not compounds
        and not any(v.get("schema") == "qwen38-prefill-overlap-stream-v1" for v in events),
        "Serial/singleton unexpectedly performed compound work",
    )
    return {"chunks": len(range(0, len(prompt) - 1, 256)), "mode": spec["prefill_mode"]}


RAW_REQUIRED = {
    "rank0.receipt.json",
    "rank1.receipt.json",
    "server.receipt.json",
    "startup.json",
    "result.json",
    "schedule_admission.json",
    "source_sha256.json",
    "controller_sha256.json",
    "peer_source.json",
    "listeners.cleanup.json",
    "actors.json",
    "rank0.job.json",
    "rank1.job.json",
    "server.job.json",
}


def audit_raw_trial(root, trusted, manifest):
    """Lossless raw025 receipt adapter; all producer dictionaries remain intact.

    The public return value intentionally contains only hashes and public metrics.
    Raw bytes remain at their original paths, bound by the parent's sealed hashes.
    No new completion/pressure/cleanup events or producer pass bits are created.
    """
    from types import SimpleNamespace

    from .cli import HISTORICAL, encoded
    from .numerical025 import audit_legacy
    from .supervisor025 import expected_pressure_policy

    policy = expected_pressure_policy(manifest)

    h.require(
        trusted.get("schema") == "qwen38-overlap-raw025-trusted-v2",
        "External raw025 trust required",
    )
    h.require(
        trusted.get("artifact_kind") in (TEST_KIND, PHYSICAL_KIND),
        "Explicit raw025 provenance required",
    )
    explicit_policy = policy == "noncritical" or trusted["artifact_kind"] == PHYSICAL_KIND
    h.require(
        trusted.get("pressure_policy", None if explicit_policy else "normal") == policy,
        "Independent trusted pressure policy differs",
    )
    phase = trusted["phase"]
    item = manifest["phases"][phase]
    h.require(
        trusted.get("manifest_sha256") == h.sha(encoded(manifest)), "External freeze SHA256 differs"
    )
    root = Path(root)
    hashes = trusted.get("raw_artifacts", {})
    h.require(RAW_REQUIRED <= set(hashes), "Missing independently sealed raw artifact")
    raw = {}
    for name, digest in hashes.items():
        h.require(Path(name).name == name and name.endswith(".json"), "Unsafe raw artifact name")
        raw[name] = h.load_bound(root / name, digest)
    sources = item["sources"]
    h.require(
        raw["source_sha256.json"] == sources
        and raw["peer_source.json"].get("source_sha256") == sources
        and raw["controller_sha256.json"] == {"supervise.py": sources["supervise.py"]},
        "Independent source/controller readback differs",
    )
    source_root = root / "sources"
    h.require(
        {p.name for p in source_root.iterdir()} == set(sources),
        "Executed source snapshot set differs",
    )
    schedule = h.load_bound(source_root / "benchmark_schedule.json", item["schedule_sha256"])
    full = h.derive_schedule(HISTORICAL)
    expected = (
        h.derive_pilot(full)["batches"][0] if phase == "pilot" else full["batches"][int(phase[-2:])]
    )
    h.require(schedule == expected, "Raw schedule rows/modes/order differ")
    prompt_ids = trusted.get("prompt_token_ids")
    h.require(
        isinstance(prompt_ids, list)
        and len(prompt_ids) == len(expected["requests"])
        and all(
            isinstance(p, list) and p and all(type(t) is int and t >= 0 for t in p)
            for p in prompt_ids
        ),
        "Independent prompt ids required",
    )
    admission = raw["schedule_admission.json"]
    h.require(
        admission
        == {
            "sha256": item["schedule_sha256"],
            "requests": expected["requests"],
            "prompt_token_ids": prompt_ids,
        },
        "Schedule admission differs from independently trusted ids/rows",
    )
    startup, final = raw["startup.json"], raw["result.json"]
    h.require(
        final.get("nonce") == startup.get("nonce")
        and final.get("status") == "success"
        and final.get("error_type") is None
        and final.get("owned_actor_cleanup_verified") is True
        and final.get("services_closed_verified") is True
        and final.get("cleanup_errors") == []
        and not final.get("pending_cleanup", False),
        "Controller cleanup/final evidence incomplete",
    )
    transports = final.get("transport_cleanup")
    controls = final.get("control_transport_cleanup")
    h.require(
        type(transports) is dict
        and set(transports) == {"rank0", "rank1", "server"}
        and type(controls) is dict
        and 1 <= len(controls) <= 128,
        "Controller transport cleanup inventory differs",
    )
    for cleanup in [*transports.values(), *controls.values()]:
        h.require(
            type(cleanup) is dict
            and cleanup.get("child_reaped") is True
            and cleanup.get("group_gone") is True
            and cleanup.get("pending_os_gpu_termination") is False,
            "Controller transport cleanup incomplete",
        )
    listeners = raw["listeners.cleanup.json"]
    endpoints = trusted.get("listener_endpoints")
    h.require(
        type(endpoints) is list
        and len(endpoints) == 3
        and all(
            type(e) is list
            and len(e) == 2
            and type(e[0]) is str
            and type(e[1]) is int
            and 1 <= e[1] <= 65535
            for e in endpoints
        )
        and len({tuple(e) for e in endpoints}) == 3
        and [e[0] for e in endpoints]
        == [
            startup["args"]["local_address"],
            startup["args"]["remote_address"],
            startup["args"]["remote_address"],
        ],
        "Listener independent endpoint identity differs",
    )
    h.require(
        listeners.get("services_closed_verified") is True
        and listeners.get("ports") == endpoints
        and listeners.get("closed") == [True, True, True],
        "Listener closure evidence incomplete",
    )
    target, small = manifest["target"], manifest["small"]
    plan = expected_plan(target, small)
    h.require(plan == manifest["numerical_plan"], "Architecture expected plan differs")
    options = dict(startup["args"])
    required_options = {
        "overlapped_prefill": True,
        "hidden_only_prefill": True,
        "require_normal_pressure": policy == "normal",
        "prefix_rollback": True,
        "hidden_gate_chunk_size": 256,
        "split": [28, 36],
        "eval_group_size": 32,
        "eval_group_schedule": None,
        "rank0_verifier": "native",
        "rank1_verifier": "slot-qmv",
        "benchmark_schedule": True,
        "benchmark_schedule_sha256": item["schedule_sha256"],
        "target_revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "target_config_sha256": target["config_sha256"],
        "target_index_sha256": target["index_sha256"],
        "small_config_sha256": small["config_sha256"],
        "small_index_sha256": small["index_sha256"],
        "repeats": item["requests"],
        "max_tokens": item["max_tokens"],
        "deadline": 900.0,
        "prompt": "tokens:[1,2,3,4,5]",
        "execution": "phased",
        "verification_gate_only": False,
        "reserve_bytes": 8 * 1024**3,
    }
    for key, value in required_options.items():
        h.require(
            options.get(key) == value
            and (type(options.get(key)) is type(value) or key == "deadline"),
            "Controller fixed options differ",
        )
    # Local paths can relocate; independently supplied metadata is authoritative.
    options.update(local_metadata=target["path"], small_model_local=small["path"])
    bindings = {
        "config.json": target["config_sha256"],
        "model.safetensors.index.json": target["index_sha256"],
    }
    h.require(
        startup.get("metadata_sha256") == final.get("metadata_sha256") == bindings,
        "Controller metadata identity differs",
    )
    ranks, phases, actors, request_metrics = [], {}, [], []
    native_outputs = trusted.get("native_outputs")
    tensor_hashes = trusted.get("tensor_sha256_by_rank")
    h.require(
        isinstance(native_outputs, list)
        and len(native_outputs) == item["requests"]
        and isinstance(tensor_hashes, list)
        and len(tensor_hashes) == 2,
        "Independent original native outputs/tensors required",
    )
    with runtime_api(source_root, sources) as api:
        require_pressure_policy(options, api, policy)
        statuses = []
        for name in ("rank0", "rank1", "server"):
            context = trusted.get("actor_contexts", {}).get(name)
            host = "local" if name == "rank0" else "remote"
            h.require(
                type(context) is dict
                and context.get("pressure_policy", None if explicit_policy else "normal") == policy
                and context.get("actor_role") == name
                and context.get("host_role") == host
                and raw["actors.json"][name].get("remote") is (host == "remote"),
                "Independent actor role/host differs",
            )
            parsed = read_actor(
                root / (name + ".receipt.json"),
                hashes[name + ".receipt.json"],
                sources,
                api,
                expected_context=context,
                expected_policy=policy,
            )
            statuses.append(parsed)
            job, actor = raw[name + ".job.json"], parsed["raw"]
            require_pressure_policy(job, api, policy)
            h.require(
                raw["actors.json"][name]["nonce"] == job["nonce"] == actor["nonce"]
                and job.get("source_sha256") == sources
                and job.get("host_role") == host
                and job.get("actor_role") == name,
                "Actor job/source/nonce differs",
            )
            actors.append(actor["result"])
        h.require(len({v["actor_nonce"] for v in statuses}) == 3, "Actor nonce reused")
        config = h.load_bound(Path(target["path"]) / "config.json", target["config_sha256"])
        index = h.load_bound(
            Path(target["path"]) / "model.safetensors.index.json", target["index_sha256"]
        )
        inventory = api("supervise").expected_inventory(config, index["weight_map"], [28, 36])
        h.require(
            set().union(*inventory.values()) and len(set().union(*inventory.values())) == 1847,
            "Target index inventory differs",
        )
        for rid in (0, 1):
            parsed = statuses[rid]
            rank, events = parsed["output"], parsed["events"]
            starts = [e for e in events if e.get("phase") == "process_start"]
            h.require(
                len(starts) == 1 and starts[0].get("pid") == parsed["raw"]["pids"]["child_pid"],
                "Actual child start identity differs",
            )
            require_pressure_policy(starts[0], api, policy)
            require_pressure_policy(rank, api, policy)
            h.require(
                type(rank.get("rank")) is int and rank["rank"] == rid,
                "Rank flag/identity differs",
            )
            import json

            h.require(
                rank.get("prompt") == options["prompt"]
                and rank.get("formatted_prompt") == json.dumps(plan["width_prompt_ids"]),
                "Raw width gate prompt differs from independent token plan",
            )
            h.require(
                set(tensor_hashes[rid]) == inventory[rid]
                and rank["weight_loading"]["tensor_sha256"] == tensor_hashes[rid],
                "Independent native tensor hashes differ",
            )
            for scope in ("small", "full"):
                evidence = rank["verification"]["groups"]["32"][scope]
                expected_scope = plan[scope][str(rid)]
                audit_legacy(evidence, expected_scope, api)
                api("overlap_gate").verify_overlap_gate(
                    evidence.get("overlapped_prefill"),
                    expected_layout=expected_scope["overlap_layout"],
                    scope=scope,
                    eval_group_size=32,
                )
                lo, hi = expected_scope["native_layout"]["owned_layers"]
                api("hidden_prefill_gate").verify_prefill_evidence(
                    evidence.get("hidden_only_prefill"),
                    scope=scope,
                    chunk_size=256,
                    layer_count=hi - lo,
                    expected_layout=expected_scope["native_layout"],
                )
            h.require(len(rank["requests"]) == item["requests"], "Raw request row count differs")
            nonces = []
            for ordinal, spec in enumerate(expected["requests"]):
                value = rank["requests"][ordinal]
                active = [e for e in events if e.get("repeat") == ordinal]
                audit_raw_request(
                    spec,
                    value,
                    active,
                    prompt_ids[ordinal],
                    plan["full"][str(rid)]["overlap_layout"],
                    rid,
                    api,
                )
                h.require(
                    {k: value.get(k) for k in ("token_ids", "finish_reason")}
                    == native_outputs[ordinal],
                    "Original native output/stop differs",
                )
                nonces.append(value["request_nonce"])
                request_metrics.append(
                    {
                        "rank": rid,
                        "id": spec["id"],
                        "prefill_mode": spec["prefill_mode"],
                        "request_nonce": value["request_nonce"],
                        "cache_epoch": value["cache_epoch"],
                        **{
                            k: value[k]
                            for k in (
                                "local_peak_mlx_bytes",
                                "local_rank_wall_seconds",
                                "accepted_drafts",
                                "verification_rounds",
                                "teacher_forwards",
                            )
                        },
                    }
                )
            h.require(len(set(nonces)) == len(nonces), "Request nonce/cache epoch reused")
            ranks.append(rank)
            phases[rid] = events
        # Never let a controller/producer flag bypass actor coverage verification.
        api("pressure_guard").verify_actor_pressures(
            actors,
            expected_bindings=[v["pressure_binding"] for v in statuses],
            expected_policy=policy,
        )
        canonical = api("supervise").verify_completed_trial(
            ranks=ranks,
            server=statuses[2]["output"],
            receipts=actors,
            expected_pressure_bindings=[v["pressure_binding"] for v in statuses],
            weight_files=api("supervise").weight_files(Path(target["path"])),
            args=SimpleNamespace(**options),
            phases=phases,
            source_sha256=sources,
            expected_inventory=inventory,
            rows=expected["requests"],
            prompt_ids=prompt_ids,
        )
        budget = (
            api("overlap_evidence").pilot_budget_check(
                [v["raw"] for v in statuses],
                requests_observed=item["requests"],
                requests_planned=14,
            )
            if phase == "pilot"
            else None
        )
    rows = []
    for spec, a, b in zip(
        expected["requests"], ranks[0]["requests"], ranks[1]["requests"], strict=True
    ):
        rows.append(
            {
                **spec,
                **{
                    k: a[k]
                    for k in (
                        "token_ids",
                        "finish_reason",
                        "first_token_seconds",
                        "wall_seconds",
                        "prompt_token_ids",
                    )
                },
                "status": "success",
                "peak_mlx_bytes_by_rank": [a["local_peak_mlx_bytes"], b["local_peak_mlx_bytes"]],
            }
        )
    audit = {
        "schema": "qwen38-overlap-raw025-audit-v2",
        "artifact_kind": trusted["artifact_kind"],
        "phase": phase,
        "manifest_sha256": trusted["manifest_sha256"],
        "raw_artifacts": hashes,
        "source_sha256": sources,
        "canonical025_valid": True,
        "architecture_metrics_valid": True,
        "original_native_bound": True,
        "canonical025_verdict": canonical,
        "pilot_budget": budget,
        "request_metrics_by_rank": request_metrics,
        "parent_review": trusted.get("parent_review"),
        "rows": rows,
    }
    if policy == "noncritical":
        audit.update(
            pressure_policy=policy,
            memory_acceptance_sha256=manifest["memory_acceptance"]["sha256"],
            pressure_observations=[
                {
                    "host_role": actor["host_role"],
                    "actor_role": actor["actor_role"],
                    "observed_levels": sorted({s["level"] for s in actor["pressure"]["samples"]}),
                    "warning_samples": sum(s["level"] == 2 for s in actor["pressure"]["samples"]),
                }
                for actor in actors
            ],
        )
    audit["physical_accepted"] = physical_acceptance(audit)
    return audit


def physical_acceptance(audit):
    """Not a constant-false placeholder: only source-bound, parent-reviewed REAL evidence.

    This predicate consumes audit_raw_trial's result, never producer pass flags.
    The caller must authenticate its trusted manifest before running the adapter.
    """
    review = audit.get("parent_review") or {}
    import re

    return bool(
        audit.get("schema") == "qwen38-overlap-raw025-audit-v2"
        and audit.get("artifact_kind") == PHYSICAL_KIND
        and audit.get("canonical025_valid") is True
        and audit.get("architecture_metrics_valid") is True
        and audit.get("original_native_bound") is True
        and RAW_REQUIRED <= set(audit.get("raw_artifacts", {}))
        and audit.get("source_sha256")
        and review.get("approved") is True
        and review.get("manifest_sha256") == audit.get("manifest_sha256")
        and review.get("phase") == audit.get("phase")
        and isinstance(review.get("review_receipt_sha256"), str)
        and re.fullmatch("[0-9a-f]{64}", review["review_receipt_sha256"])
    )


def collect_raw_trials(trials, manifest):
    """Audit sealed raw receipts before the existing exact paired analyzer."""
    from .cli import HISTORICAL

    full = h.derive_schedule(HISTORICAL)
    phases = [trust.get("phase") for _, trust in trials]
    pilot = phases == ["pilot"]
    schedule = h.derive_pilot(full) if pilot else full
    if pilot:
        measured = {r["case_id"] for r in h.flatten(schedule) if not r["warmup"]}
        schedule = {**schedule, "cases": [c for c in schedule["cases"] if c["id"] in measured]}
    audits, rows, failures = [], [], []
    for index, (root, trust) in enumerate(trials):
        try:
            audit = audit_raw_trial(root, trust, manifest)
            rows.extend(audit["rows"])
            audits.append({k: v for k, v in audit.items() if k != "rows"})
        except (ValueError, OSError, KeyError, TypeError, IndexError):
            # No raw exceptions, private startup args or uncontrolled receipt text.
            failures.append({"trial_index": index, "code": "raw025_audit_failed"})
            # Retain actual failed rows if their raw rank0 input was independently
            # sealed; never repair old mode/IDs into new025 rows or mark success.
            try:
                from .cli import public_rows

                digest = trust["raw_artifacts"]["rank0.receipt.json"]
                original = h.load_bound(Path(root) / "rank0.receipt.json", digest)
                rank = h.decode_json(original["stdout"].splitlines()[-1].encode())
                rows.extend(
                    {**r, "status": "failed"} for r in public_rows(rank.get("requests", []))
                )
            except (ValueError, OSError, KeyError, TypeError, IndexError):
                pass
    report = h.analyze(rows, schedule)
    complete = not failures and len(set(phases)) == len(phases) and report["coverage_complete"]
    physical = complete and bool(audits) and all(v["physical_accepted"] for v in audits)
    report.update(
        schema="qwen38-overlap-raw025-collection-v2",
        artifact_kind=trials[0][1].get("artifact_kind") if trials else "UNVERIFIED",
        rows=rows,
        record_audits=audits,
        evidence_errors=failures,
        evidence_validation="source_bound_raw025_valid" if not failures else "invalid",
        physical_evidence_accepted=physical,
        complete_suite=physical and not pilot,
        accepted=physical and (pilot or report["statistical_targets_met"]),
    )
    if manifest.get("pressure_policy") == "noncritical":
        report.update(
            pressure_policy="noncritical",
            pressure_observations=[
                {
                    "phase": audit["phase"],
                    "actors": audit["pressure_observations"],
                }
                for audit in audits
            ],
        )
    return report
