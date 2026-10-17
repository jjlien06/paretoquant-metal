"""Dedicated schedule evidence verification; original trial defaults stay strict."""

import hashlib
import math
import re
from collections import defaultdict

from workload_schedule import OVERLAP_SCHEMA, SCHEMA, validate_schedule


def phase_timings(events):
    """Observed local-rank wall intervals; nested, overlapping, NOT GPU attribution."""
    pending, durations = defaultdict(list), defaultdict(list)
    previous_action = None
    for e in events:
        now = e.get("elapsed_seconds")
        if type(now) not in (int, float) or not math.isfinite(now):
            continue
        phase = e.get("phase", "")
        key = (
            e.get("repeat"),
            phase.rsplit("_", 1)[0],
            e.get("layer_index"),
            e.get("start_idx"),
            e.get("end_idx"),
        )
        if phase.endswith("_before"):
            pending[key].append(now)
        elif phase.endswith("_done") and pending[key]:
            start = pending[key].pop()
            if now >= start:
                durations[key[1]].append(now - start)
        if phase == "rank_action":
            if previous_action is not None and previous_action[0] == e.get("repeat"):
                durations["inter_action_until_" + e["op"]].append(now - previous_action[1])
            previous_action = (e.get("repeat"), now)
    return {
        name: {
            "count": len(values),
            "wall_seconds": sum(values),
            "max_seconds": max(values),
            "scope": (
                "local_rank wall; synchronization included; overlapping/non-additive; "
                "inter_action includes intervening draft/control work"
            ),
        }
        for name, values in durations.items()
    }


def verify_schedule_trial(
    *,
    ranks,
    server,
    receipts,
    weight_files,
    rows,
    prompt_ids,
    args,
    phases,
    source_sha256,
    expected_inventory,
    expected_pressure_bindings=None,
):
    from mtp_audit import validate_mtp_actions
    from remote_actor import SOURCES
    from supervise import verify_mtp_evidence, verify_sources, verify_trial

    validate_schedule(
        {
            "schema": OVERLAP_SCHEMA if any("prefill_mode" in r for r in rows) else SCHEMA,
            "requests": rows,
        }
    )
    if (
        not isinstance(source_sha256, dict)
        or set(source_sha256) != set(SOURCES)
        or any(
            not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None
            for v in source_sha256.values()
        )
        or source_sha256["benchmark_schedule.json"] != args.benchmark_schedule_sha256
    ):
        raise ValueError("Missing exact registry/schedule SHA binding")
    if (
        not isinstance(phases, dict)
        or set(phases) != {0, 1}
        or len(ranks) != 2
        or sorted(r.get("rank", -1) for r in ranks) != [0, 1]
        or len(prompt_ids) != len(rows)
        or args.repeats != len(rows)
    ):
        raise ValueError("Missing trusted schedule/rank/prompt evidence")
    from overlap_prefill import validate_modes
    from pressure_guard import verify_actor_pressures

    validate_modes(args, rows)
    if getattr(args, "require_normal_pressure", False):
        verify_actor_pressures(receipts, expected_bindings=expected_pressure_bindings)
    by_rank = {r["rank"]: r for r in ranks}
    count = 0 if args.verification_gate_only else len(rows)
    for rank in ranks:
        rid = rank["rank"]
        if (
            rank.get("benchmark_schedule_sha256") != args.benchmark_schedule_sha256
            or len(rank.get("requests", [])) != count
        ):
            raise ValueError("Rank schedule hash/count mismatch")
        if count and rank.get("sidecar_resident_on_rank0") is not True:
            raise ValueError("Missing sidecar residency disclosure")
        events = phases[rid]
        if any("prefill_mode" in row for row in rows):
            modes = [
                {k: row[k] for k in ("id", "prefill_mode", "source_request_id")} for row in rows
            ]
            admitted = [
                event for event in events if event.get("phase") == "numerical_admission_done"
            ]
            if (
                rank.get("verification", {}).get("request_prefill_modes") != modes
                or len(admitted) != 1
                or admitted[0].get("request_prefill_modes") != modes
            ):
                raise ValueError("Missing trusted per-request numerical mode admission")
        starts = [e for e in events if e.get("phase") == "process_start"]
        if len(starts) != 1 or starts[0].get("rank") != rid:
            raise ValueError("Missing source process receipt")
        verify_sources(source_sha256, starts[0])
        boundaries, inside = [], None
        for e in events:
            if "repeat" not in e:
                if e.get("phase") in (
                    "request_before",
                    "request_done",
                    "rank_action",
                    "token_done",
                ):
                    raise ValueError("Unbound request/action phase")
                continue
            ordinal = e["repeat"]
            if type(ordinal) is not int or not 0 <= ordinal < count or e.get("rank") != rid:
                raise ValueError("Unexpected request phase ordinal/rank")
            spec = rows[ordinal]
            binding = {k: spec[k] for k in ("id", "case_id", "method", "warmup", "max_tokens")}
            if "prefill_mode" in spec:
                binding.update(
                    prefill_mode=spec["prefill_mode"], source_request_id=spec["source_request_id"]
                )
            binding.update(
                sample_repeat=spec["repeat"],
                eval_group_size=args.eval_group_size,
                prompt_sha256=hashlib.sha256(spec["prompt"].encode()).hexdigest(),
            )
            if any(e.get(k) != v or type(e.get(k)) is not type(v) for k, v in binding.items()):
                raise ValueError("Phase differs from admitted ordered schedule")
            if e["phase"] == "request_before":
                if inside is not None or e.get("prompt_token_ids") != prompt_ids[ordinal]:
                    raise ValueError("Request prefill prompt/order mismatch")
                inside = ordinal
                boundaries.append((e["phase"], ordinal))
            elif inside != ordinal:
                raise ValueError("Phase outside request boundaries")
            if e["phase"] == "request_done":
                if e.get("request_result") != rank["requests"][ordinal]:
                    raise ValueError("Incremental result differs from final rank result")
                boundaries.append((e["phase"], ordinal))
                inside = None
        if boundaries != [(p, i) for i in range(count) for p in ("request_before", "request_done")]:
            raise ValueError("Missing/reordered request boundaries")
        for pair in rank.get("verification", {}).get("groups", {}).values():
            if pair["small"].get("metadata_sha256") != {
                "config.json": args.small_config_sha256,
                "model.safetensors.index.json": args.small_index_sha256,
            }:
                raise ValueError("Small checkpoint binding mismatch")
        for ordinal, r in enumerate(rank.get("requests", [])):
            spec = rows[ordinal]
            expected_mode = (
                "native" if spec["method"] == "native" else getattr(args, f"rank{rid}_verifier")
            )
            if r.get("verifier_mode") != expected_mode:
                raise ValueError("Mismatched per-request verifier mode")
            for k in (
                (
                    "id",
                    "case_id",
                    "method",
                    "warmup",
                    "max_tokens",
                    "prefill_mode",
                    "source_request_id",
                )
                if "prefill_mode" in spec
                else ("id", "case_id", "method", "warmup", "max_tokens")
            ):
                if r.get(k) != spec[k] or type(r.get(k)) is not type(spec[k]):
                    raise ValueError("Result differs from admitted request")
            if (
                type(r.get("sample_repeat")) is not int
                or type(r.get("repeat")) is not int
                or r.get("sample_repeat") != spec["repeat"]
                or r.get("repeat") != ordinal
                or r.get("prompt_sha256") != hashlib.sha256(spec["prompt"].encode()).hexdigest()
                or r.get("prompt_token_ids") != prompt_ids[ordinal]
                or r.get("prompt_tokens") != len(prompt_ids[ordinal])
                or len(prompt_ids[ordinal]) + spec["max_tokens"] > 4096
                or r.get("local_rank") != rid
                or type(r.get("local_peak_mlx_bytes")) is not int
                or r["local_peak_mlx_bytes"] <= 0
                or r.get("sidecar_resident_on_rank0") is not True
            ):
                raise ValueError("Missing exact prompt/repeat/independent local memory evidence")
            tokens = r.get("token_ids")
            if not isinstance(tokens, list) or not 1 <= len(tokens) <= spec["max_tokens"]:
                raise ValueError("Token cap/result mismatch")
            if r.get("finish_reason") not in ("stop", "length") or (
                r["finish_reason"] == "length" and len(tokens) != spec["max_tokens"]
            ):
                raise ValueError("Stop/cap mismatch")
            for field in ("wall_seconds", "first_token_seconds", "local_rank_wall_seconds"):
                if (
                    type(r.get(field)) not in (int, float)
                    or not math.isfinite(r[field])
                    or r[field] <= 0
                ):
                    raise ValueError("Invalid actual timing")
            if r["wall_seconds"] < r["first_token_seconds"]:
                raise ValueError("Invalid first-token timing")
            if r.get("wall_tokens_per_second") != len(tokens) / r["wall_seconds"]:
                raise ValueError("Wall throughput timing convention mismatch")
            expected = (
                (len(tokens) - 1) / (r["wall_seconds"] - r["first_token_seconds"])
                if len(tokens) > 1
                else None
            )
            if r.get("post_first_tokens_per_second") != expected:
                raise ValueError("Post-first timing convention mismatch")
            active = [e for e in events if e.get("repeat") == ordinal]
            if "prefill_mode" in spec:
                from overlap_evidence import verify_request_modes

                verify_request_modes(
                    spec, r, active, enabled=getattr(args, "overlapped_prefill", False)
                )
                if len([e for e in active if e.get("phase") == "request_start"]) != 1:
                    raise ValueError("Missing mode-bound request_start")
            if spec["method"] == "native":
                if (
                    any(e["phase"] == "rank_action" for e in active)
                    or any(
                        r.get(k) != 0
                        for k in ("accepted_drafts", "verification_rounds", "teacher_forwards")
                    )
                    or r.get("verifier_mode") != "native"
                ):
                    raise ValueError("Native request used MTP path")
                completed = [e.get("token_id") for e in active if e["phase"] == "token_done"]
                if (
                    completed != tokens
                    or len([e for e in active if e["phase"] == "prefill_before"]) != 1
                ):
                    raise ValueError("Missing native prefill/token stage evidence")
                layers = [e.get("layer_index") for e in active if e["phase"] == "layer_done"]
                owned = list(range(rank["layer_start"], rank["layer_end"]))
                forwards = (
                    max(0, len(prompt_ids[ordinal]) - 2) // 256 + 1
                    if len(prompt_ids[ordinal]) > 1
                    else 0
                ) + len(tokens)
                if layers != owned * forwards:
                    raise ValueError("Native missing exact owned-layer forward completions")
            else:
                actions = [e for e in active if e["phase"] == "rank_action"]
                if "prefill_mode" in spec and (
                    not actions
                    or r.get("request_nonce") != actions[0].get("request")
                    or r.get("cache_epoch") != r.get("request_nonce")
                ):
                    raise ValueError("Result lacks exact request/cache epoch binding")
                # Mandatory on each physical owner's complete request slice;
                # matching rank ACKs/counts alone do not prove output progress.
                validate_mtp_actions(
                    r,
                    actions,
                    prompt_ids[ordinal],
                    expected_mode=expected_mode,
                    eos_ids=(248044, 248046),
                    hidden_only_prefill=getattr(args, "hidden_only_prefill", False),
                    prefill_mode=spec.get("prefill_mode", "serial"),
                )
                verify_count = sum(e["op"] == "verify" for e in actions)
                if (
                    r.get("verification_rounds") != verify_count
                    or r.get("teacher_forwards") != verify_count
                    or not 0 <= r.get("accepted_drafts", -1) <= verify_count
                ):
                    raise ValueError("MTP verification action/counter mismatch")
                ids = prompt_ids[ordinal]
                prefill_op = (
                    "prefill_hidden" if getattr(args, "hidden_only_prefill", False) else "forward"
                )
                expected_forward = [
                    (prefill_op, ids[s : min(s + 256, len(ids) - 1)])
                    for s in range(0, len(ids) - 1, 256)
                ] + [("forward", [ids[-1]])]
                if spec.get("prefill_mode") == "overlap":
                    expected_forward = ([("prefill_pipeline", ids)] if len(ids) > 1 else []) + [
                        ("forward", [ids[-1]])
                    ]
                forwards = [
                    (e["op"], e["tokens"])
                    for e in actions
                    if e["op"] in ("forward", "prefill_hidden", "prefill_pipeline")
                ]
                if forwards != expected_forward:
                    raise ValueError(
                        "MTP prefill differs from exact trusted prompt/chunk256/hidden mode"
                    )
    # Reuse unchanged loading/inventory/cleanup checks on real per-request slices.
    # This does not synthesize numerical receipts or equate unrelated workloads.
    for ordinal in range(max(1, count)):
        sliced = []
        for rank in ranks:
            r = dict(rank)
            if count:
                req = dict(rank["requests"][ordinal], repeat=0)
                r.update(
                    requests=[req],
                    prompt=rows[ordinal]["prompt"],
                    formatted_prompt=rows[ordinal]["prompt"],
                )
            else:
                r["requests"] = []
            sliced.append(r)
        verify_trial(
            sliced,
            server,
            receipts,
            weight_files,
            repeats=1,
            expected_inventory=expected_inventory,
            eval_group_size=args.eval_group_size,
            execution="phased",
            gate_only=not count,
        )
    admission = verify_mtp_evidence(ranks, phases, args, request_specs=rows)
    paired = {}
    for ordinal, spec in enumerate(rows[:count]):
        a, b = (by_rank[r]["requests"][ordinal] for r in (0, 1))
        if any(
            a.get(k) != b.get(k)
            for k in (
                "token_ids",
                "finish_reason",
                "text",
                "prompt_token_ids",
                "accepted_drafts",
                "verification_rounds",
                "teacher_forwards",
            )
        ):
            raise ValueError("Actual ranks disagree on request output")
        key = (spec["case_id"], spec["warmup"], spec["repeat"])
        output = (a["token_ids"], a["finish_reason"])
        if paired.setdefault(key, output) != output:
            raise ValueError("Paired native/MTP greedy tokens/stop differ")
    return {
        **admission,
        "paired_greedy_tokens_and_stop_agree": bool(count),
        "tensor_completeness": True,
        "owned_groups_gone": True,
        "benchmark_schedule_sha256": args.benchmark_schedule_sha256,
        "verified_request_count": count,
        "phase_timing_by_rank": {r: phase_timings(phases[r]) for r in (0, 1)},
        "request_phase_timing_by_rank": {
            r: [
                {
                    "id": spec["id"],
                    "method": spec["method"],
                    "repeat": ordinal,
                    "operations": phase_timings(
                        [e for e in phases[r] if e.get("repeat") == ordinal]
                    ),
                }
                for ordinal, spec in enumerate(rows[:count])
            ]
            for r in (0, 1)
        },
    }
