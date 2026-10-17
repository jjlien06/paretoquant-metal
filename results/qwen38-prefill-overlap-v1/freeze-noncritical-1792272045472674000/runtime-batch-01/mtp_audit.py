"""Stdlib-only posthoc prefix-rollback MTP action auditor (safe to copy to SOURCES).

Checks recorded output progress and rank agreement, not actual greedy predictions:
receipts lack teacher logits/hidden states. This is NOT new physical numerical
admission and does not replace the original loading/numerical/cleanup validator.
The pure function expects one rank's request-filtered actual rank_action events.
"""

import hashlib
import json
import math
import re
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _token_list(value):
    return isinstance(value, list) and all(type(t) is int and t >= 0 for t in value)


def _validate_request(request, prompt_ids, eos_ids):
    _require(isinstance(request, dict), "request must be a dictionary")
    tokens = request.get("token_ids")
    cap = request.get("max_tokens")
    _require(_token_list(tokens) and bool(tokens), "invalid output tokens")
    _require(_token_list(prompt_ids) and bool(prompt_ids), "invalid prompt tokens")
    _require(type(cap) is int and 1 <= cap <= 512 and len(tokens) <= cap, "invalid output cap")
    _require(len(prompt_ids) + cap <= 4096, "context cap exceeded")
    _require(
        isinstance(eos_ids, (list, tuple)) and all(type(t) is int and t >= 0 for t in eos_ids),
        "invalid EOS IDs",
    )
    _require(not any(t in eos_ids for t in tokens[:-1]), "output after EOS")
    reason = "stop" if tokens[-1] in eos_ids else "length"
    _require(
        request.get("finish_reason") == reason and (reason == "stop" or len(tokens) == cap),
        "finish/cap/EOS contradiction",
    )
    _require(isinstance(request.get("id"), str) and bool(request["id"]), "missing request ID")
    _require(
        type(request.get("repeat")) is int and request["repeat"] >= 0, "invalid request ordinal"
    )
    _require(
        type(request.get("local_rank")) is int and request["local_rank"] in (0, 1),
        "invalid local rank",
    )
    for field in ("wall_seconds", "first_token_seconds"):
        value = request.get(field)
        _require(
            type(value) in (int, float) and math.isfinite(value) and value > 0, f"invalid {field}"
        )
    wall, first = request["wall_seconds"], request["first_token_seconds"]
    _require(wall >= first and (len(tokens) == 1 or wall > first), "invalid first-token time")
    expected_post = (len(tokens) - 1) / (wall - first) if len(tokens) > 1 else None
    for field, expected in (
        ("wall_tokens_per_second", len(tokens) / wall),
        ("post_first_tokens_per_second", expected_post),
    ):
        value = request.get(field)
        _require(
            field in request
            and value == expected
            and (value is None or (type(value) in (int, float) and math.isfinite(value))),
            f"inconsistent {field}",
        )


def validate_mtp_actions(
    request,
    actions,
    prompt_ids,
    *,
    expected_mode,
    eos_ids=(248044, 248046),
    hidden_only_prefill=False,
    prefill_mode="serial",
):
    """Validate exact reset/prefill/snapshot/verify/commit/finish grammar."""
    _validate_request(request, prompt_ids, eos_ids)
    _require(prefill_mode in ("serial", "overlap"), "Unknown prefill mode")
    if prefill_mode == "overlap":
        _require(
            hidden_only_prefill is True and request.get("prefill_mode") == prefill_mode,
            "Missing overlap/hidden mode binding",
        )
    _require(
        expected_mode in ("native", "slot-qmv") and request.get("verifier_mode") == expected_mode,
        "wrong/missing verifier mode",
    )
    _require(request.get("method") == "mtp", "caller must skip non-MTP requests")
    _require(type(hidden_only_prefill) is bool, "hidden_only_prefill must be explicit bool")
    _require(isinstance(actions, list) and bool(actions), "missing actual rank actions")
    nonce = actions[0].get("request") if isinstance(actions[0], dict) else None
    _require(
        isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None,
        "invalid request nonce",
    )
    for sequence, event in enumerate(actions):
        _require(isinstance(event, dict), "action must be a dictionary")
        binding = {
            "phase": "rank_action",
            "id": request["id"],
            "repeat": request["repeat"],
            "rank": request["local_rank"],
            "request": nonce,
            "sequence": sequence,
            "ok": True,
        }
        if "prefill_mode" in request:
            binding.update(
                prefill_mode=prefill_mode, source_request_id=request.get("source_request_id")
            )
        _require(
            all(type(event.get(k)) is type(v) and event.get(k) == v for k, v in binding.items()),
            "action ACK/binding/sequence mismatch",
        )
        _require(
            isinstance(event.get("op"), str) and _token_list(event.get("tokens")),
            "invalid action operation/tokens",
        )
    tokens = request["token_ids"]
    cursor = rounds = accepted = index = 0

    def take(op, payload=None):
        nonlocal index
        if index >= len(actions) or actions[index]["op"] != op:
            raise ValueError(f"action {index}: expected {op}")
        value = actions[index]["tokens"]
        if payload is not None and value != payload:
            raise ValueError(f"action {index}: wrong {op} tokens")
        index += 1
        return value

    take("reset", [])
    if prefill_mode == "overlap" and len(prompt_ids) > 1:
        take("prefill_pipeline", prompt_ids)
    for start in [] if prefill_mode == "overlap" else range(0, len(prompt_ids) - 1, 256):
        take(
            "prefill_hidden" if hidden_only_prefill else "forward",
            prompt_ids[start : min(start + 256, len(prompt_ids) - 1)],
        )
    take("forward", [prompt_ids[-1]])
    while cursor < len(tokens) - 1:
        take("snapshot", [])
        pair = take("verify")
        if len(pair) != 2 or pair[0] != tokens[cursor]:
            raise ValueError("verify does not consume the current emitted token")
        rounds += 1
        if pair[1] != tokens[cursor + 1]:
            take("commit_prefix", [1])
            cursor += 1
        else:
            accepted += 1
            cursor += 1
            if cursor == request["max_tokens"] - 1 or tokens[cursor] in eos_ids:
                take("commit_prefix", [1])
            else:
                if cursor + 1 >= len(tokens):
                    raise ValueError("accepted nonterminal draft lacks emitted bonus")
                cursor += 1
    take("finish", [])
    if index != len(actions):
        raise ValueError("extra actions after generation finish")
    for field, observed in (
        ("verification_rounds", rounds),
        ("teacher_forwards", rounds),
        ("accepted_drafts", accepted),
    ):
        if type(request.get(field)) is not int or request[field] != observed:
            raise ValueError(f"{field} contradicts recorded output progress")
    return {
        "id": request["id"],
        "verification_rounds": rounds,
        "accepted_drafts": accepted,
        "output_tokens": len(tokens),
        "action_count": len(actions),
    }


SCOPE = (
    "Recorded generation progress, counters, mode, prompt/action grammar and rank equality only; "
    "no teacher logits/hidden states, so cannot prove actual greedy predictions. "
    "Not new physical numerical admission; "
    "original numerical/loading/cleanup gates remain separate. "
    "Prompt IDs are reconciled recorded evidence, not independently retokenized."
)


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def _audit_rank(rank, events, rows, expected_mode, registry, hidden_only_prefill):
    rid = rank.get("rank")
    requests = rank.get("requests")
    _require(
        type(rid) is int and rid in (0, 1) and rank.get("prefix_rollback") is True,
        "require prefix rollback on identified rank",
    )
    _require(rank.get("verifier_mode") == expected_mode, "rank verifier mode mismatch")
    _require(isinstance(requests, list) and len(requests) == len(rows), "request count mismatch")
    starts = [e for e in events if e.get("phase") == "process_start"]
    _require(
        len(starts) == 1
        and starts[0].get("rank") == rid
        and _same(starts[0].get("source_sha256"), registry),
        "process source binding mismatch",
    )
    active = [[] for _ in rows]
    inside, completed = None, 0
    for event in events:
        phase = event.get("phase")
        if "repeat" not in event:
            _require(
                phase not in ("request_before", "request_done", "rank_action", "token_done"),
                "unbound request/action event",
            )
            continue
        ordinal = event["repeat"]
        _require(
            type(ordinal) is int
            and 0 <= ordinal < len(rows)
            and type(event.get("rank")) is int
            and event["rank"] == rid,
            "event ordinal/rank mismatch",
        )
        row, request = rows[ordinal], requests[ordinal]
        binding = {k: row[k] for k in ("id", "case_id", "method", "warmup", "max_tokens")}
        binding.update(
            sample_repeat=row["repeat"],
            prompt_sha256=request.get("prompt_sha256"),
            eval_group_size=request.get("eval_group_size"),
        )
        _require(
            all(k in event and _same(event[k], v) for k, v in binding.items()),
            "event differs from schedule/request",
        )
        if phase == "request_before":
            _require(
                inside is None
                and ordinal == completed
                and _same(event.get("prompt_token_ids"), request.get("prompt_token_ids")),
                "request start/order/prompt mismatch",
            )
            inside = ordinal
        _require(inside == ordinal, "event outside ordered request boundaries")
        active[ordinal].append(event)
        if phase == "request_done":
            _require(
                _same(event.get("request_result"), request), "incremental/final result mismatch"
            )
            inside = None
            completed += 1
    _require(inside is None and completed == len(rows), "missing request finish/boundary")
    summaries, commands, nonces = [], [], set()
    for ordinal, (request, row) in enumerate(zip(requests, rows)):
        prompt = request.get("prompt_token_ids")
        _validate_request(request, prompt, (248044, 248046))
        binding = {k: row[k] for k in ("id", "case_id", "method", "warmup", "max_tokens")}
        binding.update(
            repeat=ordinal,
            sample_repeat=row["repeat"],
            local_rank=rid,
            prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
            prompt_tokens=len(prompt),
        )
        _require(
            all(k in request and _same(request[k], v) for k, v in binding.items()),
            "request differs from exact schedule/prompt",
        )
        for field in ("local_peak_mlx_bytes", "local_rank_wall_seconds"):
            value = request.get(field)
            _require(
                type(value) in ((int,) if field.endswith("bytes") else (int, float))
                and math.isfinite(value)
                and value > 0,
                "invalid independent local memory/timing",
            )
        actions = [e for e in active[ordinal] if e.get("phase") == "rank_action"]
        if row["method"] == "native":
            _require(
                not actions
                and request.get("verifier_mode") == "native"
                and all(
                    type(request.get(k)) is int and request[k] == 0
                    for k in ("accepted_drafts", "verification_rounds", "teacher_forwards")
                ),
                "native request used MTP path/counters/mode",
            )
            _require(
                _same(
                    [e.get("token_id") for e in active[ordinal] if e.get("phase") == "token_done"],
                    request["token_ids"],
                )
                and sum(e.get("phase") == "prefill_before" for e in active[ordinal]) == 1,
                "native prefill/token evidence mismatch",
            )
            summary = dict(
                id=request["id"],
                output_tokens=len(request["token_ids"]),
                verification_rounds=0,
                accepted_drafts=0,
                action_count=0,
            )
        else:
            summary = validate_mtp_actions(
                request,
                actions,
                prompt,
                expected_mode=expected_mode,
                hidden_only_prefill=hidden_only_prefill,
            )
            nonce = actions[0]["request"]
            _require(nonce not in nonces, "reused MTP request nonce")
            nonces.add(nonce)
        summaries.append(dict(summary, method=row["method"], repeat=ordinal))
        commands.extend(
            {k: e[k] for k in ("id", "repeat", "request", "sequence", "op", "tokens")}
            for e in actions
        )
    return summaries, commands


def audit_trial(trialdir, expected_modes=("native", "slot-qmv"), *, hidden_only_prefill=False):
    """Read full actor receipts, not potentially truncated transport stdout.

    Never imports/executes captured source. Never writes evidence or re-runs gates.
    CLI: python mtp_audit.py TRIAL [TRIAL ...] --output NEW_SEPARATE_FILE.json
    """
    root = Path(trialdir).resolve()
    hashes = {}

    def read(name):
        data = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return _json(data)

    _require(
        isinstance(expected_modes, (tuple, list))
        and len(expected_modes) == 2
        and all(m in ("native", "slot-qmv") for m in expected_modes),
        "invalid expected modes",
    )
    original = read("result.json")
    _require(original.get("status") == "success", "trial is not completed/successful")
    registry = read("source_sha256.json")
    _require(
        isinstance(registry, dict) and "benchmark_schedule.json" in registry,
        "missing source registry/schedule",
    )
    for name, digest in registry.items():
        _require(
            isinstance(name, str)
            and name not in (".", "..")
            and Path(name).name == name
            and "\\" not in name
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "invalid source registry name/hash",
        )
        actual = hashlib.sha256((root / "sources" / name).read_bytes()).hexdigest()
        _require(actual == digest, f"captured source SHA mismatch: {name}")
    schedule = _json((root / "sources/benchmark_schedule.json").read_bytes())
    _require(
        schedule.get("schema") == "qwen38-workload-schedule-v1"
        and isinstance(schedule.get("requests"), list)
        and bool(schedule["requests"]),
        "invalid schedule",
    )
    rows = schedule["requests"]
    _require(len({r["id"] for r in rows}) == len(rows), "duplicate schedule IDs")
    for row in rows:
        _require(row.get("method") in ("native", "mtp"), "unknown schedule method")
    _require(
        type(original.get("verified_request_count")) is int
        and original["verified_request_count"] == len(rows),
        "original request total mismatch",
    )
    ranks, summaries, commands, actor_nonces = [], [], [], []
    for rid, mode in enumerate(expected_modes):
        receipt = read(f"rank{rid}.receipt.json")
        outcome = receipt.get("result", {})
        _require(
            outcome.get("reason") == "completed"
            and type(outcome.get("returncode")) is int
            and outcome["returncode"] == 0
            and outcome.get("log_evidence_complete") is True
            and outcome.get("log_cap_exceeded") is False
            and outcome.get("nonce") == receipt.get("nonce"),
            "incomplete actor receipt",
        )
        nonce = receipt.get("nonce")
        _require(
            isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None,
            "invalid actor nonce",
        )
        actor_nonces.append(nonce)
        _require(_same(receipt.get("source_sha256"), registry), "actor source registry mismatch")
        rank = _json(receipt["stdout"])
        events = [_json(line) for line in receipt["stderr"].splitlines() if line.strip()]
        _require(all(isinstance(e, dict) for e in events), "invalid event record")
        _require(
            type(rank.get("rank")) is int
            and rank["rank"] == rid
            and rank.get("benchmark_schedule_sha256") == registry["benchmark_schedule.json"],
            "rank identity/schedule SHA mismatch",
        )
        summary, command = _audit_rank(rank, events, rows, mode, registry, hidden_only_prefill)
        ranks.append(rank)
        summaries.append(summary)
        commands.append(command)
    _require(actor_nonces[0] != actor_nonces[1], "ranks reuse actor nonce")
    _require(_same(commands[0], commands[1]), "actual rank actions disagree")
    fields = (
        "id",
        "repeat",
        "method",
        "token_ids",
        "finish_reason",
        "text",
        "prompt_token_ids",
        "accepted_drafts",
        "verification_rounds",
        "teacher_forwards",
    )
    paired = {}
    for ordinal, row in enumerate(rows):
        a, b = (rank["requests"][ordinal] for rank in ranks)
        _require(
            all(k in a and k in b and _same(a[k], b[k]) for k in fields),
            "rank output/counters disagree",
        )
        key = (row["case_id"], row["warmup"], row["repeat"])
        output = (a["token_ids"], a["finish_reason"])
        _require(_same(paired.setdefault(key, output), output), "paired native/MTP output mismatch")
    return dict(
        schema="qwen38-strict-mtp-audit-v1",
        status="verified",
        assurance_scope=SCOPE,
        trialdir=str(root),
        raw_sha256=hashes,
        source_sha256=registry,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actor_nonces=actor_nonces,
        expected_modes=list(expected_modes),
        hidden_only_prefill=hidden_only_prefill,
        verified_request_count=len(rows),
        verified_rank_request_count=2 * len(rows),
        verified_mtp_request_count=sum(r["method"] == "mtp" for r in rows),
        actual_rank_action_counts={str(r): len(commands[r]) for r in (0, 1)},
        rank_actions_agree=True,
        rank_outputs_agree=True,
        request_summaries_by_rank=summaries,
    )


def main(argv=None):
    """Print exact report JSON; optionally create one exclusive, separate file."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trialdirs", nargs="+")
    parser.add_argument(
        "--expected-modes", nargs=2, choices=("native", "slot-qmv"), default=("native", "slot-qmv")
    )
    parser.add_argument("--hidden-only-prefill", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        roots = [Path(p).resolve() for p in args.trialdirs]
        _require(len(set(roots)) == len(roots), "duplicate trial paths")
        if args.output:
            target = args.output.resolve()
            _require(
                all(target != root and root not in target.parents for root in roots),
                "audit output must be separate from original evidence",
            )
            _require(not args.output.exists(), "audit output already exists")
        reports = [
            audit_trial(root, args.expected_modes, hidden_only_prefill=args.hidden_only_prefill)
            for root in roots
        ]
        report = dict(
            schema="qwen38-strict-mtp-audit-set-v1",
            trials=reports,
            verified_trial_count=len(reports),
            verified_request_count=sum(r["verified_request_count"] for r in reports),
            verified_rank_request_count=sum(r["verified_rank_request_count"] for r in reports),
        )
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text)
        print(text, end="")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
