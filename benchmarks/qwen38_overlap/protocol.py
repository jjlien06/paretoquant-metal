"""PROPOSED normalized evidence contract. Runtime025 adapter is not admitted.

This validates recorded causality, not simultaneous GPU execution. No clock
subtraction across hosts. Unknown phases are rejected, not ignored.
"""

import json
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def token_ids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(t) is int and 0 <= t < 248320 for t in value)
    )


def audit_transcript(ranks, binding):
    """Exact ordered per-rank grammar; admits no surplus/unknown chunk events."""
    prompt = binding["prompt_token_ids"]
    mode = binding["prefill_mode"]
    require(token_ids(prompt) and len(prompt) <= 2048, "Invalid bounded prompt IDs")
    require(mode in ("serial", "overlap"), "Unknown prefill mode")
    require(type(binding["ordinal"]) is int and binding["ordinal"] >= 0, "Invalid ordinal")
    for nonce in (binding["request_nonce"], binding["cache_epoch"], *binding["actor_nonces"]):
        require(
            isinstance(nonce, str) and re.fullmatch("[0-9a-f]{32}", nonce), "Invalid nonce/epoch"
        )
    require(
        len(ranks) == 2 and all(isinstance(r, list) and len(r) <= 256 for r in ranks),
        "Two bounded rank streams required",
    )
    starts = list(range(0, len(prompt) - 1, 256))
    for rank, events in enumerate(ranks):
        index, clock = 0, -1
        common = {
            "request_id": binding["id"],
            "ordinal": binding["ordinal"],
            "rank": rank,
            "actor_nonce": binding["actor_nonces"][rank],
            "request_nonce": binding["request_nonce"],
            "cache_epoch": binding["cache_epoch"],
            "prefill_mode": mode,
        }

        def take(phase, sequence, **payload):
            nonlocal index, clock
            require(index < len(events), "Missing prefix event: " + phase)
            actual = events[index]
            expected = {**common, "phase": phase, "command_sequence": sequence, **payload}
            require(
                isinstance(actual, dict) and set(actual) == set(expected) | {"t_ns"},
                "Unknown/missing event fields",
            )
            require(
                same({k: v for k, v in actual.items() if k != "t_ns"}, expected),
                "Prefix phase/action/binding mismatch: " + phase,
            )
            require(
                type(actual["t_ns"]) is int and actual["t_ns"] >= clock and actual["t_ns"] >= 0,
                "Nonmonotonic rank-local event time",
            )
            clock = actual["t_ns"]
            index += 1

        take("reset_done", 0)
        if starts and mode == "overlap":
            take("pipeline_begin", 1)
        for chunk_index, start in enumerate(starts):
            end = min(start + 256, len(prompt) - 1)
            sequence = 1 if mode == "overlap" else chunk_index + 1
            chunk = {
                "chunk_index": chunk_index,
                "physical_start": start,
                "physical_end": end,
                "tokens": prompt[start:end],
            }
            hidden = {"shape": [1, end - start, binding["hidden_size"]], "dtype": binding["dtype"]}
            if rank == 0:
                take("hidden_receive_done", sequence, **chunk, **hidden)
                if mode == "overlap":
                    take("credit_send", sequence, **chunk)
                take("target_done", sequence, **chunk, evaluated_layers=binding["layers"][rank])
                take("hidden_norm_done", sequence, **chunk)
                if mode == "serial":
                    take("serial_ack_receive", sequence, **chunk)
                take(
                    "head_shift_done",
                    sequence,
                    **chunk,
                    shifted_tokens=prompt[start + 1 : end + 1],
                    head_position=start,
                    head_offset=end,
                )
            else:
                take("target_done", sequence, **chunk, evaluated_layers=binding["layers"][rank])
                take("hidden_send_done", sequence, **chunk, **hidden)
                if mode == "overlap":
                    take("credit_receive", sequence, **chunk)
            if mode == "serial" and rank == 1:
                take("serial_ack_send", sequence, **chunk)
        final_sequence = (2 if starts else 1) if mode == "overlap" else len(starts) + 1
        if starts and mode == "overlap":
            take("pipeline_ack_receive" if rank == 0 else "pipeline_ack_send", 1)
        take("drain_done", final_sequence - 1, active_chunks=0, pending_handoffs=0)
        take("final_forward_done", final_sequence, tokens=[prompt[-1]], target_offset=len(prompt))
        require(index == len(events), "Surplus prefix events")
    return {
        "prefix_chunks": len(starts),
        "drained": True,
        "scope": "recorded per-rank credit grammar; no physical GPU concurrency proof",
    }


EOS_IDS = (248044, 248046)


def validate_output(request):
    tokens, cap = request.get("token_ids"), request.get("max_tokens")
    require(
        token_ids(tokens) and type(cap) is int and 1 <= cap <= 512 and len(tokens) <= cap,
        "Invalid output/cap",
    )
    require(not any(t in EOS_IDS for t in tokens[:-1]), "Tokens emitted after EOS")
    reason = "stop" if tokens[-1] in EOS_IDS else "length"
    require(
        request.get("finish_reason") == reason and (reason == "stop" or len(tokens) == cap),
        "Stop/cap/EOS contradiction",
    )


def audit_decode(actions, request, binding, *, rank, start_sequence):
    """Faithful recorded-progress grammar from 024 validate_mtp_actions.

    Decode predictions are not present in receipts; this cannot certify logits.
    Verified prefix/final actions are handled separately by audit_transcript.
    """
    validate_output(request)
    require(
        isinstance(actions, list) and 1 <= len(actions) <= 2048,
        "Missing/budget-exceeded decode actions",
    )
    common = {
        "request_nonce": binding["request_nonce"],
        "cache_epoch": binding["cache_epoch"],
        "prefill_mode": binding["prefill_mode"],
        "request_id": binding["id"],
        "ordinal": binding["ordinal"],
        "rank": rank,
        "actor_nonce": binding["actor_nonces"][rank],
    }
    for index, event in enumerate(actions):
        require(
            set(event) == set(common) | {"op", "tokens", "command_sequence"},
            "Unexpected decode action fields",
        )
        require(
            same({k: event[k] for k in common}, common)
            and type(event["command_sequence"]) is int
            and event["command_sequence"] == start_sequence + index,
            "Decode binding/sequence differs",
        )
        require(
            isinstance(event["tokens"], list)
            and all(type(t) is int and t >= 0 for t in event["tokens"]),
            "Invalid action tokens",
        )
    cursor = rounds = accepted = index = 0
    tokens = request["token_ids"]

    def take(op, payload=None):
        nonlocal index
        require(
            index < len(actions) and actions[index]["op"] == op,
            "Wrong/missing decode action: " + op,
        )
        value = actions[index]["tokens"]
        require(payload is None or value == payload, "Wrong decode action tokens: " + op)
        index += 1
        return value

    while cursor < len(tokens) - 1:
        take("snapshot", [])
        pair = take("verify")
        require(
            len(pair) == 2 and pair[0] == tokens[cursor],
            "Verify does not consume current emitted token",
        )
        rounds += 1
        if pair[1] != tokens[cursor + 1]:
            take("commit_prefix", [1])
            cursor += 1
        else:
            accepted += 1
            cursor += 1
            if cursor == request["max_tokens"] - 1 or tokens[cursor] in EOS_IDS:
                take("commit_prefix", [1])
            else:
                require(cursor + 1 < len(tokens), "Accepted draft lacks bonus")
                cursor += 1
    take("finish", [])
    require(index == len(actions), "Surplus decode actions")
    counters = {
        "verification_rounds": rounds,
        "teacher_forwards": rounds,
        "accepted_drafts": accepted,
    }
    for name, value in counters.items():
        require(
            type(request.get(name)) is int and request[name] == value,
            "Decode counters contradict actual progress",
        )
    return counters
