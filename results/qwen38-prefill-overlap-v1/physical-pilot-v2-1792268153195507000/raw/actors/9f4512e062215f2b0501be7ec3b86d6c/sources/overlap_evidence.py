"""Detached source-bound causal stream verifier; not GPU occupancy proof."""

import hashlib
import json

from overlap_prefill import SCHEMA, binding, chunks, stream_seed


def require(condition, message="Invalid overlap stream evidence"):
    if not condition:
        raise ValueError(message)


def verify_overlap_stream(
    events,
    command,
    rank,
    *,
    owned_layers,
    require_native=True,
    hidden_size=None,
    dtype=None,
    eval_group_size=None,
):
    command.validate()
    stream = [e for e in events if e.get("schema") == SCHEMA]
    require(0 < len(stream) <= 2048)
    previous = stream_seed(command, rank)
    for index, event in enumerate(stream):
        payload = event.get("stream_payload")
        require(type(payload) is dict)
        require(all(type(event.get(k)) is type(v) and event[k] == v for k, v in payload.items()))
        raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        digest = hashlib.sha256((previous + raw).encode()).hexdigest()
        require(type(event.get("stream_sequence")) is int and event["stream_sequence"] == index)
        require(event.get("stream_previous") == previous and event.get("stream_digest") == digest)
        previous = digest
    cursor = 0
    for index, start, end in chunks(command.tokens):
        current = []
        while cursor < len(stream) and stream[cursor].get("chunk_index") == index:
            current.append(stream[cursor])
            cursor += 1
        expected = binding(command, rank, index, start, end)
        require(
            current
            and all(
                all(type(e.get(k)) is type(v) and e[k] == v for k, v in expected.items())
                for e in current
            )
        )
        phases = [e["phase"] for e in current]
        core = [p for p in phases if p.startswith("pipeline_")]
        require(
            core
            == (
                [
                    "pipeline_chunk_start",
                    "pipeline_receive_evaluated",
                    "pipeline_credit_sent",
                    "pipeline_target_done",
                    "pipeline_head_done",
                ]
                if rank == 0
                else ["pipeline_chunk_start", "pipeline_send_evaluated", "pipeline_credit_received"]
            )
        )
        if rank == 0:
            head = current[-1]
            require(head.get("head_position") == start and type(head.get("head_position")) is int)
            require(head.get("head_offset") == end and type(head.get("head_offset")) is int)
            require(
                type(head.get("shifted_tokens")) is list
                and all(type(t) is int for t in head["shifted_tokens"])
                and head["shifted_tokens"] == command.tokens[start + 1 : end + 1]
            )
        if require_native:
            prefix = (
                [
                    "pipeline_chunk_start",
                    "handoff_receive_before",
                    "handoff_receive_done",
                    "pipeline_receive_evaluated",
                    "pipeline_credit_sent",
                ]
                if rank == 0
                else ["pipeline_chunk_start", "embedding_before", "embedding_done"]
            )
            require(phases[: len(prefix)] == prefix)
            position, layer_cursor = len(prefix), 0
            owned = list(owned_layers)
            require(owned and owned == list(range(owned[0], owned[-1] + 1)))
            while layer_cursor < len(owned):
                require(position < len(current))
                event = current[position]
                if event["phase"] == "layer_before":
                    require(eval_group_size in (None, 1))
                    require(
                        type(event.get("layer_index")) is int
                        and event["layer_index"] == owned[layer_cursor]
                    )
                    position += 1
                    require(
                        position < len(current)
                        and current[position]["phase"] == "layer_done"
                        and type(current[position].get("layer_index")) is int
                        and current[position]["layer_index"] == owned[layer_cursor]
                    )
                    position += 1
                    layer_cursor += 1
                else:
                    lo, hi = event.get("start_idx"), event.get("end_idx")
                    require(
                        event["phase"] == "block_before"
                        and type(lo) is int
                        and type(hi) is int
                        and lo == owned[layer_cursor]
                        and lo < hi <= owned[-1] + 1
                    )
                    if eval_group_size is not None:
                        require(hi == min(lo + eval_group_size, owned[-1] + 1))
                    position += 1
                    require(position < len(current) and current[position]["phase"] == "block_done")
                    done = current[position]
                    require(
                        type(done.get("start_idx")) is int
                        and type(done.get("end_idx")) is int
                        and done["start_idx"] == lo
                        and done["end_idx"] == hi
                    )
                    position += 1
                    for layer in range(lo, hi):
                        require(
                            position < len(current)
                            and current[position]["phase"] == "layer_done"
                            and type(current[position].get("layer_index")) is int
                            and current[position]["layer_index"] == layer
                        )
                        position += 1
                        layer_cursor += 1
            tail = (
                [
                    "hidden_norm_before",
                    "hidden_norm_done",
                    "pipeline_target_done",
                    "pipeline_head_done",
                ]
                if rank == 0
                else [
                    "handoff_send_before",
                    "handoff_send_done",
                    "pipeline_send_evaluated",
                    "pipeline_credit_received",
                ]
            )
            require(phases[position:] == tail)
            handoff = next(
                e
                for e in current
                if e["phase"] == ("handoff_receive_done" if rank == 0 else "handoff_send_done")
            )
            shape = handoff.get("shape")
            require(
                type(shape) is list
                and len(shape) == 3
                and all(type(d) is int and d > 0 for d in shape)
                and shape[:2] == [1, end - start]
                and (hidden_size is None or shape[2] == hidden_size)
            )
            require(
                handoff.get("dtype") in ("float16", "bfloat16", "float32")
                and (dtype is None or handoff["dtype"] == dtype)
            )
    require(cursor == len(stream) - 1)
    final = stream[-1]
    expected_final = {
        "phase": "pipeline_prefix_done",
        "schema": SCHEMA,
        "rank": rank,
        "request_nonce": command.request,
        "command_sequence": command.sequence,
        "prefill_mode": "overlap",
        "prefix_offset": len(command.tokens) - 1,
    }
    require(all(type(final.get(k)) is type(v) and final[k] == v for k, v in expected_final.items()))
    return {
        "chunks": len(chunks(command.tokens)),
        "stream_digest": previous,
        "scope": "performed evaluated boundary grammar and causal credit bound; not GPU occupancy",
    }


def verify_compound_drain(events, command, rank):
    """Bind ACK drain after performed prefix/head and before final raw receive."""
    drained = [e for e in events if e.get("phase") == "pipeline_drained"]
    if rank == 1:
        require(not drained)
        return True
    require(len(drained) == 1)
    expected = {
        "phase": "pipeline_drained",
        "request_nonce": command.request,
        "command_sequence": command.sequence,
        "prefill_mode": "overlap",
        "pending_handoffs": 0,
        "rank": 0,
    }
    require(
        all(type(drained[0].get(k)) is type(v) and drained[0][k] == v for k, v in expected.items())
    )
    prefix = [
        e for e in events if e.get("phase") == "pipeline_prefix_done" and e.get("schema") == SCHEMA
    ]
    actions = [
        e for e in events if e.get("phase") == "rank_action" and e.get("op") == "prefill_pipeline"
    ]
    final_receives = [
        e
        for e in events
        if e.get("phase") == "handoff_receive_before" and e.get("schema") != SCHEMA
    ]
    require(len(prefix) == len(actions) == 1 and final_receives)
    require(
        events.index(prefix[0])
        < events.index(drained[0])
        < events.index(actions[0])
        < events.index(final_receives[0])
    )
    return True


def verify_request_modes(spec, result, events, *, enabled):
    mode = spec.get("prefill_mode", "serial")
    if "prefill_mode" in spec:
        for record in [result, *events]:
            require(
                record.get("prefill_mode") == mode
                and record.get("source_request_id") == spec["source_request_id"],
                "Per-request mode/source slot differs from trusted schedule",
            )
    require(mode != "overlap" or enabled is True, "Overlap without trusted capability flag")


def pilot_budget_check(receipts, *, requests_observed, requests_planned):
    """Use actual byte lengths; cannot pass from estimated/fabricated log sizes."""
    from remote_actor import LOG_CAP_BYTES, RETRIEVAL_CAP_BYTES

    require(type(requests_observed) is int and 0 < requests_observed <= 32)
    require(type(requests_planned) is int and requests_observed <= requests_planned <= 32)
    require(type(receipts) is list and len(receipts) == 3)
    sizes = []
    for receipt in receipts:
        require(
            type(receipt) is dict and all(type(receipt.get(k)) is str for k in ("stdout", "stderr"))
        )
        raw = [len(receipt[k].encode()) for k in ("stdout", "stderr")]
        require(all(n <= LOG_CAP_BYTES for n in raw), "Actual pilot log exceeds finite actor cap")
        # Conservative linear extrapolation includes setup/gates every time.
        projected = [
            (n * requests_planned + requests_observed - 1) // requests_observed for n in raw
        ]
        require(
            all(n <= LOG_CAP_BYTES for n in projected), "Pilot evidence budget blocks full schedule"
        )
        encoded = len(json.dumps(receipt, allow_nan=False).encode())
        require(
            (encoded * requests_planned + requests_observed - 1) // requests_observed
            <= RETRIEVAL_CAP_BYTES,
            "Pilot retrieval budget blocks full schedule",
        )
        sizes.append(
            {
                "actual_log_bytes": raw,
                "projected_log_bytes": projected,
                "actual_retrieval_bytes": encoded,
            }
        )
    return {
        "passed": True,
        "actor_log_cap_bytes": LOG_CAP_BYTES,
        "retrieval_cap_bytes": RETRIEVAL_CAP_BYTES,
        "requests_observed": requests_observed,
        "requests_planned": requests_planned,
        "actors": sizes,
    }
