"""Proposed normalized TEST transcript, not real 025 runtime evidence."""

from copy import deepcopy

import pytest

from .test_harness import api


def transcript_fixture(mode="overlap", length=513):
    binding = {
        "id": "TEST-request",
        "ordinal": 0,
        "request_nonce": "a" * 32,
        "cache_epoch": "b" * 32,
        "prefill_mode": mode,
        "actor_nonces": ["c" * 32, "d" * 32],
        "prompt_token_ids": list(range(length)),
        "layers": [[36, 37], [0, 1]],
        "hidden_size": 8,
        "dtype": "bfloat16",
    }
    ranks = []
    for rank in (0, 1):
        events = []

        def event(phase, seq, **payload):
            events.append(
                {
                    "phase": phase,
                    "command_sequence": seq,
                    "request_id": binding["id"],
                    "ordinal": 0,
                    "rank": rank,
                    "actor_nonce": binding["actor_nonces"][rank],
                    "request_nonce": binding["request_nonce"],
                    "cache_epoch": binding["cache_epoch"],
                    "prefill_mode": mode,
                    "t_ns": len(events) + 1,
                    **payload,
                }
            )

        event("reset_done", 0)
        if length > 1 and mode == "overlap":
            event("pipeline_begin", 1)
        bounds = list(range(0, length - 1, 256))
        for index, start in enumerate(bounds):
            end = min(start + 256, length - 1)
            seq = 1 if mode == "overlap" else index + 1
            chunk = {
                "chunk_index": index,
                "physical_start": start,
                "physical_end": end,
                "tokens": binding["prompt_token_ids"][start:end],
            }
            if rank == 0:
                event(
                    "hidden_receive_done", seq, **chunk, shape=[1, end - start, 8], dtype="bfloat16"
                )
                if mode == "overlap":
                    event("credit_send", seq, **chunk)
                event("target_done", seq, **chunk, evaluated_layers=binding["layers"][rank])
                event("hidden_norm_done", seq, **chunk)
                if mode == "serial":
                    event("serial_ack_receive", seq, **chunk)
                event(
                    "head_shift_done",
                    seq,
                    **chunk,
                    shifted_tokens=binding["prompt_token_ids"][start + 1 : end + 1],
                    head_position=start,
                    head_offset=end,
                )
            else:
                event("target_done", seq, **chunk, evaluated_layers=binding["layers"][rank])
                event("hidden_send_done", seq, **chunk, shape=[1, end - start, 8], dtype="bfloat16")
                if mode == "overlap":
                    event("credit_receive", seq, **chunk)
            if mode == "serial" and rank == 1:
                event("serial_ack_send", seq, **chunk)
        seq = (2 if bounds else 1) if mode == "overlap" else len(bounds) + 1
        if bounds and mode == "overlap":
            event("pipeline_ack_receive" if rank == 0 else "pipeline_ack_send", 1)
        event("drain_done", seq - 1, active_chunks=0, pending_handoffs=0)
        event(
            "final_forward_done",
            seq,
            tokens=[binding["prompt_token_ids"][-1]],
            target_offset=length,
        )
        ranks.append(events)
    return binding, ranks


def test_serial_ack_before_shifted_head_matches_024_control_seam():
    binding, ranks = transcript_fixture("serial", 513)
    # 024 _command drains ACK before returning hidden to head.forward.
    events = ranks[0]
    for index in range(len(events) - 1):
        if (
            events[index]["phase"] == "head_shift_done"
            and events[index + 1]["phase"] == "serial_ack_receive"
        ):
            events[index], events[index + 1] = events[index + 1], events[index]
    for index, event in enumerate(events):
        event["t_ns"] = index + 1
    assert api().audit_transcript(ranks, binding)["drained"] is True


def test_credit_aware_exact_prefix_transcript():
    h = api()
    assert hasattr(h, "audit_transcript"), "Credit-aware action parser missing"
    for mode in ("serial", "overlap"):
        for length in (1, 2, 256, 257, 258, 511, 512, 513, 1535, 1536):
            binding, events = transcript_fixture(mode, length)
            result = h.audit_transcript(events, binding)
            assert result["prefix_chunks"] == len(range(0, length - 1, 256))
            assert result["drained"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "stale_mode",
        "nonce",
        "credit_duplicate",
        "missing_credit",
        "head_shift",
        "missing_drain",
        "extra_chunk",
        "wrong_same_count",
        "layer_missing",
        "surplus",
        "premature_ack",
    ],
)
def test_credit_transcript_mutations_rejected(mutation):
    h = api()
    assert hasattr(h, "audit_transcript"), "Credit-aware action parser missing"
    binding, events = transcript_fixture()
    events = deepcopy(events)
    if mutation == "stale_mode":
        for rank in events:
            for event in rank:
                event["prefill_mode"] = "serial"
    elif mutation == "nonce":
        events[0][3]["request_nonce"] = "e" * 32
    elif mutation == "credit_duplicate":
        events[0].insert(4, deepcopy(events[0][3]))
    elif mutation == "missing_credit":
        events[1] = [e for e in events[1] if e["phase"] != "credit_receive"]
    elif mutation == "head_shift":
        next(e for e in events[0] if e["phase"] == "head_shift_done")["shifted_tokens"][0] += 1
    elif mutation == "missing_drain":
        events[0] = [e for e in events[0] if e["phase"] != "drain_done"]
    elif mutation == "extra_chunk":
        events[1].insert(-3, deepcopy(events[1][2]))
    elif mutation == "wrong_same_count":
        events[1][2]["tokens"][0] += 1
    elif mutation == "layer_missing":
        events[1][2]["evaluated_layers"] = [0]
    elif mutation == "surplus":
        events[0].append(deepcopy(events[0][-1]))
    elif mutation == "premature_ack":
        events[0].insert(2, events[0].pop(-3))
    with pytest.raises(ValueError):
        h.audit_transcript(events, binding)
