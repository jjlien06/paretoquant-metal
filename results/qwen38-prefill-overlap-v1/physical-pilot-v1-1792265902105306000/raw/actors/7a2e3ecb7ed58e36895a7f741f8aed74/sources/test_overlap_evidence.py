"""Detached stream audit mutations preserve counts but must fail ordering/state."""

from copy import deepcopy

import pytest
from test_overlap_protocol import seam


def test_performed_receive_credit_stream_and_equal_count_forgery():
    from overlap_evidence import verify_overlap_stream

    cmd, wire, streams, *_ = seam(513, eager=True)
    for rank in (0, 1):
        verdict = verify_overlap_stream(
            streams[rank], cmd, rank, owned_layers=[], require_native=False
        )
        assert verdict["chunks"] == 2
        for mutation in ("swap", "credit", "range", "nonce", "missing", "digest"):
            events = deepcopy(streams[rank])
            phase = "pipeline_credit_sent" if rank == 0 else "pipeline_credit_received"
            index = next(i for i, e in enumerate(events) if e["phase"] == phase)
            if mutation == "swap":
                events[index], events[index - 1] = events[index - 1], events[index]
            elif mutation == "credit":
                events[index]["phase"] = "pipeline_target_done"
            elif mutation == "range":
                events[index]["physical_end"] -= 1
            elif mutation == "nonce":
                events[index]["request_nonce"] = "stale"
            elif mutation == "missing":
                events.pop(index)
            else:
                events[index]["stream_digest"] = "a" * 64
            with pytest.raises(ValueError):
                verify_overlap_stream(events, cmd, rank, owned_layers=[], require_native=False)
            if mutation != "digest":
                # Recompute every hash: the state grammar, not hash syntax/counts,
                # must independently reject the equal-count causal forgery.
                from overlap_prefill import StreamState

                resealed = []
                state = StreamState(cmd, rank, resealed.append)
                for event in events:
                    if "stream_payload" in event:
                        state.emit({k: event[k] for k in event["stream_payload"]})
                with pytest.raises(ValueError):
                    verify_overlap_stream(
                        resealed, cmd, rank, owned_layers=[], require_native=False
                    )


def test_actual_compound_action_grammar_keeps_final_singleton_outside():
    from mtp_audit import validate_mtp_actions
    from test_mtp_audit import fixture

    request, actions, prompt = fixture()
    request.update(prefill_mode="overlap", source_request_id="original-slot")
    actions[1].update(op="prefill_pipeline", tokens=prompt)
    for e in actions:
        e.update(prefill_mode="overlap", source_request_id="original-slot")
    validate_mtp_actions(
        request,
        actions,
        prompt,
        expected_mode="native",
        hidden_only_prefill=True,
        prefill_mode="overlap",
    )
    for mutation in ("serial_forgery", "final_in_compound", "range_tokens", "wrong_mode"):
        bad = deepcopy(actions)
        if mutation == "serial_forgery":
            bad[1]["op"] = "prefill_hidden"
        elif mutation == "final_in_compound":
            bad.pop(2)
        elif mutation == "range_tokens":
            bad[1]["tokens"] = prompt[:-1]
        else:
            bad[1]["prefill_mode"] = "serial"
        with pytest.raises(ValueError):
            validate_mtp_actions(
                request,
                bad,
                prompt,
                expected_mode="native",
                hidden_only_prefill=True,
                prefill_mode="overlap",
            )


def test_budget_uses_actual_log_strings_and_blocks_projection_overflow(monkeypatch):
    import json

    import remote_actor
    from overlap_evidence import pilot_budget_check

    _, _, streams, *_ = seam(513, eager=True)
    # These are actual CPU fixture event bytes, never physical pilot evidence.
    raw = "\n".join(json.dumps(event) for event in streams[0])
    statuses = [{"stdout": "", "stderr": raw} for _ in range(3)]
    passed = pilot_budget_check(statuses, requests_observed=6, requests_planned=14)
    assert passed["actors"][0]["actual_log_bytes"][1] == len(raw.encode())
    monkeypatch.setattr(remote_actor, "LOG_CAP_BYTES", len(raw.encode()) + 1)
    with pytest.raises(ValueError, match="budget"):
        pilot_budget_check(statuses, requests_observed=6, requests_planned=14)
    with pytest.raises(ValueError):
        pilot_budget_check([{"stdout_bytes": 1}] * 3, requests_observed=6, requests_planned=14)


def test_one_token_bypass_has_no_prefix_or_head_work():
    from overlap_prefill import run_rank0_pipeline, run_rank1_pipeline
    from rank_control import Command

    command = Command("one", 1, "prefill_pipeline", [8])
    events = []

    def forbidden(*a, **k):
        raise AssertionError("one token did prefix work")

    run_rank0_pipeline(command, None, forbidden, forbidden, events.append)
    run_rank1_pipeline(command, None, forbidden, events.append)
    assert [e["phase"] for e in events] == ["pipeline_prefix_done"] * 2
