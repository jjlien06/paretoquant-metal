"""TEST recorded decode grammar, not teacher predictions."""

from copy import deepcopy

import pytest

from .test_harness import api
from .test_transcript import transcript_fixture


def decode_fixture():
    binding, _ = transcript_fixture()
    request = {
        "token_ids": [10, 11, 12, 248044],
        "max_tokens": 256,
        "finish_reason": "stop",
        "verification_rounds": 2,
        "teacher_forwards": 2,
        "accepted_drafts": 1,
    }
    ops = [
        ("snapshot", []),
        ("verify", [10, 99]),
        ("commit_prefix", [1]),
        ("snapshot", []),
        ("verify", [11, 12]),
        ("finish", []),
    ]
    events = [
        {
            "op": op,
            "tokens": tokens,
            "command_sequence": index + 3,
            "request_nonce": binding["request_nonce"],
            "cache_epoch": binding["cache_epoch"],
            "prefill_mode": "overlap",
            "request_id": binding["id"],
            "ordinal": 0,
            "rank": 0,
            "actor_nonce": binding["actor_nonces"][0],
        }
        for index, (op, tokens) in enumerate(ops)
    ]
    return binding, request, events


def test_decode_prefix_rollback_progress():
    h = api()
    assert hasattr(h, "audit_decode"), "Recorded decode grammar validator missing"
    binding, request, events = decode_fixture()
    assert (
        h.audit_decode(events, request, binding, rank=0, start_sequence=3)["accepted_drafts"] == 1
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_current",
        "wrong_commit",
        "wrong_snapshot",
        "missing",
        "surplus",
        "counter",
        "stale",
        "eos_middle",
        "wrong_stop",
    ],
)
def test_equal_count_wrong_actions_are_rejected(mutation):
    h = api()
    assert hasattr(h, "audit_decode"), "Recorded decode grammar validator missing"
    binding, request, events = deepcopy(decode_fixture())
    if mutation == "wrong_current":
        events[1]["tokens"][0] = 99
    elif mutation == "wrong_commit":
        events[2]["tokens"] = [2]
    elif mutation == "wrong_snapshot":
        events[0]["op"] = "reset"
    elif mutation == "missing":
        events.pop()
    elif mutation == "surplus":
        events.append(events[-1])
    elif mutation == "counter":
        request["accepted_drafts"] = 0
    elif mutation == "stale":
        events[0]["cache_epoch"] = "e" * 32
    elif mutation == "eos_middle":
        request["token_ids"][0] = 248044
    elif mutation == "wrong_stop":
        request["finish_reason"] = "length"
    with pytest.raises(ValueError):
        h.audit_decode(events, request, binding, rank=0, start_sequence=3)
