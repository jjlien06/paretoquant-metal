"""Bounded compound prefix and receiver-consumed credits; no MLX imports.

The forward adapters preserve native arithmetic. Rank0's receive_stage callback
MUST execute only after evaluated raw hidden receive and before local graph work.
Do not emit a recovery JSON frame after any raw-stream failure.
"""

import hashlib
import json

from rank_control import Command


def stream_seed(command, rank):
    raw = json.dumps(
        [command.request, command.sequence, command.tokens, rank], separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class StreamState:
    """Bounded causal chain of performed local boundaries, not self-approval."""

    def __init__(self, command, rank, progress):
        self.previous = stream_seed(command, rank)
        self.index = 0
        self.progress = progress

    def emit(self, payload):
        if self.index >= 2048:
            raise ValueError("Pipeline stream event cap")
        raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        digest = hashlib.sha256((self.previous + raw).encode()).hexdigest()
        self.progress(
            {
                **payload,
                "stream_payload": payload,
                "stream_sequence": self.index,
                "stream_previous": self.previous,
                "stream_digest": digest,
            }
        )
        self.index += 1
        self.previous = digest


CHUNK_SIZE = 256
SCHEMA = "qwen38-prefill-overlap-stream-v1"
CREDIT_SCHEMA = "qwen38-prefill-receive-credit-v1"


def validate_modes(args, rows=None):
    enabled = getattr(args, "overlapped_prefill", False)
    from pressure_guard import pressure_required

    pressure = pressure_required(args)
    if type(enabled) is not bool:
        raise ValueError("Overlap/pressure flags must be exact bool")
    new_schedule = rows is not None and any("prefill_mode" in row for row in rows)
    if new_schedule and not enabled:
        raise ValueError("Overlap schedule requires explicit capability flag")
    if enabled:
        if not pressure:
            raise ValueError("Overlap requires pressure watchdog")
        if (
            getattr(args, "hidden_only_prefill", False) is not True
            or getattr(args, "prefix_rollback", False) is not True
            or getattr(args, "hidden_gate_chunk_size", 256) != 256
        ):
            raise ValueError("Overlap requires hidden-only/prefix/native256")


def chunks(prompt):
    Command("bounds", 1, "prefill_pipeline", prompt).validate()
    return [
        (i, start, min(start + CHUNK_SIZE, len(prompt) - 1))
        for i, start in enumerate(range(0, len(prompt) - 1, CHUNK_SIZE))
    ]


def credit(command, index, start, end):
    return {
        "schema": CREDIT_SCHEMA,
        "request_nonce": command.request,
        "command_sequence": command.sequence,
        "chunk_index": index,
        "physical_start": start,
        "physical_end": end,
    }


def validate_credit(value, expected):
    if (
        type(value) is not dict
        or set(value) != set(expected)
        or any(type(value[k]) is not type(v) or value[k] != v for k, v in expected.items())
    ):
        raise ValueError("Invalid/stale/out-of-order receiver credit")


def binding(command, rank, index, start, end):
    return {
        "schema": SCHEMA,
        "rank": rank,
        "request_nonce": command.request,
        "cache_epoch": command.request,
        "command_sequence": command.sequence,
        "prefill_mode": "overlap",
        "chunk_index": index,
        "physical_start": start,
        "physical_end": end,
    }


def run_rank0_pipeline(command, channel, forward, head, progress):
    command.validate()
    state = StreamState(command, 0, progress)
    for index, start, end in chunks(command.tokens):
        bound = binding(command, 0, index, start, end)

        def emit(event):
            state.emit({**event, **bound})

        emit({"phase": "pipeline_chunk_start"})
        received = False

        def receive_stage():
            nonlocal received
            if received:
                raise ValueError("Duplicate evaluated receive boundary")
            received = True
            emit({"phase": "pipeline_receive_evaluated"})
            channel.send(credit(command, index, start, end))
            emit({"phase": "pipeline_credit_sent"})

        _, hidden = forward(command.tokens[start:end], receive_stage=receive_stage, progress=emit)
        if not received:
            raise ValueError("Missing evaluated receive boundary")
        emit({"phase": "pipeline_target_done"})
        head(hidden, command.tokens[start + 1 : end + 1], start)
        emit(
            {
                "phase": "pipeline_head_done",
                "head_position": start,
                "head_offset": end,
                "shifted_tokens": command.tokens[start + 1 : end + 1],
            }
        )
        del hidden
    state.emit(
        {
            "phase": "pipeline_prefix_done",
            "schema": SCHEMA,
            "request_nonce": command.request,
            "command_sequence": command.sequence,
            "rank": 0,
            "prefill_mode": "overlap",
            "prefix_offset": len(command.tokens) - 1,
        }
    )


def run_rank1_pipeline(command, channel, forward, progress):
    command.validate()
    state = StreamState(command, 1, progress)
    for index, start, end in chunks(command.tokens):
        bound = binding(command, 1, index, start, end)

        def emit(event):
            state.emit({**event, **bound})

        emit({"phase": "pipeline_chunk_start"})
        forward(command.tokens[start:end], progress=emit)
        emit({"phase": "pipeline_send_evaluated"})
        validate_credit(channel.receive(), credit(command, index, start, end))
        emit({"phase": "pipeline_credit_received"})
    state.emit(
        {
            "phase": "pipeline_prefix_done",
            "schema": SCHEMA,
            "request_nonce": command.request,
            "command_sequence": command.sequence,
            "rank": 1,
            "prefill_mode": "overlap",
            "prefix_offset": len(command.tokens) - 1,
        }
    )
