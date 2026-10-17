"""Bounded per-rank control contract; transport intentionally caller-owned.

Send command BEFORE rank0 enters phased_forward; wait for ack AFTER forward.
Rank1 owns only native target cache, never an MTP module or output projection.
Every request starts with reset and monotonically increasing sequence numbers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    request: str
    sequence: int
    op: str
    tokens: list

    def validate(self):
        if (
            not isinstance(self.request, str)
            or not 1 <= len(self.request) <= 64
            or type(self.sequence) is not int
            or self.sequence < 0
            or self.op
            not in (
                "reset",
                "prefill_hidden",
                "prefill_pipeline",
                "forward",
                "verify",
                "snapshot",
                "restore",
                "commit_prefix",
                "finish",
            )
            or type(self.tokens) is not list
            or len(self.tokens) > (2048 if self.op == "prefill_pipeline" else 256)
            or any(type(t) is not int or t < 0 for t in self.tokens)
            or (
                self.op
                in ("prefill_hidden", "prefill_pipeline", "forward", "verify", "commit_prefix")
            )
            != bool(self.tokens)
            or (self.op == "verify" and len(self.tokens) != 2)
            or (self.op == "commit_prefix" and self.tokens not in ([0], [1], [2]))
        ):
            raise ValueError("Invalid bounded rank command")


class Worker:
    def __init__(self, dispatch, *, overlapped_prefill=False):
        self.dispatch = dispatch
        if type(overlapped_prefill) is not bool:
            raise ValueError("Pipeline opt-in must be bool")
        self.overlapped_prefill = overlapped_prefill
        self.pipeline_used = False
        self.pending_final = None
        self.broken = False
        self.request = None
        self.sequence = -1
        self.saved = False
        self.verified = False
        self.prefilling = False
        self.finished = True

    def handle(self, command):
        if self.broken:
            raise ValueError("Rank raw stream is broken")
        command.validate()
        if self.pending_final is not None and (
            command.op != "forward" or command.tokens != [self.pending_final]
        ):
            raise ValueError("Compound drain requires exact final prompt singleton")
        if command.op == "reset":
            if command.sequence != 0 or command.request == self.request:
                raise ValueError("Reset requires fresh request and sequence0")
        elif command.request != self.request or command.sequence != self.sequence + 1:
            raise ValueError("Rank request/sequence mismatch")
        elif self.finished:
            raise ValueError("Request already finished")
        if command.op == "prefill_hidden" and not self.prefilling:
            raise ValueError("Hidden prefill must precede final prompt forward")
        if command.op == "prefill_pipeline" and (
            not self.overlapped_prefill
            or not self.prefilling
            or self.pipeline_used
            or self.sequence != 0
        ):
            raise ValueError("Pipeline requires enabled fresh prefix phase")
        if command.op == "restore" and not self.saved:
            raise ValueError("No rank-local snapshot")
        if command.op == "commit_prefix" and not self.verified:
            raise ValueError("No fresh width2 verify")
        if command.op == "prefill_pipeline":
            try:
                self.dispatch.__self__.dispatch_pipeline(command)
            except BaseException:
                self.broken = True
                raise
            self.pipeline_used = True
            self.pending_final = command.tokens[-1]
        else:
            self.dispatch(command.op, command.tokens)
            if command.op == "forward":
                self.pending_final = None
        self.verified = command.op == "verify"
        self.request = command.request
        self.sequence = command.sequence
        if command.op == "reset":
            self.saved = False
            self.pipeline_used = False
            self.prefilling = True
            self.finished = False
        elif command.op != "prefill_hidden":
            self.prefilling = False
        if command.op == "snapshot":
            self.saved = True
        if command.op == "finish":
            self.saved = False
            self.finished = True
        return {
            "request": command.request,
            "sequence": command.sequence,
            "op": command.op,
            "rank": 1,
            "ok": True,
        }
