"""Real Command/Worker/Channel seam; symbolic stages, no MLX import."""

import threading
from collections import deque

from distributed_control import Channel, serve_request
from rank_control import Command


class Link:
    """Zero-capacity rendezvous or eager wire; never Queue.put mocks."""

    def __init__(self, eager=False, timeout=2):
        self.timeout = timeout
        self.cv = threading.Condition()
        self.slots = [deque(), deque()]
        self.eager = eager
        self.closed = False
        self.sent = []

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def wait(self, predicate):
        if not self.cv.wait_for(lambda: self.closed or predicate(), timeout=self.timeout):
            raise TimeoutError("wire deadline")
        if self.closed:
            raise RuntimeError("wire closed")

    def send(self, rank, value):
        with self.cv:
            item = [value, False]
            self.slots[1 - rank].append(item)
            self.sent.append((rank, value))
            self.cv.notify_all()
            if not self.eager:
                self.wait(lambda: item[1])

    def receive(self, rank):
        with self.cv:
            self.wait(lambda: bool(self.slots[rank]))
            item = self.slots[rank].popleft()
            item[1] = True
            self.cv.notify_all()
            return item[0]


class WireChannel(Channel):
    def __init__(self, wire, rank):
        self.wire, self.rank = wire, rank

    def send(self, value):
        # Exercise actual frame encoding/decoding, not dictionary aliases.
        from distributed_control import encode_frame

        self.wire.send(self.rank, ("frame", encode_frame(value)))

    def receive(self):
        from distributed_control import decode_frame

        kind, raw = self.wire.receive(self.rank)
        assert kind == "frame", "JSON/raw-hidden stream confusion"
        return decode_frame(raw)


def start_thread(fn, errors):
    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def seam(length=513, eager=False, hold=True, follower_finished=None):
    command = Command("request-a", 1, "prefill_pipeline", [i % 100 for i in range(length)])
    command.validate()  # First RED must fail here on the missing real opcode.
    from overlap_prefill import run_rank0_pipeline, run_rank1_pipeline

    wire = Link(eager)
    channels = [WireChannel(wire, r) for r in (0, 1)]
    entered, release, next_stage = (threading.Event() for _ in range(3))
    events = [[], []]
    stages, heads, acks, errors = [], [], [], []

    class Target:
        overlapped_prefill = True

        def dispatch(self, op, tokens):
            stages.append(("op", op, list(tokens)))

        def dispatch_pipeline(self, cmd):
            def forward(tokens, *, progress):
                index = len([s for s in stages if s[0] == "rank1"])
                stages.append(("rank1", index, list(tokens)))
                if index == 1:
                    next_stage.set()
                wire.send(1, ("hidden", list(tokens)))
                progress({"phase": "handoff_send_done"})

            run_rank1_pipeline(cmd, channels[1], forward, events[1].append)

    def follower():
        serve_request(channels[1], Target(), events[1].append)

    def leader():
        reset = Command("request-a", 0, "reset", [])
        channels[0].begin(reset)
        acks.append(channels[0].finish())
        channels[0].begin(command)

        def forward(tokens, *, receive_stage, progress):
            kind, hidden = wire.receive(0)
            assert kind == "hidden" and hidden == tokens
            progress({"phase": "handoff_receive_done"})
            receive_stage()
            if not entered.is_set():
                entered.set()
                assert release.wait(2), "rank0 release deadline"
            stages.append(("rank0", len(tokens)))
            return None, hidden

        def head(hidden, shifted, position):
            heads.append((hidden, shifted, position))

        run_rank0_pipeline(command, channels[0], forward, head, events[0].append)
        acks.append(channels[0].finish())
        # Final singleton is a separate ordinary command AFTER compound drain.
        for seq, op, tokens in ((2, "forward", command.tokens[-1:]), (3, "finish", [])):
            channels[0].begin(Command(command.request, seq, op, tokens))
            acks.append(channels[0].finish())

    threads = [start_thread(follower, errors), start_thread(leader, errors)]
    try:
        assert entered.wait(2), f"rank0 did not receive: {errors}"
        if hold:
            assert next_stage.wait(1), "rank1 chunk1 must progress while rank0 chunk0 is held"
            if follower_finished is not None:
                assert follower_finished.wait(1), "mutant completion deadline"
            # Credit prevents even eager transport from building chunk2 while rank0 is held.
            assert len([s for s in stages if s[0] == "rank1"]) == min(2, (length - 2) // 256 + 1), (
                "credit bound exceeded"
            )
        release.set()
        for thread in threads:
            thread.join(3)
        assert not any(t.is_alive() for t in threads), "bounded joins failed"
        assert not errors, errors
        return command, wire, events, stages, heads, acks
    finally:
        release.set()
        wire.close()
        for thread in threads:
            thread.join(1)
        assert not any(t.is_alive() for t in threads)


def test_real_worker_zero_capacity_progress_before_rank0_completion():
    command, wire, events, stages, heads, acks = seam()
    assert [ack["op"] for ack in acks] == ["reset", "prefill_pipeline", "forward", "finish"]
    assert [(h[1], h[2]) for h in heads] == [
        (command.tokens[1:257], 0),
        (command.tokens[257:513], 256),
    ]
    assert [(s[1], s[2]) for s in stages if s[0] == "op"][-2:] == [
        ("forward", command.tokens[-1:]),
        ("finish", []),
    ]
    assert all(not slot for slot in wire.slots)
