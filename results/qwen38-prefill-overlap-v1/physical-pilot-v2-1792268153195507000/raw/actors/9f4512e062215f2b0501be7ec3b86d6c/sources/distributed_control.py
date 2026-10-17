"""Bounded CPU-stream JSON frames on the existing MLX group; no new socket."""

import json
from dataclasses import asdict

from rank_control import Command, Worker

MAX_FRAME = 65536


def encode_frame(value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf8")
    if not 1 <= len(raw) <= MAX_FRAME:
        raise ValueError("Control frame bound")
    return raw


def decode_frame(raw):
    if not 1 <= len(raw) <= MAX_FRAME:
        raise ValueError("Control frame bound")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate control key")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )
    if not isinstance(value, dict):
        raise ValueError("Control object required")
    return value


class Channel:
    def __init__(self, group):
        if group.size() != 2:
            raise ValueError("Two ranks required")
        self.group = group
        self.other = 1 - group.rank()

    def send(self, value):
        import mlx.core as mx

        raw = encode_frame(value)
        mx.eval(
            mx.distributed.send(
                mx.array([len(raw)], mx.int32), self.other, group=self.group, stream=mx.cpu
            )
        )
        mx.eval(
            mx.distributed.send(
                mx.array(list(raw), mx.uint8), self.other, group=self.group, stream=mx.cpu
            )
        )

    def receive(self):
        import mlx.core as mx

        size = mx.distributed.recv((1,), mx.int32, self.other, group=self.group, stream=mx.cpu)
        mx.eval(size)
        n = int(size.item())
        if not 1 <= n <= MAX_FRAME:
            raise ValueError("Control frame bound")
        raw = mx.distributed.recv((n,), mx.uint8, self.other, group=self.group, stream=mx.cpu)
        mx.eval(raw)
        return decode_frame(bytes(raw.tolist()))

    def begin(self, command):
        command.validate()
        self.send(asdict(command))

    def finish(self):
        return self.receive()


def serve_request(channel, target, report):
    target.channel = channel
    worker = Worker(
        target.dispatch, overlapped_prefill=getattr(target, "overlapped_prefill", False)
    )
    while True:
        command = Command(**channel.receive())
        ack = worker.handle(command)
        report({"phase": "rank_action", **ack, "tokens": command.tokens})
        channel.send(ack)
        if command.op == "finish":
            return
