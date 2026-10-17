"""Failure cleanup at actual Worker/Channel raw stream seam; bounded joins."""

import pytest
from distributed_control import decode_frame, serve_request
from overlap_prefill import credit, run_rank1_pipeline
from rank_control import Command
from test_overlap_protocol import Link, WireChannel, start_thread


@pytest.mark.parametrize(
    "fault", ["stale", "range", "bool", "opcode", "disabled", "drop", "cancel", "failure"]
)
@pytest.mark.parametrize("eager", [False, True])
def test_fault_never_emits_json_ack_into_broken_raw_stream(fault, eager):
    wire = Link(eager, timeout=0.2)
    channels = [WireChannel(wire, r) for r in (0, 1)]
    errors = []
    events = []
    cmd = Command("request-a", 1, "prefill_pipeline", [1] * 513)

    class Target:
        overlapped_prefill = fault != "disabled"

        def dispatch(self, op, tokens):
            pass

        def dispatch_pipeline(self, command):
            def forward(tokens, *, progress):
                if fault == "failure":
                    raise RuntimeError("symbolic native-stage failure")
                wire.send(1, ("hidden", tokens))

            run_rank1_pipeline(command, channels[1], forward, events.append)

    thread = start_thread(lambda: serve_request(channels[1], Target(), events.append), errors)
    try:
        channels[0].begin(Command("request-a", 0, "reset", []))
        assert channels[0].finish()["op"] == "reset"
        if fault == "opcode":
            channels[0].send(dict(request="request-a", sequence=1, op="pipeline", tokens=[1]))
        else:
            channels[0].begin(cmd)
        if fault not in ("opcode", "disabled", "failure"):
            assert wire.receive(0)[0] == "hidden"
            value = credit(cmd, 0, 0, 256)
            if fault == "stale":
                value["request_nonce"] = "stale"
            elif fault == "range":
                value["physical_end"] = 255
            elif fault == "bool":
                value["chunk_index"] = False
            if fault in ("stale", "range", "bool"):
                channels[0].send(value)
            elif fault == "cancel":
                wire.close()
            # drop intentionally sends no credit: bounded transport deadline must release.
        thread.join(1)
        assert not thread.is_alive()
        assert errors, "peer failure was silently accepted"
        sent = [decode_frame(v[1]) for r, v in wire.sent if r == 1 and v[0] == "frame"]
        assert [v.get("op") for v in sent] == ["reset"], (
            "failed raw stream emitted compound/error ACK"
        )
        assert not any(e.get("phase") == "pipeline_prefix_done" for e in events)
    finally:
        wire.close()
        thread.join(1)
        assert not thread.is_alive()
