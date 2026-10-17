"""Credit strictness, eager backpressure and real worker phase rejection."""

import pytest
from distributed_control import decode_frame, encode_frame
from rank_control import Command, Worker
from test_overlap_protocol import seam


def test_eager_witness_kills_removed_receiver_credit_wait(monkeypatch):
    import ast
    import threading
    from pathlib import Path

    import overlap_prefill

    tree = ast.parse(Path(overlap_prefill.__file__).read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_rank1_pipeline"
    )

    class RemoveCreditWait(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "validate_credit"
            ):
                return None
            return self.generic_visit(node)

    mutated = RemoveCreditWait().visit(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(vars(overlap_prefill))
    exec(
        compile(ast.fix_missing_locations(mutated), "explicit-CPU-credit-mutant", "exec"), namespace
    )
    finished = threading.Event()

    def mutant(*args):
        namespace["run_rank1_pipeline"](*args)
        finished.set()

    monkeypatch.setattr(overlap_prefill, "run_rank1_pipeline", mutant)
    with pytest.raises(AssertionError, match="credit bound exceeded"):
        seam(2048, eager=True, follower_finished=finished)


def test_eager_send_still_waits_for_consumption_credit():
    command, wire, events, stages, heads, acks = seam(2048, eager=True)
    assert len(heads) == 8
    for rank in (0, 1):
        credits = [
            e
            for e in events[rank]
            if e["phase"] == ("pipeline_credit_sent" if rank == 0 else "pipeline_credit_received")
        ]
        assert [(e["chunk_index"], e["physical_start"], e["physical_end"]) for e in credits] == [
            (i, s, min(s + 256, 2047)) for i, s in enumerate(range(0, 2047, 256))
        ]
    assert all(not slot for slot in wire.slots)


@pytest.mark.parametrize("length", [2, 255, 256, 257, 511, 512, 513, 2048])
def test_exact_chunks_shift_and_final_drain(length):
    command, wire, events, stages, heads, acks = seam(length, eager=True, hold=length > 257)
    assert [(h[1], h[2]) for h in heads] == [
        (command.tokens[s + 1 : min(s + 256, length - 1) + 1], s) for s in range(0, length - 1, 256)
    ]
    assert [a["op"] for a in acks] == ["reset", "prefill_pipeline", "forward", "finish"]


def test_only_pipeline_gets_2048_cap_and_64k_frame_unchanged():
    from dataclasses import asdict

    from distributed_control import MAX_FRAME

    assert MAX_FRAME == 65536
    cmd = Command("a" * 32, 1, "prefill_pipeline", [248046] * 2048)
    cmd.validate()
    assert len(encode_frame(asdict(cmd))) < MAX_FRAME
    for op, toks in [
        ("prefill_pipeline", []),
        ("prefill_pipeline", [1] * 2049),
        ("forward", [1] * 257),
        ("prefill_hidden", [1] * 257),
        ("verify", [1] * 3),
        ("pipeline", [1]),
        ("prefill_pipeline", [True]),
    ]:
        with pytest.raises(ValueError):
            Command("r", 1, op, toks).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_nonce", "old"),
        ("command_sequence", True),
        ("command_sequence", 2),
        ("chunk_index", 1),
        ("physical_start", 1),
        ("physical_end", 255),
        ("schema", "wrong"),
    ],
)
def test_credit_exact_types_nonce_sequence_and_range(field, value):
    from overlap_prefill import credit, validate_credit

    expected = credit(Command("r", 1, "prefill_pipeline", [1, 2]), 0, 0, 1)
    bad = dict(expected, **{field: value})
    with pytest.raises(ValueError):
        validate_credit(bad, expected)


class Target:
    def dispatch(self, op, tokens):
        pass

    def dispatch_pipeline(self, command):
        pass


def worker():
    w = Worker(Target().dispatch, overlapped_prefill=True)
    w.handle(Command("r", 0, "reset", []))
    return w


@pytest.mark.parametrize(
    "op,tokens",
    [
        ("forward", [8]),
        ("forward", [1, 9]),
        ("snapshot", []),
        ("finish", []),
        ("prefill_hidden", [1]),
        ("reset", []),
    ],
)
def test_compound_requires_exact_final_singleton_before_decode(op, tokens):
    w = worker()
    w.handle(Command("r", 1, "prefill_pipeline", [1, 9]))
    with pytest.raises(ValueError):
        w.handle(Command("new" if op == "reset" else "r", 0 if op == "reset" else 2, op, tokens))


def test_pipeline_failure_poisoned_no_json_ack_or_reset_recovery():
    class Failing(Target):
        def dispatch_pipeline(self, command):
            raise TimeoutError("lost hidden or credit")

    w = Worker(Failing().dispatch, overlapped_prefill=True)
    w.handle(Command("r", 0, "reset", []))
    with pytest.raises(TimeoutError):
        w.handle(Command("r", 1, "prefill_pipeline", [1, 9]))
    with pytest.raises(ValueError, match="broken"):
        w.handle(Command("fresh", 0, "reset", []))


def test_wrong_flag_and_duplicate_compound_reject():
    with pytest.raises(ValueError):
        Worker(Target().dispatch, overlapped_prefill=1)
    w = Worker(Target().dispatch)
    w.handle(Command("r", 0, "reset", []))
    with pytest.raises(ValueError):
        w.handle(Command("r", 1, "prefill_pipeline", [1, 9]))
    w = worker()
    w.handle(Command("r", 1, "prefill_pipeline", [1, 9]))
    with pytest.raises(ValueError):
        w.handle(Command("r", 2, "prefill_pipeline", [1, 9]))


def test_duplicate_json_keys_fail_closed():
    with pytest.raises(ValueError):
        decode_frame(b'{"request":"new","request":"stale"}')
