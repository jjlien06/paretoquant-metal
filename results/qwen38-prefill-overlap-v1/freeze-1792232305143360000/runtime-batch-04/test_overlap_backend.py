"""Execute exact backend class bodies with symbolic native stages, never MLX.

The test-local Admission is NOT verification.Admission and grants no live model
capability. It proves actual backend/Worker/Channel wiring and source audit only.
"""

import ast
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as S

import pytest
from distributed_control import serve_request
from overlap_prefill import run_rank0_pipeline, run_rank1_pipeline
from rank_control import Command
from test_overlap_protocol import Link, WireChannel, start_thread


def backend_classes(forward):
    class Admission:
        def require(self, *a, **k):
            pass

        def require_overlap(self, *a, **k):
            pass

    class KVCache:
        offset = 0

    namespace = dict(
        time=time,
        uuid=uuid,
        nullcontext=nullcontext,
        Command=Command,
        Admission=Admission,
        KVCache=KVCache,
        ownership=lambda model, group: (model.rank, 2),
        phased_forward=forward,
        run_rank0_pipeline=run_rank0_pipeline,
        run_rank1_pipeline=run_rank1_pipeline,
        mx=S(array=lambda x, dtype: x, int32="int32", argmax=lambda *a: S(item=lambda: 7)),
    )
    source = Path(__file__).with_name("mlx_backend.py").read_text()
    tree = ast.parse(source)
    classes = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]
    )
    exec(compile(classes, "mlx_backend.py", "exec"), namespace)
    return namespace, Admission()


def test_backend_rejects_wrong_mode_before_sending_and_poisoned_stream_never_resets():
    ns, admission = backend_classes(lambda *a, **k: None)
    model = S(rank=0, make_cache=lambda: [], language_model=S(args=S(vocab_size=128)))
    sent = []
    peer = S(begin=sent.append, finish=lambda: None)
    backend = ns["MTPBackend"](
        model,
        S(),
        group=S(),
        peer=peer,
        admission=admission,
        hidden_only_prefill=True,
        prefill_mode="serial",
    )
    backend.request = "r"
    with pytest.raises(ValueError):
        backend._command("prefill_pipeline", [1, 2])
    assert not sent, "wrong mode must reject before raw/control send"
    backend.prefill_mode = "overlap"
    backend.overlapped_prefill = True

    def failure(command):
        raise TimeoutError("raw transfer fixture")

    backend._pipeline = failure
    with pytest.raises(TimeoutError):
        backend._command("prefill_pipeline", [1, 2])
    count = len(sent)
    with pytest.raises(ValueError, match="broken"):
        backend._command("reset")
    assert len(sent) == count


@pytest.mark.parametrize(
    "mode,length",
    [("serial", 513), ("overlap", 1), ("overlap", 2), ("overlap", 513), ("overlap", 2048)],
)
def test_actual_backend_fresh_final_drain_and_auditable_stream(mode, length):
    wire = Link(eager=True)
    channels = [WireChannel(wire, r) for r in (0, 1)]
    prompt = [i % 100 for i in range(length)]
    shifts = []
    events = [[], []]
    errors = []
    models = [
        S(rank=r, make_cache=lambda: [S(offset=0)], language_model=S(args=S(vocab_size=128)))
        for r in (0, 1)
    ]

    def forward(model, inputs, cache, *, progress, logits=True, receive_stage=None, **kw):
        tokens = inputs[0]
        if model.rank == 1:
            progress({"phase": "embedding_before"})
            progress({"phase": "embedding_done"})
        else:
            progress({"phase": "handoff_receive_before", "tokens": len(tokens)})
            kind, received = wire.receive(0)
            assert kind == "hidden" and received == tokens
            progress(
                {"phase": "handoff_receive_done", "shape": [1, len(tokens), 4], "dtype": "float32"}
            )
            if receive_stage:
                receive_stage()
        progress({"phase": "block_before", "start_idx": 0, "end_idx": 2})
        cache[0].offset += len(tokens)
        progress({"phase": "block_done", "start_idx": 0, "end_idx": 2})
        for layer in (0, 1):
            progress({"phase": "layer_done", "layer_index": layer})
        if model.rank == 1:
            progress({"phase": "handoff_send_before"})
            wire.send(1, ("hidden", tokens))
            progress(
                {"phase": "handoff_send_done", "shape": [1, len(tokens), 4], "dtype": "float32"}
            )
            return
        if not logits:
            progress({"phase": "hidden_norm_before"})
        progress({"phase": "head_done" if logits else "hidden_norm_done"})

        class Scores:
            def __getitem__(self, key):
                return self

        return Scores() if logits else None, tokens

    ns, admission = backend_classes(forward)

    class Head:
        def forward(self, hidden, tokens, cache, *, position, logits=False):
            assert position == cache.offset
            shifts.append((position, tokens[0], hidden))
            cache.offset += len(tokens[0])

    Backend, Target = ns["MTPBackend"], ns["Rank1Target"]
    backend = Backend(
        models[0],
        Head(),
        group=S(),
        peer=channels[0],
        admission=admission,
        hidden_only_prefill=True,
        prefill_mode=mode,
        progress=events[0].append,
    )
    target = Target(
        models[1],
        S(),
        admission=admission,
        hidden_only_prefill=True,
        prefill_mode=mode,
        progress=events[1].append,
    )
    thread = start_thread(lambda: serve_request(channels[1], target, events[1].append), errors)
    try:
        assert backend.fresh(prompt)[0] == 7
        backend.finish()
        thread.join(3)
        assert not thread.is_alive() and not errors, errors
        assert backend.mtp_cache.offset == length - 1
        assert models[0].rank == 0 and backend.cache[0].offset == target.cache[0].offset == length
        assert [(s, t) for s, t, _ in shifts] == [
            (s, prompt[s + 1 : min(s + 256, length - 1) + 1]) for s in range(0, length - 1, 256)
        ]
        actions = [[e for e in stream if e["phase"] == "rank_action"] for stream in events]
        assert [(e["op"], e["tokens"]) for e in actions[0]] == [
            (e["op"], e["tokens"]) for e in actions[1]
        ]
        if mode == "overlap" and length > 1:
            from overlap_evidence import verify_overlap_stream

            cmd = Command(actions[0][1]["request"], 1, "prefill_pipeline", prompt)
            for rank in (0, 1):
                verify_overlap_stream(events[rank], cmd, rank, owned_layers=[0, 1])
                from overlap_prefill import StreamState

                forged = []
                state = StreamState(cmd, rank, forged.append)
                for event in events[rank]:
                    if "stream_payload" not in event:
                        continue
                    payload = dict(event["stream_payload"])
                    if payload["phase"] == "block_done":
                        payload["phase"] = "fake_equal_count_completion"
                    state.emit(payload)
                with pytest.raises(ValueError):
                    verify_overlap_stream(forged, cmd, rank, owned_layers=[0, 1])
            from overlap_evidence import verify_compound_drain

            verify_compound_drain(events[0], cmd, 0)
            verify_compound_drain(events[1], cmd, 1)
            from copy import deepcopy

            for fault in ("early", "nonce", "bool"):
                wrong = deepcopy(events[0])
                drain = next(e for e in wrong if e["phase"] == "pipeline_drained")
                if fault == "early":
                    wrong.remove(drain)
                    wrong.insert(0, drain)
                elif fault == "nonce":
                    drain["request_nonce"] = "stale"
                else:
                    drain["pending_handoffs"] = False
                with pytest.raises(ValueError):
                    verify_compound_drain(wrong, cmd, 0)
            phases = [e["phase"] for e in events[0]]
            assert (
                phases.index("pipeline_prefix_done")
                < phases.index("pipeline_drained")
                < max(i for i, p in enumerate(phases) if p == "head_done")
            )
        assert all(not q for q in wire.slots)
    finally:
        wire.close()
        thread.join(1)
        assert not thread.is_alive()
