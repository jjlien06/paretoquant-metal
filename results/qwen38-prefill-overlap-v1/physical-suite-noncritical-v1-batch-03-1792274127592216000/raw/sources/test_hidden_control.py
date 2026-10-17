"""Bounded prefill command and native MTP recurrence, CPU only."""

import mlx.core as mx
import pytest
from test_phased_cpu import tiny_model

mx.set_default_device(mx.cpu)


def test_explicit_prefill_op_bounds_sequence_and_finish():
    from rank_control import Command, Worker

    Command("req", 1, "prefill_hidden", [1] * 256).validate()
    for tokens in ([], [1] * 257, [-1], [True]):
        with pytest.raises(ValueError):
            Command("req", 1, "prefill_hidden", tokens).validate()
    worker = Worker(lambda op, tokens: None)
    worker.handle(Command("req", 0, "reset", []))
    ack = worker.handle(Command("req", 1, "prefill_hidden", [1, 2]))
    assert ack == dict(request="req", sequence=1, op="prefill_hidden", rank=1, ok=True)
    for command in (
        Command("other", 2, "prefill_hidden", [1]),
        Command("req", 1, "prefill_hidden", [1]),
    ):
        with pytest.raises(ValueError):
            worker.handle(command)
    worker.handle(Command("req", 2, "forward", [3]))
    with pytest.raises(ValueError):
        worker.handle(Command("req", 3, "prefill_hidden", [4]))
    worker.handle(Command("req", 3, "finish", []))
    with pytest.raises(ValueError):
        worker.handle(Command("req", 4, "forward", [4]))


@pytest.mark.parametrize("length", [1, 256, 257, 511, 512, 513])
def test_optin_prefill_retains_reference_caches_shifted_head_and_final_projection(
    length, monkeypatch
):
    from mlx_backend import MTPBackend
    from mtp_head import BoundHead, MTPHead
    from verification import Admission, cache_metrics, metric, width_gate

    model = tiny_model()
    head = BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    admission = Admission(model, "native", gate, gate, {"small": gate, "full": gate})
    prompt = [1 + i % 100 for i in range(length)]
    reference = MTPBackend(model, head, admission=admission)
    expected_token, expected_hidden = reference.fresh(prompt)
    from hidden_prefill_gate import full_gate, prefill_gate

    small = dict(
        gate,
        hidden_only_prefill=prefill_gate(
            model, head, prompt_lengths=[256, 257, 511, 512, 513], chunk_size=256
        ),
    )
    full = dict(gate, hidden_only_prefill=full_gate(model, head, chunk_size=4))
    opt_admission = Admission(
        model,
        "native",
        small,
        full,
        {"small": small, "full": full},
        hidden_only_prefill=True,
        hidden_gate_chunk_size=4,
    )
    original = type(model.language_model.lm_head).__call__
    calls, shifts = [], []

    def projection(self, x):
        if self is model.language_model.lm_head:
            calls.append(x.shape[1])
        return original(self, x)

    original_head = head.forward

    def recurrence(hidden, tokens, cache, *, position, logits=True):
        shifts.append((position, tokens.tolist()[0], cache.offset, hidden.shape[1]))
        return original_head(hidden, tokens, cache, position=position, logits=logits)

    monkeypatch.setattr(type(model.language_model.lm_head), "__call__", projection)
    monkeypatch.setattr(head, "forward", recurrence)
    events = []
    optimized = MTPBackend(
        model, head, admission=opt_admission, hidden_only_prefill=True, progress=events.append
    )
    token, hidden = optimized.fresh(prompt)
    assert token == expected_token and metric(expected_hidden, hidden)["strict_pass"]
    assert calls == [1], "no nonfinal target projection; final prompt token still projects"
    assert all(x["strict_pass"] for x in cache_metrics(reference.cache, optimized.cache))
    if length == 1:
        assert reference.mtp_cache.keys is optimized.mtp_cache.keys is None
    else:
        assert all(
            x["strict_pass"] for x in cache_metrics([reference.mtp_cache], [optimized.mtp_cache])
        )
    chunks = [prompt[s : min(s + 256, length - 1)] for s in range(0, length - 1, 256)]
    actions = [e for e in events if e["phase"] == "rank_action"]
    assert [e["op"] for e in actions] == ["reset"] + ["prefill_hidden"] * len(chunks) + ["forward"]
    assert [e["tokens"] for e in actions[1:-1]] == chunks
    assert shifts == [
        (s, prompt[s + 1 : min(s + 256, length - 1) + 1], s, len(c))
        for s, c in zip(range(0, length - 1, 256), chunks)
    ]
    assert optimized.mtp_cache.offset == length - 1
    reference.finish()
    optimized.finish()
