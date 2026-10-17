"""All-position strict admission, including head-cache and continuation proof."""

import mlx.core as mx
import pytest
from test_phased_cpu import tiny_model

mx.set_default_device(mx.cpu)


def make_head(model):
    from mtp_head import BoundHead, MTPHead

    return BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )


def test_same_process_gate_checks_boundary_chunks_all_positions_and_head_cache():
    from hidden_prefill_gate import gate_lengths, prefill_gate

    model = tiny_model()
    lengths = gate_lengths("small", 256)
    assert lengths == [256, 257, 511, 512, 513]
    assert gate_lengths("full", 256) == [513]
    assert gate_lengths("full", 8) == [17]
    for width in (0, 257, True):
        with pytest.raises(ValueError):
            gate_lengths("full", width)
    receipt = prefill_gate(
        model, make_head(model), prompt_lengths=lengths, chunk_size=256, eval_group_size=32
    )
    assert receipt["passed"] and receipt["atol"] == 1e-6 and receipt["rtol"] == 1e-5
    assert receipt["same_process"] and receipt["target_forward_mode"] == "native"
    for case, length in zip(receipt["cases"], lengths):
        assert case["prompt_tokens"] == length and case["passed"]
        for row in case["chunks"]:
            assert row["hidden"]["shape"] == [1, row["end"] - row["start"], 64]
            assert row["hidden"]["strict_pass"]
            assert row["head_offset"] == row["end"]
            assert row["head_caches"] and all(x["strict_pass"] for x in row["head_caches"])
            assert row["caches"] and row["native_caches"]
        assert case["final"]["logits"]["strict_pass"]
        assert case["final"]["native_logits"]["strict_pass"]
        assert len(case["continuation"]) == 2
        for row in case["continuation"]:
            assert row["head_logits"]["strict_pass"] and row["native_hidden"]["strict_pass"]
            assert row["head_caches"] and row["passed"]


def test_gate_rejects_wrong_early_hidden_even_if_final_position_is_unchanged(monkeypatch):
    import phased_decode
    from hidden_prefill_gate import prefill_gate

    original = phased_decode.phased_forward

    def wrong_hidden(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("logits") is False and kwargs.get("return_hidden") is True:
            scores, h = result
            h = mx.concatenate([h[:, :1] + 0.01, h[:, 1:]], axis=1)
            return scores, h
        return result

    monkeypatch.setattr(phased_decode, "phased_forward", wrong_hidden)
    model = tiny_model()
    receipt = prefill_gate(model, make_head(model), prompt_lengths=[9], chunk_size=4)
    assert not receipt["passed"]
    assert not receipt["cases"][0]["chunks"][0]["hidden"]["strict_pass"]


def test_hidden_prefill_requires_live_small_full_and_both_peer_gates():
    from hidden_prefill_gate import full_gate, prefill_gate
    from mlx_backend import MTPBackend
    from verification import Admission, width_gate

    model = tiny_model()
    head = make_head(model)
    width = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    peer = {"small": width, "full": width}
    reference = Admission(model, "native", width, width, peer)
    with pytest.raises(ValueError, match="hidden|Hidden"):
        MTPBackend(model, head, admission=reference, hidden_only_prefill=True)
    with pytest.raises(ValueError, match="hidden|Hidden"):
        Admission(model, "native", width, width, peer, hidden_only_prefill=True)
    # Reduced full gate width bounds work and remain explicit in trusted receipts.
    small = dict(
        width,
        hidden_only_prefill=prefill_gate(
            model, head, prompt_lengths=[256, 257, 511, 512, 513], chunk_size=256
        ),
    )
    full = dict(width, hidden_only_prefill=full_gate(model, head, chunk_size=4))
    admitted = Admission(
        model,
        "native",
        small,
        full,
        {"small": small, "full": full},
        hidden_only_prefill=True,
        hidden_gate_chunk_size=4,
    )
    MTPBackend(model, head, admission=admitted, hidden_only_prefill=True)
    with pytest.raises(ValueError):
        MTPBackend(model, head, admission=admitted)
    with pytest.raises(ValueError, match="hidden|Hidden"):
        Admission(
            model,
            "native",
            small,
            full,
            {"small": small, "full": width},
            hidden_only_prefill=True,
            hidden_gate_chunk_size=4,
        )


def test_gate_rejects_head_cache_corruption(monkeypatch):
    from hidden_prefill_gate import prefill_gate

    model = tiny_model()
    head = make_head(model)
    original = head.forward
    calls = []

    def corrupt(hidden, tokens, cache, **kw):
        result = original(hidden, tokens, cache, **kw)
        calls.append(cache)
        if len(calls) == 2:
            cache.keys = cache.keys + 0.01
        return result

    monkeypatch.setattr(head, "forward", corrupt)
    receipt = prefill_gate(model, head, prompt_lengths=[9], chunk_size=4)
    assert not receipt["passed"]
    assert any(not x["strict_pass"] for x in receipt["cases"][0]["chunks"][0]["head_caches"])
