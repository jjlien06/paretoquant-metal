"""New021 strict CPU integration. No physical throughput claims."""

import mlx.core as mx
import pytest
from test_phased_cpu import tiny_model

mx.set_default_device(mx.cpu)


def test_capture_admission_cannot_inherit_019_receipts():
    from verification import Admission, width_gate

    model = tiny_model()
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    with pytest.raises(ValueError, match="prefix"):
        Admission(model, "native", gate, gate, {"small": gate, "full": gate}, prefix_rollback=True)
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native", prefix_rollback=True)
    assert gate["prefix_rollback"]["passed"]
    assert gate["prefix_rollback"]["same_width"]["cache_pass"]
    assert [p["keep"] for p in gate["prefix_rollback"]["prefixes"]] == [0, 1, 2]
    live = Admission(
        model, "native", gate, gate, {"small": gate, "full": gate}, prefix_rollback=True
    )
    live.require(model, "native", prefix_rollback=True)


@pytest.mark.parametrize("terminal", ["reject", "eos", "cap", "fresh"])
def test_real_greedy_prefix_rejection_terminal_and_fresh(terminal):
    from types import SimpleNamespace

    from cache_state import assert_cache_equal
    from mlx_backend import MTPBackend
    from mtp_head import BoundHead, MTPHead
    from phased_decode import phased_generate
    from speculative import generate
    from verification import Admission, width_gate

    model = tiny_model()
    tokenizer = SimpleNamespace(eos_token_ids=[], decode=lambda ids, **kw: str(ids))
    truth = phased_generate(model, tokenizer, [1, 2, 3], max_tokens=7)["token_ids"]
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native", prefix_rollback=True)
    admission = Admission(
        model, "native", gate, gate, {"small": gate, "full": gate}, prefix_rollback=True
    )
    head = BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )
    actions = []
    backend = MTPBackend(
        model, head, admission=admission, prefix_rollback=True, progress=actions.append
    )
    limit = 2 if terminal in ("eos", "cap") else 7
    eos = [truth[1]] if terminal == "eos" else []
    for _ in range(2 if terminal == "fresh" else 1):
        index = [1]

        def draft(hidden, y):
            backend.head.forward(
                hidden, mx.array([[y]]), backend.mtp_cache, position=backend.mtp_cache.offset
            )
            token = truth[index[0]] if terminal in ("eos", "cap") else (truth[index[0]] + 1) % 128
            index[0] += 1
            return token

        backend.draft = draft
        result = generate(backend, [1, 2, 3], max_tokens=limit, eos_ids=eos)
        assert result["token_ids"] == truth[:limit]
        assert result["teacher_forwards"] == result["verification_rounds"]
        native = model.make_cache()
        for token in [1, 2, 3] + result["token_ids"][:-1]:
            out = model(mx.array([[token]]), cache=native)
            mx.eval(out, [c.state for c in native])
        assert_cache_equal(native, backend.cache)
        backend.finish()
    assert any(a.get("op") == "commit_prefix" for a in actions)


def test_commit_control_is_bounded_and_requires_fresh_verify():
    from rank_control import Command, Worker

    w = Worker(lambda *a: None)
    w.handle(Command("x", 0, "reset", []))
    with pytest.raises(ValueError):
        w.handle(Command("x", 1, "commit_prefix", [1]))
    w.handle(Command("x", 1, "verify", [4, 5]))
    w.handle(Command("x", 2, "commit_prefix", [1]))
    with pytest.raises(ValueError):
        w.handle(Command("x", 3, "commit_prefix", [1]))
    with pytest.raises(ValueError):
        Command("x", 4, "commit_prefix", [4]).validate()  # 030: [3] admitted for width3


def test_prefix_receipt_requires_all_live_parts_and_strict_hidden_logits():
    from copy import deepcopy

    from supervise import verify_prefix_evidence
    from verification import width_gate

    gate = width_gate(
        tiny_model(), [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native", prefix_rollback=True
    )["prefix_rollback"]
    verify_prefix_evidence(gate, 4)
    for mutate in ("hidden", "caches", "keep", "tolerance"):
        bad = deepcopy(gate)
        if mutate == "hidden":
            bad["prefixes"][1]["hidden"]["strict_pass"] = False
        elif mutate == "caches":
            bad["same_width"]["caches"] = []
        elif mutate == "keep":
            bad["prefixes"][1]["keep"] = 2
        else:
            bad["atol"] = 1e-3
        with pytest.raises(ValueError):
            verify_prefix_evidence(bad, 4)


def test_new_cli_and_registry_bind_opt_in():
    import subprocess
    import sys

    import remote_actor

    assert "prefix_rollback.py" in remote_actor.SOURCES
    for script in ("supervise.py", "probe.py"):
        from pathlib import Path

        p = subprocess.run(
            [sys.executable, str(Path(__file__).parent / script), "--help"],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0 and "--prefix-rollback" in p.stdout
