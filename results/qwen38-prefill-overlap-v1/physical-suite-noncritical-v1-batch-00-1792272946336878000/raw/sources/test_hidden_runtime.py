"""Opt-in admission reaches real loaded runtime; all work forced CPU."""

from types import SimpleNamespace as S

import mlx.core as mx
from test_hidden_gate import make_head
from test_phased_cpu import tiny_model

mx.set_default_device(mx.cpu)


def test_loaded_runtime_requires_and_publishes_small_full_hidden_gates():
    from hidden_prefill_gate import prefill_gate
    from mtp_runtime import run_loaded
    from verification import width_gate

    model = tiny_model()
    head = make_head(model)
    small = width_gate(
        model,
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9],
        mode="native",
        prefix_rollback=True,
        eval_group_size=32,
    )
    small["hidden_only_prefill"] = prefill_gate(
        model, head, prompt_lengths=[256, 257, 511, 512, 513], eval_group_size=32
    )
    args = S(
        verifier_mode="native",
        verification_gate_only=False,
        max_tokens=5,
        target_revision="CPU-fixture",
        prefix_rollback=True,
        hidden_only_prefill=True,
        hidden_gate_chunk_size=4,
    )
    events = []
    result = run_loaded(
        model,
        S(eos_token_ids=[], decode=lambda ids, **kw: str(ids)),
        [1, 2, 3],
        args,
        group=None,
        small={32: small},
        eval_groups=[32],
        report=events.append,
        head=head,
    )
    gate = result["verification"]
    assert gate["hidden_only_prefill"] is True
    assert gate["hidden_gate_chunk_size"] == 4
    full_hidden = gate["groups"][32]["full"]["hidden_only_prefill"]
    assert full_hidden["passed"]
    assert any(
        row["end"] - row["start"] == 256
        for case in full_hidden["representative_256"]["cases"]
        for row in case["chunks"]
    ), "full gate must exercise native width256 even with reduced forced chunks"
    assert gate["groups"][32]["full"]["prefix_rollback"]["passed"]
    assert gate["groups"][32]["peer"]["small"]["hidden_only_prefill"]["passed"]
    admit = next(e for e in events if e["phase"] == "numerical_admission_done")
    assert admit["hidden_only_prefill"] is True
    assert events.index(admit) < next(
        i for i, e in enumerate(events) if e["phase"] == "request_before"
    )
    actions = [e["op"] for e in events if e["phase"] == "rank_action"]
    assert actions[:3] == ["reset", "prefill_hidden", "forward"]
    assert actions[-1] == "finish"
    assert len(result["requests"][0]["token_ids"]) == 5


def test_small_loaded_checkpoint_gets_all_boundary_hidden_and_head_gates(monkeypatch):
    import mlx_lm
    from mtp_runtime import small_gate

    # Only IO is replaced; full gate arithmetic and live small model are real.
    model = tiny_model()
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **k: (model, None))
    import mtp_runtime

    monkeypatch.setattr(mtp_runtime, "bind_metadata", lambda *a: {"CPU-fixture": True})
    args = S(
        small_model="CPU-fixture",
        small_config_sha256="a" * 64,
        small_index_sha256="b" * 64,
        verifier_mode="native",
        prefix_rollback=True,
        hidden_only_prefill=True,
        hidden_gate_chunk_size=4,
    )
    receipt = small_gate(args, [32], lambda e: None)[32]
    assert receipt["passed"] and receipt["hidden_only_prefill"]["passed"]
    assert receipt["hidden_only_prefill"]["prompt_lengths"] == [256, 257, 511, 512, 513]
    assert receipt["hidden_only_prefill"]["chunk_size"] == 256
