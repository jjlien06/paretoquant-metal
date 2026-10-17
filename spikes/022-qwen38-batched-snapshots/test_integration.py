"""CPU-only admission and framing tests; no GPU or SSH."""

import importlib.util
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_frames_are_bounded_and_preserve_rank_commands():
    import pytest
    from distributed_control import decode_frame, encode_frame

    payload = {"request": "a", "sequence": 0, "op": "reset", "tokens": []}
    assert decode_frame(encode_frame(payload)) == payload
    with pytest.raises(ValueError):
        encode_frame({"x": "a" * 65536})
    with pytest.raises(ValueError):
        decode_frame(b"not-json")


ROOT = Path(__file__).parent


def test_verifier_scope_and_full_width_gate_cpu():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from test_phased_cpu import tiny_model
    from verification import width_gate
    from verifier import numerical_mode

    original = mx.quantized_matmul
    with numerical_mode("singleton-qmm", active=True):
        assert mx.quantized_matmul is not original
    assert mx.quantized_matmul is original
    model = tiny_model()
    receipt = width_gate(
        model, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], mode="singleton-qmm", eval_group_size=2
    )
    assert receipt["passed"] and len(receipt["windows"]) == 3
    assert all(
        w["logits"]["strict_pass"] and w["argmax_equal"] and w["cache_pass"]
        for w in receipt["windows"]
    )


def test_standalone_cli_and_exact_registry():
    assert (ROOT / "supervise.py").is_file(), "standalone supervisor missing"
    result = subprocess.run(
        [sys.executable, str(ROOT / "supervise.py"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--verification-gate-only",
        "--rank0-verifier",
        "--rank1-verifier",
        "--target-revision",
        "--small-model-local",
        "--small-model-remote",
    ):
        assert option in result.stdout
    spec = importlib.util.spec_from_file_location("actor019", ROOT / "remote_actor.py")
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    assert set(actor.source_hashes(ROOT)) == set(actor.SOURCES)
    for name in (
        "mtp_head.py",
        "mlx_backend.py",
        "cache_state.py",
        "speculative.py",
        "rank_control.py",
        "distributed_control.py",
        "verification.py",
        "verifier.py",
        "sidecar_config.json",
    ):
        assert name in actor.SOURCES


def test_supervisor_propagates_trusted_per_rank_modes_and_caps():
    import pytest
    import supervise

    options = [
        "--remote-root",
        "/remote/spikes/019",
        "--remote-source",
        "/remote/weights",
        "--local-metadata",
        "/local/meta",
        "--output",
        "/local/out",
        "--ssh-key",
        "/keys/k",
        "--split",
        "36",
        "28",
        "--target-revision",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "--target-config-sha256",
        "a" * 64,
        "--target-index-sha256",
        "b" * 64,
        "--sidecar",
        "/local/head",
        "--small-model-local",
        "/local/small",
        "--small-model-remote",
        "/remote/small",
        "--small-config-sha256",
        "c" * 64,
        "--small-index-sha256",
        "d" * 64,
        "--verification-gate-only",
    ]
    args = supervise.parser().parse_args(options)
    for rank, mode, path in [(0, "native", "/local/small"), (1, "singleton-qmm", "/remote/small")]:
        actual = supervise.probe_options(args, rank=rank)
        assert actual[actual.index("--verifier-mode") + 1] == mode
        assert actual[actual.index("--small-model") + 1] == path
        assert "--verification-gate-only" in actual
    args.max_tokens = 97
    with pytest.raises(ValueError):
        supervise.validate_args(args)


def test_live_admission_is_required_and_engine_runs_cpu():
    import mlx.core as mx
    import pytest

    mx.set_default_device(mx.cpu)
    from mlx_backend import MTPBackend
    from mtp_head import BoundHead, MTPHead
    from speculative import generate
    from test_phased_cpu import tiny_model
    from verification import Admission, width_gate

    model = tiny_model()
    head = BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )
    with pytest.raises(ValueError, match="admission"):
        MTPBackend(model, head)
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    admitted = Admission(model, "native", gate, gate, {"small": gate, "full": gate})
    backend = MTPBackend(model, head, admission=admitted)
    assert len(generate(backend, [1, 2, 3], max_tokens=5, eos_ids=[])["token_ids"]) == 5
    backend.finish()


def test_loaded_runtime_gate_only_never_loads_head_and_bench_uses_live_gate():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from types import SimpleNamespace as S

    from mtp_head import BoundHead, MTPHead
    from mtp_runtime import run_loaded
    from test_phased_cpu import tiny_model
    from verification import width_gate

    model = tiny_model()
    small = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    args = S(
        verifier_mode="native",
        verification_gate_only=True,
        max_tokens=5,
        repeats=1,
        eval_group_size=1,
        sidecar="/DO/NOT/LOAD",
        target_revision="fixture-not-production",
    )
    tokenizer = S(eos_token_ids=[], decode=lambda ids, **kw: str(ids))
    result = run_loaded(
        model,
        tokenizer,
        [1, 2, 3],
        args,
        group=None,
        small={1: small},
        eval_groups=[1],
        report=lambda e: None,
    )
    assert result["requests"] == [] and result["verification"]["passed"]
    args.verification_gate_only = False
    head = BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )
    result = run_loaded(
        model,
        tokenizer,
        [1, 2, 3],
        args,
        group=None,
        small={1: small},
        eval_groups=[1],
        report=lambda e: None,
        head=head,
    )
    assert len(result["requests"][0]["token_ids"]) == 5
    assert result["requests"][0]["wall_seconds"] > 0
    assert result["requests"][0]["first_token_seconds"] > 0
    assert result["requests"][0]["post_first_tokens_per_second"] > 0


def test_controller_rejects_missing_numerical_evidence():
    import pytest
    from supervise import verify_mtp_evidence

    with pytest.raises(ValueError):
        verify_mtp_evidence([{"rank": 0}, {"rank": 1}], {}, None)


def test_singleton_mode_is_only_used_for_explicit_verification_not_prefill(monkeypatch):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import mlx_backend
    from mtp_head import BoundHead, MTPHead
    from test_phased_cpu import tiny_model
    from verification import Admission, width_gate

    model = tiny_model()
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="singleton-qmm")
    admission = Admission(model, "singleton-qmm", gate, gate, {"small": gate, "full": gate})
    head = BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )
    calls = []
    original = mlx_backend.phased_forward

    def observe(model, inputs, cache, **kw):
        calls.append((inputs.shape[1], kw.get("verifier_mode")))
        return original(model, inputs, cache, **kw)

    monkeypatch.setattr(mlx_backend, "phased_forward", observe)
    backend = mlx_backend.MTPBackend(
        model, head, admission=admission, verifier_mode="singleton-qmm"
    )
    backend.fresh([1, 2, 3])
    backend.verify([4, 5])
    assert calls[0] == (2, "native")
    assert calls[-1] == (2, "singleton-qmm")


def test_two_rank_cpu_framing_gate_engine_and_actions(monkeypatch):
    """Actual CPU tiny target/head; queue substitutes only physical MLX link."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from types import SimpleNamespace as S

    from mtp_head import BoundHead, MTPHead
    from mtp_runtime import run_loaded
    from phased_decode import phased_generate
    from test_phased_cpu import tiny_model
    from verification import width_gate

    queues = {0: Queue(), 1: Queue()}

    def send(value, dst, *, group, stream):
        mx.eval(value)
        queues[dst].put(value)
        return value

    def recv(shape, dtype, src, *, group, stream):
        value = queues[group.rank()].get(timeout=5)
        assert value.shape == shape and value.dtype == dtype
        return value

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv", recv)
    reference = tiny_model()
    small = width_gate(reference, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    models = [tiny_model(), tiny_model()]
    groups = [S(rank=lambda r=r: r, size=lambda: 2) for r in (0, 1)]
    for model, group in zip(models, groups):
        model.model.pipeline(group, split=[2, 2])
    head = BoundHead(
        MTPHead(models[0].language_model.args),
        models[0].model.embed_tokens,
        models[0].language_model.lm_head,
    )
    mx.eval(head.module.parameters())
    args = S(
        verifier_mode="native",
        verification_gate_only=False,
        max_tokens=7,
        repeats=1,
        eval_group_size=2,
        target_revision="CPU-fixture",
    )
    tokenizer = S(eos_token_ids=[], decode=lambda ids, **kw: str(ids))
    events = {0: [], 1: []}

    def run(rank):
        mx.set_default_device(mx.cpu)
        mx.set_default_stream(mx.new_stream(mx.cpu))
        return run_loaded(
            models[rank],
            tokenizer,
            [1, 2, 3],
            args,
            group=groups[rank],
            small={2: small},
            eval_groups=[2],
            report=events[rank].append,
            head=head if rank == 0 else None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, rank) for rank in (0, 1)]
        results = [f.result(timeout=15) for f in futures]
    expected = phased_generate(reference, tokenizer, [1, 2, 3], max_tokens=7, eval_group_size=2)
    assert (
        results[0]["requests"][0]["token_ids"]
        == results[1]["requests"][0]["token_ids"]
        == expected["token_ids"]
    )
    actions = [
        [
            {k: v for k, v in e.items() if k != "rank"}
            for e in events[r]
            if e.get("phase") == "rank_action"
        ]
        for r in (0, 1)
    ]
    assert actions[0] == actions[1] and actions[0][-1]["op"] == "finish"


def test_slot_mode_dispatch_and_small_n_serial_fallback_cpu(monkeypatch):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import slot_qmv
    from verifier import numerical_mode

    calls = []
    original = mx.quantized_matmul

    def cpu_kernel(x, w, s, b, **kw):
        calls.append(w.shape[0])
        return mx.concatenate(
            [original(mx.contiguous(x[:, i : i + 1, :]), w, s, b, **kw) for i in range(2)], axis=1
        )

    monkeypatch.setattr(slot_qmv, "slot_qmv", cpu_kernel)
    x = mx.ones((1, 2, 64))
    for n in (8, 4):
        w, s, b = mx.quantize(mx.ones((n, 64)), group_size=64, bits=4)
        expected = cpu_kernel(x, w, s, b, bits=4, group_size=64)
        calls.clear()
        with numerical_mode("slot-qmv", active=True):
            actual = mx.quantized_matmul(x, w, s, b, group_size=64, bits=4)
        assert mx.array_equal(expected, actual).item()
        assert calls == ([8] if n == 8 else [])
