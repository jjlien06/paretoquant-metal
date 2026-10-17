"""CPU-only FP16 admission tests; synthetic receipts are not GPU evidence."""

import pytest
from test_schedule import args_for


def test_compute_cli_default_and_forwarding():
    from supervise import probe_options, validate_args

    for option in ("native", "float16"):
        args = args_for("--compute-dtype", option)
        validate_args(args)
        flags = probe_options(args)
        assert flags[flags.index("--compute-dtype") + 1] == option
    args = args_for()
    assert args.compute_dtype == "native"
    args.compute_dtype = "bfloat16"
    with pytest.raises(ValueError, match="compute dtype"):
        validate_args(args)


@pytest.fixture
def cpu_model():
    import mlx.core as mx
    import mlx.nn as nn

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = nn.Module()
        model.language_model = nn.Module()
        model.language_model.model = nn.Module()
        model.language_model.model.embed_tokens = nn.QuantizedEmbedding(8, 64, 32, 4)
        model.proj = nn.QuantizedLinear(64, 32, group_size=32, bits=4)
        model.apply(lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a)
        model.A_log = mx.array([0.2, 0.7], mx.float32)
        model.other_float32 = mx.array([0.3], mx.float32)
        model.already_float16 = mx.array([0.5], mx.float16)
        mx.eval(model.parameters())
        yield model
    finally:
        mx.set_default_device(previous)


def test_runtime_cast_preserves_quantization_and_source_values(cpu_model):
    import compute_dtype
    import mlx.core as mx
    from mlx.utils import tree_flatten

    before = dict(tree_flatten(cpu_model.parameters()))
    settings = (cpu_model.proj.group_size, cpu_model.proj.bits, cpu_model.proj.mode)
    receipt = compute_dtype.apply_compute_dtype(cpu_model, "float16")
    after = dict(tree_flatten(cpu_model.parameters()))
    assert receipt["requested_compute_dtype"] == "float16"
    assert receipt["actual_compute_dtype"] == "mlx.core.float16"
    assert receipt["arithmetic_changed"] is True
    assert receipt["roundtrip_exact"] is True
    assert receipt["hash_scope"] == "converted_runtime_arrays_only"
    changed = []
    for name, original in before.items():
        actual = after[name]
        if original.dtype == mx.bfloat16:
            changed.append(name)
            assert actual.dtype == mx.float16
            assert mx.array_equal(
                original.view(mx.uint16), actual.astype(mx.bfloat16).view(mx.uint16)
            )
        else:
            assert actual is original, name
    assert receipt["converted_tensor_names"] == sorted(changed)
    assert set(receipt["converted_tensor_sha256"]) == set(changed)
    assert receipt["source_parameter_layout"] != receipt["runtime_parameter_layout"]
    assert (cpu_model.proj.group_size, cpu_model.proj.bits, cpu_model.proj.mode) == settings


def test_native_does_not_cast_or_claim_changed_arithmetic(cpu_model):
    import compute_dtype
    from mlx.utils import tree_flatten

    before = dict(tree_flatten(cpu_model.parameters()))
    receipt = compute_dtype.apply_compute_dtype(cpu_model, "native")
    after = dict(tree_flatten(cpu_model.parameters()))
    assert all(after[name] is value for name, value in before.items())
    assert receipt["source_parameter_layout"] == receipt["runtime_parameter_layout"]
    assert receipt["converted_tensor_names"] == []
    assert receipt["arithmetic_changed"] is False
    assert receipt["roundtrip_exact"] is None


@pytest.mark.parametrize("value", [1e8, 1e-8, float("inf"), float("nan"), -1e8])
def test_reject_nonfinite_overflow_or_nonroundtrippable(cpu_model, value):
    import mlx.core as mx
    from compute_dtype import apply_compute_dtype

    cpu_model.bad = mx.array([value], mx.bfloat16)
    with pytest.raises(ValueError, match="finite|roundtrip"):
        apply_compute_dtype(cpu_model, "float16")


def test_rejects_protected_bf16_and_invalid_mode(cpu_model):
    import mlx.core as mx
    from compute_dtype import apply_compute_dtype

    with pytest.raises(ValueError, match="compute dtype"):
        apply_compute_dtype(cpu_model, "fp16")
    cpu_model.A_log = mx.array([1.0], mx.bfloat16)
    with pytest.raises(ValueError, match="protected"):
        apply_compute_dtype(cpu_model, "float16")


def test_probe_admits_only_attested_fp16_loading():
    from types import SimpleNamespace

    from probe import phased_options

    args = SimpleNamespace(
        phased=True,
        eval_group_size=1,
        repeats=1,
        mode="generate",
        compute_dtype="float16",
        stream_local=True,
        weight_server=None,
    )
    phased_options(args)
    args.stream_local = False
    with pytest.raises(ValueError, match="stream"):
        phased_options(args)
    args.stream_local = True
    args.compute_dtype = "invalid"
    with pytest.raises(ValueError, match="compute dtype"):
        phased_options(args)
    args.compute_dtype = "float16"
    args.mode = "collective"
    with pytest.raises(ValueError, match="generate"):
        phased_options(args)


def test_probe_cli_and_registry_include_compute_helper():
    import subprocess
    import sys
    from pathlib import Path

    from remote_actor import SOURCES, source_hashes

    root = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, str(root / "probe.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--compute-dtype {native,float16}" in result.stdout
    assert "compute_dtype.py" in SOURCES
    assert set(source_hashes(root)) == set(SOURCES)


def test_compute_preparation_keeps_source_hashes_and_emits_bound_receipt(cpu_model):
    import copy

    from compute_dtype import parameter_layout, receipt_sha256
    from probe import prepare_compute

    loading = {
        "source_parameter_layout": parameter_layout(cpu_model),
        "tensor_sha256": {"source": "a" * 64},
        "tensor_hash_scope": "serialized_source_payloads_before_runtime_cast",
    }
    before = copy.deepcopy(loading)
    events = []
    result = prepare_compute(cpu_model, "float16", loading, None, events.append)
    assert loading == before
    assert result["source_parameter_layout"] == before["source_parameter_layout"]
    assert events == [
        {
            "phase": "compute_dtype_done",
            "compute_dtype": "float16",
            "actual_compute_dtype": "mlx.core.float16",
            "compute_receipt_sha256": receipt_sha256(result),
        }
    ]
    with pytest.raises(ValueError, match="source"):
        prepare_compute(cpu_model, "float16", loading, None, events.append)


@pytest.mark.parametrize(
    "mode,peer_mode,peer_dtype",
    [
        ("float16", 1, 16),
        ("float16", 0, 16),
        ("float16", 1, 17),
        ("native", 0, 17),
        ("native", 1, 17),
        ("native", 0, 32),
    ],
)
def test_both_rank_compute_handshake_cpu(cpu_model, monkeypatch, mode, peer_mode, peer_dtype):
    import mlx.core as mx
    from compute_dtype import agree_compute, apply_compute_dtype

    receipt = apply_compute_dtype(cpu_model, mode)
    actual = [1, 16] if mode == "float16" else [0, 17]

    class Group:
        def size(self):
            return 2

    def gather(value, *, group, stream):
        assert stream == mx.cpu
        assert value.tolist() == actual
        return mx.array([*actual, peer_mode, peer_dtype], mx.int32)

    monkeypatch.setattr(mx.distributed, "all_gather", gather)
    if actual == [peer_mode, peer_dtype]:
        assert agree_compute(receipt, Group()) == [actual, actual]
    else:
        with pytest.raises(ValueError, match="ranks"):
            agree_compute(receipt, Group())


def test_cpu_stream_loader_attests_original_layout(tmp_path):
    import hashlib
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from compute_dtype import parameter_layout
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from probe import prepare_compute
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records
    from test_qwen38 import Group, tiny_config

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        config = tiny_config()
        model = Model(ModelArgs.from_dict(config))
        nn.quantize(model, group_size=32, bits=4)
        from mlx.utils import tree_map_with_path

        model.update(
            tree_map_with_path(
                # Deliberate exactly representable synthetic storage, not checkpoint parity.
                lambda p, a: (
                    mx.full(a.shape, 0.125, mx.bfloat16)
                    if mx.issubdtype(a.dtype, mx.floating) and not p.endswith("A_log")
                    else a
                ),
                model.parameters(),
            )
        )
        mx.eval(model.parameters())
        from mlx.utils import tree_flatten

        mx.save_safetensors(
            str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        (tmp_path / "config.json").write_text(json.dumps(config))
        tensors, origins = checkpoint_inventory(tmp_path)
        loaded, loading = load_from_tensors(
            config,
            Group(),
            [5],
            tensors,
            tensor_records(tensors, origins),
            budget_bytes=8 * 1024**2,
        )
        assert loading["source_parameter_layout"] == parameter_layout(loaded)
        assert loading["tensor_hash_scope"] == "serialized_source_payloads_before_runtime_cast"
        source = dict(loading["tensor_sha256"])
        compute = prepare_compute(loaded, "float16", loading, None, lambda e: None)
        assert loading["tensor_sha256"] == source
        assert compute["runtime_parameter_layout"] == parameter_layout(loaded)
        for record in tensor_records(tensors, origins):
            assert source[record["name"]] == hashlib.sha256(record["payload"]).hexdigest()
    finally:
        mx.set_default_device(previous)


def test_phased_forward_records_observed_hidden_dtype_on_cpu(cpu_model):
    from types import SimpleNamespace

    import mlx.core as mx
    from compute_dtype import apply_compute_dtype
    from mlx_lm.models.cache import KVCache
    from phased_decode import phased_forward

    apply_compute_dtype(cpu_model, "float16")

    class Layer:
        is_linear = False

        def __call__(self, h, mask, cache):
            cache.update_and_fetch(h[:, None, :, :], h[:, None, :, :])
            return h

    core = SimpleNamespace(
        pipeline_rank=0,
        pipeline_size=1,
        start_idx=0,
        end_idx=1,
        embed_tokens=cpu_model.language_model.model.embed_tokens,
        pipeline_layers=[Layer()],
        norm=lambda h: h,
    )
    model = SimpleNamespace(
        model_type="qwen3_5",
        model=core,
        language_model=SimpleNamespace(
            args=SimpleNamespace(num_hidden_layers=1, tie_word_embeddings=False),
            lm_head=lambda h: h,
        ),
    )
    events = []
    output = phased_forward(model, mx.array([[1]], mx.int32), [KVCache()], progress=events.append)
    assert output.dtype == mx.float16
    witness = next(e for e in events if e["phase"] == "embedding_done")
    assert witness["actual_compute_dtype"] == "mlx.core.float16"
