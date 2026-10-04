"""CPU-only diagnostic profiler contracts; no saved models or Metal dispatch."""

import importlib
import importlib.util

import pytest


@pytest.fixture(autouse=True)
def restore_default_device_after_each_test():
    """CPU diagnostics must not change the device of later native tests."""
    if importlib.util.find_spec("mlx") is None:
        yield
        return
    import mlx.core as mx

    previous = mx.default_device()
    try:
        yield
    finally:
        mx.set_default_device(previous)


def profiler():
    assert importlib.util.find_spec("paretoquant.profiling") is not None, "profiler missing"
    return importlib.import_module("paretoquant.profiling")


class Leaf(dict):
    def __call__(self, x):
        if x == "raise":
            raise RuntimeError("forward failed")
        return x + 1

    def as_linear(self, x):
        return x * 2


class Root:
    def __init__(self):
        self.leaf = Leaf(weight=object())

    def named_modules(self):
        return [("leaf", self.leaf)]


def test_instrumentation_preserves_parameter_tree_restores_on_exception():
    p = profiler()
    root = Root()
    original = type(root.leaf)
    parameters = dict(root.leaf)
    calls = []
    targets = {("leaf", "__call__"): "embedding", ("leaf", "as_linear"): "output_projection"}
    with pytest.raises(RuntimeError, match="forward failed"):
        with p.instrument_modules(root, targets, lambda *args: calls.append(args)):
            assert dict(root.leaf) == parameters
            assert root.leaf(3) == 4
            assert root.leaf.as_linear(3) == 6
            root.leaf("raise")
    assert type(root.leaf) is original
    assert dict(root.leaf) == parameters
    assert [call[1] for call in calls] == ["__call__", "as_linear", "__call__"]
    assert root.leaf(8) == 9


def test_target_discovery_distinguishes_tied_output_and_operator_groups():
    p = profiler()
    root = Root()
    names = [
        "model.embed_tokens",
        "model.norm",
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ]
    root.named_modules = lambda: [(name, root.leaf) for name in names]
    targets = p.operation_targets(root, tied_embeddings=True)
    assert targets[("model.embed_tokens", "as_linear")] == "output_projection"
    assert targets[("model.layers.0.self_attn", "__call__")] == "attention_inclusive"
    assert targets[("model.layers.0.self_attn.q_proj", "__call__")] == "attention_projection"
    assert targets[("model.layers.0.mlp", "__call__")] == "mlp_gate_up"
    assert ("model.layers.0.mlp.gate_proj", "__call__") not in targets


def test_capture_uses_actual_decode_inputs_and_isolated_attention_resets_cache():
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 2, bias=False)

        def __call__(self, x, mask=None, cache=None):
            keys = x.reshape(1, 1, 1, 2)
            cache.update_and_fetch(keys, keys)
            return self.q_proj(x) + cache.offset

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

        def __call__(self, tokens, cache=None):
            x = mx.ones((1, 1, 2)) * tokens[0, 0]
            return self.self_attn(x, None, cache[0])

    toy = Toy()
    original_type = type(toy.self_attn)
    weights = toy.parameters()
    targets = {
        ("self_attn", "__call__"): "attention_inclusive",
        ("self_attn.q_proj", "__call__"): "attention_projection",
    }
    operations = p.capture_decode_operations(
        toy,
        [2],
        [3, 5],
        capture_step=1,
        targets=targets,
        cache_factory=lambda model: [KVCache()],
    )
    assert type(toy.self_attn) is original_type
    assert toy.parameters() == weights
    assert operations["self_attn.__call__"].metadata["cache_offset_before"] == 2
    assert operations["self_attn.q_proj.__call__"].metadata["input_shape"] == [1, 1, 2]
    a = operations["self_attn.__call__"].function()
    b = operations["self_attn.__call__"].function()
    mx.eval(a, b)
    assert bool(mx.all(a == b).item())
    assert operations["self_attn.__call__"].metadata["cache_offset_before"] == 2


def test_mlp_gate_up_stock_closure_uses_captured_input_without_down():
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn

    class ToyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2, 3, bias=False)
            self.up_proj = nn.Linear(2, 3, bias=False)
            self.down_proj = nn.Linear(3, 2, bias=False)

    module = ToyMLP()
    x = mx.ones((1, 1, 2))
    ops = p.gate_up_operations("mlp", module, x)
    assert list(ops) == ["mlp.gate_up_stock"]
    actual = ops["mlp.gate_up_stock"].function()
    expected = nn.silu(module.gate_proj(x)) * module.up_proj(x)
    mx.eval(actual, expected)
    assert actual.shape == (1, 1, 3)
    assert bool(mx.allclose(actual, expected).item())
    assert ops["mlp.gate_up_stock"].group == "mlp_gate_up"


@pytest.mark.parametrize(
    "field,value",
    [
        ("decode_steps", True),
        ("decode_steps", 0),
        ("decode_steps", 513),
        ("repeats", 0),
        ("repeats", 201),
        ("warmup", -1),
        ("warmup", 51),
        ("capture_step", -1),
        ("capture_step", 2),
        ("full_repeats", 0),
        ("full_warmup", 11),
        ("max_prompt_tokens", 0),
    ],
)
def test_bounded_options_reject_invalid_values(field, value):
    p = profiler()
    options = {
        "decode_steps": 2,
        "capture_step": 0,
        "repeats": 2,
        "warmup": 0,
        "full_repeats": 1,
        "full_warmup": 0,
        "max_prompt_tokens": 128,
    }
    options[field] = value
    with pytest.raises(ValueError, match=field):
        p.validate_options(**options)


def test_exclusive_evidence_serializes_before_creation_never_overwrites(tmp_path):
    p = profiler()
    output = tmp_path / "evidence.json"
    with pytest.raises(ValueError):
        p.write_evidence(output, {"invalid": float("nan")})
    assert not output.exists()
    p.write_evidence(output, {"samples_ms": [1.0]})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        p.write_evidence(output, {"samples_ms": [2.0]})
    assert output.read_bytes() == original


def test_isolated_report_keeps_samples_groups_without_additive_percentages():
    p = profiler()
    operations = {
        "attn": p.Operation(lambda: 1, "attention_inclusive", {"input_shape": [1, 1, 2]}),
        "q": p.Operation(lambda: 2, "attention_projection", {}),
    }
    calls = []

    def benchmark(functions, **kwargs):
        calls.append((list(functions), kwargs))
        return {name: {"samples_ms": [1.0, 2.0], "median_ms": 1.5} for name in functions}

    report = p.time_operations(operations, repeats=2, warmup=0, benchmark=benchmark)
    assert calls == [(["attn", "q"], {"repeats": 2, "warmup": 0})]
    assert report["operations"]["attn"]["timing"]["samples_ms"] == [1.0, 2.0]
    assert report["group_members"]["attention_projection"] == ["q"]
    assert report["additive_cost_attribution"] is False
    assert "percent" not in str(report)
    assert report["measurement"] == "intrusive_capture_isolated_synchronized_wall_clock"


def test_admission_verifies_before_load_then_loaded_precision_before_fusion(tmp_path, monkeypatch):
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import json

    import mlx_lm
    import mlx_lm.models.qwen2 as qwen2

    from paretoquant import manifest, runtime

    source = tmp_path / "saved"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (source / "model.safetensors").write_bytes(b"test fixture only")
    (source / "execution_manifest.json").write_text("{}")
    events = []
    dispatch = {"model.layers.0.mlp": {"backend": "fused", "bits": 4, "rows_per_group": 4}}
    monkeypatch.setattr(manifest, "load_manifest", lambda path: {"fixture": True})
    monkeypatch.setattr(
        manifest, "validate_manifest", lambda *a, **kw: events.append(("strict", kw)) or dispatch
    )
    model = Root()
    monkeypatch.setattr(qwen2, "Model", Root)

    def load(path, **kwargs):
        events.append(("load", path, kwargs))
        return model, object()

    monkeypatch.setattr(mlx_lm, "load", load)
    monkeypatch.setattr(
        manifest, "validate_model_dispatch", lambda *a: events.append(("loaded_precision",))
    )
    monkeypatch.setattr(runtime, "install_fusion", lambda *a: events.append(("fusion",)) or ["mlp"])
    admitted, tokenizer, details = p.load_admitted_model(source, {"fixture": True}, stock=False)
    assert admitted is model
    assert [event[0] for event in events] == ["strict", "load", "loaded_precision", "fusion"]
    assert events[0][1] == {"strict_runtime": True}
    assert events[1][2]["tokenizer_config"]["local_files_only"] is True
    assert events[1][2]["trust_remote_code"] is False
    assert details["installed_fusion"] == ["mlp"]


def test_local_path_preflight_rejects_remote_missing_existing_and_overlap(tmp_path):
    p = profiler()
    with pytest.raises(ValueError, match="local"):
        p.check_profile_paths("Qwen/Qwen2-0.5B-Instruct", tmp_path / "out.json")
    source = tmp_path / "saved"
    source.mkdir()
    for name in ("config.json", "model.safetensors", "execution_manifest.json"):
        (source / name).write_text("fixture")
    with pytest.raises(ValueError, match="inside"):
        p.check_profile_paths(source, source / "out.json")
    output = tmp_path / "out.json"
    output.write_text("existing evidence")
    with pytest.raises(ValueError, match="new"):
        p.check_profile_paths(source, output)


def test_path_preflight_normalizes_parent_components_before_overlap_check(tmp_path):
    p = profiler()
    source = tmp_path / "saved"
    source.mkdir()
    for name in ("config.json", "model.safetensors", "execution_manifest.json"):
        (source / name).write_text("path-admission fixture, not model weights")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    disguised_output = alternate / ".." / "saved" / "new-evidence.json"
    with pytest.raises(ValueError, match="inside"):
        p.check_profile_paths(source, disguised_output)
    assert not (source / "new-evidence.json").exists()


def test_profile_orchestration_retains_chat_ids_schedule_hashes_writes_last(tmp_path, monkeypatch):
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import json

    from paretoquant import evaluation

    source = tmp_path / "saved"
    source.mkdir()
    for name in ("config.json", "model.safetensors", "execution_manifest.json", "tokenizer.json"):
        (source / name).write_text("fixture bytes; mock loader only")
    output = tmp_path / "out.json"
    events = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "fixture prompt"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "<chat>fixture prompt</chat>"

        def encode(self, text):
            assert text == "<chat>fixture prompt</chat>"
            return [7, 8]

    def step(name):
        assert not output.exists()
        events.append(name)

    monkeypatch.setattr(p, "environment_snapshot", lambda: step("environment") or {"fixture": True})
    monkeypatch.setattr(p, "execution_hashes", lambda: {"fixture.py": "a" * 64})
    monkeypatch.setattr(
        p,
        "load_admitted_model",
        lambda *a, **kw: (object(), Tokenizer(), {"installed_fusion": [], "stock_override": False}),
    )
    monkeypatch.setattr(
        evaluation, "reference_schedule", lambda *a, **kw: step("schedule") or [3, 4]
    )
    monkeypatch.setattr(
        evaluation,
        "cached_decode_benchmark",
        lambda *a, **kw: step("full_uninstrumented") or {"admitted": {"decode_samples_ms": [9.0]}},
    )
    ops = {"q": p.Operation(lambda: None, "attention_projection", {})}
    monkeypatch.setattr(p, "capture_decode_operations", lambda *a, **kw: step("capture") or ops)
    monkeypatch.setattr(
        p, "time_operations", lambda *a, **kw: step("isolated") or {"fixture_samples": [1.0]}
    )
    report = p.profile_saved_decode(
        source,
        output,
        prompt="fixture prompt",
        decode_steps=2,
        repeats=1,
        warmup=0,
        full_repeats=1,
        full_warmup=0,
    )
    assert events == [
        "environment",
        "schedule",
        "full_uninstrumented",
        "capture",
        "isolated",
        "environment",
    ]
    assert report["prompt"]["token_ids"] == [7, 8]
    assert report["schedule"]["token_ids"] == [3, 4]
    assert report["schedule"]["capture_token_id"] == 3
    assert report["full_model_uninstrumented"]["decode_samples_ms"] == [9.0]
    assert set(report["hashes"]["model_files"]) == {
        "config.json",
        "model.safetensors",
        "execution_manifest.json",
        "tokenizer.json",
    }
    assert report["hashes"]["execution_files"] == {"fixture.py": "a" * 64}
    assert json.loads(output.read_text()) == report


def test_environment_does_not_invoke_git(monkeypatch):
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import subprocess

    calls = []

    def run(args, **kwargs):
        assert args[0] != "git"
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 0, "fixture" if kwargs.get("text") else b"fixture", ""
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "fixture CPU"})
    result = p.environment_snapshot()
    assert result["device"]["device_name"] == "fixture CPU"
    assert result["swap_usage"] == "fixture"
    assert "git_revision" not in result


def test_cli_is_runnable_and_rejects_invalid_counts_before_model_load(monkeypatch):
    p = profiler()
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "profile_decode.py"
    assert script.is_file(), "runnable profiler script missing"
    spec = importlib.util.spec_from_file_location("profile_decode_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        p,
        "profile_saved_decode",
        lambda *a, **kw: pytest.fail("invalid options reached model execution"),
    )
    with pytest.raises(SystemExit) as exc:
        module.main(["--model", "missing", "--output", "unused", "--repeats", "0"])
    assert exc.value.code == 1


def test_quantized_tied_embedding_both_methods_restore_without_parameter_changes():
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(8, 32, group_size=32, bits=4)

    toy = Toy()
    original_type = type(toy.embed_tokens)
    parameters = toy.parameters()
    calls = []
    targets = {
        ("embed_tokens", "__call__"): "embedding",
        ("embed_tokens", "as_linear"): "output_projection",
    }
    with p.instrument_modules(toy, targets, lambda *a: calls.append(a)):
        logits = toy.embed_tokens.as_linear(toy.embed_tokens(mx.array([[2]])))
        mx.eval(logits)
        assert logits.shape == (1, 1, 8)
        assert toy.parameters() == parameters
    assert type(toy.embed_tokens) is original_type
    assert toy.parameters() == parameters
    assert [call[1] for call in calls] == ["__call__", "as_linear"]


def test_callback_exception_restores_instrumented_instance():
    p = profiler()
    root = Root()

    def before(*args):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        with p.instrument_modules(root, {("leaf", "__call__"): "embedding"}, before):
            root.leaf(2)
    assert type(root.leaf) is Leaf
    assert root.leaf(2) == 3


def test_real_execution_hashes_include_runner_module_kernels_and_installed_qwen2():
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    hashes = p.execution_hashes()
    assert any(path.endswith("/scripts/profile_decode.py") for path in hashes)
    assert any(path.endswith("/paretoquant/profiling.py") for path in hashes)
    assert any(path.endswith("/models/qwen2.py") for path in hashes)
    assert any(path.endswith("/models/cache.py") for path in hashes)
    assert any(path.endswith("/kernels/gate_up_packed.metal") for path in hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_custom_gate_up_replay_uses_captured_dynamic_weights_and_runtime_eligibility(monkeypatch):
    """Custom operation is mocked: no Metal kernel is built or dispatched."""
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    from mlx_lm.models.activations import swiglu

    from paretoquant import runtime

    class Projection:
        bits = 4
        group_size = 32

        def __init__(self, multiplier):
            self.multiplier = multiplier
            self.weight = mx.zeros((32, 4), dtype=mx.uint32)
            self.scales = mx.ones((32, 1), dtype=mx.float32)
            self.biases = mx.zeros((32, 1), dtype=mx.float32)

        def __call__(self, x):
            return x * self.multiplier

    class MockFusion:
        rows_per_group = 8
        gate_proj = Projection(2)
        up_proj = Projection(3)

    calls = []
    module = MockFusion()
    x = mx.ones((1, 1, 32), dtype=mx.float32)

    def custom(actual_x, gate, up, **kwargs):
        calls.append((actual_x, gate, up, kwargs))
        return swiglu(module.gate_proj(actual_x), module.up_proj(actual_x))

    monkeypatch.setattr(runtime, "FusedMLP", MockFusion)
    monkeypatch.setattr(runtime, "compiled_fused_gate_up", custom)
    ops = p.gate_up_operations("mlp", module, x)
    assert list(ops) == ["mlp.gate_up_stock", "mlp.gate_up_fused"]
    fused = ops["mlp.gate_up_fused"].function()
    stock = ops["mlp.gate_up_stock"].function()
    mx.eval(fused, stock)
    assert bool(mx.allclose(fused, stock).item())
    assert calls[0][0] is x
    assert calls[0][1][0] is module.gate_proj.weight
    assert calls[0][2][0] is module.up_proj.weight
    assert calls[0][3] == {"bits": 4, "group_size": 32, "rows_per_group": 8}
    assert ops["mlp.gate_up_fused"].metadata["active_decode_backend"] == "fused"
    fallback = p.gate_up_operations("mlp", module, mx.ones((1, 2, 32)))
    assert list(fallback) == ["mlp.gate_up_stock"]
    assert fallback["mlp.gate_up_stock"].metadata["active_decode_backend"] == "stock"


@pytest.mark.metal
@pytest.mark.integration
def test_opt_in_local_native_profile(tmp_path):
    """Parent-only sequential opt-in; never enabled by the default CPU run."""
    import os

    source = os.environ.get("PARETOQUANT_PROFILE_NATIVE_MODEL")
    if not source:
        pytest.skip("set PARETOQUANT_PROFILE_NATIVE_MODEL explicitly for sequential GPU validation")
    p = profiler()
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.gpu)
    report = p.profile_saved_decode(
        source,
        tmp_path / "native-profile.json",
        decode_steps=1,
        capture_step=0,
        repeats=1,
        warmup=0,
        full_repeats=1,
        full_warmup=0,
    )
    groups = report["isolated_diagnostics"]["group_members"]
    assert {
        "attention_inclusive",
        "attention_projection",
        "mlp_gate_up",
        "mlp_down",
        "embedding",
        "output_projection",
        "norm",
    } <= set(groups)
    assert report["full_model_uninstrumented"]["decode_steps"] == 1
    assert report["additive_cost_attribution"] is False
    assert report["hashes"]["model_files"]["execution_manifest.json"]
