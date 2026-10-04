"""CLI smoke tests do not download models or claim benchmark performance."""

import json
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest


def _tokenizer_saving_case(tmp_path, monkeypatch, *, named=False):
    """Real local tokenizer I/O; stub only model work and measured costs."""
    mx = pytest.importorskip("mlx.core")
    lm = pytest.importorskip("mlx_lm")
    from mlx_lm import utils
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from paretoquant import benchmark, evaluation, pipeline
    from paretoquant.allocator import Option

    template = (
        "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}|assistant:{% endif %}"
    )
    templates = {"default": template, "alternate": "ALT:" + template} if named else template
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", chat_template=templates
    )
    source = tmp_path / "source"
    tokenizer.save_pretrained(source)
    assert (source / "chat_template.jinja").read_text() == template
    assert "chat_template" not in json.loads((source / "tokenizer_config.json").read_text())
    if named:
        assert (
            source / "additional_chat_templates/alternate.jinja"
        ).read_text() == "ALT:" + template
    (source / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (source / "model.safetensors").write_bytes(b"stub model weights")
    (source / "generation_config.json").write_text('{"eos_token_id": 0}')
    (source / "LICENSE").write_text("Test model license\n")
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    load_tokenizer = utils.load_tokenizer
    loaded = load_tokenizer(source, tokenizer_config_extra={"local_files_only": True})
    model = SimpleNamespace(named_modules=lambda: [], set_dtype=lambda _: None, parameters=list)
    environment = {"device": {"device_name": "test device"}, "mlx": "test", "mlx_lm": "test"}
    measured = {
        "environment": environment,
        "units": [
            {
                "name": "model.layers.0.mlp",
                "options": [asdict(Option("q4:stock", 4, 20, 1.0, 0.0))],
                "measurements": {"4": {"validation": {"stock": {"median_ms": 1.0}}}},
            }
        ],
    }

    def save_model(destination, _model):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.safetensors").write_bytes(b"stub saved model weights")

    monkeypatch.setattr(mx, "eval", lambda *a: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 0)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    monkeypatch.setattr(lm, "load", lambda *a: (model, loaded))
    monkeypatch.setattr(lm, "generate", lambda *a, **k: "stub model response")
    monkeypatch.setattr(utils, "save_model", save_model)
    monkeypatch.setattr(benchmark, "environment", lambda: environment)
    monkeypatch.setattr(evaluation, "text_nll", lambda *a: {"mean_nll": 1.0, "token_count": 2})
    monkeypatch.setattr(evaluation, "reference_schedule", lambda *a, **k: [1])
    monkeypatch.setattr(
        evaluation,
        "cached_decode_benchmark",
        lambda models, *a, **k: {name: {"median_decode_tokens_per_second": 1.0} for name in models},
    )
    monkeypatch.setattr(pipeline, "calibrate_inputs", lambda *a: {})
    monkeypatch.setattr(pipeline, "profile_units", lambda *a, **k: measured)
    monkeypatch.setattr(pipeline, "model_bytes", lambda *a: 100)
    monkeypatch.setattr(pipeline, "apply_plan", lambda model, config, choices: (model, config, {}))
    return SimpleNamespace(
        source=source,
        tokenizer=loaded,
        load_tokenizer=load_tokenizer,
        before=before,
        measured=measured,
        environment=environment,
    )


@pytest.mark.parametrize("named", [False, True], ids=["standalone", "named"])
def test_run_saved_tokenizer_preserves_chat_prompt(tmp_path, monkeypatch, named):
    from paretoquant import cli as cli_module
    from paretoquant import manifest

    case = _tokenizer_saving_case(tmp_path, monkeypatch, named=named)
    expected = cli_module._chat_prompt(case.tokenizer, "hello world")
    assert expected == "user:hello world|assistant:"
    output = tmp_path / "run"
    assert (
        cli_module.main(
            [
                "run",
                "--model",
                str(case.source),
                "--output",
                str(output),
                "--decode-steps",
                "1",
                "--profile-repeats",
                "1",
                "--decode-repeats",
                "1",
            ]
        )
        == 0
    )
    saved = output / "model"
    restored = case.load_tokenizer(saved, tokenizer_config_extra={"local_files_only": True})
    assert cli_module._chat_prompt(restored, "hello world") == expected
    assert restored.chat_template == case.tokenizer.chat_template
    if named:
        messages = [{"role": "user", "content": "hello world"}]
        assert (
            restored.apply_chat_template(
                messages, chat_template="alternate", tokenize=False, add_generation_prompt=True
            )
            == "ALT:" + expected
        )
    for filename in ("generation_config.json", "LICENSE"):
        assert (saved / filename).read_bytes() == case.before[filename]
    assert {
        str(p.relative_to(case.source)): p.read_bytes()
        for p in case.source.rglob("*")
        if p.is_file()
    } == case.before
    assert (
        manifest.verify_manifest(
            saved, manifest.load_manifest(saved / "execution_manifest.json"), case.environment
        )
        == {}
    )


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "paretoquant.cli", *args], capture_output=True, text=True, timeout=30
    )


def test_help_lists_runnable_commands():
    result = cli("--help")
    assert result.returncode == 0
    assert "run" in result.stdout and "generate" in result.stdout and "replay" in result.stdout


def test_doctor_returns_real_device_metadata():
    pytest.importorskip("mlx.core")
    result = cli("doctor")
    assert result.returncode == 0
    metadata = json.loads(result.stdout)
    assert "device" in metadata
    assert "mlx" in metadata


def test_run_refuses_missing_local_source():
    result = cli("run", "--model", "/path/that/does/not/exist")
    assert result.returncode == 2
    assert "local" in result.stderr.lower()


def test_generate_refuses_missing_local_model():
    result = cli("generate", "--model", "/path/that/does/not/exist", "--prompt", "test")
    assert result.returncode == 2
    assert "local" in result.stderr.lower()


def test_counter_collection_accepts_installed_fused_wrappers():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.models.qwen2 import MLP

    from paretoquant import cli as cli_module
    from paretoquant.runtime import install_fusion

    model = nn.Module()
    model.mlp = MLP(64, 128)
    nn.quantize(model, group_size=64, bits=4)
    install_fusion(model, {"mlp": {"backend": "fused"}})
    mx.eval(model.mlp(mx.ones((1, 1, 64))))
    mx.eval(model.mlp(mx.ones((1, 2, 64))))

    assert cli_module._runtime_counters(model) == {"mlp": {"fused_calls": 1, "stock_calls": 1}}


@pytest.fixture
def saved_artifact(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "saved"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"toy weights, not a real model")
    profile = {"device": {"device_name": "test device"}, "mlx": "0.32.3", "mlx_lm": "0.32.0"}
    dispatch = {"model.layers.0.mlp": {"backend": "fused", "bits": 4, "rows_per_group": 2}}
    return SimpleNamespace(source=source, profile=profile, dispatch=dispatch)


@pytest.fixture
def fake_execution(monkeypatch):
    """Avoid model loading/Metal work; integrity validation uses real files."""
    from types import ModuleType, SimpleNamespace

    events = []
    projection = SimpleNamespace(bits=4)
    pair = SimpleNamespace(gate_proj=projection, up_proj=projection)
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.mlp", pair)])
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "chat prompt")
    lm = ModuleType("mlx_lm")
    lm.load = lambda *a, **k: (events.append("load") or model, tokenizer)
    lm.generate = lambda *a, **k: events.append("generate") or "actual stub response"
    benchmark = ModuleType("paretoquant.benchmark")
    benchmark.environment = lambda: {
        "device": {"device_name": "test device"},
        "mlx": "0.32.3",
        "mlx_lm": "0.32.0",
    }
    runtime = ModuleType("paretoquant.runtime")
    runtime.install_fusion = lambda *a: events.append("fusion") or ["model.layers.0.mlp"]
    monkeypatch.setitem(sys.modules, "mlx_lm", lm)
    monkeypatch.setitem(sys.modules, "paretoquant.benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "paretoquant.runtime", runtime)
    return SimpleNamespace(events=events, model=model)


def write_bound_manifest(artifact):
    from paretoquant import manifest

    data = manifest.create_manifest(artifact.source, artifact.dispatch, artifact.profile)
    path = artifact.source / "execution_manifest.json"
    path.write_text(json.dumps(data))
    return data


def test_saving_execution_manifest_binds_final_saved_bytes(saved_artifact):
    from paretoquant import cli as cli_module
    from paretoquant import manifest

    cli_module._save_execution_manifest(
        saved_artifact.source, saved_artifact.dispatch, saved_artifact.profile
    )
    data = manifest.load_manifest(saved_artifact.source / "execution_manifest.json")
    assert data["schema_version"] == 2
    assert manifest.verify_manifest(saved_artifact.source, data, saved_artifact.profile)


def test_generation_fuses_only_verified_v2(saved_artifact, fake_execution, capsys):
    from paretoquant import cli as cli_module

    write_bound_manifest(saved_artifact)
    assert (
        cli_module.main(["generate", "--model", str(saved_artifact.source), "--prompt", "test"])
        == 0
    )
    assert fake_execution.events == ["load", "fusion", "generate"]
    assert "actual stub response" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change,reason",
    [
        ("legacy", "seal"),
        ("weights", "SHA-256"),
        ("config", "SHA-256"),
        ("kernel", "Metal kernel"),
        ("metadata", "hardware/runtime"),
        ("bad_json", "JSON"),
        ("unknown_dispatch_field", "unknown"),
        ("loaded_bits", "loaded model bits"),
        ("missing", "manifest"),
    ],
)
def test_generation_falls_back_to_stock_with_reason(
    saved_artifact, fake_execution, capsys, monkeypatch, change, reason
):
    from paretoquant import cli as cli_module
    from paretoquant import manifest

    data = write_bound_manifest(saved_artifact)
    path = saved_artifact.source / "execution_manifest.json"
    if change == "legacy":
        data["schema_version"] = 1
        del data["model_sha256"], data["kernel_sha256"]
    elif change in ("weights", "config"):
        target = saved_artifact.source / (
            "model.safetensors" if change == "weights" else "config.json"
        )
        target.write_bytes(target.read_bytes() + b"\n")
    elif change == "kernel":
        monkeypatch.setattr(manifest, "kernel_hashes", lambda: {"kernels/gate_up.metal": "0" * 64})
    elif change == "metadata":
        data["mlx"] = "other"
    elif change == "unknown_dispatch_field":
        data["dispatch"]["model.layers.0.mlp"]["unknown"] = 1
    elif change == "loaded_bits":
        fake_execution.model.named_modules()[0][1].gate_proj.bits = 3
    if change == "bad_json":
        path.write_text("{broken")
    elif change == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps(data))
    assert (
        cli_module.main(["generate", "--model", str(saved_artifact.source), "--prompt", "test"])
        == 0
    )
    assert fake_execution.events == ["load", "generate"]
    stderr = capsys.readouterr().err
    assert "stock" in stderr.lower() and reason in stderr


def test_explicit_stock_skips_even_invalid_manifest(saved_artifact, fake_execution, capsys):
    from paretoquant import cli as cli_module

    (saved_artifact.source / "execution_manifest.json").write_text("{broken")
    assert (
        cli_module.main(
            ["generate", "--model", str(saved_artifact.source), "--prompt", "test", "--stock"]
        )
        == 0
    )
    assert fake_execution.events == ["load", "generate"]
    assert not capsys.readouterr().err


@pytest.mark.parametrize(
    "change,reason",
    [
        ("legacy", "seal"),
        ("weights", "SHA-256"),
        ("kernel", "Metal kernel"),
        ("metadata", "hardware/runtime"),
        ("unknown_dispatch_field", "unknown"),
    ],
)
def test_replay_rejects_unverified_dispatch_before_load_or_output_creation(
    saved_artifact, fake_execution, capsys, monkeypatch, tmp_path, change, reason
):
    from paretoquant import cli as cli_module
    from paretoquant import manifest

    data = write_bound_manifest(saved_artifact)
    if change == "legacy":
        data["schema_version"] = 1
        del data["model_sha256"], data["kernel_sha256"]
    elif change == "weights":
        (saved_artifact.source / "model.safetensors").write_bytes(b"changed")
    elif change == "kernel":
        monkeypatch.setattr(manifest, "kernel_hashes", lambda: {"kernels/gate_up.metal": "0" * 64})
    elif change == "metadata":
        data["mlx"] = "other"
    else:
        data["dispatch"]["model.layers.0.mlp"]["unknown"] = 1
    (saved_artifact.source / "execution_manifest.json").write_text(json.dumps(data))
    output = tmp_path / "replay"
    assert (
        cli_module.main(["replay", "--model", str(saved_artifact.source), "--output", str(output)])
        == 2
    )
    assert reason in capsys.readouterr().err
    assert fake_execution.events == []
    assert not output.exists()


def test_seal_copies_artifacts_preserves_old_manifest_and_profile_without_profiling(
    saved_artifact, tmp_path, capsys
):
    import hashlib

    from paretoquant import cli as cli_module
    from paretoquant import manifest

    data = write_bound_manifest(saved_artifact)
    data["schema_version"] = 1
    del data["model_sha256"], data["kernel_sha256"]
    data["mlx"] = "old-profile-version"
    path = saved_artifact.source / "execution_manifest.json"
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    (saved_artifact.source / "tokenizer.json").write_text('{"toy":true}')
    before = {p.name: p.read_bytes() for p in saved_artifact.source.iterdir()}
    output = tmp_path / "sealed-model"
    assert (
        cli_module.main(["seal", "--model", str(saved_artifact.source), "--output", str(output)])
        == 0
    )
    assert {p.name: p.read_bytes() for p in saved_artifact.source.iterdir()} == before
    sealed = manifest.load_manifest(output / "execution_manifest.json")
    assert sealed["schema_version"] == 2
    for field in ("dispatch", "profile_device", "mlx", "mlx_lm"):
        assert sealed[field] == data[field]
    assert sealed["seal"] == {
        "source_manifest_sha256": hashlib.sha256(original).hexdigest(),
        "profiling_performed": False,
    }
    archived = output / f"execution_manifest.original-{hashlib.sha256(original).hexdigest()}.json"
    assert archived.read_bytes() == original
    assert (output / "model.safetensors").read_bytes() == before["model.safetensors"]
    assert (output / "tokenizer.json").read_bytes() == before["tokenizer.json"]
    assert manifest.validate_manifest(output, sealed, saved_artifact.profile, strict_runtime=False)
    with pytest.raises(manifest.ManifestError, match="hardware/runtime"):
        manifest.validate_manifest(output, sealed, saved_artifact.profile)
    report = capsys.readouterr().out
    assert "not" in report and "profil" in report.lower()


@pytest.mark.parametrize("destination", ["same", "inside", "ancestor", "existing"])
def test_seal_refuses_destinations_that_could_overwrite_evidence(
    saved_artifact, tmp_path, capsys, destination
):
    from paretoquant import cli as cli_module

    write_bound_manifest(saved_artifact)
    output = {
        "same": saved_artifact.source,
        "inside": saved_artifact.source / "sealed",
        "ancestor": tmp_path,
        "existing": tmp_path / "existing",
    }[destination]
    if destination == "existing":
        output.mkdir()
        (output / "evidence").write_text("keep")
    assert (
        cli_module.main(["seal", "--model", str(saved_artifact.source), "--output", str(output)])
        == 2
    )
    assert "output" in capsys.readouterr().err.lower()
    assert (saved_artifact.source / "execution_manifest.json").exists()
    if destination == "existing":
        assert (output / "evidence").read_text() == "keep"
    if destination == "inside":
        assert not output.exists()


def test_seal_requires_explicit_output():
    result = cli("seal", "--model", "/irrelevant")
    assert result.returncode == 2
    assert "--output" in result.stderr


def test_seal_refuses_to_reauthorize_changed_v2_weights(saved_artifact, tmp_path, capsys):
    from paretoquant import cli as cli_module

    write_bound_manifest(saved_artifact)
    (saved_artifact.source / "model.safetensors").write_bytes(b"changed")
    output = tmp_path / "sealed"
    assert (
        cli_module.main(["seal", "--model", str(saved_artifact.source), "--output", str(output)])
        == 2
    )
    assert "SHA-256" in capsys.readouterr().err
    assert not output.exists()
