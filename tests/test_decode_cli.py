"""Runner metadata fixtures are admission tests, never inference results."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("change", ["legacy", "weights", "kernel", "runtime"])
def test_decode_runner_rejects_invalid_manifest_before_loading_model(tmp_path, monkeypatch, change):
    pytest.importorskip("mlx.core")
    import mlx_lm

    from paretoquant.benchmark import environment
    from paretoquant.manifest import ManifestError, create_manifest

    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen2"}')
    (source / "model.safetensors").write_bytes(b"invalid inference weights, admission fixture")
    manifest = create_manifest(source, {}, environment())
    if change == "legacy":
        manifest["schema_version"] = 1
        del manifest["model_sha256"], manifest["kernel_sha256"]
    elif change == "weights":
        (source / "model.safetensors").write_bytes(b"changed")
    elif change == "kernel":
        manifest["kernel_sha256"]["kernels/gate_up_packed.metal"] = "0" * 64
    else:
        manifest["mlx"] = "different"
    (source / "execution_manifest.json").write_text(json.dumps(manifest))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("invalid manifest must be rejected before model loading")

    monkeypatch.setattr(mlx_lm, "load", forbidden_load)
    runner = runpy.run_path(str(ROOT / "scripts/benchmark_decode.py"), run_name="admission_unit")
    output = tmp_path / "result.json"
    with pytest.raises(ManifestError):
        runner["main"](["--model", str(source), "--output", str(output)])
    assert not output.exists()
