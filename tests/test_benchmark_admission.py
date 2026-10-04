"""Benchmark admission checks; metadata fixtures are not measurements."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["003-ollama-head-to-head", "006-framework-head-to-head"])
def runner(request):
    pytest.importorskip("mlx.core")
    return runpy.run_path(
        str(ROOT / "spikes" / request.param / "main.py"), run_name="admission_unit"
    )


@pytest.fixture
def bound_source(tmp_path):
    from paretoquant.benchmark import environment
    from paretoquant.manifest import create_manifest

    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                },
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"admission fixture, never inference weights")
    dispatch = {"model.layers.0.mlp": {"backend": "fused", "bits": 4, "rows_per_group": 4}}
    data = create_manifest(source, dispatch, environment())
    (source / "execution_manifest.json").write_text(json.dumps(data))
    return source, data


@pytest.mark.parametrize("change", ["legacy", "weights", "kernel", "runtime"])
def test_benchmark_refuses_unbound_dispatch(runner, bound_source, change):
    from paretoquant.manifest import ManifestError

    source, data = bound_source
    if change == "legacy":
        data["schema_version"] = 1
        del data["model_sha256"], data["kernel_sha256"]
    elif change == "weights":
        (source / "model.safetensors").write_bytes(b"changed")
    elif change == "kernel":
        data["kernel_sha256"]["kernels/gate_up_packed.metal"] = "0" * 64
    else:
        data["mlx"] = "changed runtime"
    (source / "execution_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ManifestError):
        runner["load_verified_dispatch"](source)


def test_bound_admission_preserves_dispatch(runner, bound_source):
    source, data = bound_source
    assert runner["load_verified_dispatch"](source) == data["dispatch"]


@pytest.mark.parametrize("change", ["legacy", "weights", "kernel", "runtime"])
def test_invalid_dispatch_refused_by_entrypoint_before_network_or_output(
    runner, bound_source, tmp_path, monkeypatch, change
):
    import urllib.request

    from paretoquant.manifest import ManifestError

    source, data = bound_source
    if change == "legacy":
        data["schema_version"] = 1
        del data["model_sha256"], data["kernel_sha256"]
    elif change == "weights":
        (source / "model.safetensors").write_bytes(b"changed")
    elif change == "kernel":
        data["kernel_sha256"]["kernels/gate_up_packed.metal"] = "0" * 64
    else:
        data["mlx"] = "changed runtime"
    (source / "execution_manifest.json").write_text(json.dumps(data))

    def forbidden_network(*args, **kwargs):
        raise AssertionError("invalid dispatch must be rejected before any network request")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    output = tmp_path / "measurements.json"
    argv = ["--model", str(source), "--output", str(output)]
    if "validate_server_model" in runner:
        argv += ["--engine", "vllm_metal"]
    with pytest.raises(ManifestError):
        runner["main"](argv)
    assert not output.exists()
