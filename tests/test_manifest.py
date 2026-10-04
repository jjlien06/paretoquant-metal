"""CPU-only integrity tests; toy bytes are never represented as model weights."""

import copy
import hashlib
import json

import pytest

from paretoquant import manifest


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    config = {
        "model_type": "qwen2",
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model-00001-of-00002.safetensors").write_bytes(b"toy shard one")
    (source / "model-00002-of-00002.safetensors").write_bytes(b"toy shard two")
    return source


@pytest.fixture
def profile():
    return {
        "device": {"device_name": "test device", "architecture": "test", "memory_size": 1024},
        "mlx": "0.32.3",
        "mlx_lm": "0.32.0",
    }


@pytest.fixture
def dispatch():
    return {"model.layers.0.mlp": {"backend": "fused", "bits": 4, "rows_per_group": 4}}


def test_bound_manifest_roundtrip_covers_every_shard_config_and_shipped_kernel(
    artifact, profile, dispatch
):
    data = manifest.create_manifest(artifact, dispatch, profile)
    assert data["schema_version"] == 2
    assert set(data["model_sha256"]) == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert set(data["kernel_sha256"]) == {"kernels/gate_up.metal", "kernels/gate_up_packed.metal"}
    assert (
        data["model_sha256"]["config.json"]
        == hashlib.sha256((artifact / "config.json").read_bytes()).hexdigest()
    )
    path = artifact / "execution_manifest.json"
    path.write_text(json.dumps(data))
    restored = manifest.load_manifest(path)
    assert restored == data
    assert manifest.verify_manifest(artifact, restored, profile) == dispatch
    dispatch["model.layers.0.mlp"]["bits"] = 3
    profile["device"]["device_name"] = "mutated"
    assert restored["dispatch"]["model.layers.0.mlp"]["bits"] == 4
    assert restored["profile_device"]["device_name"] == "test device"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("schema_version", 2.0),
        ("schema_version", "2"),
        ("schema_version", 3),
        ("dispatch", []),
        ("dispatch", None),
        ("mlx", None),
        ("mlx", ""),
        ("mlx_lm", 32),
        ("profile_device", []),
        ("profile_device", {}),
        ("profile_device", {"device_name": 7}),
        ("profile_device", {"device_name": "device", "memory_size": float("nan")}),
        ("model_sha256", []),
        ("model_sha256", {}),
        ("kernel_sha256", {}),
    ],
)
def test_schema_rejects_invalid_types_and_versions(artifact, profile, dispatch, field, value):
    data = manifest.create_manifest(artifact, dispatch, profile)
    data[field] = value
    with pytest.raises(manifest.ManifestError):
        manifest.validate_schema(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "metal"),
        ("backend", None),
        ("bits", True),
        ("bits", "4"),
        ("bits", 4.0),
        ("bits", 5),
        ("bits", 8),
        ("rows_per_group", True),
        ("rows_per_group", 4.0),
        ("rows_per_group", 3),
        ("typo_backend", "fused"),
        ("group_size", 64),
    ],
)
def test_dispatch_rejects_invalid_or_unknown_fields(artifact, profile, dispatch, field, value):
    data = manifest.create_manifest(artifact, dispatch, profile)
    data["dispatch"]["model.layers.0.mlp"][field] = value
    with pytest.raises(manifest.ManifestError):
        manifest.validate_schema(data)


@pytest.mark.parametrize("path", ["", "/model.mlp", "../mlp", "model..mlp", "a/b", "a\\b"])
def test_dispatch_rejects_malformed_module_paths(artifact, profile, dispatch, path):
    data = manifest.create_manifest(artifact, dispatch, profile)
    data["dispatch"] = {path: dispatch["model.layers.0.mlp"]}
    with pytest.raises(manifest.ManifestError):
        manifest.validate_schema(data)


@pytest.mark.parametrize(
    "field,path,digest",
    [
        ("model_sha256", "../model.safetensors", "a" * 64),
        ("model_sha256", "/model.safetensors", "a" * 64),
        ("model_sha256", "other.json", "a" * 64),
        ("model_sha256", "nested/model.safetensors", "a" * 64),
        ("model_sha256", "model.safetensors", "a" * 63),
        ("model_sha256", "model.safetensors", "A" * 64),
        ("model_sha256", "model.safetensors", None),
        ("kernel_sha256", "kernels/../gate.metal", "a" * 64),
        ("kernel_sha256", "/kernels/gate.metal", "a" * 64),
        ("kernel_sha256", "kernels/gate.py", "a" * 64),
        ("kernel_sha256", "kernels/gate.metal", "z" * 64),
    ],
)
def test_schema_rejects_malformed_hashes_and_file_paths(
    artifact, profile, dispatch, field, path, digest
):
    data = manifest.create_manifest(artifact, dispatch, profile)
    data[field][path] = digest
    with pytest.raises(manifest.ManifestError):
        manifest.validate_schema(data)


def test_schema_rejects_missing_and_unknown_fields(artifact, profile, dispatch):
    data = manifest.create_manifest(artifact, dispatch, profile)
    for field in data:
        incomplete = copy.deepcopy(data)
        del incomplete[field]
        with pytest.raises(manifest.ManifestError):
            manifest.validate_schema(incomplete)
    data["use_fusion"] = True
    with pytest.raises(manifest.ManifestError):
        manifest.validate_schema(data)


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":2,"schema_version":1}',
        '{"schema_version":NaN}',
        "{broken",
        "[]",
    ],
)
def test_load_rejects_duplicate_fields_nonfinite_json_or_invalid_schema(tmp_path, text):
    path = tmp_path / "manifest.json"
    path.write_text(text)
    with pytest.raises(manifest.ManifestError):
        manifest.load_manifest(path)


def test_legacy_requires_explicit_reseal_even_when_metadata_matches(artifact, profile, dispatch):
    data = {
        "schema_version": 1,
        "dispatch": dispatch,
        "profile_device": profile["device"],
        "mlx": profile["mlx"],
        "mlx_lm": profile["mlx_lm"],
    }
    with pytest.raises(manifest.ManifestError, match="regenerat.*seal"):
        manifest.verify_manifest(artifact, data, profile)
    assert manifest.validate_schema(data, allow_legacy=True) == data


@pytest.mark.parametrize("change", ["weights", "config", "extra_shard", "missing_shard"])
def test_verify_rejects_any_changed_artifact(artifact, profile, dispatch, change):
    data = manifest.create_manifest(artifact, dispatch, profile)
    if change == "weights":
        (artifact / "model-00002-of-00002.safetensors").write_bytes(b"different")
    elif change == "config":
        with (artifact / "config.json").open("a") as handle:
            handle.write("\n")
    elif change == "extra_shard":
        (artifact / "model-extra.safetensors").write_bytes(b"new shard")
    else:
        (artifact / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(manifest.ManifestError, match="model artifact"):
        manifest.verify_manifest(artifact, data, profile)


def test_verify_rejects_changed_shipped_kernel(artifact, profile, dispatch, monkeypatch):
    data = manifest.create_manifest(artifact, dispatch, profile)
    changed = dict(data["kernel_sha256"])
    changed["kernels/gate_up.metal"] = "0" * 64
    monkeypatch.setattr(manifest, "kernel_hashes", lambda: changed)
    with pytest.raises(manifest.ManifestError, match="Metal kernel"):
        manifest.verify_manifest(artifact, data, profile)


@pytest.mark.parametrize("field", ["mlx", "mlx_lm", "device_name", "architecture", "memory_size"])
def test_verify_rejects_incompatible_profile_metadata(artifact, profile, dispatch, field):
    data = manifest.create_manifest(artifact, dispatch, profile)
    if field in ("mlx", "mlx_lm"):
        profile[field] = "different"
    else:
        profile["device"][field] = "different"
    with pytest.raises(manifest.ManifestError, match="hardware/runtime"):
        manifest.verify_manifest(artifact, data, profile)


@pytest.mark.parametrize("change", ["bits", "up_bits", "mode", "group_size", "model_type"])
def test_create_rejects_dispatch_incompatible_with_saved_config(
    artifact, profile, dispatch, change
):
    config = json.loads((artifact / "config.json").read_text())
    if change == "bits":
        config["quantization"]["bits"] = 3
    elif change == "up_bits":
        config["quantization"]["model.layers.0.mlp.up_proj"] = {
            "bits": 3,
            "group_size": 64,
            "mode": "affine",
        }
    elif change == "model_type":
        config["model_type"] = "llama"
    else:
        config["quantization"][change] = "mxfp4" if change == "mode" else 16
    (artifact / "config.json").write_text(json.dumps(config))
    with pytest.raises(manifest.ManifestError, match="config"):
        manifest.create_manifest(artifact, dispatch, profile)


def test_verify_rechecks_dispatch_precision_not_just_hashes(artifact, profile, dispatch):
    data = manifest.create_manifest(artifact, dispatch, profile)
    data["dispatch"]["model.layers.0.mlp"]["bits"] = 3
    with pytest.raises(manifest.ManifestError, match="config"):
        manifest.verify_manifest(artifact, data, profile)


@pytest.mark.parametrize("name", ["config.json", "model-00001-of-00002.safetensors"])
def test_binding_rejects_symlink_artifacts(artifact, profile, dispatch, name, tmp_path):
    original = artifact / name
    outside = tmp_path / "external"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(manifest.ManifestError, match="regular.*file"):
        manifest.create_manifest(artifact, dispatch, profile)


def test_loaded_model_precision_is_checked_before_installing_wrappers(dispatch):
    from types import SimpleNamespace

    pair = SimpleNamespace(gate_proj=SimpleNamespace(bits=3), up_proj=SimpleNamespace(bits=3))
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.mlp", pair)])
    with pytest.raises(manifest.ManifestError, match="loaded.*bits"):
        manifest.validate_model_dispatch(model, dispatch)
    pair.gate_proj.bits = pair.up_proj.bits = 4
    manifest.validate_model_dispatch(model, dispatch)
    with pytest.raises(manifest.ManifestError, match="Unknown module"):
        manifest.validate_model_dispatch(model, {"other.mlp": dispatch["model.layers.0.mlp"]})


def test_public_validation_api_checks_artifacts_with_optional_runtime_check(
    artifact, profile, dispatch
):
    data = manifest.create_manifest(artifact, dispatch, profile)
    assert manifest.validate_manifest(artifact, data, profile) == dispatch
    other_runtime = {**profile, "mlx": "other"}
    with pytest.raises(manifest.ManifestError, match="hardware/runtime"):
        manifest.validate_manifest(artifact, data, other_runtime)
    assert (
        manifest.validate_manifest(artifact, data, other_runtime, strict_runtime=False) == dispatch
    )
    (artifact / "model-00001-of-00002.safetensors").write_bytes(b"changed")
    with pytest.raises(manifest.ManifestError, match="SHA-256"):
        manifest.validate_manifest(artifact, data, other_runtime, strict_runtime=False)


@pytest.mark.parametrize("environment", [None, [], {}, {"device": [], "mlx": "1", "mlx_lm": "1"}])
def test_public_api_rejects_invalid_environment_metadata(artifact, profile, dispatch, environment):
    data = manifest.create_manifest(artifact, dispatch, profile)
    with pytest.raises(manifest.ManifestError, match="environment|device"):
        manifest.create_manifest(artifact, dispatch, environment)
    with pytest.raises(manifest.ManifestError, match="environment|device"):
        manifest.validate_manifest(artifact, data, environment)


@pytest.mark.parametrize("strict", [None, 0, 1, "false"])
def test_runtime_validation_flag_requires_boolean(artifact, profile, dispatch, strict):
    data = manifest.create_manifest(artifact, dispatch, profile)
    with pytest.raises(manifest.ManifestError, match="strict_runtime"):
        manifest.validate_manifest(artifact, data, profile, strict_runtime=strict)


@pytest.mark.parametrize(
    "seal",
    [
        [],
        {},
        {"source_manifest_sha256": "a" * 64, "profiling_performed": True},
        {"source_manifest_sha256": "bad", "profiling_performed": False},
        {"source_manifest_sha256": "a" * 64, "profiling_performed": 0},
        {"source_manifest_sha256": "a" * 64, "profiling_performed": False, "unknown": 1},
    ],
)
def test_seal_provenance_is_strict_and_cannot_claim_fresh_profiling(
    artifact, profile, dispatch, seal
):
    data = manifest.create_manifest(artifact, dispatch, profile)
    data["seal"] = seal
    with pytest.raises(manifest.ManifestError):
        manifest.validate_schema(data)


def test_tiny_saved_mlx_model_manifest_roundtrip_does_not_serialize_fused_wrappers(
    tmp_path, profile
):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from mlx_lm.utils import load_model, save_config, save_model

    from paretoquant.runtime import FusedMLP, install_fusion

    config = {
        "model_type": "qwen2",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        "num_key_value_heads": 2,
    }
    source = tmp_path / "tiny"
    with mx.stream(mx.cpu):
        model = Model(ModelArgs.from_dict(config))
        model.set_dtype(mx.float16)
        nn.quantize(model, group_size=64, bits=4)
        mx.eval(model.parameters())
        config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
        dispatch = {"model.layers.0.mlp": {"backend": "fused", "bits": 4, "rows_per_group": 2}}
        install_fusion(model, dispatch)
        save_model(source, model)
        save_config(config, source / "config.json")
        sealed = manifest.create_manifest(source, dispatch, profile)
        (source / "execution_manifest.json").write_text(json.dumps(sealed))
        reloaded, _ = load_model(source)
        assert not any(type(m) is FusedMLP for _, m in reloaded.named_modules())
        restored = manifest.load_manifest(source / "execution_manifest.json")
        validated = manifest.validate_manifest(source, restored, profile)
        manifest.validate_model_dispatch(reloaded, validated)
        before = manifest.model_hashes(source)
        assert install_fusion(reloaded, validated) == ["model.layers.0.mlp"]
        assert reloaded.model.layers[0].mlp.rows_per_group == 2
        assert manifest.model_hashes(source) == before


@pytest.mark.integration
def test_existing_mixed_model_integrity_roundtrip_opt_in(tmp_path):
    """Read/copy real saved evidence only when explicitly enabled and no trials are running."""
    import os
    import re
    import subprocess
    from pathlib import Path

    if os.environ.get("PARETOQUANT_VERIFY_SAVED_MODEL") != "1":
        pytest.skip("set PARETOQUANT_VERIFY_SAVED_MODEL=1 when hardware trials are idle")
    processes = subprocess.run(
        ["ps", "-axo", "args"], capture_output=True, text=True, check=True
    ).stdout
    if re.search(r"paretoquant(?:\.cli)?\s+(?:run|replay)\b|run_scaling\.py", processes):
        pytest.skip("hardware benchmark process is running")
    from paretoquant.cli import main

    source = Path(__file__).resolve().parents[1] / "artifacts/m2pro-mixed-v2/model"
    if not source.is_dir():
        pytest.skip("saved mixed model is not present")
    old = manifest.load_manifest(source / "execution_manifest.json", allow_legacy=True)
    before = manifest.model_hashes(source)
    output = tmp_path / "sealed-real-model"
    assert main(["seal", "--model", str(source), "--output", str(output)]) == 0
    sealed = manifest.load_manifest(output / "execution_manifest.json")
    profile = {"device": old["profile_device"], "mlx": old["mlx"], "mlx_lm": old["mlx_lm"]}
    assert manifest.validate_manifest(output, sealed, profile) == old["dispatch"]
    assert manifest.model_hashes(source) == before == manifest.model_hashes(output)
