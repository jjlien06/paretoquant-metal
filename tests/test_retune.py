"""CPU policy/safety tests; GPU integration is explicitly opt-in."""

import importlib
import importlib.util

import numpy as np
import pytest


def retune_module():
    assert importlib.util.find_spec("paretoquant.retune") is not None, "retuning is missing"
    return importlib.import_module("paretoquant.retune")


def timing(values):
    import statistics

    return {"samples_ms": values, "median_ms": statistics.median(values)}


def test_dispatch_requires_fresh_validation_margin_not_tuning_win():
    retune = retune_module()
    validation = {"stock": timing([1.0, 1.0, 1.0]), "fused:rpg2": timing([0.96, 0.97, 0.98])}
    decision = retune.select_dispatch(validation, "fused:rpg2", bits=3)
    assert decision["backend"] == "stock"
    validation["fused:rpg2"] = timing([0.89, 0.90, 0.91])
    assert retune.select_dispatch(validation, "fused:rpg2", bits=3) == {
        "backend": "fused",
        "rows_per_group": 2,
        "bits": 3,
    }


def test_numerical_validation_reports_errors_and_rejects_nonfinite_or_wrong_shapes():
    retune = retune_module()
    expected = np.array([[1.0, 2.0, 0.0]], dtype=np.float16)
    result = retune.numerical_errors(expected + np.float16(0.005), expected)
    assert result["valid"]
    assert result["max_absolute_error"] > 0
    assert result["relative_mse"] > 0
    assert result["elements"] == 3
    assert not retune.numerical_errors(np.array([[1.0, 2.0, 0.3]]), expected)["valid"]
    assert not retune.numerical_errors(np.full_like(expected, np.nan), expected)["valid"]
    assert not retune.numerical_errors(np.full_like(expected, np.inf), expected)["valid"]
    assert not retune.numerical_errors(expected.reshape(-1), expected)["valid"]


@pytest.fixture
def saved_fixture(tmp_path):
    """Toy file bytes for CPU safety checks only, not runnable model data."""
    import json

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_act": "silu",
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"toy test bytes")
    (source / "tokenizer.json").write_bytes(b"toy tokenizer")
    (source / "execution_manifest.json").write_text('{"schema_version":1,"dispatch":"untrusted"}')
    return source


@pytest.mark.parametrize(
    "case",
    [
        "same",
        "descendant",
        "ancestor",
        "nonempty",
        "symlink",
        "symlink_parent",
        "source_symlink",
        "nested_source_symlink",
    ],
)
def test_output_safety_refuses_overlap_nonempty_and_symlinks(saved_fixture, tmp_path, case):
    retune = retune_module()
    source, output = saved_fixture, tmp_path / "new"
    if case == "same":
        output = source
    elif case == "descendant":
        output = source / "new"
    elif case == "ancestor":
        output = source.parent
    elif case == "nonempty":
        output.mkdir()
        (output / "keep").write_text("important")
    elif case == "symlink":
        output.symlink_to(source, target_is_directory=True)
    elif case == "symlink_parent":
        parent = tmp_path / "link"
        parent.symlink_to(tmp_path, target_is_directory=True)
        output = parent / "new"
    elif case == "source_symlink":
        source = tmp_path / "source-link"
        source.symlink_to(saved_fixture, target_is_directory=True)
    else:
        (source / "linked").symlink_to(source / "tokenizer.json")
    before = (saved_fixture / "model.safetensors").read_bytes()
    with pytest.raises(ValueError):
        retune.check_paths(source, output)
    assert (saved_fixture / "model.safetensors").read_bytes() == before


def test_safe_paths_and_snapshot_hashing_are_read_only(saved_fixture, tmp_path):
    retune = retune_module()
    output = tmp_path / "new"
    assert retune.check_paths(saved_fixture, output) == (saved_fixture.resolve(), output.resolve())
    assert not output.exists()
    hashes = retune.snapshot_hashes(saved_fixture)
    assert set(hashes) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "execution_manifest.json",
    }
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize(
    "field,value",
    [("mode", "mxfp4"), ("mode", "nvfp4"), ("bits", 5), ("bits", True), ("group_size", 16)],
)
def test_saved_config_rejects_unsupported_quantization_without_writes(saved_fixture, field, value):
    import json

    retune = retune_module()
    config = json.loads((saved_fixture / "config.json").read_text())
    config["quantization"][field] = value
    (saved_fixture / "config.json").write_text(json.dumps(config))
    before = retune.snapshot_hashes(saved_fixture)
    with pytest.raises(ValueError, match="quantization"):
        retune.read_saved_config(saved_fixture)
    assert retune.snapshot_hashes(saved_fixture) == before


@pytest.mark.parametrize("bits", [3, 4, 6])
def test_saved_config_accepts_affine_saved_bits_without_conversion(saved_fixture, bits):
    import json

    retune = retune_module()
    config = json.loads((saved_fixture / "config.json").read_text())
    config["quantization"]["model.layers.0.mlp.gate_proj"] = {
        "bits": bits,
        "group_size": 64,
        "mode": "affine",
    }
    (saved_fixture / "config.json").write_text(json.dumps(config))
    assert retune.read_saved_config(saved_fixture) == config


@pytest.mark.parametrize("invalid", [False, True])
def test_pair_profiles_every_launch_then_fresh_best_only_and_falls_back_if_invalid(invalid):
    retune = retune_module()
    inputs = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float16)
    numerical_calls, rounds = [], []

    def stock(x, gate, up, **kwargs):
        assert gate == "saved gate" and up == "saved up"
        assert kwargs == {"bits": 6, "group_size": 64}
        return x * 2

    def fused(x, gate, up, *, rows_per_group, **kwargs):
        numerical_calls.append(rows_per_group)
        value = stock(x, gate, up, **kwargs)
        return value + 100 if invalid else value

    def benchmark(functions, *, repeats, warmup):
        # Explicit fabricated timing fixtures ONLY for the CPU selection-policy test.
        rounds.append((tuple(functions), repeats, warmup))
        for fn in functions.values():
            fn()
        medians = {
            "stock": 1.0,
            "fused:rpg1": 0.8,
            "fused:rpg2": 0.7,
            "fused:rpg4": 0.6,
            "fused:rpg8": 0.9,
        }
        if len(rounds) == 2:
            medians["fused:rpg4"] = 1.1  # tuning wins must not leak into validation
        return {name: timing([medians[name]] * repeats) for name in functions}

    pair = retune.profile_pair(
        inputs,
        "saved gate",
        "saved up",
        bits=6,
        group_size=64,
        repeats=20,
        warmup=3,
        stock_function=stock,
        fused_function=fused,
        benchmark=benchmark,
        evaluate=np.asarray,
    )
    assert pair["dispatch"]["backend"] == "stock"
    assert set(pair["numerical_validation"]) == {"1", "2", "4", "8"}
    assert all(rpg in numerical_calls for rpg in (1, 2, 4, 8))
    assert pair["calibration_input_count"] == 3
    assert rounds[0][1:] == (20, 3)
    assert rounds[1][1:] == (20, 3)
    if invalid:
        assert rounds == [(("stock",), 20, 3), (("stock",), 20, 3)]
        assert pair["best_tuning_candidate"] is None
        assert not any(v["valid"] for v in pair["numerical_validation"].values())
    else:
        assert rounds[0][0] == ("stock", "fused:rpg1", "fused:rpg2", "fused:rpg4", "fused:rpg8")
        assert rounds[1][0] == ("stock", "fused:rpg4")
        assert pair["best_tuning_candidate"] == "fused:rpg4"
        assert all(v["valid"] for v in pair["numerical_validation"].values())


@pytest.mark.parametrize(
    "values",
    [[1.0], [float("nan")] * 3, [float("inf")] * 3, [0.0] * 3, [-1.0] * 3, [0.95] * 3, [1.1] * 3],
)
def test_selection_keeps_stock_on_invalid_slow_or_margin_boundary(values):
    validation = {"stock": timing([1.0] * 3), "fused:rpg8": timing(values)}
    assert retune_module().select_dispatch(validation, "fused:rpg8", bits=4)["backend"] == "stock"


@pytest.mark.parametrize("margin", [-0.01, 1.0, float("nan"), float("inf"), True])
def test_selection_rejects_invalid_margin(margin):
    with pytest.raises(ValueError, match="min_improvement"):
        retune_module().select_dispatch({}, None, bits=4, min_improvement=margin)


def test_zero_margin_is_explicit_and_sample_medians_are_recomputed():
    validation = {"stock": timing([1.0] * 3), "fused:rpg1": timing([0.99] * 3)}
    validation["fused:rpg1"]["median_ms"] = 99.0
    assert (
        retune_module().select_dispatch(validation, "fused:rpg1", bits=4, min_improvement=0)[
            "backend"
        ]
        == "fused"
    )


def test_publish_copies_exact_snapshot_archives_untrusted_legacy_and_binds_fresh_environment(
    saved_fixture, tmp_path
):
    import json

    from paretoquant.manifest import load_manifest, validate_manifest

    retune = retune_module()
    original = retune.snapshot_hashes(saved_fixture)
    environment = {"device": {"device_name": "CPU test fixture"}, "mlx": "test", "mlx_lm": "test"}
    dispatch = {"model.layers.0.mlp": {"backend": "stock", "bits": 4, "rows_per_group": 4}}
    profile = {
        "environment": environment,
        "dispatch": dispatch,
        "source_file_sha256": original,
        "units": [],
    }
    output = tmp_path / "published"
    retune.publish_retune(saved_fixture, output, profile)
    assert retune.snapshot_hashes(saved_fixture) == original
    for filename in original:
        if filename != "execution_manifest.json":
            assert (output / "model" / filename).read_bytes() == (
                saved_fixture / filename
            ).read_bytes()
    archived = output / "provenance" / "source_execution_manifest.json"
    assert archived.read_bytes() == (saved_fixture / "execution_manifest.json").read_bytes()
    fresh = load_manifest(output / "model" / "execution_manifest.json")
    assert fresh["schema_version"] == 2
    assert fresh["profile_device"] == environment["device"]
    assert validate_manifest(output / "model", fresh, environment) == dispatch
    report = json.loads((output / "retune_profile.json").read_text())
    assert report["source_file_sha256"] == report["copied_source_file_sha256"] == original
    assert report["source_payload_sha256"] == report["new_payload_sha256"]
    assert report["original_dispatch_trusted"] is False
    assert report["source_execution_manifest_sha256"] == original["execution_manifest.json"]
    assert report["kernel_sha256"] == fresh["kernel_sha256"]


def test_calibration_local_json_records_exact_file_and_settings(tmp_path):
    import hashlib

    retune = retune_module()
    path = tmp_path / "local.json"
    path.write_text('["a real calibration sentence", "another sentence"]\n')
    texts, metadata = retune.read_calibration(path)
    assert texts == ["a real calibration sentence", "another sentence"]
    assert metadata["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["max_tokens"] == 128
    assert metadata["samples_per_text"] == 16
    assert len(metadata["calibration_sha256"]) == 64
    for invalid in ["[]", '[""]', "[12]", '{"texts":[]}', '["ok", NaN]']:
        path.write_text(invalid)
        with pytest.raises(ValueError):
            retune.read_calibration(path)


def test_existing_v2_source_binding_is_preserved_but_dispatch_and_runtime_are_not_trusted(
    saved_fixture,
):
    import json

    from paretoquant.manifest import model_hashes

    retune = retune_module()
    original = {
        "schema_version": 2,
        "model_sha256": model_hashes(saved_fixture),
        "dispatch": "deliberately ignored",
        "kernel_sha256": "obsolete",
        "profile_device": "obsolete",
    }
    (saved_fixture / "execution_manifest.json").write_text(json.dumps(original))
    binding = retune.source_binding(saved_fixture)
    assert binding["model_sha256"] == original["model_sha256"]
    assert binding["verified"] is True
    (saved_fixture / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source.*binding"):
        retune.source_binding(saved_fixture)


def test_retune_rejects_unsafe_output_before_loading_gpu(saved_fixture):
    with pytest.raises(ValueError, match="overlap"):
        retune_module().retune_saved_model(saved_fixture, saved_fixture)


@pytest.mark.integration
@pytest.mark.metal
@pytest.mark.parametrize("bits", [3, 4, 6])
def test_tiny_saved_qwen_gpu_retune_opt_in(tmp_path, bits):
    """Real tiny random model/tokenizer fixtures; no model-quality or speed claim."""
    import json
    import os

    if os.environ.get("PARETOQUANT_RETUNE_GPU") != "1":
        pytest.skip("set PARETOQUANT_RETUNE_GPU=1 with the GPU idle")
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from mlx_lm.utils import load_model, load_tokenizer, quantize_model, save_config, save_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from paretoquant.manifest import load_manifest, validate_manifest

    retune = retune_module()
    source = tmp_path / "tiny-saved"
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
    with mx.stream(mx.cpu):
        mx.random.seed(91)
        model = Model(ModelArgs.from_dict(config))
        model.set_dtype(mx.float16)
        model, config = quantize_model(model, config, 64, bits)
        mx.eval(model.parameters())
        save_model(source, model)
        save_config(config, source / "config.json")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "tiny": 2, "test": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(source)
    (source / "execution_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dispatch": {"bogus.layer": {"backend": "fused", "bits": 9}},
            }
        )
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text('["a tiny test a tiny test"]')
    before = retune.snapshot_hashes(source)
    report = retune.retune_saved_model(
        source,
        tmp_path / "retuned",
        calibration=calibration,
        repeats=2,
        warmup=0,
        min_improvement=0.99,
        verbose=False,
    )
    assert retune.snapshot_hashes(source) == before
    assert report["counts"] == {
        "pairs": 1,
        "fused_pairs": 0,
        "stock_pairs": 1,
        "bits": {str(bits): 1},
        "numerically_valid_candidates": 4,
    }
    assert report["source_code_sha256"]
    assert report["source_binding"]["schema_version"] == 1
    unit = report["units"][0]
    assert unit["bits"] == bits
    # AutoTokenizer may select Qwen's backend; assert actual captured token count,
    # not an assumed WordLevel count from the pre-save tokenizer fixture.
    encoded_count = len(load_tokenizer(source).encode("a tiny test a tiny test"))
    assert unit["calibration_input_count"] == min(encoded_count, 16)
    assert all(error["valid"] for error in unit["numerical_validation"].values())
    assert len(unit["validation"]["stock"]["samples_ms"]) == 2
    saved = tmp_path / "retuned" / "model"
    fresh = load_manifest(saved / "execution_manifest.json")
    assert validate_manifest(saved, fresh, report["environment"]) == report["dispatch"]
    loaded, _ = load_model(saved)
    assert loaded.model.layers[0].mlp.gate_proj.bits == bits
    assert type(loaded.model.layers[0].mlp).__name__ == "MLP"


def test_retune_script_has_required_flags_and_rejects_overlap_before_gpu(saved_fixture):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/retune_model.py"
    assert script.is_file(), "saved-model retune script is missing"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert help_result.returncode == 0
    for flag in (
        "--model",
        "--output",
        "--repeats",
        "--warmup",
        "--min-improvement",
        "--calibration",
    ):
        assert flag in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(saved_fixture),
            "--output",
            str(saved_fixture),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "overlap" in result.stderr
