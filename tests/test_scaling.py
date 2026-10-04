"""Opt-in real-checkpoint acceptance for the bounded-residency scaling runner."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("named", [False, True], ids=["standalone", "named"])
def test_scaling_saved_tokenizer_preserves_chat_prompt(tmp_path, monkeypatch, named):
    import hashlib
    import runpy
    from types import SimpleNamespace

    from test_cli import _tokenizer_saving_case

    from paretoquant.cli import _chat_prompt, _source_hashes
    from paretoquant.manifest import kernel_hashes

    case = _tokenizer_saving_case(tmp_path, monkeypatch, named=named)
    runner = runpy.run_path(str(ROOT / "scripts/run_scaling.py"), run_name="scaling_test")
    profile = {
        **case.measured,
        "source_weights_sha256": _source_hashes(case.source),
        "source_config_sha256": hashlib.sha256(
            (case.source / "config.json").read_bytes()
        ).hexdigest(),
        "profile_kernel_sha256": kernel_hashes(),
        "fixed_parameter_bytes": 80,
        "uniform4_parameter_bytes": 100,
        "reference_smoke_quality": {"mean_nll": 1.0, "token_count": 2},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    output = tmp_path / "scaled"
    assert (
        runner["evaluate"](
            SimpleNamespace(
                model=case.source,
                profile=profile_path,
                output=output,
                memory_fraction=1.0,
                latency_factor=1.0,
                max_states=100,
                decode_steps=1,
                repeats=1,
            )
        )
        == 0
    )
    expected = _chat_prompt(case.tokenizer, "hello world")
    for variant in ("uniform4", "mixed"):
        saved = output / variant / "model"
        restored = case.load_tokenizer(saved, tokenizer_config_extra={"local_files_only": True})
        assert _chat_prompt(restored, "hello world") == expected
        assert restored.chat_template == case.tokenizer.chat_template
        assert (saved / "generation_config.json").read_bytes() == case.before[
            "generation_config.json"
        ]
    assert {
        str(p.relative_to(case.source)): p.read_bytes()
        for p in case.source.rglob("*")
        if p.is_file()
    } == case.before


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("PARETOQUANT_RUN_SCALING") != "1", reason="opt-in real checkpoint")
def test_profile_real_checkpoint_records_actual_costs(tmp_path):
    source = ROOT / "models/qwen2.5-0.5b-instruct"
    if not source.is_dir():
        pytest.skip("local reference absent")
    output = tmp_path / "profile"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_scaling.py"),
            "profile",
            "--model",
            str(source),
            "--output",
            str(output),
            "--profile-repeats",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    data = json.loads((output / "profile.json").read_text())
    assert len(data["units"]) == 24
    assert data["uniform4_parameter_bytes"] > data["fixed_parameter_bytes"] > 0
    assert data["source_weights_sha256"]
    assert data["source_config_sha256"]
    assert set(data["profile_kernel_sha256"]) == {
        "kernels/gate_up.metal",
        "kernels/gate_up_packed.metal",
    }
    assert all({o["bits"] for o in row["options"]} == {3, 4, 6} for row in data["units"])
    assert all(
        len(row["measurements"]["4"]["validation"]["stock"]["samples_ms"]) == 2
        for row in data["units"]
    )
    evaluated = tmp_path / "evaluated"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_scaling.py"),
            "evaluate",
            "--model",
            str(source),
            "--profile",
            str(output / "profile.json"),
            "--output",
            str(evaluated),
            "--memory-fraction",
            "0.94",
            "--latency-factor",
            "1.15",
            "--decode-steps",
            "8",
            "--repeats",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((evaluated / "results.json").read_text())
    baseline, mixed = result["variants"]["uniform4"], result["variants"]["mixed"]
    assert mixed["parameter_bytes"] <= result["budget_bytes"] < baseline["parameter_bytes"]
    assert baseline["generation"] and mixed["generation"]
    assert len(mixed["timing"]["decode_samples_ms"]) == 2
    assert mixed["quality"]["token_count"] == baseline["quality"]["token_count"] > 0
    manifest = json.loads((evaluated / "mixed/model/execution_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    data["profile_kernel_sha256"]["kernels/gate_up_packed.metal"] = "0" * 64
    stale = tmp_path / "stale-profile.json"
    stale.write_text(json.dumps(data))
    refused = tmp_path / "refused"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_scaling.py"),
            "evaluate",
            "--model",
            str(source),
            "--profile",
            str(stale),
            "--output",
            str(refused),
            "--decode-steps",
            "8",
            "--repeats",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert process.returncode == 2
    assert "kernel" in process.stderr.lower()
    assert not refused.exists()
