"""Live opt-in acceptance test; actual MLX model, no synthesized API output."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    return runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="admission_unit")


@pytest.mark.parametrize("root", ["/different/checkpoint", None, "", "relative/model"])
def test_vllm_refuses_conflicting_or_unobservable_root(runner, tmp_path, root):
    # Explicit metadata fixture for admission, never benchmark measurements.
    models = {"data": [{"id": "paretoquant-mixed", "root": root, "max_model_len": 1024}]}
    with pytest.raises(RuntimeError, match="root"):
        runner["validate_server_model"](models, tmp_path, 1024)


def test_import_does_not_start_benchmark():
    runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="admission_unit")


@pytest.mark.parametrize("context", [None, 4096, "1024", True])
def test_vllm_refuses_wrong_context(runner, tmp_path, context):
    models = {
        "data": [{"id": "paretoquant-mixed", "root": str(tmp_path), "max_model_len": context}]
    }
    with pytest.raises(RuntimeError, match="context"):
        runner["validate_server_model"](models, tmp_path, 1024)


@pytest.mark.parametrize("count", [0, 2])
def test_vllm_refuses_absent_or_ambiguous_alias(runner, tmp_path, count):
    item = {"id": "paretoquant-mixed", "root": str(tmp_path), "max_model_len": 1024}
    with pytest.raises(RuntimeError, match="exactly one"):
        runner["validate_server_model"]({"data": [item] * count}, tmp_path, 1024)


def test_matching_server_root_is_not_weight_or_settings_attestation(runner, tmp_path):
    models = {"data": [{"id": "paretoquant-mixed", "root": str(tmp_path), "max_model_len": 1024}]}
    data = runner["validate_server_model"](models, tmp_path, 1024)
    assert data["server_root_matches_local_model"] is True
    assert data["max_model_len"] == 1024
    for field in (
        "same_saved_mixed_weights",
        "server_loaded_weight_sha256",
        "prefix_caching",
        "max_num_seqs",
    ):
        assert data[field] is None
    assert data["requested_settings"]["max_num_seqs"] == 1


@pytest.mark.skipif(os.environ.get("PARETOQUANT_LIVE_MLX") != "1", reason="live MLX opt-in")
def test_live_runner_measures_actual_mixed_model(tmp_path):
    script = Path(__file__).with_name("main.py")
    output = tmp_path / "mlx.json"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--engine",
            "paretoquant_mlx",
            "--output",
            str(output),
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--max-tokens",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    data = json.loads(output.read_text())
    assert data["engine"] == "paretoquant_mlx"
    assert data["metadata"]["fused_pair_count"] > 0
    assert len(data["trials"]) == 6
    assert len(data["summaries"]) == 3
    assert len(data["warmups"]) == 3
    for row in data["trials"]:
        assert row["generated_tokens"] == 8
        assert len(row["token_ids"]) == 8
        assert row["wall_seconds"] > 0
        assert row["wall_tokens_per_second"] > 0
        assert row["response"]
    for summary in data["summaries"]:
        assert summary["repeats"] == 2
        assert summary["median_wall_tokens_per_second"] > 0


@pytest.mark.skipif(os.environ.get("PARETOQUANT_LIVE_VLLM") != "1", reason="live vLLM opt-in")
def test_live_vllm_runner_counts_real_api_generations(tmp_path):
    script = Path(__file__).with_name("main.py")
    output = tmp_path / "vllm.json"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--engine",
            "vllm_metal",
            "--output",
            str(output),
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--max-tokens",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    data = json.loads(output.read_text())
    assert data["metadata"]["server_root_matches_local_model"] is True
    assert data["metadata"]["same_saved_mixed_weights"] is None
    assert data["metadata"]["server_loaded_weight_sha256"] is None
    assert len(data["trials"]) == 6
    for row in data["trials"]:
        assert row["generated_tokens"] == 8
        assert row["wall_tokens_per_second"] > 0
        assert row["response"]
