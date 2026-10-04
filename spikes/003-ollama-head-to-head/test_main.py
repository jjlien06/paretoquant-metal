"""Opt-in live integration test; requires the downloaded model and local Ollama."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


def test_import_does_not_start_benchmark():
    runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="admission_unit")


@pytest.mark.skipif(os.environ.get("PARETOQUANT_LIVE_OLLAMA") != "1", reason="live Ollama opt-in")
def test_live_comparison_records_real_paired_generations(tmp_path):
    script = Path(__file__).with_name("main.py")
    output = tmp_path / "comparison.json"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
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
    assert data["ollama_details"]["quantization_level"] == "Q4_K_M"
    assert data["ollama_model_info"]["general.finetune"] == "Instruct"
    assert data["installed_fused_pair_count"] > 0
    assert data["quantization_matched"] is False
    assert len(data["trials"]) == 12
    assert len(data["summaries"]) == 3
    assert data["runtime_fused_calls"] > 0
    assert data["ollama_ps"]["models"][0]["size_vram"] > 0
    for trial in data["trials"]:
        assert 0 < trial["generated_tokens"] <= 8
        assert trial["wall_seconds"] > 0
        assert trial["wall_tokens_per_second"] > 0
        assert trial["response"]
        assert trial["engine"] in ("paretoquant_mlx", "ollama")
    for summary in data["summaries"]:
        assert summary["paretoquant_mlx"]["repeats"] == 2
        assert summary["ollama"]["repeats"] == 2
        assert summary["ratio_mlx_over_ollama_wall_tps"] > 0
