"""Opt-in real TensorFlow benchmark acceptance, not a mocked generation."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_native_tensorflow_benchmark_retains_actual_token_counts(tmp_path):
    output = tmp_path / "tensorflow.json"
    run = subprocess.run(
        [str(ROOT / "artifacts/tensorflow-metal-probe-py312/bin/python"),
         str(ROOT / "spikes/005-tensorflow-metal/probe.py"),
         "--cpu-cache-updates", "--new-tokens", "8", "--output", str(output),
         "--benchmark-inputs", str(ROOT / "results/ollama-head-to-head-v1/comparison.json"),
         "--benchmark-repeats", "2", "--benchmark-warmup", "1"],
        capture_output=True, text=True, timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    data = json.loads(output.read_text())
    assert data["generation"]["all_weights_verified"]
    assert data["generation"]["parameter_count"] == 494032768
    benchmark = data["benchmark"]
    assert benchmark["mode"] == "native_keras_qwen_hybrid_cpu_cache_metal_forward"
    assert len(benchmark["trials"]) == 6
    assert len(benchmark["warmups"]) == 3
    assert benchmark["cpu_cache_update_devices"]
    for row in benchmark["trials"]:
        assert row["generated_tokens"] == 8
        assert len(row["token_ids"]) == 8
        assert row["wall_tokens_per_second"] > 0
        assert row["response"]


def test_native_tensorflow_graph_mode_generates_actual_tokens(tmp_path):
    output = tmp_path / "graph.json"
    run = subprocess.run(
        [str(ROOT / "artifacts/tensorflow-metal-probe-py312/bin/python"),
         str(ROOT / "spikes/005-tensorflow-metal/probe.py"),
         "--cpu-cache-updates", "--graph-generation", "--new-tokens", "8",
         "--output", str(output)],
        capture_output=True, text=True, timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    data = json.loads(output.read_text())
    assert data["status"] == "ready"
    assert data["generation"]["run_eagerly"] is False
    assert len(data["generation"]["new_token_ids"]) == 8
    assert data["generation"]["all_weights_verified"]
