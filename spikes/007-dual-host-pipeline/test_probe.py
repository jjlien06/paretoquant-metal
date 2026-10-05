"""Native subprocess smoke tests for the standalone two-host spike."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SAVED = ROOT.parents[1] / "artifacts/m2pro-saved-retune-v1/model"


@pytest.mark.skipif(not SAVED.exists(), reason="Native local saved checkpoint required")
def test_single_host_greedy_generation_records_real_tokens():
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "probe.py"),
            "--mode",
            "generate",
            "--single-host",
            "--model",
            str(SAVED),
            "--max-tokens",
            "8",
            "--repeats",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stderr
    record = json.loads(run.stdout)
    assert record["world_size"] == 1
    assert record["layer_start"] == 0
    assert record["layer_end"] == 24
    assert record["parameter_bytes"] > 0
    assert len(record["requests"]) == 2
    assert record["requests"][0]["token_ids"] == record["requests"][1]["token_ids"]
    assert all(len(r["token_ids"]) == 8 and r["wall_seconds"] > 0 for r in record["requests"])
    assert record["trust_remote_code"] is False


def test_single_host_stream_loading_matches_ordinary_saved_generation():
    if not SAVED.exists():
        pytest.skip("Native local saved checkpoint required")
    base = [
        sys.executable,
        str(ROOT / "probe.py"),
        "--mode",
        "generate",
        "--single-host",
        "--model",
        str(SAVED),
        "--max-tokens",
        "8",
        "--repeats",
        "1",
    ]
    ordinary = subprocess.run(base, capture_output=True, text=True, timeout=90)
    streamed = subprocess.run(base + ["--stream-local"], capture_output=True, text=True, timeout=90)
    assert ordinary.returncode == 0, ordinary.stderr
    assert streamed.returncode == 0, streamed.stderr
    one, two = json.loads(ordinary.stdout), json.loads(streamed.stdout)
    assert one["requests"][0]["token_ids"] == two["requests"][0]["token_ids"]
    assert two["weight_loading"]["all_retained_tensors_loaded"] is True
    assert one["parameter_bytes"] == two["parameter_bytes"]
    assert two["wired_before_load_bytes"] == two["device"]["max_recommended_working_set_size"]


@pytest.mark.skipif(not SAVED.exists(), reason="Native local saved checkpoint required")
def test_two_rank_cpu_boundaries_match_single_host_generation(tmp_path):
    base = [
        sys.executable,
        str(ROOT / "probe.py"),
        "--mode",
        "generate",
        "--model",
        str(SAVED),
        "--stream-local",
        "--max-tokens",
        "8",
        "--repeats",
        "1",
    ]
    control = subprocess.run(base + ["--single-host"], capture_output=True, text=True, timeout=90)
    assert control.returncode == 0, control.stderr
    expected = json.loads(control.stdout)["requests"][0]["token_ids"]
    held = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        ports = [sock.getsockname()[1] for sock in held]
    finally:
        for sock in held:
            sock.close()
    hostfile = tmp_path / "ring.json"
    hostfile.write_text(json.dumps([[f"127.0.0.1:{port}"] for port in ports]))
    processes, handles = [], []
    try:
        for rank in range(2):
            stdout = (tmp_path / f"rank{rank}.json").open("wb")
            stderr = (tmp_path / f"rank{rank}.stderr").open("wb")
            handles.extend([stdout, stderr])
            processes.append(
                subprocess.Popen(
                    base + ["--split", "12", "12", "--cpu-communication"],
                    env=dict(os.environ, MLX_HOSTFILE=str(hostfile), MLX_RANK=str(rank)),
                    stdout=stdout,
                    stderr=stderr,
                )
            )
        for rank, process in enumerate(processes):
            process.wait(timeout=90)
            assert process.returncode == 0, (tmp_path / f"rank{rank}.stderr").read_text()
        for handle in handles:
            handle.close()
        records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
        assert all(r["requests"][0]["token_ids"] == expected for r in records)
        assert all(r["communication"]["mode"] == "synchronous_cpu_boundaries" for r in records)
        assert all(r["communication"]["calls"]["all_gather"] > 0 for r in records)
        assert records[0]["communication"]["calls"]["recv_like"] > 0
        assert records[1]["communication"]["calls"]["send"] > 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
        for handle in handles:
            if not handle.closed:
                handle.close()


def test_two_rank_collectives_are_real_and_fail_closed(tmp_path):
    held = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        ports = [s.getsockname()[1] for s in held]
    finally:
        for sock in held:
            sock.close()
    hostfile = tmp_path / "ring.json"
    hostfile.write_text(json.dumps([[f"127.0.0.1:{p}"] for p in ports]))
    processes = []
    try:
        for rank in range(2):
            env = dict(os.environ, MLX_HOSTFILE=str(hostfile), MLX_RANK=str(rank))
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(ROOT / "probe.py"), "--mode", "collective"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        records = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=40)
            assert process.returncode == 0, stderr
            records.append(json.loads(stdout))
        assert [r["rank"] for r in records] == [0, 1]
        assert all(r["world_size"] == 2 for r in records)
        assert all(r["all_sum"] == 3 for r in records)
        assert all(r["all_gather"] == [0, 1] for r in records)
        assert records[0]["received"] == [21.0, 22.0]
        assert records[1]["received"] == [11.0, 12.0]
        assert all(r["mlx_version"] == "0.32.3" for r in records)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
