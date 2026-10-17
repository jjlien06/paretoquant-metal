"""Explicit per-rank CLI and immutable complete source registry."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as S

import pytest

ROOT = Path(__file__).parent


def test_hidden_cli_default_control_optin_both_ranks_and_chunk_bounds():
    import supervise

    for name in ("probe.py", "supervise.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / name), "--help"], capture_output=True, text=True, check=True
        )
        assert "--hidden-only-prefill" in result.stdout
        assert "--hidden-gate-chunk-size" in result.stdout
    args = S(
        max_tokens=5,
        repeats=1,
        execution="phased",
        eval_group_size=32,
        eval_group_schedule=None,
        target_revision="10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        target_config_sha256="a" * 64,
        target_index_sha256="b" * 64,
        small_config_sha256="c" * 64,
        small_index_sha256="d" * 64,
        split=[28, 36],
        reserve_bytes=0,
        deadline=900,
        local_address="127.0.0.1",
        remote_address="127.0.0.2",
        remote_root="/remote/024",
        remote_source="/weights",
        local_metadata="/meta",
        output="/out",
        ssh_key="/keys/k",
        ssh_target="host",
        prompt="tokens:[1,2,3]",
        rank0_verifier="native",
        rank1_verifier="singleton-qmm",
        prefix_rollback=True,
        verification_gate_only=True,
        sidecar="/head",
        small_model_local="/small",
        small_model_remote="/remote/small",
        hidden_only_prefill=True,
        hidden_gate_chunk_size=256,
    )
    supervise.validate_args(args)
    for rank in (0, 1):
        opts = supervise.probe_options(args, rank=rank)
        assert "--hidden-only-prefill" in opts
        assert opts[opts.index("--hidden-gate-chunk-size") + 1] == "256"
    for value in (0, 257, True):
        with pytest.raises(ValueError):
            supervise.validate_args(S(**{**vars(args), "hidden_gate_chunk_size": value}))
    args.hidden_only_prefill = False
    args.deadline = 240
    for rank in (0, 1):
        opts = supervise.probe_options(args, rank=rank)
        assert "--hidden-only-prefill" not in opts


def test_fixed_registry_covers_every_code_config_kernel_and_empty_template():
    import json

    from remote_actor import SOURCES, source_hashes

    expected = {p.name for p in ROOT.iterdir() if p.suffix in (".py", ".json", ".metal")}
    assert set(SOURCES) == expected
    assert len(SOURCES) == len(set(SOURCES))
    assert set(source_hashes(ROOT)) == expected
    assert json.loads((ROOT / "benchmark_schedule.json").read_text())["requests"] == []
