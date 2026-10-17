"""CLI dry-run is structurally no hardware/no writes, not physical admission."""

import sys
from pathlib import Path


def argv():
    return [
        "--remote-root",
        "/remote/025",
        "--remote-source",
        "/weights",
        "--local-metadata",
        "/meta",
        "--output",
        "/never-created",
        "--ssh-key",
        "/keys/k",
        "--split",
        "28",
        "36",
        "--target-revision",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "--target-config-sha256",
        "a" * 64,
        "--target-index-sha256",
        "b" * 64,
        "--sidecar",
        "/head",
        "--small-model-local",
        "/small",
        "--small-model-remote",
        "/remote/small",
        "--small-config-sha256",
        "c" * 64,
        "--small-index-sha256",
        "d" * 64,
        "--max-tokens",
        "5",
        "--repeats",
        "1",
        "--eval-group-size",
        "32",
        "--rank0-verifier",
        "native",
        "--rank1-verifier",
        "slot-qmv",
        "--prefix-rollback",
        "--hidden-only-prefill",
        "--overlapped-prefill",
        "--require-normal-pressure",
        "--verification-gate-only",
        "--dry-run",
    ]


def test_overlap_cli_frozen_profile_cannot_change_rank_math_or_placement():
    from copy import deepcopy

    import pytest
    import supervise

    args = supervise.parser().parse_args(argv())
    supervise.validate_args(args)
    for field, value in (
        ("rank1_verifier", "native"),
        ("rank0_verifier", "slot-qmv"),
        ("split", [36, 28]),
        ("eval_group_size", 16),
    ):
        bad = deepcopy(args)
        setattr(bad, field, value)
        with pytest.raises(ValueError, match="profile"):
            supervise.validate_args(bad)


def test_cli_dryrun_no_transport_model_or_output(monkeypatch, capsys):
    import supervise

    def forbidden(*a, **k):
        raise AssertionError("dry run entered live run")

    monkeypatch.setattr(supervise, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["supervise.py", *argv()])
    assert supervise.main() == 0
    import json

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run" and result["hardware_executed"] is False
    assert result["overlapped_prefill"] is True and result["require_normal_pressure"] is True
    assert result["prefill_modes"] == ["serial", "overlap"]
    for opts in result["rank_probe_options"]:
        assert "--overlapped-prefill" in opts and "--require-normal-pressure" in opts
    assert not Path("/never-created").exists()
