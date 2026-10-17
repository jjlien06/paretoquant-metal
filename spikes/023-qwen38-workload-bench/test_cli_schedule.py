import hashlib
import json

import pytest
from test_workload_schedule import schedule


def test_schedule_cli_bounds_hash_and_rank_propagation(monkeypatch, tmp_path):
    from types import SimpleNamespace as S

    import supervise
    import workload_schedule as w

    data = json.dumps(schedule()).encode()
    path = tmp_path / w.NAME
    path.write_bytes(data)
    monkeypatch.setattr(supervise, "ROOT", tmp_path)
    args = S(
        benchmark_schedule=True,
        benchmark_schedule_sha256=hashlib.sha256(data).hexdigest(),
        max_tokens=5,
        repeats=2,
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
        remote_root="/remote/023",
        remote_source="/weights",
        local_metadata="/meta",
        output="/out",
        ssh_key="/keys/k",
        ssh_target="host",
        prompt="tokens:[1,2,3]",
        rank0_verifier="native",
        rank1_verifier="singleton-qmm",
        prefix_rollback=True,
        verification_gate_only=False,
        sidecar="/head",
        small_model_local="/small",
        small_model_remote="/remote/small",
    )
    supervise.validate_args(args)
    for rank in (0, 1):
        opts = supervise.probe_options(args, rank=rank)
        assert "--benchmark-schedule" in opts
        assert opts[opts.index("--benchmark-schedule-sha256") + 1] == args.benchmark_schedule_sha256
    for field, value in (
        ("repeats", 3),
        ("max_tokens", 4),
        ("deadline", 901),
        ("split", [36, 28]),
        ("eval_group_size", 16),
        ("prefix_rollback", False),
    ):
        bad = S(**vars(args))
        setattr(bad, field, value)
        with pytest.raises(ValueError):
            supervise.validate_args(bad)
    args.benchmark_schedule_sha256 = "0" * 64
    with pytest.raises(ValueError):
        supervise.validate_args(args)
