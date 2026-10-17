"""Scheduling/admission tests; synthetic receipts are not benchmark evidence."""

import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def args_for(*extra):
    from supervise import parser

    return parser().parse_args(
        [
            "--remote-root",
            "/a/b/spikes/012-qwen38-scheduling",
            "--remote-source",
            "/model",
            "--local-metadata",
            "/metadata",
            "--output",
            "/out",
            "--ssh-key",
            "/key",
            "--split",
            "2",
            "3",
            *extra,
        ]
    )


def test_schedule_admission_and_cli_forwarding():
    from phased_decode import validate_eval_schedule
    from probe import phased_options
    from supervise import probe_options, validate_args

    assert validate_eval_schedule(4, None, 3, True) == [4, 4, 4]
    assert validate_eval_schedule(4, [], 3, True) == [4, 4, 4]
    assert validate_eval_schedule(4, [1, 32, 16], 3, True) == [1, 32, 16]
    assert phased_options(SimpleNamespace(phased=True, eval_group_size=32, repeats=1)) == {
        "eval_group_size": 32
    }
    args = args_for(
        "--repeats", "3", "--eval-group-size", "4", "--eval-group-schedule", "1", "32", "16"
    )
    validate_args(args)
    options = probe_options(args)
    start = options.index("--eval-group-schedule")
    assert options[start + 1 : start + 4] == ["1", "32", "16"]
    assert "--phased" in options
    validate_args(args_for("--eval-group-schedule"))


@pytest.mark.parametrize(
    "size,schedule,repeats,phased",
    [
        (1, [3], 1, True),
        (1, [True], 1, True),
        (1, [1], 2, True),
        (1, [1], 1, False),
        (1, None, 33, True),
        (1, [], 0, True),
        (True, None, 1, True),
        (4, None, 1, False),
        (1, "1", 1, True),
    ],
)
def test_schedule_rejects_invalid_admission(size, schedule, repeats, phased):
    from phased_decode import validate_eval_schedule

    with pytest.raises(ValueError):
        validate_eval_schedule(size, schedule, repeats, phased)


def evidence():
    from remote_actor import source_hashes

    hashes = source_hashes(Path(__file__).parent)
    inventory, ranks, phases = {}, [], {}
    for rank in (0, 1):
        layer = 1 - rank
        names = {
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "language_model.lm_head.weight",
            f"language_model.model.layers.{layer}.weight",
        }
        inventory[rank] = names
        tensors = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        ranks.append(
            dict(
                rank=rank,
                world_size=2,
                eval_group_size=4,
                eval_group_schedule=[1, 32],
                execution="phased",
                prompt="p",
                formatted_prompt="p",
                layer_start=layer,
                layer_end=layer + 1,
                parameter_bytes=len(names) * 4,
                parameter_layout=[dict(name=n, bytes=4) for n in sorted(names)],
                weight_loading=dict(
                    all_retained_tensors_loaded=True,
                    tensor_count=len(names),
                    tensor_sha256=tensors,
                    parameter_bytes=len(names) * 4,
                ),
                requests=[
                    dict(repeat=i, eval_group_size=g, prompt_tokens=2, token_ids=[7, 8])
                    for i, g in enumerate([1, 32])
                ],
            )
        )
        phases[rank] = [dict(phase="process_start", rank=rank, source_sha256=hashes)]
        for i, g in enumerate([1, 32]):
            phases[rank].extend(
                dict(phase=p, rank=rank, repeat=i, eval_group_size=g)
                for p in ("request_before", "token_done", "request_done")
            )
    server = dict(completed=True, rank=0, selected_tensor_count=4, selected_bytes=16)
    receipts = [
        dict(reason="completed", cleanup=dict(child_reaped=True, group_gone=True)) for _ in range(3)
    ]
    return (
        ranks,
        server,
        receipts,
        dict(
            repeats=2,
            expected_inventory=inventory,
            eval_group_size=4,
            eval_group_schedule=[1, 32],
            phases=phases,
            source_sha256=hashes,
            execution="phased",
        ),
    )


def test_verifier_binds_admitted_schedule_and_source_phases():
    from supervise import verify_trial

    ranks, server, receipts, options = evidence()
    assert verify_trial(ranks, server, receipts, [], **options)["tokens_agree"]


@pytest.mark.parametrize(
    "tamper",
    [
        "request",
        "both_ranks",
        "order",
        "rank_schedule",
        "phase",
        "missing_phase",
        "phase_order",
        "source",
        "missing_source",
        "execution",
        "top_level",
        "missing_request_group",
        "admitted_schedule",
    ],
)
def test_verifier_rejects_schedule_tampering(tamper):
    from supervise import verify_trial

    ranks, server, receipts, options = copy.deepcopy(evidence())
    if tamper == "request":
        ranks[0]["requests"][1]["eval_group_size"] = 16
    elif tamper == "both_ranks":
        for r in ranks:
            r["requests"][1]["eval_group_size"] = 16
    elif tamper == "order":
        ranks[0]["requests"].reverse()
    elif tamper == "rank_schedule":
        ranks[0]["eval_group_schedule"] = [32, 1]
    elif tamper == "phase":
        options["phases"][0][-1]["eval_group_size"] = 16
    elif tamper == "missing_phase":
        options["phases"][0].pop()
    elif tamper == "phase_order":
        options["phases"][0][1], options["phases"][0][3] = (
            options["phases"][0][3],
            options["phases"][0][1],
        )
    elif tamper == "source":
        options["phases"][0][0]["source_sha256"] = {}
    elif tamper == "missing_source":
        options["source_sha256"] = None
    elif tamper == "execution":
        for r in ranks:
            r["execution"] = "upstream"
    elif tamper == "top_level":
        ranks[0]["eval_group_size"] = 1
    elif tamper == "missing_request_group":
        ranks[0]["requests"][0].pop("eval_group_size")
    elif tamper == "admitted_schedule":
        options["eval_group_schedule"] = [1, 16]
    with pytest.raises(ValueError):
        verify_trial(ranks, server, receipts, [], **options)


@pytest.mark.parametrize("script", ["probe.py", "supervise.py"])
@pytest.mark.parametrize(
    "extra",
    [
        ["--eval-group-schedule", "3"],
        ["--eval-group-schedule", "1", "2", "--repeats", "1"],
        ["--repeats", "33"],
    ],
)
def test_invalid_cli_fails_before_any_work(script, extra):
    root = Path(__file__).parent
    base = (
        ["--mode", "generate", "--model", "/nonexistent", "--phased"]
        if script == "probe.py"
        else [
            "--remote-root",
            "/none",
            "--remote-source",
            "/none",
            "--local-metadata",
            "/none",
            "--output",
            "/none",
            "--ssh-key",
            "/none",
            "--split",
            "2",
            "3",
        ]
    )
    result = subprocess.run(
        [sys.executable, str(root / script), *base, *extra],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "process_start" not in result.stderr


def test_source_registry_unchanged():
    from remote_actor import SOURCES

    assert set(SOURCES) == {
        "probe.py",
        "stream_weights.py",
        "stream_server.py",
        "shard_plan.py",
        "phased_decode.py",
        "cpu_communication.py",
        "remote_actor.py",
    }
