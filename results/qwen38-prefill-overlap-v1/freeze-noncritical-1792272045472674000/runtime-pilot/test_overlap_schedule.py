"""Versioned two-MTP-mode schedule admission without hardware imports."""

from copy import deepcopy
from types import SimpleNamespace as S

import pytest
from workload_schedule import validate_schedule


def schedule():
    return {
        "schema": "qwen38-prefill-overlap-schedule-v1",
        "requests": [
            dict(
                id="r-" + mode,
                case_id="case",
                method="mtp",
                prefill_mode=mode,
                source_request_id="old-" + mode,
                prompt="exact bytes",
                max_tokens=32,
                warmup=False,
                repeat=0,
            )
            for mode in ("serial", "overlap")
        ],
    }


def test_new_schema_two_mtp_modes_old_schema_unchanged():
    rows = validate_schedule(schedule())
    assert [r["prefill_mode"] for r in rows] == ["serial", "overlap"]
    old = deepcopy(schedule())
    old["schema"] = "qwen38-workload-schedule-v1"
    for r, m in zip(old["requests"], ("native", "mtp")):
        r.pop("prefill_mode")
        r.pop("source_request_id")
        r["method"] = m
    assert len(validate_schedule(old)) == 2


@pytest.mark.parametrize(
    "mutation", ["same_mode", "wrong_method", "duplicate_source", "missing_mode", "extra_field"]
)
def test_new_schema_fail_closed(mutation):
    value = schedule()
    rows = value["requests"]
    if mutation == "same_mode":
        rows[1]["prefill_mode"] = "serial"
    elif mutation == "wrong_method":
        rows[0]["method"] = "native"
    elif mutation == "duplicate_source":
        rows[1]["source_request_id"] = rows[0]["source_request_id"]
    elif mutation == "missing_mode":
        rows[0].pop("prefill_mode")
    else:
        rows[0]["extra"] = True
    with pytest.raises(ValueError):
        validate_schedule(value)


def test_new_schedule_prompt_cap_rejects_before_any_backend_or_model_import():
    from workload_runtime import tokenize_requests

    rows = schedule()["requests"]
    tokenizer = S(
        apply_chat_template=lambda *a, **k: "formatted", encode=lambda *a, **k: [1] * 2049
    )
    with pytest.raises(ValueError, match="2048"):
        tokenize_requests(tokenizer, rows)
    tokenizer.encode = lambda *a, **k: [1] * 2048
    assert [len(ids) for ids in tokenize_requests(tokenizer, rows)] == [2048, 2048]


def test_runtime_missing_enable_flag_rejects_before_mlx_import():
    from workload_runtime import run_requests

    args = S(
        overlapped_prefill=False,
        hidden_only_prefill=True,
        prefix_rollback=True,
        hidden_gate_chunk_size=256,
        require_normal_pressure=True,
    )
    with pytest.raises(ValueError, match="flag"):
        run_requests(
            None,
            None,
            schedule()["requests"],
            args,
            group=None,
            channel=None,
            head=None,
            admissions={},
            eval_groups=[32, 32],
            report=lambda e: None,
        )


def test_overlap_schedule_requires_flag_before_hardware():
    from overlap_prefill import validate_modes

    args = S(
        overlapped_prefill=False,
        hidden_only_prefill=True,
        prefix_rollback=True,
        hidden_gate_chunk_size=256,
        require_normal_pressure=True,
    )
    with pytest.raises(ValueError, match="flag"):
        validate_modes(args, schedule()["requests"])
    args.overlapped_prefill = True
    validate_modes(args, schedule()["requests"])
    args.require_normal_pressure = False
    with pytest.raises(ValueError, match="pressure"):
        validate_modes(args, schedule()["requests"])
