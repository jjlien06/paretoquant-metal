"""Dedicated fail-closed schedule verification, genuine CPU fixture evidence."""

from copy import deepcopy
from types import SimpleNamespace as S

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)


def evidence():
    from pathlib import Path

    from mtp_head import BoundHead, MTPHead
    from mtp_runtime import run_loaded
    from remote_actor import source_hashes
    from test_phased_cpu import tiny_model
    from test_workload_schedule import schedule
    from verification import width_gate

    model = tiny_model()
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native", prefix_rollback=True)
    args = S(
        verifier_mode="native",
        verification_gate_only=False,
        max_tokens=5,
        repeats=2,
        eval_group_size=1,
        eval_group_schedule=None,
        target_revision="fixture",
        prefix_rollback=True,
        rank0_verifier="native",
        rank1_verifier="native",
        target_config_sha256="a" * 64,
        target_index_sha256="b" * 64,
        small_config_sha256="c" * 64,
        small_index_sha256="d" * 64,
        benchmark_schedule_sha256="e" * 64,
    )
    rows = schedule()["requests"]
    events = []
    result = run_loaded(
        model,
        S(eos_token_ids=[], decode=lambda ids, **kw: str(ids)),
        [1, 2, 3],
        args,
        group=None,
        small={1: gate},
        eval_groups=[1, 1],
        report=events.append,
        head=BoundHead(
            MTPHead(model.language_model.args),
            model.model.embed_tokens,
            model.language_model.lm_head,
        ),
        request_specs=rows,
    )
    # Distributed receipt envelope fixture, not a physical two-rank/GPU claim.
    hashes = source_hashes(Path(__file__).parent)
    args.benchmark_schedule_sha256 = hashes["benchmark_schedule.json"]
    ranks, phases = [], {}
    names = {f"language_model.model.layers.{i}.weight" for i in range(4)}
    inventory = {0: frozenset(names), 1: frozenset(names)}
    for rank in (0, 1):
        r = deepcopy(result)
        r.update(
            rank=rank,
            world_size=2,
            execution="phased",
            eval_group_size=1,
            eval_group_schedule=[],
            prompt="gate seed",
            formatted_prompt="gate seed",
            layer_start=0,
            layer_end=4,
            parameter_bytes=4,
            parameter_layout=[{"name": n, "bytes": 1} for n in names],
            weight_loading={
                "tensor_sha256": {n: "f" * 64 for n in names},
                "all_retained_tensors_loaded": True,
                "tensor_count": 4,
                "parameter_bytes": 4,
            },
            target_binding={"config.json": "a" * 64, "model.safetensors.index.json": "b" * 64},
            target_revision="fixture",
            verifier_mode="native",
            prefix_rollback=True,
            benchmark_schedule_sha256=args.benchmark_schedule_sha256,
        )
        for pair in r["verification"]["groups"].values():
            pair["small"]["metadata_sha256"] = {
                "config.json": "c" * 64,
                "model.safetensors.index.json": "d" * 64,
            }
        for req in r["requests"]:
            req["local_rank"] = rank
        p = [{"phase": "process_start", "rank": rank, "source_sha256": hashes}]
        for i, e in enumerate(deepcopy(events)):
            e.update(rank=rank, elapsed_seconds=(i + 1) * 0.001)
            if e.get("phase") == "request_done":
                e["request_result"] = deepcopy(r["requests"][e["repeat"]])
            p.append(e)
        phases[rank] = p
        ranks.append(r)
    return dict(
        ranks=ranks,
        phases=phases,
        rows=rows,
        prompt_ids=[[1, 2, 3]] * 2,
        args=args,
        source_sha256=hashes,
        expected_inventory=inventory,
        server={"completed": True, "rank": 0, "selected_tensor_count": 4, "selected_bytes": 4},
        receipts=[{"reason": "completed", "cleanup": {"child_reaped": True, "group_gone": True}}]
        * 3,
        weight_files=[],
    )


def test_dedicated_verifier_preserves_gates_pairs_rank_memory_and_stage_actions():
    from workload_evidence import verify_schedule_trial

    good = evidence()
    summary = verify_schedule_trial(**good)
    assert summary["paired_greedy_tokens_and_stop_agree"]
    assert summary["strict_numerical_admission"]
    assert summary["phase_timing_by_rank"][0]
    for mutate in (
        "tokens",
        "stop",
        "prompt",
        "cap",
        "warmup",
        "sample",
        "peak",
        "boundary",
        "action",
        "gate",
        "inventory",
        "source",
        "small_binding",
        "native_action",
    ):
        bad = deepcopy(good)
        r = bad["ranks"][1]["requests"][0]
        if mutate == "tokens":
            r["token_ids"][0] += 1
        elif mutate == "stop":
            r["finish_reason"] = "stop"
        elif mutate == "prompt":
            r["prompt_token_ids"] = [9]
        elif mutate == "cap":
            r["max_tokens"] = 9
        elif mutate == "warmup":
            r["warmup"] = True
        elif mutate == "sample":
            r["sample_repeat"] = 1
        elif mutate == "peak":
            r["local_rank"] = 0
        elif mutate == "boundary":
            bad["phases"][1] = [e for e in bad["phases"][1] if e["phase"] != "request_done"]
        elif mutate == "action":
            bad["phases"][1] = [e for e in bad["phases"][1] if e.get("op") != "finish"]
        elif mutate == "gate":
            bad["ranks"][1]["verification"]["groups"][1]["full"]["atol"] = 1e-3
        elif mutate == "inventory":
            bad["ranks"][1]["weight_loading"]["tensor_sha256"] = {}
        elif mutate == "source":
            bad["phases"][1][0]["source_sha256"] = {}
        elif mutate == "small_binding":
            bad["ranks"][1]["verification"]["groups"][1]["small"]["metadata_sha256"] = {}
        else:
            action = deepcopy(next(e for e in bad["phases"][1] if e["phase"] == "rank_action"))
            action["repeat"] = 0
            bad["phases"][1].insert(2, action)
        with pytest.raises(ValueError, match=".+"):
            verify_schedule_trial(**bad)


@pytest.mark.parametrize("mutation", ["zero_rounds", "accepted_counter", "verify_current"])
def test_review_generation_progress_mutations_rejected_by_schedule_integration(mutation):
    from workload_evidence import verify_schedule_trial

    good = evidence()
    verify_schedule_trial(**good)
    bad = deepcopy(good)
    # Mutate both owners and their incremental result copies so rank agreement,
    # original receipt binding and old loose action-count checks still agree.
    for rid in (0, 1):
        request = bad["ranks"][rid]["requests"][1]
        events = bad["phases"][rid]
        if mutation == "zero_rounds":
            request.update(verification_rounds=0, teacher_forwards=0, accepted_drafts=0)
            events[:] = [
                e
                for e in events
                if not (
                    e.get("repeat") == 1
                    and e.get("phase") == "rank_action"
                    and e["op"] in ("snapshot", "verify", "commit_prefix")
                )
            ]
            for sequence, e in enumerate(e for e in events if e.get("phase") == "rank_action"):
                e["sequence"] = sequence
        elif mutation == "accepted_counter":
            request["accepted_drafts"] = 1
        else:
            verify = [e for e in events if e.get("op") == "verify"]
            verify[1]["tokens"][0] = verify[0]["tokens"][0]
        next(e for e in events if e.get("phase") == "request_done" and e.get("repeat") == 1)[
            "request_result"
        ] = deepcopy(request)
    with pytest.raises(ValueError, match=".+"):
        verify_schedule_trial(**bad)
