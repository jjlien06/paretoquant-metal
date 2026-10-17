"""Real tiny CPU execution of mixed same-loaded-model requests."""

import hashlib
from types import SimpleNamespace as S

import mlx.core as mx

mx.set_default_device(mx.cpu)


def test_mixed_loaded_schedule_native_is_independent_fresh_and_incremental(monkeypatch):
    from mtp_head import BoundHead, MTPHead
    from mtp_runtime import run_loaded
    from test_phased_cpu import tiny_model
    from test_workload_schedule import schedule
    from verification import width_gate

    model = tiny_model()
    head = BoundHead(
        MTPHead(model.language_model.args), model.model.embed_tokens, model.language_model.lm_head
    )
    small = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native", prefix_rollback=True)
    rows = schedule()["requests"]
    rows += [
        dict(r, id=r["id"] + "2", case_id="b", prompt="tokens:[2,3,4]", max_tokens=2)
        for r in rows[:]
    ]
    args = S(
        verifier_mode="native",
        verification_gate_only=False,
        max_tokens=5,
        repeats=4,
        eval_group_size=1,
        target_revision="fixture",
        prefix_rollback=True,
    )
    events = []
    result = run_loaded(
        model,
        S(eos_token_ids=[], decode=lambda ids, **kw: str(ids)),
        [1, 2, 3],
        args,
        group=None,
        small={1: small},
        eval_groups=[1] * 4,
        report=events.append,
        head=head,
        request_specs=rows,
    )
    requests = result["requests"]
    assert len(requests) == 4
    assert requests[0]["token_ids"] == requests[1]["token_ids"]
    assert requests[2]["token_ids"] == requests[3]["token_ids"]
    for i, (r, spec) in enumerate(zip(requests, rows)):
        assert r["repeat"] == i and r["sample_repeat"] == spec["repeat"]
        assert all(r[k] == spec[k] for k in ("id", "case_id", "method", "max_tokens", "warmup"))
        assert r["prompt_sha256"] == hashlib.sha256(spec["prompt"].encode()).hexdigest()
        assert r["prompt_token_ids"] == ([1, 2, 3] if i < 2 else [2, 3, 4])
        assert r["local_peak_mlx_bytes"] > 0 and r["local_rank"] == 0
        assert r["wall_seconds"] > r["first_token_seconds"] > 0
        assert r["post_first_tokens_per_second"] > 0
        active = [e for e in events if e.get("repeat") == i]
        assert active[0]["phase"] == "request_before"
        assert active[-1]["phase"] == "request_done"
        assert active[-1]["request_result"] == r
        assert bool([e for e in active if e["phase"] == "rank_action"]) == (spec["method"] == "mtp")
        if spec["method"] == "native":
            assert (r["accepted_drafts"], r["verification_rounds"], r["teacher_forwards"]) == (
                0,
                0,
                0,
            )
    assert events.index(
        next(e for e in events if e["phase"] == "numerical_admission_done")
    ) < events.index(next(e for e in events if e["phase"] == "request_before"))
