"""Real CPU sharded models; only physical link is replaced by queues."""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace as S

import mlx.core as mx
from test_phased_cpu import tiny_model
from test_workload_schedule import schedule

mx.set_default_device(mx.cpu)


def test_two_rank_mixed_native_mtp_schedule_has_no_channel_cross_talk(monkeypatch):
    from mtp_head import BoundHead, MTPHead
    from mtp_runtime import run_loaded
    from verification import width_gate

    queues = {0: Queue(), 1: Queue()}

    def send(value, dst, *, group, stream):
        mx.eval(value)
        queues[dst].put(value)
        return value

    def recv(shape, dtype, src, *, group, stream):
        value = queues[group.rank()].get(timeout=10)
        assert value.shape == shape and value.dtype == dtype
        return value

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv", recv)
    small = width_gate(
        tiny_model(),
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9],
        mode="native",
        eval_group_size=32,
        prefix_rollback=True,
    )
    models = [tiny_model(), tiny_model()]
    groups = [S(rank=lambda r=r: r, size=lambda: 2) for r in (0, 1)]
    for model, group in zip(models, groups):
        model.model.pipeline(group, split=[2, 2])
    head = BoundHead(
        MTPHead(models[0].language_model.args),
        models[0].model.embed_tokens,
        models[0].language_model.lm_head,
    )
    mx.eval(head.module.parameters())
    rows = schedule()["requests"]
    rows += [dict(rows[1], id="mtp-r1", repeat=1), dict(rows[0], id="native-r1", repeat=1)]
    events = {0: [], 1: []}
    args = S(
        verifier_mode="native",
        verification_gate_only=False,
        max_tokens=5,
        repeats=4,
        eval_group_size=32,
        target_revision="CPU-fixture",
        prefix_rollback=True,
    )
    tokenizer = S(eos_token_ids=[], decode=lambda ids, **kw: str(ids))

    def run(rank):
        mx.set_default_device(mx.cpu)
        mx.set_default_stream(mx.new_stream(mx.cpu))
        return run_loaded(
            models[rank],
            tokenizer,
            [1, 2, 3],
            args,
            group=groups[rank],
            small={32: small},
            eval_groups=[32] * 4,
            report=events[rank].append,
            head=head if rank == 0 else None,
            request_specs=rows,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, rank) for rank in (0, 1)]
        results = [f.result(timeout=30) for f in futures]
    for ordinal, spec in enumerate(rows):
        a, b = (r["requests"][ordinal] for r in results)
        assert a["token_ids"] == b["token_ids"] == results[0]["requests"][0]["token_ids"]
        assert a["local_rank"] == 0 and b["local_rank"] == 1
        assert a["local_rank_wall_seconds"] != b["local_rank_wall_seconds"]
        for rank in (0, 1):
            active = [e for e in events[rank] if e.get("repeat") == ordinal]
            assert active[0]["phase"] == "request_before" and active[-1]["phase"] == "request_done"
            assert any(e["phase"] == "rank_action" for e in active) == (spec["method"] == "mtp")
    assert queues[0].empty() and queues[1].empty()
