"""No previous request cache may survive into the native memory window."""

from types import SimpleNamespace as S

import mlx.core as mx

mx.set_default_device(mx.cpu)


def test_common_gc_is_outside_request_timing_before_next_peak_reset(monkeypatch):
    import gc

    from mtp_head import BoundHead, MTPHead
    from test_phased_cpu import tiny_model
    from test_workload_schedule import schedule
    from verification import Admission, width_gate
    from workload_runtime import run_requests

    model = tiny_model()
    gate = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native", prefix_rollback=True)
    admission = Admission(
        model, "native", gate, gate, {"small": gate, "full": gate}, prefix_rollback=True
    )
    rows = list(reversed(schedule()["requests"]))  # MTP immediately before native.
    events, collections, resets = [], [], []
    real_collect, real_reset = gc.collect, mx.reset_peak_memory

    def collect():
        collections.append(events[-1]["phase"])
        return real_collect()

    def reset():
        resets.append(len(collections))
        return real_reset()

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(mx, "reset_peak_memory", reset)
    result = run_requests(
        model,
        S(eos_token_ids=[], decode=lambda ids, **kw: str(ids)),
        rows,
        S(verifier_mode="native", prefix_rollback=True),
        group=None,
        channel=None,
        head=BoundHead(
            MTPHead(model.language_model.args),
            model.model.embed_tokens,
            model.language_model.lm_head,
        ),
        admissions={1: admission},
        eval_groups=[1, 1],
        report=events.append,
    )
    assert len(result) == 2
    assert collections == ["request_done", "request_done"]
    assert resets == [0, 1]
