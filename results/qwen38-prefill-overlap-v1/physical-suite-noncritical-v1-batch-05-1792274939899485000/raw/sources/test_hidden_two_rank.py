"""Real CPU two-rank models; only the physical link uses queues.

Loading/watchdog envelopes below are UNIT fixtures, not physical-run evidence.
Every numerical gate, generated token, cache and op/ACK is actually executed.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from types import SimpleNamespace as S

import mlx.core as mx
import numpy as np
import pytest
from test_hidden_gate import make_head
from test_phased_cpu import tiny_model
from test_workload_schedule import schedule

mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("gate_width", [4, 256])
def test_two_rank_optin_reference_native_outputs_strict_gates_and_schedule_verification(
    monkeypatch, gate_width, tmp_path
):
    from hidden_prefill_gate import prefill_gate
    from mlx.utils import tree_flatten
    from mtp_runtime import run_loaded
    from remote_actor import source_hashes
    from verification import width_gate
    from workload_evidence import verify_schedule_trial

    queues = {0: Queue(), 1: Queue()}

    def send(value, dst, *, group, stream):
        mx.eval(value)
        queues[dst].put(value)
        return value

    def recv(shape, dtype, src, *, group, stream):
        value = queues[group.rank()].get(timeout=15)
        assert value.shape == shape and value.dtype == dtype
        return value

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv", recv)
    small = {}
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    config = asdict(tiny_model().language_model.args)
    config["dtype"] = "float32"
    (metadata / "config.json").write_text(json.dumps(config))
    (metadata / "model.safetensors.index.json").write_text("{}")
    config_sha = hashlib.sha256((metadata / "config.json").read_bytes()).hexdigest()
    index_sha = hashlib.sha256((metadata / "model.safetensors.index.json").read_bytes()).hexdigest()
    for rank in (0, 1):
        sm = tiny_model()
        small[rank] = width_gate(
            sm,
            [1, 2, 3],
            [4, 5, 6, 7, 8, 9],
            mode="native",
            eval_group_size=32,
            prefix_rollback=True,
        )
        small[rank]["hidden_only_prefill"] = prefill_gate(
            sm, make_head(sm), prompt_lengths=[256, 257, 511, 512, 513], eval_group_size=32
        )
        small[rank]["metadata_sha256"] = {
            "config.json": config_sha,
            "model.safetensors.index.json": index_sha,
        }
    models = [tiny_model(), tiny_model()]
    groups = [S(rank=lambda r=r: r, size=lambda: 2) for r in (0, 1)]
    for model, group in zip(models, groups):
        model.model.pipeline(group, split=[2, 2])
    head = make_head(models[0])
    mx.eval(head.module.parameters())
    prompt = [1 + i % 100 for i in range(257)]
    rows = [dict(r, prompt="tokens:" + str(prompt)) for r in schedule()["requests"]]
    rows += [
        dict(r, id=r["id"] + "-single", case_id="single", prompt="tokens:[1]") for r in rows[:]
    ]
    hashes = source_hashes(Path(__file__).parent)
    args = S(
        verifier_mode="native",
        verification_gate_only=False,
        max_tokens=5,
        repeats=4,
        eval_group_size=32,
        eval_group_schedule=None,
        target_revision="CPU-fixture",
        prefix_rollback=True,
        hidden_gate_chunk_size=gate_width,
        rank0_verifier="native",
        rank1_verifier="native",
        target_config_sha256=config_sha,
        target_index_sha256=index_sha,
        small_config_sha256=config_sha,
        small_index_sha256=index_sha,
        local_metadata=str(metadata),
        small_model_local=str(metadata),
        benchmark_schedule_sha256=hashes["benchmark_schedule.json"],
    )
    by_mode = {}
    for enabled in (False, True):
        args.hidden_only_prefill = enabled
        began = time.perf_counter()
        events = {
            r: [
                {
                    "phase": "process_start",
                    "rank": r,
                    "source_sha256": hashes,
                    "hidden_only_prefill": enabled,
                    "hidden_gate_chunk_size": gate_width,
                }
            ]
            for r in (0, 1)
        }

        def run(rank):
            mx.set_default_device(mx.cpu)
            mx.set_default_stream(mx.new_stream(mx.cpu))

            def report(event):
                events[rank].append(
                    {**event, "rank": rank, "elapsed_seconds": time.perf_counter() - began}
                )

            return run_loaded(
                models[rank],
                S(eos_token_ids=[], decode=lambda ids, **kw: str(ids)),
                [1, 2, 3],
                args,
                group=groups[rank],
                small={32: small[rank]},
                eval_groups=[32] * 4,
                report=report,
                head=head if rank == 0 else None,
                request_specs=rows,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, r) for r in (0, 1)]
            results = [f.result(timeout=60) for f in futures]
        assert queues[0].empty() and queues[1].empty()
        for ordinal in range(4):
            a, b = (r["requests"][ordinal] for r in results)
            assert a["token_ids"] == b["token_ids"]
        assert results[0]["requests"][0]["token_ids"] == results[0]["requests"][1]["token_ids"]
        assert results[0]["requests"][2]["token_ids"] == results[0]["requests"][3]["token_ids"]
        for rank in (0, 1):
            for ordinal in (1, 3):
                ops = [
                    e["op"]
                    for e in events[rank]
                    if e.get("repeat") == ordinal and e["phase"] == "rank_action"
                ]
                expected = (
                    ["reset"]
                    + (["prefill_hidden" if enabled else "forward"] if ordinal == 1 else [])
                    + ["forward"]
                )
                assert ops[: len(expected)] == expected and ops[-1] == "finish"
                if ordinal == 3:
                    assert "prefill_hidden" not in ops
        by_mode[enabled] = results
        # Exact numerical/phase records in a declared controller-envelope fixture.
        inventory = {}
        for rank, result in enumerate(results):
            parameters = tree_flatten(models[rank].parameters())
            layout = [{"name": k, "bytes": v.nbytes} for k, v in parameters]
            tensor_hashes = {
                k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in parameters
            }
            inventory[rank] = frozenset(tensor_hashes)
            result.update(
                rank=rank,
                world_size=2,
                execution="phased",
                eval_group_size=32,
                eval_group_schedule=[],
                prompt="gate seed",
                formatted_prompt="gate seed",
                layer_start=models[rank].model.start_idx,
                layer_end=models[rank].model.end_idx or 4,
                parameter_bytes=sum(p["bytes"] for p in layout),
                parameter_layout=layout,
                weight_loading=dict(
                    tensor_sha256=tensor_hashes,
                    all_retained_tensors_loaded=True,
                    tensor_count=len(layout),
                    parameter_bytes=sum(p["bytes"] for p in layout),
                ),
                target_binding={
                    "config.json": config_sha,
                    "model.safetensors.index.json": index_sha,
                },
                target_revision="CPU-fixture",
                verifier_mode="native",
                prefix_rollback=True,
                hidden_only_prefill=enabled,
                hidden_gate_chunk_size=gate_width,
                benchmark_schedule_sha256=args.benchmark_schedule_sha256,
            )
        proof = dict(
            ranks=results,
            phases=events,
            rows=rows,
            prompt_ids=[prompt, prompt, [1], [1]],
            args=args,
            source_sha256=hashes,
            expected_inventory=inventory,
            server=dict(
                completed=True,
                rank=0,
                selected_tensor_count=len(results[0]["parameter_layout"]),
                selected_bytes=results[0]["parameter_bytes"],
            ),
            weight_files=[],
            receipts=[dict(reason="completed", cleanup=dict(child_reaped=True, group_gone=True))]
            * 3,
        )
        verified = verify_schedule_trial(**proof)
        assert verified["paired_greedy_tokens_and_stop_agree"]
        if enabled:
            # Controller must also validate detached process receipts against
            # trusted checkpoint metadata, not a same-interpreter registry.
            verify_schedule_trial(**deepcopy(proof))
            for mutate in (
                "op",
                "flag",
                "head",
                "peer",
                "hidden",
                "sequence",
                "finish",
                "projection",
                "layer",
            ):
                bad = deepcopy(proof)
                if mutate == "op":
                    e = next(e for e in bad["phases"][1] if e.get("op") == "prefill_hidden")
                    e["op"] = "forward"
                elif mutate == "flag":
                    bad["ranks"][1]["hidden_only_prefill"] = False
                elif mutate == "head":
                    bad["ranks"][1]["verification"]["groups"][32]["full"]["hidden_only_prefill"][
                        "cases"
                    ][0]["chunks"][0]["head_caches"] = []
                elif mutate == "peer":
                    bad["ranks"][1]["verification"]["groups"][32]["peer"]["full"][
                        "hidden_only_prefill"
                    ]["passed"] = False
                elif mutate == "hidden":
                    bad["ranks"][0]["verification"]["groups"][32]["full"]["hidden_only_prefill"][
                        "cases"
                    ][0]["chunks"][0]["hidden"]["shape"][1] = 1
                elif mutate == "sequence":
                    next(e for e in bad["phases"][1] if e.get("op") == "prefill_hidden")[
                        "sequence"
                    ] += 1
                elif mutate == "finish":
                    bad["phases"][1] = [e for e in bad["phases"][1] if e.get("op") != "finish"]
                elif mutate == "projection":
                    i = next(
                        i for i, e in enumerate(bad["phases"][0]) if e.get("op") == "prefill_hidden"
                    )
                    j = max(
                        j
                        for j, e in enumerate(bad["phases"][0][:i])
                        if e.get("phase") == "hidden_norm_done"
                    )
                    bad["phases"][0][j]["phase"] = "head_done"
                else:
                    i = next(
                        i for i, e in enumerate(bad["phases"][1]) if e.get("op") == "prefill_hidden"
                    )
                    j = max(
                        j
                        for j, e in enumerate(bad["phases"][1][:i])
                        if e.get("phase") == "layer_done"
                    )
                    bad["phases"][1].pop(j)
                with pytest.raises(ValueError):
                    verify_schedule_trial(**bad)
    for ordinal in range(4):
        a, b = (by_mode[mode][0]["requests"][ordinal] for mode in (False, True))
        assert all(
            a[k] == b[k]
            for k in ("token_ids", "finish_reason", "accepted_drafts", "teacher_forwards")
        )
