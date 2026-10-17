"""TEST symbolic evaluated seams using actual025 producer helper APIs, not GPUs."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from .test_integration025 import RUNTIME, registry


def producer_request(api, rank, spec, prompt, layout):
    """Construct explicit TEST data; run actual pipeline/credit codec helpers."""
    nonce = "b" * 32
    command = api("rank_control").Command(nonce, 1, "prefill_pipeline", prompt)
    events, credits = [], []
    lo, hi = layout["owned_layers"]

    def forward(tokens, *, progress, receive_stage=None):
        if rank == 0:
            progress({"phase": "handoff_receive_before"})
            progress(
                {
                    "phase": "handoff_receive_done",
                    "shape": [1, len(tokens), layout["hidden_size"]],
                    "dtype": layout["dtype"],
                }
            )
            receive_stage()
        else:
            progress({"phase": "embedding_before"})
            progress({"phase": "embedding_done"})
        for start in range(lo, hi, 32):
            end = min(start + 32, hi)
            progress({"phase": "block_before", "start_idx": start, "end_idx": end})
            progress({"phase": "block_done", "start_idx": start, "end_idx": end})
            for layer in range(start, end):
                progress({"phase": "layer_done", "layer_index": layer})
        if rank == 0:
            progress({"phase": "hidden_norm_before"})
            progress({"phase": "hidden_norm_done"})
            return None, "TEST_SYMBOLIC_HIDDEN_NO_MODEL_ARRAY"
        progress({"phase": "handoff_send_before"})
        progress(
            {
                "phase": "handoff_send_done",
                "shape": [1, len(tokens), layout["hidden_size"]],
                "dtype": layout["dtype"],
            }
        )

    module = api("overlap_prefill")
    if rank == 0:
        module.run_rank0_pipeline(
            command,
            SimpleNamespace(send=credits.append),
            forward,
            lambda *args: None,
            events.append,
        )
    else:
        credits = [module.credit(command, *row) for row in module.chunks(prompt)]
        module.run_rank1_pipeline(
            command, SimpleNamespace(receive=lambda: credits.pop(0)), forward, events.append
        )
    result = {
        **spec,
        "repeat": 0,
        "sample_repeat": spec["repeat"],
        "local_rank": rank,
        "token_ids": [248044],
        "finish_reason": "stop",
        "verifier_mode": ["native", "slot-qmv"][rank],
        "wall_seconds": 2.0,
        "first_token_seconds": 1.0,
        "wall_tokens_per_second": 0.5,
        "post_first_tokens_per_second": None,
        "accepted_drafts": 0,
        "verification_rounds": 0,
        "teacher_forwards": 0,
        "request_nonce": nonce,
        "cache_epoch": nonce,
        "prompt_token_ids": prompt,
        "local_peak_mlx_bytes": 100 + rank,
        "local_rank_wall_seconds": 3.0 + rank,
    }
    common = {
        "id": spec["id"],
        "prefill_mode": "overlap",
        "source_request_id": spec["source_request_id"],
        "repeat": 0,
        "rank": rank,
    }

    def action(sequence, op, tokens):
        return {
            **common,
            "phase": "rank_action",
            "request": nonce,
            "sequence": sequence,
            "op": op,
            "tokens": tokens,
            "ok": True,
        }

    suffix = []
    if rank == 0:
        suffix.append(
            {
                "phase": "pipeline_drained",
                "request_nonce": nonce,
                "command_sequence": 1,
                "prefill_mode": "overlap",
                "pending_handoffs": 0,
                "rank": 0,
            }
        )
    events = [
        action(0, "reset", []),
        *events,
        *suffix,
        action(1, "prefill_pipeline", prompt),
        {"phase": "handoff_receive_before", "rank": rank},
        action(2, "forward", [prompt[-1]]),
        action(3, "finish", []),
    ]
    for event in events:
        event.update(common)
    return result, events


def test_source_bound_raw_request_real_codec_both_ranks_and_resealed_mutations():
    from benchmarks.qwen38_overlap import integration025 as i

    spec = {
        "id": "TEST-row",
        "case_id": "TEST-case",
        "method": "mtp",
        "prompt": "TEST",
        "max_tokens": 32,
        "warmup": True,
        "repeat": 0,
        "prefill_mode": "overlap",
        "source_request_id": "TEST-slot",
    }
    prompt = [n % 100 for n in range(513)]
    with i.runtime_api(RUNTIME, registry()) as api:
        for rank in (0, 1):
            layout = {
                "owned_layers": [36, 64] if rank == 0 else [0, 36],
                "hidden_size": 5120,
                "dtype": "bfloat16",
            }
            result, events = producer_request(api, rank, spec, prompt, layout)
            assert (
                i.audit_raw_request(spec, result, events, prompt, layout, rank, api)["chunks"] == 2
            )
            for mutation in ("credit", "head_shift", "mode", "nonce", "counter"):
                bad, row = deepcopy(events), deepcopy(result)
                if mutation in ("credit", "head_shift"):
                    if mutation == "head_shift" and rank == 1:
                        continue
                    state = api("overlap_prefill").StreamState(
                        api("rank_control").Command("b" * 32, 1, "prefill_pipeline", prompt),
                        rank,
                        lambda _: None,
                    )
                    for event in bad:
                        if "stream_payload" not in event:
                            continue
                        payload = dict(event["stream_payload"])
                        if mutation == "credit" and "credit" in payload["phase"]:
                            payload["phase"] = "TEST_wrong_credit"
                        if mutation == "head_shift" and payload["phase"] == "pipeline_head_done":
                            payload["shifted_tokens"] = [999] * len(payload["shifted_tokens"])
                        rebuilt = []
                        state.progress = rebuilt.append
                        state.emit(payload)
                        event.clear()
                        event.update(
                            rebuilt[0],
                            id=spec["id"],
                            source_request_id=spec["source_request_id"],
                            repeat=0,
                        )
                elif mutation == "mode":
                    row["prefill_mode"] = "serial"
                elif mutation == "nonce":
                    row["request_nonce"] = "c" * 32
                else:
                    row["accepted_drafts"] = 1
                with pytest.raises(ValueError):
                    i.audit_raw_request(spec, row, bad, prompt, layout, rank, api)
