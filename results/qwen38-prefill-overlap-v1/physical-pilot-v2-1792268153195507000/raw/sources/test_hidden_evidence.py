"""Fail-closed hidden-prefill evidence contracts; CPU fixture envelopes only."""

from copy import deepcopy

import pytest


def test_schedule_rejects_hidden_optin_without_matching_actual_flag_and_gates():
    from test_workload_evidence import evidence
    from workload_evidence import verify_schedule_trial

    good = evidence()
    verify_schedule_trial(**good)
    bad = deepcopy(good)
    bad["args"].hidden_only_prefill = True
    bad["args"].hidden_gate_chunk_size = 256
    with pytest.raises(ValueError, match="hidden|Hidden"):
        verify_schedule_trial(**bad)


def test_request_verifier_mode_and_wall_tps_are_bound():
    from test_workload_evidence import evidence
    from workload_evidence import verify_schedule_trial

    good = evidence()
    for field, value, error in (
        ("verifier_mode", "slot-qmv", "per-request verifier"),
        ("wall_tokens_per_second", 0.01, "Wall throughput"),
    ):
        bad = deepcopy(good)
        request = bad["ranks"][1]["requests"][1]
        request[field] = value
        done = next(
            e for e in bad["phases"][1] if e.get("phase") == "request_done" and e.get("repeat") == 1
        )
        done["request_result"] = deepcopy(request)
        with pytest.raises(ValueError, match=error):
            verify_schedule_trial(**bad)


def test_hidden_evidence_validator_rejects_missing_positions_heads_caches_and_continuation():
    from hidden_prefill_gate import full_gate, verify_prefill_evidence
    from test_hidden_gate import make_head
    from test_phased_cpu import tiny_model

    model = tiny_model()
    evidence = full_gate(model, make_head(model), chunk_size=4)
    verify_prefill_evidence(evidence, scope="full", chunk_size=4, layer_count=4)
    for mutate in ("positions", "head", "native", "continuation", "tolerance", "boundary", "norm"):
        bad = deepcopy(evidence)
        row = bad["cases"][0]["chunks"][0]
        if mutate == "positions":
            row["hidden"]["shape"][1] = 1
        elif mutate == "head":
            row["head_caches"] = []
        elif mutate == "native":
            row["native_caches"] = []
        elif mutate == "continuation":
            bad["cases"][0]["continuation"] = []
        elif mutate == "tolerance":
            bad["atol"] = 1e-3
        elif mutate == "boundary":
            bad["cases"][0]["chunks"] = bad["cases"][0]["chunks"][:1]
        else:
            row["hidden"]["strict_pass"] = False
        with pytest.raises(ValueError, match="hidden|Hidden"):
            verify_prefill_evidence(bad, scope="full", chunk_size=4, layer_count=4)
