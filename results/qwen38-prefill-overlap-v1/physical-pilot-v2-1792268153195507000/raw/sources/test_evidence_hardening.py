"""Additional fail-closed evidence boundaries (tiny CPU fixture)."""

from copy import deepcopy

import pytest
from test_workload_evidence import evidence


def test_schedule_verifier_binds_registry_digest_and_counter_types():
    from workload_evidence import verify_schedule_trial

    good = evidence()
    verify_schedule_trial(**good)
    for change in (
        "registry",
        "digest",
        "ordinal_type",
        "sample_type",
        "timing",
        "action_width",
        "native_layers",
        "text",
        "cap_tokens",
    ):
        bad = deepcopy(good)
        if change == "registry":
            bad["source_sha256"].pop("workload_runtime.py")
            for p in bad["phases"].values():
                p[0]["source_sha256"] = bad["source_sha256"]
        elif change == "digest":
            bad["args"].benchmark_schedule_sha256 = "a" * 64
            for r in bad["ranks"]:
                r["benchmark_schedule_sha256"] = "a" * 64
        elif change == "ordinal_type":
            bad["ranks"][0]["requests"][0]["repeat"] = False
        elif change == "sample_type":
            bad["ranks"][0]["requests"][0]["sample_repeat"] = False
        elif change == "timing":
            bad["ranks"][0]["requests"][0]["wall_seconds"] = float("nan")
        elif change == "native_layers":
            bad["phases"][0] = [
                e
                for e in bad["phases"][0]
                if not (e.get("repeat") == 0 and e.get("phase") == "layer_done")
            ]
        elif change == "text":
            bad["ranks"][1]["requests"][0]["text"] += "tampered"
        elif change == "cap_tokens":
            bad["ranks"][0]["requests"][0]["token_ids"] *= 5
        else:
            for p in bad["phases"].values():
                next(e for e in p if e.get("op") == "verify")["tokens"] = [1]
        with pytest.raises(ValueError):
            verify_schedule_trial(**bad)
