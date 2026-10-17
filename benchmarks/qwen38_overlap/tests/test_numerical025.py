"""Read-only preserved024 control proves the old width/prefix inventory codec."""

from copy import deepcopy

import pytest

from benchmarks.qwen38_overlap import harness as h

from .test_integration025 import ROOT, RUNTIME, metadata, registry


@pytest.fixture(scope="module")
def preserved_small():
    raw = h.decode_json(
        h.read_bytes(
            ROOT / "results/qwen38-broad-v1/prefill-experiment/candidate-v1/rank0.receipt.json"
        )
    )
    rank = h.decode_json(raw["stdout"].splitlines()[-1].encode())
    return rank["verification"]["groups"]["32"]["small"]


def test_real_preserved024_width_prefix_metrics_against_independent_small_plan(preserved_small):
    from benchmarks.qwen38_overlap import integration025 as i
    from benchmarks.qwen38_overlap.numerical025 import audit_legacy

    plan = i.expected_plan(
        metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"),
        metadata(
            ROOT.parent.parent / ".hermes/cache/scratch/paretoquant-qwen38-preflight/small-control"
        ),
    )
    with i.runtime_api(RUNTIME, registry()) as api:
        audit_legacy(preserved_small, plan["small"]["0"], api)
        for mutation in ("gdn", "kv", "offset", "nan", "logits", "prefix"):
            bad = deepcopy(preserved_small)
            row = bad["windows"][0]
            if mutation == "gdn":
                row["caches"] = [v for v in row["caches"] if v["part"] != "state.0.1"]
            elif mutation == "kv":
                row["caches"] = [v for v in row["caches"] if v["part"] != "values"]
            elif mutation == "offset":
                next(v for v in row["caches"] if v["part"] == "offset")["value"] = 999
            elif mutation == "nan":
                row["logits"]["max_abs"] = float("nan")
            elif mutation == "logits":
                row["logits"]["shape"][1] = 1
            else:
                bad["prefix_rollback"]["prefixes"][1]["caches"].pop()
            with pytest.raises(ValueError):
                audit_legacy(bad, plan["small"]["0"], api)
