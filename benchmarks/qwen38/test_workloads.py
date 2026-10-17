"""Authored synthetic workloads, not fabricated model outputs."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name("workloads.py")
    assert path.exists(), "Workload manifest generator has not been implemented"
    spec = importlib.util.spec_from_file_location("qwen38_workloads", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_six_categories_three_lengths_and_paired_counterbalance():
    w = module()

    # A deterministic toy tokenizer exercises scheduling without loading a model.
    def encode(s):
        return list(s.encode("utf-8"))

    def format_prompt(s):
        return "<user>" + s + "</user><assistant>"

    manifest = w.build_manifest(encode, format_prompt, levels=(None, 1024, 2048))
    assert len(manifest["cases"]) == 18
    assert len(manifest["batches"]) == 6
    assert len({c["id"] for c in manifest["cases"]}) == 18
    assert len({c["category"] for c in manifest["cases"]}) == 6
    requests = [r for b in manifest["batches"] for r in b["requests"]]
    assert len(requests) == 84
    assert len({r["id"] for r in requests}) == len(requests)
    for case in manifest["cases"]:
        measured = [r for r in requests if r["case_id"] == case["id"] and not r["warmup"]]
        assert sorted((r["method"], r["repeat"]) for r in measured) == [
            ("mtp", 0),
            ("mtp", 1),
            ("native", 0),
            ("native", 1),
        ]
        assert [r["method"] for r in measured] in (
            ["native", "mtp", "mtp", "native"],
            ["mtp", "native", "native", "mtp"],
        )
        assert all(r["prompt"] == case["prompt"] for r in measured)
        assert all(r["max_tokens"] == case["max_tokens"] for r in measured)
    for batch in manifest["batches"]:
        assert len(batch["requests"]) == 14
        assert sum(r["warmup"] for r in batch["requests"]) == 2
        assert batch["schema"] == "qwen38-workload-schedule-v1"
