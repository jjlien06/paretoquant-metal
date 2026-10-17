"""Source-pinned integration tests. No MLX, child, transport or physical approval."""

import importlib
from pathlib import Path

import pytest

from benchmarks.qwen38_overlap import harness as h

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "spikes/025-qwen38-overlapped-prefill"


def registry():
    # Read the actual constant without executing an arbitrary producer module.
    import ast

    tree = ast.parse((RUNTIME / "remote_actor.py").read_text())
    names = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SOURCES" for t in n.targets)
    )
    return {name: h.sha(h.read_bytes(RUNTIME / name)) for name in names}


def test_source_pinned_real_public_verifiers_and_independent_layout(tmp_path):
    from benchmarks.qwen38_overlap import integration025 as i

    hashes = registry()
    with i.runtime_api(RUNTIME, hashes) as api:
        assert api("overlap_gate").lengths("small") == [1, 2, 255, 256, 257, 511, 512, 513]
        assert api("supervise").verify_completed_trial.__code__.co_filename == str(
            RUNTIME / "supervise.py"
        )
        assert api("overlap_evidence").verify_overlap_stream.__module__ == "overlap_evidence"
    import sys

    assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
    hashes["overlap_evidence.py"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        with i.runtime_api(RUNTIME, hashes):
            pytest.fail("wrong source reached code execution")
    assert importlib.util.find_spec("benchmarks.qwen38_overlap.integration025") is not None


def metadata(root):
    return {
        "path": str(root),
        "config_sha256": h.sha(h.read_bytes(root / "config.json")),
        "index_sha256": h.sha(h.read_bytes(root / "model.safetensors.index.json")),
    }


def test_architecture_expected_plan_canonical_shapes_and_513_not_257():
    from benchmarks.qwen38_overlap import integration025 as i

    target = metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only")
    small = metadata(
        Path("/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control")
    )
    plan = i.expected_plan(target, small)
    assert plan["schema"] == "qwen38-overlap-numerical-plan-v2"
    assert "passed" not in str(plan)
    with i.runtime_api(RUNTIME, registry()) as api:
        from types import SimpleNamespace

        options = SimpleNamespace(
            local_metadata=target["path"],
            small_model_local=small["path"],
            split=[28, 36],
            target_config_sha256=target["config_sha256"],
            target_index_sha256=target["index_sha256"],
            small_config_sha256=small["config_sha256"],
            small_index_sha256=small["index_sha256"],
        )
        for scope in ("small", "full"):
            for rank in (0, 1):
                assert plan[scope][str(rank)]["overlap_layout"] == api(
                    "overlap_gate"
                ).metadata_layout(options, scope=scope, rank=rank)
    rank0 = plan["full"]["0"]
    assert rank0["overlap_layout"]["owned_layers"] == [36, 64]
    assert rank0["overlap_layout"]["target"][0]["shape"] == [1, 3, 10240]
    assert rank0["overlap_layout"]["target"][1]["shape"] == [1, 48, 128, 128]
    assert rank0["overlap_cases"][0]["intervals"] == [[0, 256], [256, 512]]
    assert rank0["width_windows"][0]["target_offset"] == 7
    assert rank0["prefix_boundaries"][1]["target_offset"] == 5
    target["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        i.expected_plan(target, small)


def test_raw_actor_lossless_identity_pressure_and_incomplete_rejected(tmp_path):
    from benchmarks.qwen38_overlap import integration025 as i
    from benchmarks.qwen38_overlap.cli import encoded

    hashes = registry()
    with i.runtime_api(RUNTIME, hashes) as api:
        from .fixtures025 import actor_documents

        nonce = "a" * 32
        raw, context = actor_documents(api, tmp_path / nonce, hashes)
        raw["unknown_future_field"] = {"must_survive": [1, 2]}
        path = tmp_path / "TEST-rank0.receipt.json"
        path.write_bytes(encoded(raw))
        value = i.read_actor(path, h.sha(path.read_bytes()), hashes, api, expected_context=context)
        assert value["raw"] == raw
        assert value["input_sha256"] == h.sha(path.read_bytes())
        assert value["events"][0]["pid"] == 101
        assert value["actor_nonce"] == nonce
        for field, changed in (("log_evidence_complete", False), ("returncode", True)):
            bad = h.decode_json(encoded(raw))
            bad["result"][field] = changed
            path.write_bytes(encoded(bad))
            with pytest.raises(ValueError):
                i.read_actor(path, h.sha(path.read_bytes()), hashes, api, expected_context=context)
        raw["result"]["pressure"]["samples"][1]["level"] = 2
        path.write_bytes(encoded(raw))
        with pytest.raises(ValueError, match="pressure"):
            i.read_actor(path, h.sha(path.read_bytes()), hashes, api, expected_context=context)
