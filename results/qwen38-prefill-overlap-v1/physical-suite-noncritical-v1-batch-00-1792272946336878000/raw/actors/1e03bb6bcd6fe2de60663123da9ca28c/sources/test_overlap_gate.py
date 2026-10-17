"""Detached numerical admission validator fixtures; NOT physical results."""

from copy import deepcopy
from types import SimpleNamespace as S

import pytest


def layout():
    from hidden_prefill_gate import architecture_layout

    args = S(
        num_key_value_heads=1,
        head_dim=2,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_num_value_heads=1,
        linear_value_head_dim=2,
        hidden_size=4,
        vocab_size=128,
    )
    result = architecture_layout(
        args, ["ArraysCache", "KVCache"], dtype="float32", head_dtype="float32"
    )
    result.update(rank=0, owned_layers=[2, 4])
    return result


def metric(shape):
    return dict(
        shape=shape,
        dtype="float32",
        finite=True,
        exact=True,
        max_abs=0.0,
        max_tolerance=1e-6,
        max_tolerance_ratio=0.0,
        strict_pass=True,
    )


def parts(spec, offset):
    return [
        dict(
            layer=p["layer"],
            part=p["part"],
            strict_pass=True,
            exact=True,
            max_abs=0.0,
            value=offset,
        )
        if p["part"] == "offset"
        else dict(
            metric([offset if d is None else d for d in p["shape"]]),
            layer=p["layer"],
            part=p["part"],
        )
        for p in spec
    ]


def evidence():
    expected_layout = layout()
    prompt = [1 + i % 100 for i in range(513)]

    def row(offset):
        return dict(
            passed=True,
            caches=parts(expected_layout["target"], offset),
            head_caches=parts(expected_layout["head"], offset),
        )

    chunks = []
    for s, e in ((0, 256), (256, 512)):
        chunks.append(
            dict(
                row(e),
                start=s,
                end=e,
                hidden=metric([1, 256, 4]),
                shifted_tokens=prompt[s + 1 : e + 1],
            )
        )
    final = dict(
        row(513),
        head_caches=parts(expected_layout["head"], 512),
        hidden=metric([1, 1, 4]),
        logits=metric([1, 1, 128]),
        argmax_equal=True,
    )
    cont = [
        dict(
            row(514 + i),
            head_caches=parts(expected_layout["head"], 513 + i),
            hidden=metric([1, 1, 4]),
            logits=metric([1, 1, 128]),
            head_logits=metric([1, 1, 128]),
            argmax_equal=True,
        )
        for i in range(2)
    ]
    return dict(
        schema="qwen38-prefill-overlap-gate-v1",
        passed=True,
        same_process=True,
        physical_gpu=True,
        scope="full",
        modes=["serial", "overlap"],
        chunk_size=256,
        prompt_lengths=[513],
        rank=0,
        owned_layers=[2, 4],
        atol=1e-6,
        rtol=1e-5,
        eval_group_size=32,
        cases=[dict(prompt_tokens=513, chunks=chunks, final=final, continuation=cont)],
    )


def test_trusted_json_metadata_layout_never_imports_mlx(tmp_path):
    import hashlib
    import json

    from overlap_gate import metadata_layout

    config = dict(
        num_hidden_layers=4,
        hidden_size=4,
        vocab_size=128,
        num_key_value_heads=1,
        head_dim=2,
        full_attention_interval=2,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_num_value_heads=1,
        linear_value_head_dim=2,
        dtype="float32",
    )
    data = json.dumps(config).encode()
    (tmp_path / "config.json").write_bytes(data)
    (tmp_path / "model.safetensors.index.json").write_bytes(b"{}")
    options = S(
        local_metadata=str(tmp_path),
        small_model_local=str(tmp_path),
        split=[2, 2],
        target_config_sha256=hashlib.sha256(data).hexdigest(),
        small_config_sha256=hashlib.sha256(data).hexdigest(),
        target_index_sha256=hashlib.sha256(b"{}").hexdigest(),
        small_index_sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    assert metadata_layout(options, scope="full", rank=0) == layout()
    assert metadata_layout(options, scope="small", rank=1)["owned_layers"] == [0, 2]
    native = metadata_layout(options, scope="small", rank=1, native_small=True)
    assert native["owned_layers"] == [0, 4] and native["rank"] == 0
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        metadata_layout(options, scope="full", rank=0)


def test_live_gate_wired_small_two_rank_full513_and_admission_modes():
    from pathlib import Path

    root = Path(__file__).parent
    runtime = (root / "mtp_runtime.py").read_text()
    assert runtime.count("pipeline_gate(") >= 2
    assert 'scope="small"' in runtime and 'scope="full"' in runtime
    assert "overlap_small" in runtime and "overlapped_prefill=overlap_enabled" in runtime
    assert "small_gate(args, eval_groups, report, group=group)" in (root / "probe.py").read_text()
    admission = (root / "verification.py").read_text()
    assert "def require_overlap(" in admission and "require_live_overlap(" in admission


def test_complete_full513_inventory_and_live_capability_not_json_mintable():
    from overlap_gate import require_live_overlap, verify_overlap_gate

    verify_overlap_gate(evidence(), expected_layout=layout(), scope="full", eval_group_size=32)
    with pytest.raises(ValueError, match="live"):
        require_live_overlap(evidence(), S(), 32)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_gdn",
        "missing_kv_value",
        "head_offset",
        "duplicate",
        "shape",
        "bool_shape",
        "bool_shift",
        "dtype",
        "nan",
        "tolerance",
        "not513",
        "bool",
        "missing_chunk",
        "shift",
    ],
)
def test_full513_validator_rejects_false_pass_inventory_and_schedule(mutation):
    from overlap_gate import verify_overlap_gate

    e = evidence()
    row = e["cases"][0]["chunks"][0]
    if mutation == "missing_gdn":
        row["caches"].pop(0)
    elif mutation == "missing_kv_value":
        row["caches"] = [p for p in row["caches"] if p["part"] != "values"]
    elif mutation == "head_offset":
        row["head_caches"][-1]["value"] = 255
    elif mutation == "duplicate":
        row["caches"][1] = deepcopy(row["caches"][0])
    elif mutation == "shape":
        row["hidden"]["shape"] = [1, 255, 4]
    elif mutation == "bool_shape":
        row["hidden"]["shape"] = [True, 256, 4]
    elif mutation == "bool_shift":
        row["shifted_tokens"][99] = True
    elif mutation == "dtype":
        row["hidden"]["dtype"] = "float16"
    elif mutation == "nan":
        row["hidden"]["max_abs"] = float("nan")
    elif mutation == "tolerance":
        e["rtol"] = 1.0
    elif mutation == "not513":
        e["prompt_lengths"] = [257]
    elif mutation == "bool":
        row["caches"][-1]["value"] = True
    elif mutation == "missing_chunk":
        e["cases"][0]["chunks"].pop()
    else:
        row["shifted_tokens"][0] = 999
    with pytest.raises(ValueError):
        verify_overlap_gate(e, expected_layout=layout(), scope="full", eval_group_size=32)
