"""Real CPU receipts, independently supplied native layout, no GPU work."""

from copy import deepcopy

import mlx.core as mx
import pytest
from test_hidden_gate import make_head
from test_phased_cpu import tiny_model

mx.set_default_device(mx.cpu)


@pytest.fixture(scope="module")
def proof():
    from hidden_prefill_gate import full_gate

    model = tiny_model()
    return model, full_gate(model, make_head(model), chunk_size=4)


def rows(evidence):
    for case in evidence["cases"]:
        yield from case["chunks"]
        yield case["final"]
        yield from case["continuation"]


def validate(bad, original):
    from hidden_prefill_gate import trusted_layout, verify_prefill_evidence

    verify_prefill_evidence(
        bad,
        scope="full",
        chunk_size=4,
        layer_count=4,
        expected_layout=trusted_layout(original),
    )


def test_review_reproduction_one_part_per_layer_must_not_admit(proof):
    _, evidence = proof
    validate(evidence, evidence)
    bad = deepcopy(evidence)
    for row in rows(bad):
        for field in ("caches", "native_caches"):
            first = {}
            for part in row[field]:
                first.setdefault(part["layer"], part)
            row[field] = list(first.values())
    with pytest.raises(ValueError, match="hidden|Hidden"):
        validate(bad, evidence)


@pytest.mark.parametrize(
    "field,location",
    [
        (field, location)
        for field in ("caches", "native_caches", "head_caches")
        for location in ("chunk", "final", "continuation", "native256")
        if (field, location) != ("head_caches", "final")
    ],
)
def test_every_individual_part_deletion_duplicate_and_unknown_rejected(proof, field, location):
    _, evidence = proof
    original = evidence["representative_256"] if location == "native256" else evidence
    row = original["cases"][0][
        {
            "chunk": "chunks",
            "native256": "chunks",
            "final": "final",
            "continuation": "continuation",
        }[location]
    ]
    if location != "final":
        row = row[0]

    for index in range(len(row[field])):
        for mutation in ("delete", "duplicate", "unknown", "replace_with_duplicate"):
            bad = deepcopy(evidence)
            owner = bad["representative_256"] if location == "native256" else bad
            target = owner["cases"][0][
                {
                    "chunk": "chunks",
                    "native256": "chunks",
                    "final": "final",
                    "continuation": "continuation",
                }[location]
            ]
            if location != "final":
                target = target[0]
            parts = target[field]
            if mutation == "delete":
                del parts[index]
            elif mutation == "duplicate":
                parts.append(deepcopy(parts[index]))
            elif mutation == "unknown":
                parts[index]["part"] = "unrecognized"
            else:
                parts[index] = deepcopy(parts[(index + 1) % len(parts)])
            with pytest.raises(ValueError, match="hidden|Hidden"):
                validate(bad, evidence)


@pytest.mark.parametrize(
    "field,part",
    [
        (field, part)
        for field in ("caches", "native_caches", "head_caches")
        for part in ("state.0.0", "state.0.1", "keys", "values")
        if not (field == "head_caches" and part.startswith("state"))
    ],
)
@pytest.mark.parametrize(
    "mutation",
    ["nan", "inf", "negative", "huge", "dtype", "shape", "finite", "scaled", "bound", "exact"],
)
def test_forged_strict_pass_does_not_hide_bad_metric(proof, field, part, mutation):
    _, evidence = proof

    bad = deepcopy(evidence)
    metric = next(p for p in bad["cases"][0]["chunks"][0][field] if p["part"] == part)
    assert metric["strict_pass"] is True
    if mutation in ("nan", "inf", "negative", "huge"):
        metric["max_abs"] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -1,
            "huge": 1e20,
        }[mutation]
    elif mutation == "dtype":
        metric["dtype"] = "int32"
    elif mutation == "shape":
        metric["shape"][-1] += 1
    elif mutation == "finite":
        metric["finite"] = False
    elif mutation == "scaled":
        metric["max_tolerance_ratio"] = 1.01
    elif mutation == "bound":
        metric["max_tolerance"] = -1
    else:
        metric["exact"] = True
        metric["max_abs"] = 1e-8
    with pytest.raises(ValueError, match="hidden|Hidden"):
        validate(bad, evidence)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_native_metric_rejects_nonfinite_source_arrays(value):
    from verification import metric

    x = mx.array([value], mx.float32)
    assert metric(x, x)["strict_pass"] is False


@pytest.mark.parametrize("mutation", ["float", "bool", "inexact", "error", "missing_value"])
def test_offsets_are_exact_integral_receipts(proof, mutation):
    _, evidence = proof
    bad = deepcopy(evidence)
    for field in ("caches", "native_caches", "head_caches"):
        p = next(p for p in bad["cases"][0]["chunks"][0][field] if p["part"] == "offset")
        if mutation == "float":
            p["value"] = float(p["value"])
        elif mutation == "bool":
            p["value"] = True
        elif mutation == "inexact":
            p["exact"] = False
        elif mutation == "error":
            p["max_abs"] = 1
        else:
            del p["value"]
    with pytest.raises(ValueError, match="hidden|Hidden"):
        validate(bad, evidence)


def test_serialized_receipt_uses_independently_supplied_full_native_architecture(proof):
    import json

    from hidden_prefill_gate import trusted_layout, verify_prefill_evidence

    _, original = proof
    detached = json.loads(json.dumps(original))
    verify_prefill_evidence(
        detached,
        scope="full",
        chunk_size=4,
        layer_count=4,
        expected_layout=trusted_layout(original),
    )
    with pytest.raises(ValueError):
        verify_prefill_evidence(detached, scope="full", chunk_size=4, layer_count=4)


def test_admission_rejects_truncated_parts_on_live_receipts_and_other_model(proof):
    from hidden_prefill_gate import prefill_gate
    from mlx_backend import MTPBackend
    from verification import Admission, width_gate

    model, full_hidden = proof
    width = width_gate(model, [1, 2, 3], [4, 5, 6, 7, 8, 9], mode="native")
    small_hidden = prefill_gate(model, make_head(model), prompt_lengths=[256, 257, 511, 512, 513])
    small = dict(width, hidden_only_prefill=small_hidden)
    full = dict(width, hidden_only_prefill=full_hidden)
    peer = {"small": small, "full": full}

    def admit(target=model):
        return Admission(
            target, "native", small, full, peer, hidden_only_prefill=True, hidden_gate_chunk_size=4
        )

    MTPBackend(model, make_head(model), admission=admit(), hidden_only_prefill=True)
    with pytest.raises(ValueError, match="hidden|Hidden"):
        admit(tiny_model())
    saved_owned = full["owned_layers"]
    full["owned_layers"] = [10, 14]
    try:
        with pytest.raises(ValueError, match="hidden|Hidden"):
            admit()
    finally:
        full["owned_layers"] = saved_owned
    original_native256 = full_hidden["representative_256"]
    full_hidden["representative_256"] = deepcopy(original_native256)
    try:
        with pytest.raises(ValueError, match="hidden|Hidden"):
            admit()
    finally:
        full_hidden["representative_256"] = original_native256
    for scope in (small_hidden, full_hidden, full_hidden["representative_256"]):
        for row in rows(scope):
            for field in ("caches", "native_caches", "head_caches"):
                if field not in row:
                    continue
                original_parts = row[field]
                for index in range(len(original_parts)):
                    row[field] = original_parts[:index] + original_parts[index + 1 :]
                    try:
                        with pytest.raises(ValueError, match="hidden|Hidden"):
                            admit()
                    finally:
                        row[field] = original_parts


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("scope", ["small", "full"])
def test_hash_bound_metadata_layout_matches_warm_native_classes_shapes_and_dtypes(
    tmp_path, dtype, scope
):
    import hashlib
    import json
    from dataclasses import asdict
    from types import SimpleNamespace as S

    import mlx.nn as nn
    from hidden_prefill_gate import (
        full_gate,
        metadata_layout,
        trusted_layout,
        verify_prefill_evidence,
    )

    model = tiny_model()
    model.set_dtype(getattr(mx, dtype))
    head = make_head(model)
    nn.quantize(head.module, group_size=64, bits=4)
    if scope == "full":
        head.module.set_dtype(getattr(mx, dtype))
    config = asdict(model.language_model.args)
    config["dtype"] = dtype
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    config_sha = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    index_sha = hashlib.sha256((tmp_path / "model.safetensors.index.json").read_bytes()).hexdigest()
    options = S(
        local_metadata=str(tmp_path),
        small_model_local=str(tmp_path),
        target_config_sha256=config_sha,
        target_index_sha256=index_sha,
        small_config_sha256=config_sha,
        small_index_sha256=index_sha,
    )
    layout = metadata_layout(options, scope=scope, owned_layers=[0, 4], rank=0)
    # Reduced full workload has the same cache layout as small; the independent
    # CPU dtype test need not repeat all five small boundary cases again.
    receipt = full_gate(model, head, chunk_size=4)
    assert layout == trusted_layout(receipt)
    verify_prefill_evidence(
        receipt, scope="full", chunk_size=4, layer_count=4, expected_layout=layout
    )
    (tmp_path / "config.json").write_text(json.dumps(dict(config, hidden_size=128)))
    with pytest.raises(ValueError, match="digest"):
        metadata_layout(options, scope=scope, owned_layers=[0, 4], rank=0)
