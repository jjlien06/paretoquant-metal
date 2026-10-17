"""CPU-only rounded-cast diagnostics. No quality or physical-host claim."""

import hashlib

import pytest

pytest_plugins = ["test_compute_dtype"]


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_rounded_preserves_bf16_a_log_exactly_and_casts_only_norm(cpu_model, dtype_name):
    import mlx.core as mx
    import numpy as np
    from compute_dtype import apply_compute_dtype
    from mlx.utils import tree_flatten

    cpu_model.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
    dtype = getattr(mx, dtype_name)
    cpu_model.A_log = mx.array([0.2, 1e-9], dtype)
    cpu_model.norm = mx.array([0.125, 1e-9], mx.bfloat16)
    cpu_model.cast_predicate = lambda path: not path.endswith("A_log")
    before = dict(tree_flatten(cpu_model.parameters()))
    bits = np.asarray(before["A_log"].view(mx.uint16)).tobytes()
    digest = hashlib.sha256(bits).hexdigest()

    result = apply_compute_dtype(cpu_model, "float16-rounded")

    after = dict(tree_flatten(cpu_model.parameters()))
    assert after["A_log"] is before["A_log"]
    assert after["A_log"].dtype == dtype
    assert np.asarray(after["A_log"].view(mx.uint16)).tobytes() == bits
    assert (
        hashlib.sha256(np.asarray(after["A_log"].view(mx.uint16)).tobytes()).hexdigest() == digest
    )
    assert after["norm"].dtype == mx.float16
    assert all(after[n] is a for n, a in before.items() if n != "norm")
    assert result["converted_tensor_names"] == ["norm"]
    assert set(result["converted_tensor_sha256"]) == {"norm"}
    assert set(result["conversion_diagnostics"]) == {"norm"}
    assert result["conversion_diagnostics"]["norm"]["element_count"] == 2
    source = {r["name"]: r for r in result["source_parameter_layout"]}
    runtime = {r["name"]: r for r in result["runtime_parameter_layout"]}
    assert source["A_log"] == runtime["A_log"]


def test_tiny_conv_exact_rejects_rounded_records_drift_and_runs(cpu_model):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from compute_dtype import apply_compute_dtype

    cpu_model.conv1d = nn.Conv1d(1, 1, 1, bias=False)
    cpu_model.conv1d.weight = mx.array([[[1e-9]]], mx.bfloat16)
    original = cpu_model.conv1d.weight
    with pytest.raises(ValueError, match="roundtrip is not exact: conv1d.weight"):
        apply_compute_dtype(cpu_model, "float16")
    receipt = apply_compute_dtype(cpu_model, "float16-rounded")
    drift = receipt["conversion_diagnostics"]["conv1d.weight"]
    assert receipt["cast_policy"] == "finite-fp16-rounding-v1"
    assert receipt["roundtrip_exact"] is False
    assert receipt["precision_equivalent"] is False
    assert (
        drift["source_sha256"]
        == hashlib.sha256(np.asarray(original.view(mx.uint16)).tobytes()).hexdigest()
    )
    assert (
        drift["runtime_sha256"]
        == hashlib.sha256(np.asarray(cpu_model.conv1d.weight).tobytes()).hexdigest()
    )
    assert drift["element_count"] == 1
    assert drift["numerically_changed_count"] == 1
    assert drift["roundtrip_exact_count"] == 0
    assert drift["max_abs_error"] == original.astype(mx.float32).item()
    assert drift["max_rel_error"] == 1.0
    output = cpu_model.conv1d(mx.ones((1, 2, 1), mx.float16))
    mx.eval(output)
    assert output.dtype == mx.float16
    assert mx.all(mx.isfinite(output)).item()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e8, -1e8])
def test_rounded_refuses_nonfinite_and_overflow(cpu_model, value):
    import mlx.core as mx
    from compute_dtype import apply_compute_dtype

    cpu_model.bad = mx.array([value], mx.bfloat16)
    with pytest.raises(ValueError, match="Nonfinite/overflow"):
        apply_compute_dtype(cpu_model, "float16-rounded")


def test_rounded_protects_packed_f32_and_cast_predicate(cpu_model):
    import mlx.core as mx
    from compute_dtype import apply_compute_dtype
    from mlx.utils import tree_flatten

    before = dict(tree_flatten(cpu_model.parameters()))
    result = apply_compute_dtype(cpu_model, "float16-rounded")
    after = dict(tree_flatten(cpu_model.parameters()))
    for name, a in before.items():
        if a.dtype != mx.bfloat16:
            assert after[name] is a
    assert set(result["conversion_diagnostics"]) == set(result["converted_tensor_names"])
    cpu_model.A_log = mx.array([1.0], mx.bfloat16)
    protected = cpu_model.A_log
    apply_compute_dtype(cpu_model, "float16-rounded")
    assert cpu_model.A_log is protected
    cpu_model.blocked = mx.array([0.125], mx.bfloat16)
    cpu_model.cast_predicate = lambda path: path != "blocked"
    with pytest.raises(ValueError, match="protected"):
        apply_compute_dtype(cpu_model, "float16-rounded")


def test_rounded_cli_forwarding_and_probe_admission():
    from types import SimpleNamespace

    from probe import phased_options
    from supervise import probe_options, validate_args
    from test_schedule import args_for

    args = args_for("--compute-dtype", "float16-rounded")
    validate_args(args)
    flags = probe_options(args)
    assert flags[flags.index("--compute-dtype") + 1] == "float16-rounded"
    probe = SimpleNamespace(
        compute_dtype="float16-rounded",
        mode="generate",
        phased=True,
        stream_local=True,
        weight_server=None,
        repeats=1,
    )
    phased_options(probe)
    probe.stream_local = False
    with pytest.raises(ValueError, match="stream"):
        phased_options(probe)


def rounded_evidence():
    from compute_dtype import receipt_sha256
    from test_compute_receipts import compute_evidence

    ranks, server, receipts, options = compute_evidence()
    options["compute_dtype"] = "float16-rounded"
    for rank in ranks:
        rank["compute_dtype"] = "float16-rounded"
        compute = rank["compute"]
        compute.update(
            requested_compute_dtype="float16-rounded",
            cast_policy="finite-fp16-rounding-v1",
            roundtrip_exact=False,
            peer_compute_codes=[[2, 16], [2, 16]],
        )
        compute["conversion_diagnostics"] = {
            n: dict(
                source_sha256=rank["weight_loading"]["tensor_sha256"][n],
                runtime_sha256=compute["converted_tensor_sha256"][n],
                element_count=2,
                numerically_changed_count=1,
                roundtrip_exact_count=1,
                max_abs_error=1e-9,
                max_rel_error=1.0,
            )
            for n in compute["converted_tensor_names"]
        }
        for event in options["phases"][rank["rank"]]:
            if "compute_dtype" in event:
                event["compute_dtype"] = "float16-rounded"
            if event["phase"] == "compute_dtype_done":
                event["compute_receipt_sha256"] = receipt_sha256(compute)
    return ranks, server, receipts, options


def test_gate_accepts_explicit_rounded_policy_and_binds_both_ranks():
    from supervise import verify_trial

    ranks, server, receipts, options = rounded_evidence()
    result = verify_trial(ranks, server, receipts, [], **options)
    assert result["compute_dtype"] == "float16-rounded"
    assert result["precision_equivalent"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("cast_policy", "exact"),
        ("roundtrip_exact", True),
        ("precision_equivalent", True),
        ("source_sha256", "a" * 64),
        ("runtime_sha256", "c" * 64),
        ("element_count", 3),
        ("numerically_changed_count", -1),
        ("roundtrip_exact_count", 2),
        ("max_abs_error", float("inf")),
        ("max_rel_error", -1),
        ("peer_compute_codes", [[1, 16], [1, 16]]),
        ("conversion_diagnostics", {}),
    ],
)
def test_rounded_gate_rejects_tampering(field, value):
    from compute_dtype import verify_compute_receipt

    ranks, _, _, options = rounded_evidence()
    rank = ranks[0]
    compute = rank["compute"]
    diagnostic = next(iter(compute["conversion_diagnostics"].values()))
    (diagnostic if field in diagnostic else compute)[field] = value
    with pytest.raises(ValueError):
        verify_compute_receipt(rank, "float16-rounded", options["expected_inventory"][0])
