"""Synthetic gate receipts; NOT numerical or physical benchmark evidence."""

import copy

import pytest


def compute_evidence(mode="float16", *, base=None):
    from compute_dtype import receipt_sha256
    from test_schedule import evidence

    ranks, server, receipts, options = copy.deepcopy(
        evidence(with_compute=False) if base is None else base
    )
    for rank in ranks:
        source = [
            {
                "name": r["name"],
                "bytes": 4,
                "shape": [2 if mode == "float16" else 1],
                "dtype": "mlx.core.bfloat16" if mode == "float16" else "mlx.core.float32",
            }
            for r in sorted(rank["parameter_layout"], key=lambda row: row["name"])
        ]
        runtime = copy.deepcopy(source)
        if mode == "float16":
            for row in runtime:
                row["dtype"] = "mlx.core.float16"
        rank["parameter_layout"] = runtime
        rank["compute_dtype"] = mode
        rank["weight_loading"].update(
            source_parameter_layout=source,
            tensor_hash_scope="serialized_source_payloads_before_runtime_cast",
        )
        actual = "mlx.core.float16" if mode == "float16" else "mlx.core.float32"
        converted = [r["name"] for r in source] if mode == "float16" else []
        rank["compute"] = dict(
            requested_compute_dtype=mode,
            actual_compute_dtype=actual,
            arithmetic_changed=bool(converted),
            precision_equivalent=False if converted else None,
            roundtrip_exact=True if converted else None,
            hash_scope="converted_runtime_arrays_only",
            converted_tensor_names=converted,
            converted_tensor_sha256={n: "b" * 64 for n in converted},
            source_parameter_layout=source,
            runtime_parameter_layout=runtime,
            peer_compute_codes=[[1, 16], [1, 16]] if converted else [[0, 32], [0, 32]],
        )
        phases = options["phases"][rank["rank"]]
        phases[0]["compute_dtype"] = mode
        phases.insert(
            1,
            dict(
                phase="compute_dtype_done",
                rank=rank["rank"],
                compute_dtype=mode,
                actual_compute_dtype=actual,
                compute_receipt_sha256=receipt_sha256(rank["compute"]),
            ),
        )
        phases.insert(
            2,
            dict(
                phase="embedding_done" if rank["rank"] == 1 else "handoff_receive_done",
                rank=rank["rank"],
                actual_compute_dtype=actual,
            ),
        )
    options["compute_dtype"] = mode
    return ranks, server, receipts, options


@pytest.mark.parametrize("mode", ["native", "float16"])
def test_gate_binds_admitted_and_actual_compute(mode):
    from supervise import verify_trial

    ranks, server, receipts, options = compute_evidence(mode)
    result = verify_trial(ranks, server, receipts, [], **options)
    assert result["compute_dtype"] == mode
    assert result["actual_compute_dtype"] == ranks[0]["compute"]["actual_compute_dtype"]


@pytest.mark.parametrize(
    "tamper",
    [
        "admission",
        "both_modes",
        "missing_compute",
        "missing_actual",
        "actual",
        "start",
        "phase_hash",
        "missing_phase",
        "runtime_inventory",
        "source_inventory",
        "hash_scope",
        "runtime_hash",
        "roundtrip",
        "precision",
        "changed",
        "peer",
        "witness",
        "missing_witness",
        "arithmetic",
        "shape_bytes",
        "packed_dtype",
    ],
)
def test_gate_rejects_forged_compute_receipts(tamper):
    from compute_dtype import receipt_sha256
    from supervise import verify_trial

    ranks, server, receipts, options = compute_evidence()
    rank = ranks[0]
    compute = rank["compute"]
    phases = options["phases"][0]
    if tamper == "admission":
        options["compute_dtype"] = "native"
    elif tamper == "both_modes":
        for r in ranks:
            r["compute_dtype"] = "native"
    elif tamper == "missing_compute":
        rank.pop("compute")
    elif tamper == "missing_actual":
        compute.pop("actual_compute_dtype")
    elif tamper == "actual":
        compute["actual_compute_dtype"] = "mlx.core.bfloat16"
    elif tamper == "start":
        phases[0]["compute_dtype"] = "native"
    elif tamper == "phase_hash":
        phases[1]["compute_receipt_sha256"] = "a" * 64
    elif tamper == "missing_phase":
        phases.pop(1)
    elif tamper == "runtime_inventory":
        compute["runtime_parameter_layout"][0]["dtype"] = "mlx.core.bfloat16"
    elif tamper == "source_inventory":
        rank["weight_loading"]["source_parameter_layout"] = []
    elif tamper == "hash_scope":
        rank["weight_loading"]["tensor_hash_scope"] = "runtime_parameters"
    elif tamper == "runtime_hash":
        compute["converted_tensor_sha256"].pop(next(iter(compute["converted_tensor_sha256"])))
    elif tamper == "roundtrip":
        compute["roundtrip_exact"] = False
    elif tamper == "precision":
        compute["precision_equivalent"] = True
    elif tamper == "changed":
        compute["converted_tensor_names"] = []
    elif tamper == "peer":
        compute["peer_compute_codes"][1][1] = 17
    elif tamper == "witness":
        phases[2]["actual_compute_dtype"] = "mlx.core.bfloat16"
    elif tamper == "missing_witness":
        phases.pop(2)
    elif tamper == "arithmetic":
        compute["arithmetic_changed"] = False
    elif tamper == "shape_bytes":
        compute["source_parameter_layout"][0]["shape"] = [1]
    elif tamper == "packed_dtype":
        # Forge internally consistent inventories; a U32 pack must never cast.
        compute["source_parameter_layout"][0].update(dtype="mlx.core.uint32", shape=[1])
    if tamper not in {"phase_hash", "missing_phase"} and "compute" in rank:
        for event in phases:
            if event["phase"] == "compute_dtype_done":
                event["compute_receipt_sha256"] = receipt_sha256(compute)
    with pytest.raises(ValueError):
        verify_trial(ranks, server, receipts, [], **options)
