"""Synthetic TEST cache inventory; never numerical/model evidence."""

from copy import deepcopy

import pytest

from .test_harness import api


def inventory_fixture():
    expected = []
    for tree in ("target", "native_target", "head"):
        for part in ("keys", "values", "offset"):
            expected.append(
                {
                    "tree": tree,
                    "layer": 0,
                    "part": part,
                    "shape": [] if part == "offset" else [1, 2, 256, 4],
                    "dtype": "int" if part == "offset" else "bfloat16",
                    "offset": 256,
                }
            )
    for tree in ("target", "native_target"):
        for part in ("state.0.0", "state.0.1"):
            expected.append(
                {
                    "tree": tree,
                    "layer": 1,
                    "part": part,
                    "shape": [1, 2, 4],
                    "dtype": "float32",
                    "offset": 256,
                }
            )
    metrics = [
        {
            **row,
            "reference_finite": True,
            "candidate_finite": True,
            "atol": 1e-6,
            "rtol": 1e-5,
            "max_abs_error": 0.0,
            "max_tolerance": 1e-6,
            "max_error_over_tolerance": 0.0,
        }
        for row in expected
    ]
    return expected, metrics


def test_exact_every_cache_part_metrics():
    h = api()
    assert hasattr(h, "audit_numeric_inventory"), (
        "Exact independent numeric inventory validator missing"
    )
    expected, metrics = inventory_fixture()
    assert h.audit_numeric_inventory(metrics, expected)["parts"] == 13
    for index in range(13):
        missing = deepcopy(metrics)
        missing.pop(index)
        with pytest.raises(ValueError):
            h.audit_numeric_inventory(missing, expected)
        replaced = deepcopy(metrics)
        replaced[index] = deepcopy(replaced[(index + 1) % len(replaced)])
        with pytest.raises(ValueError):
            h.audit_numeric_inventory(replaced, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [1]),
        ("dtype", "float16"),
        ("offset", 255),
        ("reference_finite", False),
        ("candidate_finite", False),
        ("atol", 1e-5),
        ("rtol", 1e-4),
        ("max_abs_error", -1),
        ("max_abs_error", float("nan")),
        ("max_error_over_tolerance", 1.01),
        ("max_tolerance", 1e-8),
        ("max_abs_error", 0.1),
    ],
)
def test_forged_strict_pass_cannot_replace_metrics(field, value):
    h = api()
    assert hasattr(h, "audit_numeric_inventory"), (
        "Exact independent numeric inventory validator missing"
    )
    expected, metrics = inventory_fixture()
    metrics[0][field] = value
    metrics[0]["strict_pass"] = True
    with pytest.raises(ValueError):
        h.audit_numeric_inventory(metrics, expected)
