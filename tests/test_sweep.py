"""Synthetic CPU-only fixtures; these tests are not hardware measurements."""

import hashlib
import importlib
import json

import pytest


def fixture_profile():
    return {
        "scope": "coupled_gate_up_pairs",
        "quality_proxy": "relative_MSE_of_gate_up_activation_on_calibration_inputs",
        "fixed_parameter_bytes": 100,
        "uniform4_parameter_bytes": 120,
        "units": [{
            "name": "unit<&\"",
            "options": [
                {"label": "q3:stock", "bits": 3, "memory_bytes": 10,
                 "latency_ms": 2.0, "loss": 0.8, "backend": "stock"},
                {"label": "q4:stock", "bits": 4, "memory_bytes": 20,
                 "latency_ms": 1.0, "loss": 0.2, "backend": "stock"},
                {"label": "q6:stock", "bits": 6, "memory_bytes": 30,
                 "latency_ms": 1.5, "loss": 0.01, "backend": "stock"},
                {"label": "q6:fused:rpg1", "bits": 6, "memory_bytes": 30,
                 "latency_ms": 0.7, "loss": 0.01, "backend": "fused"},
            ],
        }],
    }


def write_profile(tmp_path, profile=None):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(fixture_profile() if profile is None else profile))
    return path


def test_exact_sweep_uses_shared_uniform4_reference_and_fixed_model_bytes(tmp_path):
    sweep = importlib.import_module("paretoquant.sweep")
    path = write_profile(tmp_path)
    result = sweep.sweep_profile(path, [1.1, 1.0], [1.0])
    assert result["profile_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["fixed_parameter_bytes_source"] == "profile.fixed_parameter_bytes"
    assert result["reference"]["parameter_bytes"] == 120
    assert result["reference"]["predicted_gate_up_latency_sum_ms"] == 1.0
    assert len(result["rows"]) == 4
    assert [(r["memory_fraction"], r["strategy"]) for r in result["rows"]] == [
        (1.0, "stock"), (1.0, "fusion_aware"),
        (1.1, "stock"), (1.1, "fusion_aware"),
    ]
    stock, fused = result["rows"][-2:]
    assert stock["status"] == fused["status"] == "feasible"
    assert stock["feasible"] is fused["feasible"] is True
    assert stock["solution"]["surrogate_loss"] == 0.2
    assert fused["solution"]["surrogate_loss"] == 0.01
    assert fused["solution"]["parameter_bytes"] == 130
    assert fused["total_parameter_budget_bytes"] == 132
    assert fused["gate_up_parameter_budget_bytes"] == 32
    assert fused["solution"]["predicted_gate_up_latency_sum_ms"] == 0.7
    assert fused["solution"]["bits_histogram"] == {"6": 1}
    assert fused["solution"]["backend_histogram"] == {"fused": 1}
    assert fused["solution"]["exact"] is True
    assert "not a quality or perplexity measurement" in result["surrogate_loss_note"]
    assert "not measured full-model latency" in result["latency_note"]


def test_infeasible_and_capped_frontier_are_distinct_with_no_solution(tmp_path):
    from paretoquant.sweep import sweep_profile

    path = write_profile(tmp_path)
    result = sweep_profile(path, [0.8, 1.1], [0.6, 2.0], max_states=1)
    assert len(result["rows"]) == 8
    small = result["rows"][0]
    assert small["status"] == "infeasible"
    assert small["feasible"] is False
    assert small["error"]["type"] == "InfeasibleBudget"
    assert small["solution"] is None
    latency = result["rows"][4]
    assert latency["status"] == "infeasible"
    assert latency["error"]["kind"] == "infeasible_budget"
    capped = result["rows"][-1]
    assert capped["status"] == "error"
    assert capped["feasible"] is None  # Overflow does NOT prove infeasibility.
    assert capped["error"]["kind"] == "capped_frontier"
    assert capped["error"]["type"] == "FrontierOverflow"
    assert "max_states=1" in capped["error"]["message"]
    assert capped["solution"] is None


@pytest.mark.parametrize("field,values", [
    ("memory_fractions", []), ("memory_fractions", [1, 1.0]),
    ("memory_fractions", [True]), ("memory_fractions", [0]),
    ("memory_fractions", [-1]), ("memory_fractions", [float("nan")]),
    ("memory_fractions", [float("inf")]), ("memory_fractions", ["1"]),
    ("memory_fractions", [10**1000]), ("memory_fractions", "1"),
    ("latency_factors", [1, 1.0]), ("latency_factors", [False]),
    ("latency_factors", [float("inf")]), ("latency_factors", [0]),
    ("max_states", True), ("max_states", 0), ("max_states", 1.5),
])
def test_invalid_grid_rejected_before_allocation(tmp_path, field, values):
    from paretoquant.sweep import sweep_profile

    kwargs = {"memory_fractions": [1], "latency_factors": [1], "max_states": 10000}
    kwargs[field] = values
    with pytest.raises(ValueError, match=field):
        sweep_profile(write_profile(tmp_path), **kwargs)


@pytest.mark.parametrize("change", [
    lambda p: p.pop("fixed_parameter_bytes"),
    lambda p: p.update(fixed_parameter_bytes=True),
    lambda p: p.update(fixed_parameter_bytes=-1),
    lambda p: p.update(uniform4_parameter_bytes=121),
    lambda p: p.update(uniform4_parameter_bytes=120.0),
    lambda p: p.update(scope="full_model"),
    lambda p: p.update(units=[]),
    lambda p: p["units"].append(p["units"][0]),
    lambda p: p["units"][0].update(options=[]),
    lambda p: p["units"][0]["options"].append(p["units"][0]["options"][0]),
    lambda p: p["units"][0]["options"].pop(1),
    lambda p: p["units"][0]["options"][0].update(bits=True),
    lambda p: p["units"][0]["options"][0].update(memory_bytes=10.0),
    lambda p: p["units"][0]["options"][0].update(latency_ms=float("nan")),
    lambda p: p["units"][0]["options"][0].update(loss=-1),
    lambda p: p["units"][0]["options"][0].update(backend="unknown"),
])
def test_invalid_profile_rejected_instead_of_reported_infeasible(tmp_path, change):
    from paretoquant.sweep import sweep_profile

    profile = fixture_profile()
    change(profile)
    with pytest.raises(ValueError):
        sweep_profile(write_profile(tmp_path, profile), [1], [1])


def test_duplicate_json_keys_rejected(tmp_path):
    from paretoquant.sweep import sweep_profile

    path = tmp_path / "profile.json"
    path.write_text('{"scope":"a","scope":"b"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        sweep_profile(path, [1], [1])


def test_input_order_is_irrelevant_and_aggregate_overflow_is_recorded(tmp_path):
    from paretoquant.sweep import sweep_profile

    profile = fixture_profile()
    second = json.loads(json.dumps(profile["units"][0]))
    second["name"] = "second"
    profile["units"].append(second)
    profile["uniform4_parameter_bytes"] = 140
    path = write_profile(tmp_path, profile)
    before = sweep_profile(path, [1.1, 1], [1.5, 1])
    profile["units"].reverse()
    for unit in profile["units"]:
        unit["options"].reverse()
    after = sweep_profile(write_profile(tmp_path, profile), [1, 1.1], [1, 1.5])
    assert before["rows"] == after["rows"]
    for unit in profile["units"]:
        for option in unit["options"]:
            option["loss"] = 1e308
    result = sweep_profile(write_profile(tmp_path, profile), [1], [1])
    for row in result["rows"]:
        assert row["status"] == "error"
        assert row["feasible"] is None
        assert row["solution"] is None
        assert row["error"]["kind"] == "allocation_error"
        assert row["error"]["type"] == "ValueError"
        assert "nonfinite" in row["error"]["message"]


def test_svg_is_self_contained_escaped_and_plots_only_feasible_rows(tmp_path):
    import xml.etree.ElementTree as ET

    from paretoquant.sweep import frontier_svg, sweep_profile

    result = sweep_profile(write_profile(tmp_path), [0.8, 1, 1.1], [1])
    title = '<bad attr="x"> & </text>'
    svg = frontier_svg(result, title=title)
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.find("s:title", ns).text == title
    points = [g for g in root.findall(".//s:g", ns) if g.get("class") == "point"]
    assert len(points) == sum(r["status"] == "feasible" for r in result["rows"])
    assert len(points) == 4
    assert all("surrogate loss=" in g.find("s:title", ns).text for g in points)
    assert all("unit<&\"" in g.find("s:title", ns).text for g in points)
    assert all(g.get("data-fraction") is not None for g in points)
    assert "Resident model parameters (MiB)" in svg
    assert "Predicted gate/up latency sum (ms)" in svg
    assert "&lt;bad" in svg and "&amp;" in svg and "&quot;" in svg
    assert '<bad attr="x">' not in svg
    assert "href=" not in svg and "<image" not in svg
    assert root.find("s:script", ns) is not None
    buttons = [g for g in root.findall(".//s:g", ns) if g.get("role") == "button"]
    assert len(buttons) == 3
    assert all(b.get("tabindex") == "0" for b in buttons)


def test_svg_handles_no_feasible_or_constant_axis_values(tmp_path):
    import xml.etree.ElementTree as ET

    from paretoquant.sweep import frontier_svg, sweep_profile

    path = write_profile(tmp_path)
    for fractions in ([0.8], [1]):
        svg = frontier_svg(sweep_profile(path, fractions, [1]))
        ET.fromstring(svg)
        assert "nan" not in svg.lower() and "infinity" not in svg.lower()
    assert "No feasible allocations" in frontier_svg(sweep_profile(path, [0.8], [1]))


def test_save_artifacts_deterministic_and_refuses_nonempty_output(tmp_path):
    from paretoquant.sweep import sweep_profile, write_sweep

    result = sweep_profile(write_profile(tmp_path), [1, 1.1], [1])
    first = tmp_path / "first"
    second = tmp_path / "second"
    second.mkdir()  # An existing empty directory is safe.
    paths = write_sweep(result, first)
    write_sweep(result, second)
    assert [p.name for p in paths] == ["sweep.json", "frontier.svg"]
    assert json.loads((first / "sweep.json").read_text()) == result
    assert (first / "frontier.svg").read_bytes() == (second / "frontier.svg").read_bytes()
    assert (first / "sweep.json").read_bytes() == (second / "sweep.json").read_bytes()
    with pytest.raises(FileExistsError, match="nonempty"):
        write_sweep(result, first)
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / ".keep").write_text("keep")
    with pytest.raises(FileExistsError, match="nonempty"):
        write_sweep(result, hidden)
    assert (hidden / ".keep").read_text() == "keep"


def run_script(*args):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": ""}
    return subprocess.run(
        [sys.executable, "-S", str(root / "scripts" / "sweep_profile.py"), *map(str, args)],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )


def test_memory_budget_floor_is_independent_of_decimal_context(tmp_path):
    from decimal import localcontext

    from paretoquant.sweep import sweep_profile

    with localcontext() as context:
        context.prec = 2
        result = sweep_profile(write_profile(tmp_path), [1.1, 1.109], [1])
    assert [r["total_parameter_budget_bytes"] for r in result["rows"]] == [132, 132, 133, 133]


def test_script_runs_with_stdlib_only_and_rejects_nonempty_before_profile_read(tmp_path):
    profile = write_profile(tmp_path)
    output = tmp_path / "cli-output"
    completed = run_script("--profile", profile, "--output", output,
                           "--memory-fractions", "1", "1.1", "--latency-factors", "1")
    assert completed.returncode == 0, completed.stderr
    assert "4 rows" in completed.stdout
    assert len(json.loads((output / "sweep.json").read_text())["rows"]) == 4
    before = (output / "sweep.json").read_bytes()
    refused = run_script("--profile", "missing-profile", "--output", output)
    assert refused.returncode == 2
    assert "nonempty" in refused.stderr
    assert (output / "sweep.json").read_bytes() == before


def test_script_accepts_comma_lists_and_reports_capped_rows_as_errors(tmp_path):
    output = tmp_path / "cli-output"
    completed = run_script("--profile", write_profile(tmp_path), "--output", output,
                           "--memory-fractions", "1,1.1", "--latency-factors", "1,2",
                           "--max-states", "1")
    assert completed.returncode == 1, completed.stderr
    assert "8 rows" in completed.stdout
    rows = json.loads((output / "sweep.json").read_text())["rows"]
    assert len(rows) == 8
    assert any(r["error"] and r["error"]["kind"] == "capped_frontier" for r in rows)


@pytest.mark.parametrize("option,value", [
    ("--memory-fractions", "1,1"), ("--memory-fractions", "nan"),
    ("--memory-fractions", "true"), ("--memory-fractions", "1,"),
    ("--latency-factors", "0"), ("--latency-factors", "inf"),
    ("--max-states", "0"),
])
def test_script_invalid_inputs_write_no_artifacts(tmp_path, option, value):
    output = tmp_path / "invalid"
    completed = run_script("--profile", write_profile(tmp_path), "--output", output, option, value)
    assert completed.returncode == 2
    assert not output.exists()
