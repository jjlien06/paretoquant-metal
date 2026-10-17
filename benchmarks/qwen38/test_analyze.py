"""CPU-only tests. All request data in this file are SYNTHETIC fixtures.

Observed TDD checkpoints (pytest output, not benchmark evidence):
missing module: 1 failed -> 1 passed; schedule checks: 19 failed -> 20 passed;
row validation: 32 failed -> 55 passed; statistics: 6 failed -> 61 passed;
envelope coverage/failures: 9 failed -> 70 passed; CLI/exports: 6 failed ->
76 passed; acceptance regression: 1 failed -> 77 passed; numeric/path
regressions: 2 failed -> 79 passed. Each RED was run before its implementation.

Run from repository root with the parent interpreter and explicit scratch:
TMPDIR=/Users/jeremylien/.hermes/cache/scratch .venv/bin/python -m pytest \
benchmarks/qwen38/test_analyze.py -q \
--basetemp /Users/jeremylien/.hermes/cache/scratch/qwen38-analyze-pytest
"""

import copy
import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=1)
def api():
    spec = importlib.util.spec_from_file_location(
        "qwen38_analyze_cpu_tests", Path(__file__).with_name("analyze.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_real_json_csv_and_markdown_exclusively(tmp_path):
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path

    evidence = synthetic_evidence()
    source = tmp_path / "synthetic-evidence.json"
    source.write_text(json.dumps(evidence), encoding="utf-8")
    outputs = [tmp_path / name for name in ("analysis.json", "analysis.csv", "analysis.md")]
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze.py")),
        str(source),
        "--json",
        str(outputs[0]),
        "--csv",
        str(outputs[1]),
        "--markdown",
        str(outputs[2]),
        "--exclusive",
    ]
    done = subprocess.run(command, text=True, capture_output=True)
    assert done.returncode == 0, done.stderr
    report = json.loads(outputs[0].read_text())
    assert report["coverage"]["observed_count"] == 4
    csv_rows = list(csv.DictReader(outputs[1].read_text().splitlines()))
    assert {r["record_type"] for r in csv_rows} >= {"case", "request", "metadata"}
    markdown = outputs[2].read_text()
    assert "synthetic-case" in markdown
    assert "sample stdev" in markdown
    assert "Failures (0)" in markdown
    assert "confidence interval" in markdown
    before = [p.read_bytes() for p in outputs]
    again = subprocess.run(command, text=True, capture_output=True)
    assert again.returncode != 0
    assert [p.read_bytes() for p in outputs] == before
    assert json.loads(source.read_text()) == evidence


@pytest.mark.parametrize("marker", ["=", "+", "-", "@"])
@pytest.mark.parametrize("whitespace", ["", " \t", "\r\n"])
def test_csv_neutralizes_formula_cells_without_mutating_json(marker, whitespace):
    import csv
    import io
    import json

    attack = whitespace + marker + "1+1"
    expected, rows = synthetic_suite(case_id=attack)
    evidence = dict(
        synthetic_evidence(),
        expected_requests=expected,
        rows=rows,
        failures=[dict(case_id=attack, id=attack, method=attack)],
    )
    before_evidence = copy.deepcopy(evidence)
    report = api().analyze_evidence(evidence)
    report[attack] = -1  # Untrusted metric name and formula-like JSON numeric cell.
    before_json = json.dumps(report, ensure_ascii=False)
    exported = list(csv.DictReader(io.StringIO(api().render_csv(report))))

    assert all(
        not cell.lstrip().startswith(("=", "+", "-", "@"))
        for row in exported
        for cell in row.values()
    )
    request = next(row for row in exported if row["record_type"] == "request")
    assert request["case_id"] == "'" + attack
    assert request["id"] == "'" + rows[0]["id"]
    failure = next(row for row in exported if row["record_type"] == "failure")
    assert all(failure[key] == "'" + attack for key in ("case_id", "id", "method"))
    metric = next(row for row in exported if row["metric"] == "'" + attack)
    assert metric["value"] == "'-1"
    assert evidence == before_evidence
    assert json.dumps(report, ensure_ascii=False) == before_json


def test_cli_partial_stdout_and_renderers_retain_all_failures(tmp_path):
    import csv
    import io
    import json
    import subprocess
    import sys
    from pathlib import Path

    evidence = synthetic_evidence()
    missing = evidence["rows"].pop()["id"]
    evidence["failures"] = [{"id": missing, "error": "synthetic failure"}]
    source = tmp_path / "synthetic-partial.json"
    source.write_text(json.dumps(evidence), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).with_name("analyze.py")), str(source)]
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert not rejected.stdout
    assert "missing" in rejected.stderr
    done = subprocess.run(command + ["--allow-partial"], text=True, capture_output=True)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["overall"] is None
    assert report["coverage"]["missing_ids"] == [missing]
    markdown = api().render_markdown(report)
    assert "not a full-suite" in markdown
    assert "Failures (1)" in markdown and "synthetic failure" in markdown
    assert missing in markdown
    csv_rows = list(csv.DictReader(io.StringIO(api().render_csv(report))))
    failures = [r for r in csv_rows if r["record_type"] == "failure"]
    assert len(failures) == 1
    assert "synthetic failure" in failures[0]["value"]


@pytest.mark.parametrize(
    "bad_json", ['{"schema": "x", "schema": "x"}', '{"x": NaN}', '{"x": 1e999}']
)
def test_cli_rejects_duplicate_keys_and_nonfinite_json(tmp_path, bad_json):
    import subprocess
    import sys
    from pathlib import Path

    source = tmp_path / "invalid.json"
    source.write_text(bad_json)
    output = tmp_path / "must-not-exist.json"
    done = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze.py")),
            str(source),
            "--json",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert done.returncode != 0
    assert not output.exists()


def test_cli_cannot_overwrite_input_or_alias_outputs(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    source = tmp_path / "synthetic.json"
    original = json.dumps(synthetic_evidence())
    source.write_text(original)
    command = [sys.executable, str(Path(__file__).with_name("analyze.py")), str(source)]
    assert subprocess.run(command + ["--json", str(source)], capture_output=True).returncode != 0
    assert source.read_text() == original
    dest = tmp_path / "alias"
    assert (
        subprocess.run(
            command + ["--json", str(dest), "--csv", str(dest)], capture_output=True
        ).returncode
        != 0
    )
    assert not dest.exists()


def test_summary_extreme_finite_timings_do_not_create_nonfinite_statistics():
    import json

    _, rows = synthetic_suite()
    for row in rows:
        row.update(wall_seconds=1e308, first_token_seconds=1e307)
    report = api().summarize_requests(rows)
    assert report["cases"]["synthetic-case"]["native"]["wall_seconds"]["median"] == 1e308
    json.dumps(report, allow_nan=False)


def test_cli_rejects_hardlink_alias_of_input(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    source = tmp_path / "synthetic.json"
    original = json.dumps(synthetic_evidence())
    source.write_text(original)
    alias = tmp_path / "hardlink.json"
    os.link(source, alias)
    done = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze.py")),
            str(source),
            "--json",
            str(alias),
        ],
        capture_output=True,
    )
    assert done.returncode != 0
    assert source.read_text() == original


def test_summary_uses_per_request_statistics_and_paired_ratios():
    import math

    _, rows = synthetic_suite()
    rows[2]["wall_seconds"] = 5.0
    result = api().summarize_requests(rows)
    case = result["cases"]["synthetic-case"]
    native = case["native"]
    assert native["count"] == 2
    assert native["post_first_tps"] == dict(
        count=2, median=0.75, min=0.5, max=1.0, stdev=pytest.approx(math.sqrt(0.125))
    )
    assert native["first_token_seconds"]["median"] == 1.0
    assert native["wall_seconds"]["median"] == 4.0
    assert native["whole_request_tps"]["median"] == 0.8
    assert native["post_first_below_10_percent"] == 100.0
    assert [p["decode_speedup"] for p in case["paired"]["repeats"]] == [2.0, 4.0]
    assert case["paired"]["decode_speedup"]["median"] == 3.0
    assert case["paired"]["whole_request_speedup"]["median"] == 2.0
    assert result["overall"]["decode"]["geometric_mean"] == pytest.approx(3.0)
    assert "p95" not in result["overall"]["interpretation"]


def test_acceptance_is_round_weighted_and_rank_peaks_remain_separate():
    _, rows = synthetic_suite()
    rows[3].update(
        accepted_drafts=0,
        verification_rounds=2,
        teacher_forwards=2,
        peak_mlx_bytes_by_rank=[300, 150],
    )
    mtp = api().summarize_requests(rows)["cases"]["synthetic-case"]["mtp"]
    assert mtp["acceptance"] == dict(
        accepted_drafts=1,
        verification_rounds=3,
        teacher_forwards=3,
        weighted_rate=pytest.approx(1 / 3),
        rows_with_counters=2,
        rows_without_counters=0,
    )
    assert mtp["peak_mlx_bytes_by_rank"] == [300, 200]


def test_geometric_mean_weights_cases_not_rows_or_tokens():
    import math

    _, first = synthetic_suite()
    _, second = synthetic_suite(repeats=4, case_id="synthetic-second")
    for row in second:
        if row["method"] == "mtp":
            row["wall_seconds"] = 5.0
    result = api().summarize_requests(first + second)
    assert result["overall"]["decode"]["geometric_mean"] == pytest.approx(math.sqrt(2 * 0.5))
    assert result["overall"]["decode"]["case_count"] == 2
    assert result["overall"]["decode"]["lower_tail_min"] == 0.5
    assert "pooled_tps" not in result["overall"]


def test_explicit_first_loaded_warmups_do_not_pollute_measured_statistics():
    _, rows = synthetic_suite()
    warmups = [dict(r, id="warm:" + r["id"], warmup=True, wall_seconds=100.0) for r in rows[:2]]
    result = api().summarize_requests(warmups + rows)
    assert result["warmup_count"] == 2
    assert result["measured_count"] == 4
    assert result["cases"]["synthetic-case"]["native"]["wall_seconds"]["median"] == 3.0
    assert [r["id"] for r in result["warmups"]] == [r["id"] for r in warmups]


def test_single_token_eos_excluded_only_from_decode_metric():
    expected, rows = synthetic_suite()
    for row in rows:
        row.update(token_ids=[99], finish_reason="stop", wall_seconds=1.0, first_token_seconds=1.0)
        if row["method"] == "mtp":
            row.update(accepted_drafts=0, verification_rounds=0, teacher_forwards=0)
    assert api().validate_pairs(rows, expected, eos_token_ids=[99])["complete"]
    result = api().summarize_requests(rows)
    native = result["cases"]["synthetic-case"]["native"]
    assert native["count"] == 2
    assert native["post_first_tps"] == dict(count=0, median=None, min=None, max=None, stdev=None)
    assert native["post_first_excluded_count"] == 2
    assert native["post_first_below_10_percent"] is None
    assert native["whole_request_tps"]["count"] == 2
    assert result["requests"][0]["post_first_tps"] is None
    assert result["overall"]["decode"]["geometric_mean"] is None
    assert result["overall"]["whole_request"]["geometric_mean"] == 1.0


def test_missing_counters_and_unpaired_rows_are_explicit_not_imputed():
    _, rows = synthetic_suite(repeats=1)
    for key in ("accepted_drafts", "verification_rounds", "teacher_forwards"):
        rows[1].pop(key)
    result = api().summarize_requests(rows[1:])
    case = result["cases"]["synthetic-case"]
    assert case["native"]["count"] == 0
    assert case["mtp"]["post_first_tps"]["stdev"] is None
    assert case["mtp"]["acceptance"]["weighted_rate"] is None
    assert case["mtp"]["acceptance"]["rows_without_counters"] == 1
    assert case["paired"]["incomplete_repeats"] == [
        {"sample_repeat": 0, "missing_methods": ["native"]}
    ]
    with pytest.raises(ValueError):
        api().summarize_requests([dict(rows[1], wall_seconds=float("nan"))])


def synthetic_suite(repeats=2, case_id="synthetic-case", prompt="Synthetic 中文 prompt"):
    expected, rows = [], []
    for repeat in range(repeats):
        for method in ("native", "mtp"):
            ident = f"{case_id}:{repeat}:{method}"
            expected.append(
                dict(
                    id=ident,
                    case_id=case_id,
                    method=method,
                    prompt=prompt,
                    max_tokens=3,
                    warmup=False,
                    repeat=repeat,
                )
            )
            row = dict(
                id=ident,
                case_id=case_id,
                method=method,
                warmup=False,
                sample_repeat=repeat,
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                max_tokens=3,
                token_ids=[10, 11, 12],
                finish_reason="length",
                wall_seconds=3.0 if method == "native" else 2.0,
                first_token_seconds=1.0,
                prompt_tokens=7,
                peak_mlx_bytes_by_rank=[100, 200],
            )
            if method == "mtp":
                row.update(accepted_drafts=1, verification_rounds=1, teacher_forwards=1)
            rows.append(row)
    return expected, rows


def test_validate_exact_schedule_and_explicit_warmups():
    expected, rows = synthetic_suite()
    coverage = api().validate_pairs(rows, expected)
    assert coverage["complete"] is True
    assert coverage["expected_count"] == coverage["observed_count"] == 4
    assert coverage["measured_count"] == 4
    assert coverage["warmup_count"] == 0  # Repeat zero is not implicitly warm.
    assert coverage["missing_ids"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "extra",
        "method",
        "case_id",
        "sample_repeat",
        "warmup",
        "prompt_sha256",
        "max_tokens",
        "prompt_tokens",
        "tokens",
        "finish",
        "repeat_output",
    ],
)
def test_schedule_and_pair_mismatches_fail_closed(mutation):
    expected, rows = synthetic_suite()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(dict(rows[0], id="unscheduled"))
    elif mutation == "tokens":
        rows[1]["token_ids"] = [10, 11, 99]
    elif mutation == "repeat_output":
        rows[2]["token_ids"] = rows[3]["token_ids"] = [10, 11, 99]
    elif mutation == "finish":
        rows[1]["finish_reason"] = "stop"
    elif mutation == "prompt_tokens":
        rows[1]["prompt_tokens"] = 8
    else:
        replacement = {
            "method": "other",
            "case_id": "other",
            "sample_repeat": 9,
            "warmup": True,
            "prompt_sha256": "0" * 64,
            "max_tokens": 4,
        }
        rows[0][mutation] = replacement[mutation]
    with pytest.raises(ValueError):
        api().validate_pairs(rows, expected)


def test_partial_mode_lists_all_missing_ids_but_still_rejects_extras():
    expected, rows = synthetic_suite()
    with pytest.raises(ValueError, match="missing"):
        api().validate_pairs(rows[:1], expected)
    result = api().validate_pairs(rows[:1], expected, allow_partial=True)
    assert result["complete"] is False
    assert result["missing_ids"] == sorted(r["id"] for r in expected[1:])
    with pytest.raises(ValueError):
        api().validate_pairs(rows + [dict(rows[0], id="bad")], expected, allow_partial=True)


@pytest.mark.parametrize(
    "mutation", ["duplicate_id", "duplicate_slot", "unpaired", "case_prompt", "repeat_cap"]
)
def test_bad_expected_schedule_is_rejected(mutation):
    expected, rows = synthetic_suite()
    if mutation == "duplicate_id":
        expected[1]["id"] = expected[0]["id"]
    elif mutation == "duplicate_slot":
        expected[1]["method"] = "native"
    elif mutation == "unpaired":
        expected.pop()
        rows.pop()
    elif mutation == "case_prompt":
        expected[2]["prompt"] = expected[3]["prompt"] = "different"
    else:
        expected[2]["max_tokens"] = expected[3]["max_tokens"] = 4
    with pytest.raises(ValueError):
        api().validate_pairs(rows, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_seconds", float("nan")),
        ("wall_seconds", float("inf")),
        ("wall_seconds", 0),
        ("wall_seconds", -1),
        ("wall_seconds", True),
        ("first_token_seconds", float("nan")),
        ("first_token_seconds", -1),
        ("first_token_seconds", 0),
        ("first_token_seconds", 3.0),
        ("first_token_seconds", 4.0),
        ("token_ids", []),
        ("token_ids", [True, 2, 3]),
        ("token_ids", [-1, 2, 3]),
        ("token_ids", [1.0, 2, 3]),
        ("token_ids", [1, 2, 3, 4]),
        ("token_ids", [1, 2]),
        ("finish_reason", "error"),
        ("prompt_tokens", 0),
        ("prompt_tokens", True),
        ("max_tokens", True),
        ("sample_repeat", True),
        ("warmup", 0),
        ("peak_mlx_bytes_by_rank", [1]),
        ("peak_mlx_bytes_by_rank", [1, -1]),
        ("peak_mlx_bytes_by_rank", [1, 2.0]),
    ],
)
def test_invalid_normalized_rows_are_rejected(field, value):
    expected, rows = synthetic_suite()
    for row in rows:
        row[field] = value
    with pytest.raises(ValueError):
        api().validate_pairs(rows, expected)


@pytest.mark.parametrize(
    "counters",
    [
        dict(accepted_drafts=2, verification_rounds=1, teacher_forwards=1),
        dict(accepted_drafts=-1, verification_rounds=1, teacher_forwards=1),
        dict(accepted_drafts=True, verification_rounds=1, teacher_forwards=1),
        dict(accepted_drafts=1, verification_rounds=1, teacher_forwards=0),
        dict(accepted_drafts=1, verification_rounds=1, teacher_forwards=3),
        dict(accepted_drafts=0, verification_rounds=0, teacher_forwards=0),
        dict(accepted_drafts=2, verification_rounds=2, teacher_forwards=2),
        dict(accepted_drafts=1),
    ],
)
def test_invalid_one_draft_acceptance_counts_fail(counters):
    expected, rows = synthetic_suite()
    for row in rows:
        if row["method"] == "mtp":
            for key in ("accepted_drafts", "verification_rounds", "teacher_forwards"):
                row.pop(key)
            row.update(counters)
    with pytest.raises(ValueError):
        api().validate_pairs(rows, expected)


def test_native_acceptance_is_not_mtp_acceptance():
    expected, rows = synthetic_suite()
    rows[0].update(accepted_drafts=1, verification_rounds=1, teacher_forwards=1)
    with pytest.raises(ValueError):
        api().validate_pairs(rows, expected)


def test_optional_eos_ids_check_terminal_token_and_early_eos():
    expected, rows = synthetic_suite()
    for row in rows:
        row["finish_reason"] = "stop"
    assert api().validate_pairs(rows, expected, eos_token_ids=[12])["complete"]
    with pytest.raises(ValueError, match="EOS"):
        api().validate_pairs(rows, expected, eos_token_ids=[99])
    for row in rows:
        row["finish_reason"] = "length"
    with pytest.raises(ValueError, match="EOS"):
        api().validate_pairs(rows, expected, eos_token_ids=[11])


def test_empty_and_malformed_schedules_rejected_as_value_errors():
    for expected in ([], [dict(id="x")], None):
        with pytest.raises(ValueError):
            api().validate_pairs([], expected)


def synthetic_evidence() -> dict:
    expected, rows = synthetic_suite()
    return dict(
        schema="qwen38-workload-evidence-v1",
        expected_requests=expected,
        rows=rows,
        failures=[],
        provenance={"synthetic_test_fixture": True},
    )


def test_envelope_preserves_provenance_without_inventing_verified_gates():
    evidence = synthetic_evidence()
    result = api().analyze_evidence(evidence)
    assert result["schema"] == "qwen38-workload-analysis-v1"
    assert result["provenance"] == evidence["provenance"]
    assert result["coverage_status"] == "complete"
    assert result["full_suite_performance_claim"] is True
    assert result["runtime_verification"] == "not_performed_by_analyzer"
    assert result["eos_token_identity_checked"] is False
    assert not any(k.endswith("_verified") for k in result)
    assert evidence == synthetic_evidence()  # No caller mutations.


def test_partial_report_counts_entirely_absent_cases_and_preserves_failures():
    evidence = synthetic_evidence()
    more_expected, _ = synthetic_suite(case_id="synthetic-missing-case")
    evidence["expected_requests"].extend(more_expected)
    evidence["rows"].pop()
    evidence["failures"] = [{"id": more_expected[0]["id"], "error": "synthetic failure"}]
    with pytest.raises(ValueError, match="missing"):
        api().analyze_evidence(evidence)
    result = api().analyze_evidence(evidence, allow_partial=True)
    assert result["coverage_status"] == "partial"
    assert result["coverage"]["expected_count"] == 8
    assert len(result["coverage"]["missing_ids"]) == 5
    assert result["coverage"]["expected_case_count"] == 2
    assert result["cases"]["synthetic-missing-case"]["native"]["count"] == 0
    assert result["cases"]["synthetic-missing-case"]["coverage"]["expected_count"] == 4
    assert result["failures"] == evidence["failures"]
    assert result["failure_count"] == 1
    assert result["full_suite_performance_claim"] is False
    assert result["overall"] is None
    assert result["observed_only_overall"]["decode"]["case_count"] == 1
    assert "not a full-suite" in result["performance_scope"]


@pytest.mark.parametrize("partial", [False, True])
def test_expected_measured_and_warmup_case_counts_preserve_total(partial):
    evidence = synthetic_evidence()
    warm_expected, warm_rows = synthetic_suite(repeats=1, case_id="synthetic-warmup-only")
    for entry in warm_expected + warm_rows:
        entry["warmup"] = True
    # A case can appear in both sets; count unique IDs, never add disjointly.
    shared_expected = [
        dict(r, id="warm:" + r["id"], warmup=True) for r in evidence["expected_requests"][:2]
    ]
    shared_rows = [dict(r, id="warm:" + r["id"], warmup=True) for r in evidence["rows"][:2]]
    evidence["expected_requests"].extend(warm_expected + shared_expected)
    evidence["rows"].extend(([] if partial else warm_rows) + shared_rows)
    report = api().analyze_evidence(evidence, allow_partial=partial)
    coverage = report["coverage"]
    assert coverage["expected_case_count"] == 2  # Existing all-request case count stays compatible.
    assert coverage["expected_measured_case_count"] == 1
    assert coverage["expected_warmup_case_count"] == 2
    assert coverage["expected_measured_count"] == 4
    assert coverage["expected_warmup_count"] == 4
    markdown = api().render_markdown(report)
    assert "2 requested cases total (1 measured; 2 warmup; case sets may overlap)" in markdown


def test_complete_rows_with_failures_do_not_make_fullsuite_claims():
    evidence = synthetic_evidence()
    evidence["failures"] = [{"stage": "cleanup", "error": "synthetic cleanup error"}]
    report = api().analyze_evidence(evidence)
    assert report["coverage_status"] == "complete_with_failures"
    assert report["full_suite_performance_claim"] is False
    assert report["overall"] is None
    assert report["failures"] == evidence["failures"]


def test_envelope_optional_eos_token_ids_are_actually_checked():
    evidence = synthetic_evidence()
    evidence["provenance"]["eos_token_ids"] = [99]
    assert api().analyze_evidence(evidence)["eos_token_identity_checked"] is True
    evidence["rows"][0]["finish_reason"] = "stop"
    with pytest.raises(ValueError, match="EOS"):
        api().analyze_evidence(evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong"),
        ("failures", None),
        ("provenance", []),
        ("rows", None),
        ("expected_requests", []),
    ],
)
def test_malformed_envelopes_rejected(field, value):
    evidence = synthetic_evidence()
    evidence[field] = value
    with pytest.raises(ValueError):
        api().analyze_evidence(evidence)
