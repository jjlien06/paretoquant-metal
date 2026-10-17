"""Standalone versioned CPU-only overlap benchmark harness. No runtime imports."""

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from .protocol import audit_decode as audit_decode
from .protocol import audit_transcript as audit_transcript


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path, limit=128 * 1024 * 1024):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink(), "Symlink input is not admitted")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= limit,
            "Input budget exceeded/not regular",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        require(len(data) <= limit, "Input budget exceeded")
        return data
    finally:
        os.close(fd)


def decode_json(data):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), "Nonfinite JSON number")
        return value

    def bad_constant(text):
        raise ValueError("Nonfinite JSON constant: " + text)

    try:
        return json.loads(
            data, object_pairs_hook=unique, parse_float=finite_float, parse_constant=bad_constant
        )
    except (RecursionError, UnicodeError) as error:
        raise ValueError("Invalid bounded JSON") from error


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_bound(path, digest, limit=128 * 1024 * 1024):
    require(
        isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "Invalid expected SHA256"
    )
    data = read_bytes(path, limit)
    require(sha(data) == digest, "External SHA256 binding differs")
    return decode_json(data)


SCHEMA = "qwen38-prefill-overlap-schedule-v1"
HISTORICAL_SHA = "b20a600c3889d62cdf15a7faa6c2ca90611084ceae15b9a8dea7f7823d5a29e6"
DESIGN_SHA = "710703230b23461193337e78c8a38e1c72675e19f6d747c614c14c7211ac95f0"


def derive_schedule(manifest_path):
    data = Path(manifest_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != HISTORICAL_SHA:
        raise ValueError("Historical manifest bytes differ")
    original = json.loads(data)
    batches = []
    for batch in original["batches"]:
        requests = []
        for source in batch["requests"]:
            mode = {"native": "serial", "mtp": "overlap"}[source["method"]]
            requests.append(
                {
                    **source,
                    "id": "overlap-v1-" + source["id"],
                    "method": "mtp",
                    "prefill_mode": mode,
                    "source_request_id": source["id"],
                }
            )
        batches.append({"schema": SCHEMA, "requests": requests})
    return {
        "schema": SCHEMA,
        "historical_manifest_sha256": HISTORICAL_SHA,
        "design_sha256": DESIGN_SHA,
        "population": "full84",
        "cases": original["cases"],
        "batches": batches,
    }


def derive_pilot(full):
    rows = list(full["batches"][0]["requests"][:2])
    for case, modes in (
        ("code-1536", ("serial", "overlap")),
        ("structured-1536", ("overlap", "serial")),
    ):
        for mode in modes:
            rows.append(
                next(
                    r
                    for b in full["batches"]
                    for r in b["requests"]
                    if r["case_id"] == case and r["repeat"] == 0 and r["prefill_mode"] == mode
                )
            )
    return {
        **full,
        "population": "pilot6",
        "batches": [
            {
                "schema": SCHEMA,
                "requests": [{**r, "id": "pilot-v1-" + r["source_request_id"]} for r in rows],
            }
        ],
    }


METRICS = ("first_token_seconds", "wall_seconds")
LONG_IDS = tuple(
    c + "-1536" for c in ("code", "explanation", "math", "chinese", "spanish", "structured")
)


def flatten(schedule):
    return [r for batch in schedule["batches"] for r in batch["requests"]]


def positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def analyze(rows, schedule):
    """Only descriptive statistics; never promotes timings to physical evidence."""
    from statistics import median

    expected = {r["id"]: r for r in flatten(schedule)}
    by_id, failures, extras, duplicates = {}, [], [], []
    for row in rows:
        rid = row.get("id")
        if rid not in expected:
            extras.append(rid)
            continue
        if rid in by_id:
            duplicates.append(rid)
        by_id.setdefault(rid, row)
        spec = expected[rid]
        error = None
        if any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in spec.items()):
            error = "schedule_binding"
        elif row.get("status") != "success":
            error = "failed_request"
        elif not all(positive(row.get(k)) for k in METRICS):
            error = "nonpositive_or_nonfinite_metrics"
        elif row["wall_seconds"] < row["first_token_seconds"]:
            error = "wall_less_than_ttft"
        if error:
            failures.append({"id": rid, "error": error})
    invalid = {r["id"] for r in failures} | set(duplicates)
    missing = sorted(set(expected) - set(by_id))
    cases, regressions = [], []
    for case in schedule["cases"]:
        cid = case["id"]
        specs = [r for r in expected.values() if r["case_id"] == cid and not r["warmup"]]
        repeats = sorted({s["repeat"] for s in specs})
        record = {
            "case_id": cid,
            "formatted_prompt_tokens": case["formatted_prompt_tokens"],
            "target_long": cid in LONG_IDS,
            "complete": bool(specs),
            "thresholds_met": False,
        }
        for metric, threshold in zip(METRICS, (0.70, 0.85), strict=True):
            ratios = []
            for repeat in repeats:
                pair = {s["prefill_mode"]: s for s in specs if s["repeat"] == repeat}
                if set(pair) != {"serial", "overlap"} or any(
                    s["id"] not in by_id or s["id"] in invalid for s in pair.values()
                ):
                    record["complete"] = False
                    continue
                reference, candidate = (by_id[pair[m]["id"]][metric] for m in ("serial", "overlap"))
                ratio = candidate / reference
                if not positive(ratio):
                    record["complete"] = False
                    continue
                ratios.append(ratio)
            value = median(ratios) if ratios and record["complete"] else None
            record[metric] = {
                "repeat_ids": repeats,
                "paired_ratios": ratios,
                "median_ratio": value,
                "reduction": None if value is None else 1 - value,
                "range": [min(ratios), max(ratios)] if ratios else None,
                "required_max_ratio": threshold if cid in LONG_IDS else None,
                "threshold_met": value is not None and value <= threshold,
            }
            if any(r > 1 for r in ratios):
                regressions.append(
                    {
                        "case_id": cid,
                        "metric": metric,
                        "paired_ratios": ratios,
                        "median_ratio": value,
                    }
                )
        record["thresholds_met"] = record["complete"] and all(
            record[m]["threshold_met"] for m in METRICS
        )
        cases.append(record)
    complete = not (failures or missing or extras or duplicates) and all(
        c["complete"] for c in cases
    )
    targets = [c for c in cases if c["target_long"]]
    return {
        "schema": "qwen38-prefill-overlap-analysis-v1",
        "population": schedule["population"],
        "attempted_rows": rows,
        "accepted": False,
        "evidence_validation": "not_evaluated",
        "coverage_complete": complete,
        "expected_requests": len(expected),
        "observed_requests": len(rows),
        "missing_ids": missing,
        "extra_ids": extras,
        "duplicate_ids": duplicates,
        "failed_rows": failures,
        "cases": cases,
        "regressions": regressions,
        "statistical_targets_met": complete
        and len(targets) == 6
        and all(c["thresholds_met"] for c in targets),
        "statistics": (
            "Descriptive median of paired repeat ratios; "
            "no confidence intervals; no pooled override."
        ),
    }


def audit_pressure(guard, expected_source, *, max_gap):
    require(guard.get("require_normal_pressure") is True, "Pressure guard absent/disabled")
    require(guard.get("guard_source_sha256") == expected_source, "Pressure guard source differs")
    require(positive(max_gap) and 0.5 <= max_gap <= 1.0, "Unbounded pressure sampling gap")
    require(
        type(guard.get("interval_seconds")) in (int, float) and guard["interval_seconds"] == 0.5,
        "Pressure interval must be 0.5 seconds",
    )
    start, end = guard.get("spawn_seconds"), guard.get("execution_end_seconds")
    require(positive(start) and positive(end) and end >= start, "Missing execution interval")
    samples = guard.get("samples")
    require(
        isinstance(samples, list) and 2 <= len(samples) <= 10000,
        "Missing/budget-exceeded pressure measurements",
    )
    previous, times = None, []
    for sample in samples:
        require(
            set(sample) == {"monotonic_seconds", "pressure", "read_error"},
            "Missing pressure measurement fields",
        )
        current = sample["monotonic_seconds"]
        require(positive(current), "Nonfinite pressure timestamp")
        require(
            type(sample["pressure"]) is int
            and sample["pressure"] == 1
            and sample["read_error"] is None,
            "Abnormal pressure or read error",
        )
        require(
            previous is None or 0 < current - previous <= max_gap, "Uncovered pressure interval"
        )
        times.append(current)
        previous = current
    require(
        times[0] <= start <= times[-1] and times[0] <= end <= times[-1],
        "Uncovered execution interval",
    )
    require(
        start - times[0] <= max_gap and times[-1] - end <= max_gap,
        "Pressure boundary measurement stale",
    )
    return {
        "scope": "sampled_not_continuous",
        "sample_count": len(samples),
        "max_gap_seconds": max_gap,
    }


def audit_numeric_inventory(actual, expected):
    from .protocol import same

    require(
        isinstance(expected, list) and bool(expected) and isinstance(actual, list),
        "Missing numerical inventory",
    )
    fields = ("tree", "layer", "part", "shape", "dtype", "offset")
    keys, observed = {}, set()
    for spec in expected:
        key = (spec["tree"], spec["layer"], spec["part"])
        require(key not in keys and set(spec) == set(fields), "Invalid trusted numerical plan")
        keys[key] = spec
    for row in actual:
        key = (row.get("tree"), row.get("layer"), row.get("part"))
        require(key in keys and key not in observed, "Unknown/duplicate/replaced cache part")
        observed.add(key)
        require(
            same({k: row.get(k) for k in fields}, keys[key]),
            "Cache shape/dtype/offset inventory differs",
        )
        require(
            row.get("reference_finite") is True and row.get("candidate_finite") is True,
            "Nonfinite/missing reference/candidate",
        )
        require(
            type(row.get("atol")) is float
            and row["atol"] == 1e-6
            and type(row.get("rtol")) is float
            and row["rtol"] == 1e-5,
            "Tolerance changed",
        )
        error, tolerance, ratio = (
            row.get(k) for k in ("max_abs_error", "max_tolerance", "max_error_over_tolerance")
        )
        require(
            all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in (error, tolerance, ratio)
            ),
            "Invalid numeric comparison metrics",
        )
        require(
            tolerance >= 1e-6
            and ratio <= 1
            and error <= tolerance
            and ratio >= error / tolerance
            and ((error == 0) == (ratio == 0)),
            "Inconsistent/failed strict metrics",
        )
    require(observed == set(keys), "Missing cache/logit/hidden inventory parts")
    return {
        "parts": len(observed),
        "scope": "recorded metrics only; live numerical capability separate",
    }


def audit_record(record, trusted, schedule, record_root):
    from .evidence import audit_record as collect_record

    return collect_record(record, trusted, schedule, record_root)
