"""CPU-only aggregation of normalized Qwen3.8 evidence (no runtime verification).

Public API:
    validate_pairs(rows, expected_requests, *, allow_partial=False,
                   eos_token_ids=None) -> coverage; raises EvidenceError(ValueError).
    summarize_requests(rows) -> observed descriptive statistics (no schedule claim).
    analyze_evidence(envelope, *, allow_partial=False) -> schedule-aware report.
    render_csv(report), render_markdown(report) -> export strings.

CLI: analyze.py EVIDENCE [--json FILE] [--csv FILE] [--markdown FILE]
     [--exclusive] [--allow-partial]. Without output paths, JSON goes to stdout.
Expected requests must span ALL batches, with globally unique IDs. Malformed
rows are rejected, not silently dropped. Failed attempts belong in the separate
failures array; any failure suppresses the full-suite aggregate. Fixture data
exist only in test_analyze.py; this module never fabricates benchmark evidence.

Whole-request throughput is emitted token events / wall_seconds; post-first is
(events - 1) / (wall_seconds - first_token_seconds). EOS events count as emitted.
Optional MTP counters must be a complete triple for the one-draft/two-slot engine.
Memory peaks are process-reported per-rank maxima, not additive request allocations.

Limitations: this consumes already rank-reconciled normalized evidence. It does
not load actors, tokenize prompts independently, verify gate/checkpoint/placement
receipts, or certify sidecar residency. Prompt token counts must agree across a
case; only supplied provenance.eos_token_ids enables EOS identity validation.
Absent EOS metadata, stop/length semantics and token equality are checked without
claiming EOS identity was established. Provenance is preserved without promotion.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_COUNTERS = ("accepted_drafts", "verification_rounds", "teacher_forwards")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _identity(row, repeat_key):
    for key in ("id", "case_id"):
        _require(isinstance(row.get(key), str) and row[key], f"invalid {key}")
    _require(row.get("method") in ("native", "mtp"), "invalid method")
    _require(type(row.get("warmup")) is bool, "warmup must be explicit bool")
    _require(_integer(row.get(repeat_key)), f"invalid {repeat_key}")
    _require(_integer(row.get("max_tokens"), 1), "invalid max_tokens")


def _validate_rows(rows, eos_token_ids=None):
    _require(isinstance(rows, list), "rows must be a list")
    if eos_token_ids is not None:
        _require(
            isinstance(eos_token_ids, (list, tuple))
            and eos_token_ids
            and all(_integer(t) for t in eos_token_ids),
            "invalid EOS token IDs",
        )
    ids, slots = set(), set()
    for row in rows:
        _require(isinstance(row, dict), "invalid row")
        _identity(row, "sample_repeat")
        ident = row["id"]
        slot = _slot(row, "sample_repeat")
        _require(ident not in ids and slot not in slots, f"duplicate row/slot: {ident}")
        ids.add(ident)
        slots.add(slot)
        _require(
            isinstance(row.get("prompt_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"]),
            f"{ident}: invalid prompt_sha256",
        )
        _require(_integer(row.get("prompt_tokens"), 1), f"{ident}: invalid prompt_tokens")
        tokens = row.get("token_ids")
        _require(
            isinstance(tokens, list) and tokens and all(_integer(t) for t in tokens),
            f"{ident}: actual token_ids required",
        )
        n = len(tokens)
        _require(n <= row["max_tokens"], f"{ident}: output exceeds cap")
        _require(row.get("finish_reason") in ("stop", "length"), f"{ident}: invalid finish_reason")
        _require(
            row["finish_reason"] != "length" or n == row["max_tokens"],
            f"{ident}: length stop before cap",
        )
        if eos_token_ids is not None:
            _require(not any(t in eos_token_ids for t in tokens[:-1]), f"{ident}: early EOS")
            _require(
                (tokens[-1] in eos_token_ids) == (row["finish_reason"] == "stop"),
                f"{ident}: EOS/finish_reason mismatch",
            )
        for key in ("wall_seconds", "first_token_seconds"):
            value = row.get(key)
            _require(
                type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"{ident}: invalid {key}",
            )
        wall, first = row["wall_seconds"], row["first_token_seconds"]
        _require(wall >= first and (n == 1 or wall > first), f"{ident}: impossible timing")
        if n > 1:
            _require(
                math.isfinite((n - 1) / (wall - first)), f"{ident}: nonfinite decode throughput"
            )
        _require(math.isfinite(n / wall), f"{ident}: nonfinite whole-request throughput")
        peaks = row.get("peak_mlx_bytes_by_rank")
        _require(
            isinstance(peaks, list) and len(peaks) == 2 and all(_integer(p) for p in peaks),
            f"{ident}: require two nonnegative integer rank peaks",
        )
        present = [key in row for key in _COUNTERS]
        _require(not any(present) or all(present), f"{ident}: incomplete acceptance counters")
        if all(present):
            accepted, rounds, teacher = (row[key] for key in _COUNTERS)
            _require(
                all(_integer(row[key]) for key in _COUNTERS),
                f"{ident}: invalid acceptance counters",
            )
            if row["method"] == "native":
                _require(
                    (accepted, rounds, teacher) == (0, 0, 0), f"{ident}: native acceptance counters"
                )
            else:
                _require(
                    accepted <= rounds
                    and (n - 1 == rounds + accepted or accepted > 0 and n == rounds + accepted),
                    f"{ident}: impossible one-draft acceptance/output counts",
                )
                _require(rounds <= teacher <= 2 * rounds, f"{ident}: impossible teacher_forwards")
    _consistent(rows)


class EvidenceError(ValueError):
    """Malformed, inconsistent, or incomplete benchmark evidence."""


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _slot(row, repeat_key):
    return row["case_id"], row["warmup"], row[repeat_key], row["method"]


def _consistent(rows):
    """Compare actual token IDs and metadata across methods AND repeats."""
    cases = {}
    for row in rows:
        signature = tuple(
            row[key] if key != "token_ids" else tuple(row[key])
            for key in (
                "prompt_sha256",
                "max_tokens",
                "prompt_tokens",
                "token_ids",
                "finish_reason",
            )
        )
        previous = cases.setdefault(row["case_id"], signature)
        _require(signature == previous, f"case output/metadata mismatch: {row['id']}")


def validate_pairs(rows, expected_requests, *, allow_partial=False, eos_token_ids=None):
    """Validate all scheduled rows; only explicit partial mode tolerates missing IDs.

    Expected entries use ``repeat``; evidence uses ``sample_repeat``. IDs must be
    globally unique across batches. No inference about warmups or verified gates.
    """
    _validate_rows(rows, eos_token_ids)
    _require(
        isinstance(expected_requests, list) and expected_requests,
        "expected_requests must be nonempty",
    )
    expected, slots, case_metadata, groups = {}, set(), {}, defaultdict(set)
    for entry in expected_requests:
        _require(isinstance(entry, dict), "invalid expected entry")
        _require(
            all(
                key in entry
                for key in ("id", "case_id", "method", "warmup", "repeat", "prompt", "max_tokens")
            ),
            "expected entry missing fields",
        )
        _identity(entry, "repeat")
        _require(
            isinstance(entry["prompt"], str) and entry["prompt"],
            "prompt must be nonempty UTF-8 text",
        )
        _require(entry["id"] not in expected, f"duplicate expected id: {entry['id']}")
        slot = _slot(entry, "repeat")
        _require(slot not in slots, f"duplicate expected slot: {slot}")
        slots.add(slot)
        expected[entry["id"]] = entry
        groups[slot[:3]].add(entry["method"])
        metadata = entry["prompt"], entry["max_tokens"]
        _require(
            case_metadata.setdefault(entry["case_id"], metadata) == metadata,
            f"case prompt/cap mismatch: {entry['case_id']}",
        )
    _require(
        all(methods == {"native", "mtp"} for methods in groups.values()),
        "unpaired expected schedule",
    )
    seen = set()
    for row in rows:
        _require(isinstance(row, dict) and "id" in row, "row missing id")
        ident = row["id"]
        _require(ident not in seen, f"duplicate row: {ident}")
        _require(ident in expected, f"extra row: {ident}")
        seen.add(ident)
        entry = expected[ident]
        for key in ("case_id", "method", "warmup", "max_tokens"):
            _require(row.get(key) == entry[key], f"{ident}: schedule {key} mismatch")
        _require(row.get("sample_repeat") == entry["repeat"], f"{ident}: repeat mismatch")
        _require(
            row.get("prompt_sha256") == hashlib.sha256(entry["prompt"].encode("utf-8")).hexdigest(),
            f"{ident}: prompt_sha256 mismatch",
        )
    missing = sorted(set(expected) - seen)
    _require(allow_partial or not missing, f"missing request IDs: {missing}")
    return {
        "complete": not missing,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "measured_count": sum(not r["warmup"] for r in rows),
        "warmup_count": sum(r["warmup"] for r in rows),
        "missing_ids": missing,
    }


def _stats(values):
    values = sorted(v for v in values if v is not None)
    median = None
    if values:
        low, high = values[(len(values) - 1) // 2], values[len(values) // 2]
        median = low + (high - low) / 2  # Positive metrics: avoid overflow in (low + high) / 2.
    return {
        "count": len(values),
        "median": median,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def _request_metrics(row):
    n = len(row["token_ids"])
    return dict(
        row,
        output_tokens=n,
        post_first_tps=(n - 1) / (row["wall_seconds"] - row["first_token_seconds"])
        if n > 1
        else None,
        whole_request_tps=n / row["wall_seconds"],
    )


def _method_summary(rows) -> dict:
    result: dict = {"count": len(rows)}
    for metric in ("post_first_tps", "first_token_seconds", "wall_seconds", "whole_request_tps"):
        result[metric] = _stats([r[metric] for r in rows])
    decode = [r["post_first_tps"] for r in rows if r["post_first_tps"] is not None]
    result["post_first_excluded_count"] = len(rows) - len(decode)
    result["post_first_below_10_count"] = sum(v < 10 for v in decode)
    result["post_first_below_10_percent"] = (
        100 * sum(v < 10 for v in decode) / len(decode) if decode else None
    )
    result["peak_mlx_bytes_by_rank"] = (
        [max(r["peak_mlx_bytes_by_rank"][rank] for r in rows) for rank in (0, 1)]
        if rows
        else [None, None]
    )
    result["acceptance"] = None
    if rows and rows[0]["method"] == "mtp":
        counters = [r for r in rows if all(key in r for key in _COUNTERS)]
        sums = {key: sum(r[key] for r in counters) for key in _COUNTERS}
        result["acceptance"] = dict(
            sums,
            weighted_rate=sums["accepted_drafts"] / sums["verification_rounds"]
            if sums["verification_rounds"]
            else None,
            rows_with_counters=len(counters),
            rows_without_counters=len(rows) - len(counters),
        )
    return result


def _case_summary(rows) -> dict:
    methods: dict = {
        method: _method_summary([r for r in rows if r["method"] == method])
        for method in ("native", "mtp")
    }
    repeats = defaultdict(dict)
    for row in rows:
        repeats[row["sample_repeat"]][row["method"]] = row
    paired, incomplete = [], []
    for repeat, pair in sorted(repeats.items()):
        missing = sorted({"native", "mtp"} - set(pair))
        if missing:
            incomplete.append({"sample_repeat": repeat, "missing_methods": missing})
            continue
        native, mtp = pair["native"], pair["mtp"]
        decode = (
            mtp["post_first_tps"] / native["post_first_tps"]
            if native["post_first_tps"] is not None
            else None
        )
        whole = native["wall_seconds"] / mtp["wall_seconds"]
        _require(
            (decode is None or math.isfinite(decode) and decode > 0)
            and math.isfinite(whole)
            and whole > 0,
            "nonfinite/impossible paired speedup",
        )
        paired.append(
            {
                "sample_repeat": repeat,
                "native_id": native["id"],
                "mtp_id": mtp["id"],
                "decode_speedup": decode,
                "whole_request_speedup": whole,
            }
        )
    methods["paired"] = {
        "repeats": paired,
        "incomplete_repeats": incomplete,
        "decode_speedup": _stats([p["decode_speedup"] for p in paired]),
        "whole_request_speedup": _stats([p["whole_request_speedup"] for p in paired]),
    }
    return methods


def _overall(cases):
    result: dict = {
        "interpretation": "Equal case weight: geometric mean of each case's median paired ratio. "
        "Lower-tail minimum is descriptive, not a confidence interval; "
        "tiny repeat counts do not establish tail reliability. "
        "No cross-case pooled tokens/sec; "
        "no quality scoring or runtime/numerical gate verification."
    }
    for label, metric in (("decode", "decode_speedup"), ("whole_request", "whole_request_speedup")):
        values = [
            case["paired"][metric]["median"]
            for case in cases.values()
            if case["paired"][metric]["median"] is not None
        ]
        result[label] = {
            "case_count": len(values),
            "excluded_case_count": len(cases) - len(values),
            "geometric_mean": math.exp(statistics.fmean(math.log(v) for v in values))
            if values
            else None,
            "lower_tail_min": min(values) if values else None,
            "case_speedup_statistics": _stats(values),
        }
    return result


def summarize_requests(rows):
    """Summarize VALID normalized evidence; validate_pairs checks schedule coverage.

    Every explicit warmup is retained separately, never inferred from repeat=0.
    Single-token outputs contribute wall/TTFT/whole metrics but not post-first
    decode. Stdev is sample stdev (undefined for n<2); no tail/CI claims.
    """
    _validate_rows(rows)
    enriched = [_request_metrics(row) for row in rows]
    measured = [r for r in enriched if not r["warmup"]]
    cases = {
        case_id: _case_summary([r for r in measured if r["case_id"] == case_id])
        for case_id in sorted({r["case_id"] for r in rows})
    }
    return {
        "observed_count": len(rows),
        "measured_count": len(measured),
        "warmup_count": len(rows) - len(measured),
        "requests": measured,
        "warmups": [r for r in enriched if r["warmup"]],
        "cases": cases,
        "overall": _overall(cases),
        "coverage_status": "schedule_not_checked",
    }


def analyze_evidence(evidence, *, allow_partial=False):
    """Validate an evidence envelope, retaining failures and every expected case.

    Runtime provenance is passed through, never upgraded to verified status.
    ``overall`` is suppressed for missing rows or ANY recorded failure. In that
    situation ``observed_only_overall`` is descriptive survivor-only evidence,
    explicitly NOT a full-suite performance claim or failure-adjusted rate.
    Optional provenance.eos_token_ids enables actual EOS membership checking;
    absent tokenizer metadata only stop/length consistency can be checked.
    Coverage's legacy expected_case_count includes all scheduled case IDs;
    expected_measured_case_count and expected_warmup_case_count distinguish the
    two unique-ID sets (which may overlap), not the number of request rows.
    """
    _require(
        isinstance(evidence, dict) and evidence.get("schema") == "qwen38-workload-evidence-v1",
        "invalid evidence schema",
    )
    _require(
        all(key in evidence for key in ("expected_requests", "rows", "failures", "provenance")),
        "evidence missing required fields",
    )
    _require(
        isinstance(evidence["failures"], list)
        and all(isinstance(f, dict) for f in evidence["failures"]),
        "failures must be a list of records",
    )
    _require(isinstance(evidence["provenance"], dict), "provenance must be an object")
    eos_ids = evidence["provenance"].get("eos_token_ids")
    coverage = validate_pairs(
        evidence["rows"],
        evidence["expected_requests"],
        allow_partial=allow_partial,
        eos_token_ids=eos_ids,
    )
    result = summarize_requests(evidence["rows"])
    expected_cases = sorted({r["case_id"] for r in evidence["expected_requests"]})
    coverage.update(
        expected_case_count=len(expected_cases),
        expected_measured_case_count=len(
            {r["case_id"] for r in evidence["expected_requests"] if not r["warmup"]}
        ),
        expected_warmup_case_count=len(
            {r["case_id"] for r in evidence["expected_requests"] if r["warmup"]}
        ),
        observed_case_count=len(result["cases"]),
        expected_measured_count=sum(not r["warmup"] for r in evidence["expected_requests"]),
        expected_warmup_count=sum(r["warmup"] for r in evidence["expected_requests"]),
    )
    for case_id in expected_cases:
        case = result["cases"].setdefault(case_id, _case_summary([]))
        expected = [r for r in evidence["expected_requests"] if r["case_id"] == case_id]
        observed = [r for r in evidence["rows"] if r["case_id"] == case_id]
        missing_ids = sorted(set(r["id"] for r in expected) - set(r["id"] for r in observed))
        case["coverage"] = {
            "expected_count": len(expected),
            "observed_count": len(observed),
            "missing_ids": missing_ids,
            "expected_measured_count": sum(not r["warmup"] for r in expected),
            "expected_warmup_count": sum(r["warmup"] for r in expected),
        }
        for method in ("native", "mtp"):
            method_expected = [r for r in expected if r["method"] == method and not r["warmup"]]
            case[method]["expected_count"] = len(method_expected)
            case[method]["missing_ids"] = [
                r["id"] for r in method_expected if r["id"] in missing_ids
            ]
    failures = evidence["failures"]
    complete = coverage["complete"] and not failures
    result.update(
        schema="qwen38-workload-analysis-v1",
        coverage=coverage,
        coverage_status="complete"
        if complete
        else "complete_with_failures"
        if coverage["complete"]
        else "partial",
        failures=failures,
        failure_count=len(failures),
        provenance=evidence["provenance"],
        runtime_verification="not_performed_by_analyzer",
        eos_token_identity_checked=eos_ids is not None,
        full_suite_performance_claim=complete,
        performance_scope=(
            "Complete scheduled observed requests; not a quality or runtime verification claim."
        )
        if complete
        else "Observed successful rows only; not a full-suite performance claim. "
        "Missing requests and failures have no imputed throughput and remain listed separately.",
    )
    result["overall"] = _overall(result["cases"])
    if not complete:
        result["observed_only_overall"] = result["overall"]
        result["overall"] = None
    return result


def _json(value, **kwargs):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs)


def _flatten(value, prefix=""):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


def render_csv(report):
    """Long-form CSV: typed records, dotted metric paths, JSON-encoded values.

    Lists (including missing IDs and actual token IDs) remain intact JSON cells;
    failures have one record per original failure, never discarded from export.
    Formula-like cells receive a leading apostrophe, including after whitespace;
    this spreadsheet presentation escape never changes the JSON report/evidence.
    """
    buffer = io.StringIO(newline="")
    fieldnames: list[str] = ["record_type", "case_id", "method", "id", "metric", "value"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    def write_row(record):
        writer.writerow(
            {
                key: "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
                for key, value in record.items()
            }
        )

    def emit(kind, value, case_id="", method="", ident=""):
        for metric, leaf in _flatten(value):
            write_row(
                dict(
                    record_type=kind,
                    case_id=case_id,
                    method=method,
                    id=ident,
                    metric=metric,
                    value=_json(leaf),
                )
            )

    for key, value in report.items():
        if key not in ("cases", "requests", "warmups", "failures"):
            emit("metadata", {key: value})
    for case_id, case in report["cases"].items():
        for method in ("native", "mtp"):
            emit("case", case[method], case_id, method)
        emit("pair", case["paired"], case_id)
        if "coverage" in case:
            emit("coverage", case["coverage"], case_id)
    for kind, key in (("request", "requests"), ("warmup", "warmups")):
        for row in report[key]:
            emit(kind, row, row["case_id"], row["method"], row["id"])
    for failure in report.get("failures", []):
        write_row(
            dict(
                record_type="failure",
                case_id=failure.get("case_id", ""),
                method=failure.get("method", ""),
                id=failure.get("id", ""),
                metric="failure",
                value=_json(failure),
            )
        )
    return buffer.getvalue()


def _cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report):
    """Human report; JSON/CSV retain full precision and every normalized row."""
    coverage = report["coverage"]
    lines = [
        "# Qwen3.8 workload evidence analysis",
        "",
        f"**Coverage: {report['coverage_status']}** — "
        f"{coverage['observed_count']}/{coverage['expected_count']} requests; "
        f"{report['measured_count']} measured, {report['warmup_count']} explicit warmups; "
        f"{coverage['expected_case_count']} requested cases total "
        f"({coverage['expected_measured_case_count']} measured; "
        f"{coverage['expected_warmup_case_count']} warmup; case sets may overlap).",
        "",
        report["performance_scope"],
        "",
        "Statistics are descriptive: sample stdev is undefined for n<2. "
        "Lower-tail minima are not a confidence interval. "
        "Two repeats cannot establish reliable tail latency. Cases are never pooled as tokens/sec. "
        "Token events include any emitted EOS. "
        "Single-token outputs are excluded only from post-first decode.",
        "",
        "No quality scoring or runtime verification is performed. "
        "Rank agreement, checkpoint identity, placement, eval32, gate receipts, cleanup, "
        "and resident-sidecar memory baselines must be established upstream.",
        "",
        "EOS token identity checked using supplied metadata: "
        f"{report['eos_token_identity_checked']}.",
        "",
        "## Per-case measured statistics",
        "",
        "| Case | Method | Metric | n | Median | Min | Max | Sample stdev |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for case_id, case in report["cases"].items():
        for method in ("native", "mtp"):
            summary = case[method]
            for metric in (
                "post_first_tps",
                "first_token_seconds",
                "wall_seconds",
                "whole_request_tps",
            ):
                stats = summary[metric]
                lines.append(
                    "| "
                    + " | ".join(
                        _cell(v)
                        for v in (
                            case_id,
                            method,
                            metric,
                            stats["count"],
                            stats["median"],
                            stats["min"],
                            stats["max"],
                            stats["stdev"],
                        )
                    )
                    + " |"
                )
    lines += [
        "",
        "## Coverage, acceptance, and memory",
        "",
        "| Case | Method | Observed / expected measured | Decode excluded | Decode below 10 (%) "
        "| Acceptance accepted / rounds | Rank 0 peak bytes | Rank 1 peak bytes |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for case_id, case in report["cases"].items():
        for method in ("native", "mtp"):
            summary = case[method]
            acceptance = summary["acceptance"]
            acceptance_text = (
                "not supplied"
                if acceptance is None
                else (
                    f"{acceptance['accepted_drafts']} / {acceptance['verification_rounds']} "
                    f"(rate {_cell(acceptance['weighted_rate'])}; "
                    f"{acceptance['rows_without_counters']} rows without counters)"
                )
            )
            values = (
                case_id,
                method,
                f"{summary['count']} / {summary['expected_count']}",
                summary["post_first_excluded_count"],
                summary["post_first_below_10_percent"],
                acceptance_text,
                *summary["peak_mlx_bytes_by_rank"],
            )
            lines.append("| " + " | ".join(_cell(v) for v in values) + " |")
    lines += [
        "",
        "## Paired ratios (MTP / native throughput)",
        "",
        "| Case | Repeat | Decode speedup | Whole-request speedup |",
        "|---|---:|---:|---:|",
    ]
    for case_id, case in report["cases"].items():
        for pair in case["paired"]["repeats"]:
            lines.append(
                "| "
                + " | ".join(
                    _cell(v)
                    for v in (
                        case_id,
                        pair["sample_repeat"],
                        pair["decode_speedup"],
                        pair["whole_request_speedup"],
                    )
                )
                + " |"
            )
        lines.append(
            "| "
            + " | ".join(
                _cell(v)
                for v in (
                    case_id,
                    "median",
                    case["paired"]["decode_speedup"]["median"],
                    case["paired"]["whole_request_speedup"]["median"],
                )
            )
            + " |"
        )
    overall = (
        report["overall"] if report["overall"] is not None else report.get("observed_only_overall")
    )
    lines += [
        "",
        "## Equal-case aggregate"
        if report["overall"] is not None
        else "## Observed-only aggregate — NOT full suite",
        "",
        "```json",
        _json(overall, indent=2),
        "```",
        "",
        "## Missing request IDs",
        "",
        "```json",
        _json(coverage["missing_ids"], indent=2),
        "```",
        "",
        f"## Failures ({report['failure_count']})",
        "",
        "```json",
        _json(report["failures"], indent=2),
        "```",
        "",
        "## Explicit first-loaded warmups (excluded from measured statistics)",
        "",
        "```json",
        _json(report["warmups"], indent=2),
        "```",
        "",
        "## Provenance (as supplied, not verified here)",
        "",
        "```json",
        _json(report["provenance"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise EvidenceError(f"nonfinite JSON number: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence", type=Path, help="parent-normalized qwen38-workload-evidence-v1 JSON"
    )
    parser.add_argument("--json", "--output-json", type=Path, dest="json_path")
    parser.add_argument(
        "--csv", "--output-csv", type=Path, dest="csv_path", help="long-form metrics; JSON values"
    )
    parser.add_argument("--markdown", "--output-markdown", type=Path, dest="markdown_path")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="list every missing ID; suppress full-suite aggregate",
    )
    parser.add_argument(
        "--exclusive", action="store_true", help="create outputs with mode x; never overwrite"
    )
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(
            args.evidence.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_constant,
        )
        _json(evidence)  # Also rejects exponent overflow (e.g. 1e999) anywhere in the envelope.
        report = analyze_evidence(evidence, allow_partial=args.allow_partial)
        outputs = []
        for path, renderer in (
            (args.json_path, lambda value: _json(value, indent=2) + "\n"),
            (args.csv_path, render_csv),
            (args.markdown_path, render_markdown),
        ):
            if path is not None:
                outputs.append((path, renderer(report)))
        resolved = [path.resolve() for path, _ in outputs]
        _require(len(set(resolved)) == len(resolved), "output paths must be distinct")
        _require(args.evidence.resolve() not in resolved, "cannot overwrite evidence input")
        existing = [path for path, _ in outputs if path.exists()]
        _require(
            not any(path.samefile(args.evidence) for path in existing),
            "cannot overwrite evidence input via hardlink",
        )
        _require(
            not any(
                left.samefile(right)
                for i, left in enumerate(existing)
                for right in existing[i + 1 :]
            ),
            "output paths must not alias via hardlinks",
        )
        if args.exclusive:
            _require(
                not any(path.exists() for path, _ in outputs), "exclusive output already exists"
            )
        for path, content in outputs:
            with path.open("x" if args.exclusive else "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
        if not outputs:
            print(_json(report, indent=2))
    except (ValueError, OSError, OverflowError) as exc:
        parser.exit(2, f"evidence rejected: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
