"""CPU-only plan/collect/analyze/audit CLI; never starts SSH or a model."""

import argparse
import json
from pathlib import Path

from . import harness as h
from .evidence import INTEGRATION_STATUS

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "results/qwen38-broad-v1/workloads/manifest.json"
FLAGS = [
    "--hidden-only-prefill",
    "--overlapped-prefill",
    "--require-normal-pressure",
    "--prefix-rollback",
    "--benchmark-schedule",
]


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def exclusive_dir(path):
    path = Path(path).absolute()
    h.require(all(not p.is_symlink() for p in (path, *path.parents)), "Symlink output not admitted")
    path.mkdir(parents=True, exist_ok=False)
    return path


def plan(out_dir=None):
    full = h.derive_schedule(HISTORICAL)
    pilot = h.derive_pilot(full)
    artifacts = {"schedule.json": encoded(full), "pilot.json": encoded(pilot)}
    artifacts.update(
        {f"batch-{i:02d}.json": encoded(batch) for i, batch in enumerate(full["batches"])}
    )
    rows = h.flatten(full)
    result = {
        "schema": "qwen38-overlap-plan-v1",
        "requests": len(rows),
        "measured": sum(not r["warmup"] for r in rows),
        "warmups": sum(r["warmup"] for r in rows),
        "pilot_requests": len(h.flatten(pilot)),
        "batch_sizes": [len(b["requests"]) for b in full["batches"]],
        "historical_manifest_sha256": h.HISTORICAL_SHA,
        "design_sha256": h.DESIGN_SHA,
        "artifact_sha256": {name: h.sha(data) for name, data in artifacts.items()},
        "mandatory_supervisor_flags": FLAGS,
        "execution_permitted": False,
        "integration_status": INTEGRATION_STATUS,
        "conventions": {
            "loading": (
                "One loaded process pair per category batch; "
                "loading/gates excluded from request timing."
            ),
            "warmup": (
                "Two original short queue prompts cap32 per loaded batch; "
                "no first-long-shape exclusion."
            ),
            "prefill": (
                "Both MTP hidden-only; serial versus overlap; "
                "native label denotes historical ORDER SLOT only."
            ),
            "timing": (
                "Rank0 inclusive first_token_seconds and wall_seconds; "
                "phases non-additive; no cross-host clock subtraction."
            ),
            "EOS": (
                "EOS IDs248044/248046 included as output events; stop iff final EOS; "
                "length iff exact original cap and no EOS."
            ),
            "statistics": (
                "Two paired repeats descriptive, not CI/tail/general-workload guarantees."
            ),
            "pressure": "Mandatory 0.5s both-host sampled guard, not a continuous guarantee.",
        },
    }
    if out_dir is not None:
        output = exclusive_dir(out_dir)
        for name, data in {**artifacts, "freeze.json": encoded(result)}.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
    return result


def execute_guarded(value, supervisor=None):
    h.require(supervisor is not None, "parent_only_guarded_supervisor_unavailable")
    # Only a parent-installed guarded supervisor may act on this non-approved plan.
    # No subprocess fallback, private profile argv or executable-path option.
    return supervisor(value)


def bound_schedule(path, digest):
    value = h.load_bound(path, digest, limit=8 * 1024 * 1024)
    full = h.derive_schedule(HISTORICAL)
    h.require(
        any(encoded(value) == encoded(expected) for expected in (full, h.derive_pilot(full))),
        "Schedule is not the exact frozen full84 or separate pilot6",
    )
    return value


def collect(args):
    schedule = bound_schedule(args.schedule, args.schedule_sha256)
    trusted = h.load_bound(args.trusted, args.trusted_sha256, limit=32 * 1024 * 1024)
    h.require(
        trusted.get("schedule_sha256") == args.schedule_sha256,
        "External trusted schedule SHA differs",
    )
    h.require(
        1 <= len(args.record) <= 6 and len(args.record) == len(args.record_sha256),
        "Require 1..6 explicit record path/hash pairs",
    )
    h.require(len(set(args.record)) == len(args.record), "Duplicate record input")
    audits, rows, failures, batch_indices = [], [], [], []
    for index, (path, digest) in enumerate(zip(args.record, args.record_sha256, strict=True)):
        record = h.load_bound(path, digest, limit=32 * 1024 * 1024)
        try:
            manifest = trusted
            if trusted.get("schema") == "qwen38-overlap-trusted-suite-v1":
                manifest = trusted["batches"][str(record["batch_index"])]
            audit = h.audit_record(record, manifest, schedule, Path(path).parent)
            audits.append({k: v for k, v in audit.items() if k != "rows"})
            rows.extend(audit["rows"])
            batch_indices.append(audit["batch_index"])
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            message = str(error) if isinstance(error, ValueError) else type(error).__name__
            failures.append({"record_index": index, "error": message})
            # Preserve failed/partial attempted rows, but only public row fields.
            for rank in record.get("ranks", [])[:1]:
                rows.extend(public_rows(rank.get("requests", [])))
    report = h.analyze(rows, schedule)
    contract_valid = not failures and len(set(batch_indices)) == len(batch_indices)
    report.update(
        evidence_validation="contract_valid_integration_pending" if contract_valid else "invalid",
        evidence_contract_valid=contract_valid,
        record_audits=audits,
        evidence_errors=failures,
        integration_status=INTEGRATION_STATUS,
        accepted=False,
        complete_suite=False,
        artifact_kind=trusted.get("artifact_kind", "MIXED_UNVERIFIED"),
        inputs={
            "trusted_sha256": args.trusted_sha256,
            "schedule_sha256": args.schedule_sha256,
            "record_sha256": args.record_sha256,
        },
    )
    return report


def public_rows(rows):
    fields = {
        "id",
        "case_id",
        "method",
        "prompt",
        "max_tokens",
        "warmup",
        "repeat",
        "prefill_mode",
        "source_request_id",
        "status",
        "first_token_seconds",
        "wall_seconds",
        "token_ids",
        "finish_reason",
        "prompt_token_ids",
        "local_peak_mlx_bytes",
        "peak_mlx_bytes_by_rank",
        "accepted_drafts",
        "verification_rounds",
        "teacher_forwards",
    }
    return [{k: v for k, v in r.items() if k in fields} for r in rows if isinstance(r, dict)]


def csv_safe(value):
    text = str(value)
    return (
        "'" + text
        if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n"))
        else text
    )


def write_reports(report, out_dir):
    import csv
    import io

    output = exclusive_dir(out_dir)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "case_id",
            "formatted_prompt_tokens",
            "target_long",
            "complete",
            "metric",
            "repeat_ids",
            "paired_ratios",
            "median_ratio",
            "elapsed_reduction",
            "required_max_ratio",
            "threshold_met",
        ]
    )
    for case in report["cases"]:
        for metric in h.METRICS:
            row = case[metric]
            writer.writerow(
                [
                    csv_safe(v)
                    for v in (
                        case["case_id"],
                        case["formatted_prompt_tokens"],
                        case["target_long"],
                        case["complete"],
                        metric,
                        json.dumps(row["repeat_ids"]),
                        json.dumps(row["paired_ratios"]),
                        row["median_ratio"],
                        row["reduction"],
                        row["required_max_ratio"],
                        row["threshold_met"],
                    )
                ]
            )
    markdown = [
        "# Qwen3.8 overlap descriptive report",
        "",
        "Artifact kind: " + report.get("artifact_kind", "UNVERIFIED_INPUT"),
        "",
        "**Not physical acceptance.** Evidence status: " + report["evidence_validation"],
        "",
        "Two repeats are descriptive, not confidence intervals or tail-latency guarantees.",
        (
            "Elapsed reduction = 1 - median(paired candidate/reference ratios). "
            "No pooled pass override."
        ),
        "",
        (
            "Loading/gates excluded; two queue warmups per batch included in coverage "
            "but not estimators."
        ),
        (
            "No first-long-shape exclusion. Inclusive rank0 TTFT/request wall; "
            "phases are non-additive."
        ),
        "Both modes MTP hidden-only + prefix rollback; serialized MTP is never native width-one.",
        (
            "Original caps retained. EOS248044/248046 counts as an output event; "
            "length-capped is not natural EOS."
        ),
        (
            "Pressure evidence is sampled, never a continuous guarantee. "
            "No simultaneous GPU-kernel claim."
        ),
        "",
        "| Case | TTFT ratio | Wall ratio | Complete | Long thresholds |",
        "|---|---:|---:|---|---|",
    ]
    if report.get("pressure_policy") == "noncritical":
        markdown[2:2] = [
            "Pressure policy: **noncritical** (amended levels 1/2; critical4 abort; "
            "not normal-pressure qualification).",
            "",
            *[
                f"- {entry['phase']} / {actor['host_role']} / {actor['actor_role']}: "
                f"observed_levels={actor['observed_levels']}, "
                f"warning_samples={actor['warning_samples']}"
                for entry in report.get("pressure_observations", [])
                for actor in entry["actors"]
            ],
            "",
        ]
    for case in report["cases"]:
        name = str(case["case_id"]).replace("|", "\\|").replace("\n", " ")
        markdown.append(
            f"| {name} | {case[h.METRICS[0]]['median_ratio']} | "
            f"{case[h.METRICS[1]]['median_ratio']} | {case['complete']} | "
            f"{case['thresholds_met'] if case['target_long'] else 'not targeted'} |"
        )
    markdown.extend(["", "## Regressions (all observed repeat ratios >1)", ""])
    for regression in report["regressions"]:
        markdown.append(
            f"- {regression['case_id']} / {regression['metric']}: {regression['paired_ratios']}"
        )
    if not report["regressions"]:
        markdown.append("None among the valid recorded pairs; missing/invalid pairs are not gains.")
    markdown.extend(
        [
            "",
            f"Coverage complete: {report['coverage_complete']}; "
            f"missing={len(report['missing_ids'])}, failed={len(report['failed_rows'])}, "
            f"duplicate={len(report['duplicate_ids'])}, extra={len(report['extra_ids'])}.",
            "",
            (
                "Full-suite acceptance: **false**. "
                "Runtime adapter and independent review remain pending."
            ),
        ]
    )
    artifacts = {
        "report.json": encoded(report),
        "report.csv": stream.getvalue().encode(),
        "report.md": ("\n".join(markdown) + "\n").encode(),
    }
    for name, data in {
        **artifacts,
        "artifacts.json": encoded({"artifact_sha256": {k: h.sha(v) for k, v in artifacts.items()}}),
    }.items():
        with (output / name).open("xb") as destination:
            destination.write(data)


def analyze_input(args):
    schedule = bound_schedule(args.schedule, args.schedule_sha256)
    value = h.load_bound(args.input, args.input_sha256, limit=32 * 1024 * 1024)
    rows = value.get("rows", value.get("attempted_rows"))
    h.require(
        isinstance(rows, list) and len(rows) <= 168 and all(isinstance(r, dict) for r in rows),
        "Bounded explicit dictionary row list required",
    )
    report = h.analyze(public_rows(rows), schedule)
    report.update(
        artifact_kind=value.get("artifact_kind", "UNVERIFIED_INPUT"),
        integration_status=INTEGRATION_STATUS,
        inputs={"schedule_sha256": args.schedule_sha256, "input_sha256": args.input_sha256},
    )
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--out-dir")
    p.add_argument("--execute", action="store_true")
    for name in ("collect", "audit"):
        p = commands.add_parser(name)
        p.add_argument("--schedule", required=True)
        p.add_argument("--schedule-sha256", required=True)
        p.add_argument("--trusted", required=True)
        p.add_argument("--trusted-sha256", required=True)
        p.add_argument("--record", action="append", required=True)
        p.add_argument("--record-sha256", action="append", required=True)
        p.add_argument("--out-dir")
    p = commands.add_parser("analyze")
    p.add_argument("--schedule", required=True)
    p.add_argument("--schedule-sha256", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--input-sha256", required=True)
    p.add_argument("--out-dir")
    p = commands.add_parser("collect025")
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--raw-dir", action="append", required=True)
    p.add_argument("--trusted", action="append", required=True)
    p.add_argument("--trusted-sha256", action="append", required=True)
    p.add_argument("--out-dir")
    p = commands.add_parser("freeze025")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-config-sha256", required=True)
    p.add_argument("--target-index-sha256", required=True)
    p.add_argument("--small-metadata", required=True)
    p.add_argument("--small-config-sha256", required=True)
    p.add_argument("--small-index-sha256", required=True)
    p.add_argument("--pressure-policy", choices=("normal", "noncritical"), default="normal")
    p.add_argument("--memory-acceptance")
    p.add_argument("--memory-acceptance-sha256")
    p = commands.add_parser("run025")
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--profile")
    p.add_argument("--execute", action="store_true")
    return result


def main(argv=None, *, supervisor=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "collect025":
            from .integration025 import collect_raw_trials
            from .supervisor025 import read_freeze

            manifest = read_freeze(args.manifest, args.manifest_sha256)
            h.require(
                1 <= len(args.raw_dir) <= 6
                and len(args.raw_dir) == len(args.trusted) == len(args.trusted_sha256),
                "Require exact raw/trust/hash pairs",
            )
            trials = []
            for root, path, digest in zip(
                args.raw_dir, args.trusted, args.trusted_sha256, strict=True
            ):
                trusted = h.load_bound(path, digest, limit=32 * 1024**2)
                h.require(
                    trusted.get("manifest_sha256") == args.manifest_sha256,
                    "Independent trust/freeze binding differs",
                )
                trials.append((root, trusted))
            result = collect_raw_trials(trials, manifest)
            code = 0 if result["accepted"] else 2
            if args.out_dir:
                write_reports(result, args.out_dir)
        elif args.command == "freeze025":
            from .supervisor025 import freeze

            result = freeze(
                args.out_dir,
                {
                    "path": str(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"),
                    "config_sha256": args.target_config_sha256,
                    "index_sha256": args.target_index_sha256,
                },
                {
                    "path": args.small_metadata,
                    "config_sha256": args.small_config_sha256,
                    "index_sha256": args.small_index_sha256,
                },
                pressure_policy=args.pressure_policy,
                memory_acceptance=args.memory_acceptance,
                memory_acceptance_sha256=args.memory_acceptance_sha256,
            )
            code = 0
        elif args.command == "run025":
            from .supervisor025 import guarded_run

            result = guarded_run(
                args.manifest,
                args.manifest_sha256,
                args.phase,
                args.out_dir,
                execute=args.execute,
                profile_path=args.profile,
                parent_guard=supervisor,
            )
            code = 0
        elif args.command == "plan":
            result = plan(args.out_dir)
            if args.execute:
                result = execute_guarded(result, supervisor)
            code = 0
        else:
            result = analyze_input(args) if args.command == "analyze" else collect(args)
            code = 0 if args.command == "analyze" and result["coverage_complete"] else 2
            if args.out_dir:
                write_reports(result, args.out_dir)
        print(encoded(result).decode(), end="")
        return code
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        # Never serialize profile/argv/raw receipt bodies or arbitrary exception paths.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(encoded({"accepted": False, "error": message}).decode(), end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
