"""Parent-owned025 supervisor hook. Plans are inert; no automatic admission.

Only explicit execute + an injected parent guard can reach a subprocess. Private
profile arguments are read only after that guard and are never returned/logged.
"""

from pathlib import Path

from . import harness as h
from . import integration025 as i
from .cli import HISTORICAL, encoded, exclusive_dir, plan

PHASES = ("pilot", *(f"batch-{n:02d}" for n in range(6)))
DEADLINE = 900
ORIGINAL_ACCEPTANCE_SHA = "22b37500f5141e8a105b336751b32f27eb08c101b4ffcca81e0f206d078143af"


def expected_pressure_policy(manifest):
    """Validate external freeze/amendment, never infer policy from raw receipts."""
    policy = manifest.get("pressure_policy", "normal")
    h.require(policy in ("normal", "noncritical"), "Unknown frozen pressure policy")
    amendment = manifest.get("memory_acceptance")
    if policy == "normal":
        h.require(amendment is None, "Normal policy cannot carry a pressure amendment")
        return policy
    h.require(type(amendment) is dict, "Noncritical requires external memory acceptance")
    data = amendment.get("bytes_utf8")
    h.require(type(data) is str, "Missing exact amendment bytes")
    data = data.encode("utf-8")
    h.require(
        len(data) <= 64 * 1024 and h.sha(data) == amendment.get("sha256"),
        "Memory acceptance SHA256 differs",
    )
    value = h.decode_json(data)
    h.require(type(value) is dict, "Memory acceptance must be an object")
    expected = {
        "schema": "qwen38-overlap-memory-acceptance-amendment-v1",
        "original_acceptance_sha256": ORIGINAL_ACCEPTANCE_SHA,
        "pressure_policy": "noncritical",
        "allowed_pressure_levels": [1, 2],
        "abort_pressure_levels": [4],
    }
    for key, wanted in expected.items():
        h.require(
            encoded(value.get(key)) == encoded(wanted), "Memory acceptance policy/binding differs"
        )
    original = i.ROOT / "results/qwen38-prefill-overlap-v1/acceptance.json"
    h.require(
        h.sha(h.read_bytes(original)) == ORIGINAL_ACCEPTANCE_SHA,
        "Original acceptance source differs",
    )
    return policy


def freeze(
    out_dir,
    target,
    small,
    *,
    pressure_policy="normal",
    memory_acceptance=None,
    memory_acceptance_sha256=None,
):
    policy_fields = {}
    h.require(pressure_policy in ("normal", "noncritical"), "Unknown pressure policy")
    if pressure_policy == "noncritical":
        h.require(
            memory_acceptance is not None and memory_acceptance_sha256 is not None,
            "Noncritical requires memory acceptance path/SHA256 pair",
        )
        h.load_bound(memory_acceptance, memory_acceptance_sha256, limit=64 * 1024)
        data = h.read_bytes(memory_acceptance, 64 * 1024)
        policy_fields = {
            "pressure_policy": pressure_policy,
            "memory_acceptance": {
                "sha256": memory_acceptance_sha256,
                "bytes_utf8": data.decode("utf-8"),
            },
        }
        expected_pressure_policy(policy_fields)
    else:
        h.require(
            memory_acceptance is None and memory_acceptance_sha256 is None,
            "Memory amendment requires explicit noncritical policy",
        )
    numerical = i.expected_plan(target, small)
    h.require(
        Path(target["path"]).resolve()
        == (i.ROOT / "models/qwen3.8-27b-4bit-tokenizer-only").resolve(),
        "Require canonical target metadata directory",
    )
    full = h.derive_schedule(HISTORICAL)
    pilot = h.derive_pilot(full)["batches"][0]
    schedules = {"pilot": pilot, **{f"batch-{n:02d}": b for n, b in enumerate(full["batches"])}}
    # Validate all source bytes and architecture before creating an exclusive tree.
    names = i.source_names(i.RUNTIME)
    runtime = {n: h.read_bytes(i.RUNTIME / n, 2 * 1024**2) for n in names}
    output = exclusive_dir(out_dir)
    plan(output / "schedules")
    manifest = {
        "schema": "qwen38-overlap-integration-freeze-v2",
        "artifact_kind": "PUBLIC_PLAN_NOT_APPROVAL",
        "hardware_executed": False,
        "execution_permitted": False,
        "target": target,
        "small": small,
        "numerical_plan": numerical,
        "numerical_plan_sha256": h.sha(encoded(numerical)),
        "phases": {},
        "harness_sources": {
            str(p.relative_to(i.ROOT)): h.sha(h.read_bytes(p))
            for p in sorted((i.ROOT / "benchmarks/qwen38_overlap").glob("*.py"))
        },
        **policy_fields,
    }
    for phase, schedule in schedules.items():
        snapshot = output / ("runtime-" + phase)
        snapshot.mkdir()
        actual = {**runtime, "benchmark_schedule.json": encoded(schedule)}
        for name, data in actual.items():
            with (snapshot / name).open("xb") as handle:
                handle.write(data)
        manifest["phases"][phase] = {
            "source_dir": snapshot.name,
            "sources": {n: h.sha(v) for n, v in actual.items()},
            "schedule_sha256": h.sha(actual["benchmark_schedule.json"]),
            "requests": len(schedule["requests"]),
            "max_tokens": max(r["max_tokens"] for r in schedule["requests"]),
            "deadline_seconds": DEADLINE,
        }
    with (output / "integration-freeze.json").open("xb") as handle:
        handle.write(encoded(manifest))
    return manifest


def read_freeze(path, digest):
    manifest = h.load_bound(path, digest, limit=8 * 1024**2)
    expected_pressure_policy(manifest)
    h.require(
        manifest.get("schema") == "qwen38-overlap-integration-freeze-v2"
        and set(manifest["phases"]) == set(PHASES),
        "Invalid integration freeze",
    )
    h.require(
        manifest["numerical_plan"] == i.expected_plan(manifest["target"], manifest["small"])
        and manifest["numerical_plan_sha256"] == h.sha(encoded(manifest["numerical_plan"])),
        "Independent numerical plan differs",
    )
    expected_harness = {
        str(p.relative_to(i.ROOT)) for p in (i.ROOT / "benchmarks/qwen38_overlap").glob("*.py")
    }
    h.require(
        set(manifest["harness_sources"]) == expected_harness, "Require exact harness source set"
    )
    for name, expected in manifest["harness_sources"].items():
        h.require(
            name.startswith("benchmarks/qwen38_overlap/")
            and ".." not in Path(name).parts
            and h.sha(h.read_bytes(i.ROOT / name)) == expected,
            "Frozen harness source differs",
        )
    return manifest


def run_plan(path, digest, phase, out_dir):
    manifest = read_freeze(path, digest)
    h.require(phase in PHASES, "Unknown frozen phase")
    item = manifest["phases"][phase]
    source = Path(path).absolute().parent / item["source_dir"]
    h.require(Path(item["source_dir"]).name == item["source_dir"], "Unsafe frozen source path")
    with i.runtime_api(source, item["sources"]) as api:
        schedule = h.load_bound(source / "benchmark_schedule.json", item["schedule_sha256"])
        api("workload_schedule").validate_schedule(schedule)
    full = h.derive_schedule(HISTORICAL)
    expected = (
        h.derive_pilot(full)["batches"][0] if phase == "pilot" else full["batches"][int(phase[-2:])]
    )
    h.require(
        schedule == expected
        and item["requests"] == len(expected["requests"])
        and item["deadline_seconds"] == DEADLINE,
        "Frozen schedule rows/count/deadline differs",
    )
    target, small = manifest["target"], manifest["small"]
    argv = [
        str(i.ROOT / ".venv/bin/python"),
        str(source / "supervise.py"),
        "--output",
        str(Path(out_dir).absolute() / "raw"),
        "--local-metadata",
        target["path"],
        "--split",
        "28",
        "36",
        "--eval-group-size",
        "32",
        "--rank0-verifier",
        "native",
        "--rank1-verifier",
        "slot-qmv",
        "--execution",
        "phased",
        "--repeats",
        str(item["requests"]),
        "--max-tokens",
        str(item["max_tokens"]),
        "--deadline",
        str(DEADLINE),
        "--reserve-bytes",
        str(8 * 1024**3),
        "--prompt",
        "tokens:[1,2,3,4,5]",
        "--target-revision",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "--target-config-sha256",
        target["config_sha256"],
        "--target-index-sha256",
        target["index_sha256"],
        "--small-model-local",
        small["path"],
        "--small-config-sha256",
        small["config_sha256"],
        "--small-index-sha256",
        small["index_sha256"],
        "--overlapped-prefill",
        "--hidden-only-prefill",
        "--require-normal-pressure",
        "--hidden-gate-chunk-size",
        "256",
        "--prefix-rollback",
        "--benchmark-schedule",
        "--benchmark-schedule-sha256",
        item["schedule_sha256"],
    ]
    policy = expected_pressure_policy(manifest)
    fields = {}
    if policy == "noncritical":
        offset = argv.index("--require-normal-pressure")
        argv[offset : offset + 1] = ["--pressure-policy", "noncritical"]
        fields = {
            "pressure_policy": policy,
            "memory_acceptance_sha256": manifest["memory_acceptance"]["sha256"],
        }
    return {
        "schema": "qwen38-overlap-supervisor-plan-v2",
        "manifest_sha256": digest,
        "phase": phase,
        "requests": item["requests"],
        "deadline_seconds": DEADLINE,
        "public_argv": argv,
        "hardware_executed": False,
        "accepted": False,
        "execution_permitted": False,
        **fields,
    }


def guarded_run(
    path,
    digest,
    phase,
    out_dir,
    *,
    execute=False,
    profile_path=None,
    parent_guard=None,
    runner=None,
    pilot_admission=None,
):
    result = run_plan(path, digest, phase, out_dir)
    if not execute:
        return result
    h.require(parent_guard is not None, "parent_only_guarded_supervisor_unavailable")
    h.require(
        phase == "pilot" or pilot_admission is not None,
        "Source-bound real pilot admission required before batches",
    )
    if phase != "pilot":
        h.require(
            type(pilot_admission) is tuple and len(pilot_admission) == 3,
            "Require pilot raw directory + external trust path/SHA256, not a passed flag",
        )
        pilot_root, pilot_trust_path, pilot_trust_sha = pilot_admission
        pilot_trust = h.load_bound(pilot_trust_path, pilot_trust_sha, limit=32 * 1024**2)
        h.require(
            pilot_trust.get("phase") == "pilot" and pilot_trust.get("manifest_sha256") == digest,
            "Independent pilot identity differs",
        )
        pilot = i.audit_raw_trial(pilot_root, pilot_trust, read_freeze(path, digest))
        h.require(
            pilot["physical_accepted"] is True
            and pilot.get("pilot_budget", {}).get("passed") is True,
            "Real source-bound pilot/budget not admitted",
        )
    from copy import deepcopy

    authorization = parent_guard(deepcopy(result))
    h.require(
        type(authorization) is dict
        and authorization.get("approved") is True
        and authorization.get("manifest_sha256") == digest
        and authorization.get("phase") == phase,
        "Parent approval does not bind this exact plan",
    )
    fixture = authorization.get("artifact_kind") == i.TEST_KIND
    h.require(
        fixture or authorization.get("artifact_kind") == "PARENT_REVIEWED_PHYSICAL_APPROVAL",
        "Missing explicit parent approval kind",
    )
    h.require(not fixture or runner is not None, "TEST approval cannot start the actual supervisor")
    h.require(profile_path is not None, "Explicit private profile path required only for execute")
    # Bounded, exact whitelist; no arbitrary extra argv or shell interpolation.
    profile = h.decode_json(h.read_bytes(profile_path, 64 * 1024))
    private_keys = {
        "remote_source",
        "ssh_key",
        "ssh_target",
        "local_address",
        "remote_address",
        "ssh_host_key_alias",
        "small_model_remote",
        "sidecar",
        "remote_root_by_phase",
    }
    h.require(
        set(profile) == private_keys and set(profile["remote_root_by_phase"]) == set(PHASES),
        "Private profile key set differs",
    )
    remote_roots = profile.pop("remote_root_by_phase")
    h.require(
        len(set(remote_roots.values())) == len(PHASES),
        "Remote code directories must be distinct per phase",
    )
    profile["remote_root"] = remote_roots[phase]
    h.require(
        all(type(v) is str and v and "\x00" not in v for v in profile.values()),
        "Invalid private profile values",
    )
    argv = list(result["public_argv"])
    for key, value in profile.items():
        argv.extend(["--" + key.replace("_", "-"), value])
    # Parse/validate the exact real025 options without running its transport.
    source = Path(argv[1]).parent
    sources = read_freeze(path, digest)["phases"][phase]["sources"]
    from contextlib import redirect_stderr
    from io import StringIO

    with i.runtime_api(source, sources) as api:
        try:
            with redirect_stderr(StringIO()):
                options = api("supervise").parser().parse_args(argv[2:])
                api("supervise").validate_args(options)
        except (ValueError, OSError, SystemExit):
            raise ValueError("Private profile/options validation failed") from None
    output = exclusive_dir(out_dir)
    with (output / "plan.json").open("xb") as handle:
        handle.write(encoded(result))
    if runner is None:
        import subprocess

        runner = subprocess.run
    try:
        completion = runner(argv, timeout=DEADLINE + 30, check=False, capture_output=True)
        returncode = completion.returncode
    except Exception:
        # TimeoutExpired/OSError may contain credential-bearing argv; discard it.
        return {
            "schema": "qwen38-overlap-guarded-run-v2",
            "accepted": False,
            "phase": phase,
            "manifest_sha256": digest,
            "status": "supervisor_exception_cleanup_unverified",
            "hardware_executed": not fixture,
        }
    # Controller exit0 is NOT evidence acceptance. Readback/collector is separate.
    return {
        "schema": "qwen38-overlap-guarded-run-v2",
        "accepted": False,
        "phase": phase,
        "manifest_sha256": digest,
        "returncode": returncode,
        "status": "raw_receipt_audit_required",
        "hardware_executed": not fixture,
    }
