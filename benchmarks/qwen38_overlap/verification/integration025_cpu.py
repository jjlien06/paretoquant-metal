"""CPU integration verification runner; never launches hardware or subprocesses.

Run from the repository root with PYTHONDONTWRITEBYTECODE=1 and project Python.
All emitted fixture data is explicitly TEST-only and goes to a new scratch tree.
"""

import argparse
import importlib.util
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ["TMPDIR"] = "/Users/jeremylien/.hermes/cache/scratch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    output = Path(args.run_dir).absolute()
    scratch = Path(os.environ["TMPDIR"])
    if scratch not in output.parents or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Use a new contained scratch run directory")
    output.mkdir(mode=0o700)
    runtime = ROOT / "spikes/025-qwen38-overlapped-prefill"
    spec = importlib.util.spec_from_file_location(
        "integration_cpu_guard", runtime / "cpu_only_runner.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install_guard()  # Do not call main(): its runtime ledger is immutable here.
    denied = []

    def audit(event, arguments):
        if event in ("socket.__new__", "os.kill", "os.killpg", "subprocess.Popen"):
            denied.append(event)
            raise RuntimeError("CPU verification denied external activity: " + event)

    sys.addaudithook(audit)
    import pytest

    from benchmarks.qwen38_overlap import harness as h
    from benchmarks.qwen38_overlap.cli import encoded
    from benchmarks.qwen38_overlap.cli import main as cli

    acceptance = h.decode_json(
        (ROOT / "results/qwen38-prefill-overlap-v1/acceptance.json").read_bytes()
    )
    preserved = acceptance["source_preservation_sha256"]
    mismatches = [
        name for name, digest in preserved.items() if h.sha(h.read_bytes(ROOT / name)) != digest
    ]
    original = ROOT / "results/qwen38-prefill-overlap-v1/cpu-harness-implementation.json"
    original_sha = h.sha(h.read_bytes(original))
    assert not mismatches, mismatches
    assert original_sha == "3c7a024aba420891de1d9cf7f974e9331e7bea886fe4ec48d53fb0284ca3a513"
    arguments = [
        "benchmarks/qwen38_overlap/tests",
        "-q",
        "--import-mode=importlib",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "--junitxml=" + str(output / "pytest.xml"),
        "--basetemp=" + str(output / "pytest-tmp"),
    ]
    with (
        (output / "pytest.log").open("x") as stream,
        redirect_stdout(stream),
        redirect_stderr(stream),
    ):
        exit_code = int(pytest.main(arguments))
    tree = ElementTree.parse(output / "pytest.xml")
    tests = tree.findall(".//testcase")
    old = [test for test in tests if "025" not in test.attrib["classname"]]
    counts = {
        "total": len(tests),
        "original_harness": len(old),
        "integration025": len(tests) - len(old),
        "failures": len(tree.findall(".//failure")),
        "errors": len(tree.findall(".//error")),
        "skipped": len(tree.findall(".//skipped")),
    }
    assert exit_code == 0 and counts["original_harness"] == 95, counts
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0, counts

    from benchmarks.qwen38_overlap.tests.test_complete025 import complete_raw
    from benchmarks.qwen38_overlap.tests.test_integration025 import metadata
    from benchmarks.qwen38_overlap.tests.test_supervisor025 import SMALL

    fixture_parent = output / "TEST-standalone-cli"
    fixture_parent.mkdir()
    root, trust, manifest = complete_raw.__wrapped__(fixture_parent)
    target, small = metadata(ROOT / "models/qwen3.8-27b-4bit-tokenizer-only"), metadata(SMALL)
    freeze_path = output / "cli-freeze/integration-freeze.json"
    invocations = []

    def invoke(name, argv, expected):
        with (
            (output / (name + ".stdout.json")).open("x") as stdout,
            (output / (name + ".stderr")).open("x") as stderr,
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli(argv)
        invocations.append(
            dict(
                entrypoint="benchmarks.qwen38_overlap.cli.main",
                argv=argv,
                exit_code=code,
                expected_exit_code=expected,
            )
        )
        assert code == expected, invocations[-1]
        return h.decode_json((output / (name + ".stdout.json")).read_bytes())

    frozen = invoke(
        "freeze025",
        [
            "freeze025",
            "--out-dir",
            str(freeze_path.parent),
            "--target-config-sha256",
            target["config_sha256"],
            "--target-index-sha256",
            target["index_sha256"],
            "--small-metadata",
            small["path"],
            "--small-config-sha256",
            small["config_sha256"],
            "--small-index-sha256",
            small["index_sha256"],
        ],
        0,
    )
    freeze_sha = h.sha(freeze_path.read_bytes())
    for phase in frozen["phases"]:
        planned = invoke(
            "run025-" + phase,
            [
                "run025",
                "--manifest",
                str(freeze_path),
                "--manifest-sha256",
                freeze_sha,
                "--phase",
                phase,
                "--out-dir",
                str(output / ("never-" + phase)),
                "--profile",
                str(output / "PRIVATE_NONEXISTENT_PROFILE.json"),
            ],
            0,
        )
        assert planned["hardware_executed"] is False
        assert not (output / ("never-" + phase)).exists()
    blocked = invoke(
        "run025-execute-refused",
        [
            "run025",
            "--manifest",
            str(freeze_path),
            "--manifest-sha256",
            freeze_sha,
            "--phase",
            "pilot",
            "--out-dir",
            str(output / "never-execute"),
            "--profile",
            str(output / "PRIVATE_NONEXISTENT_PROFILE.json"),
            "--execute",
        ],
        2,
    )
    assert blocked["error"] == "parent_only_guarded_supervisor_unavailable"
    trust_path = output / "TEST-trust.json"
    trust_path.write_bytes(encoded(trust))
    fixture_freeze = fixture_parent / "freeze/integration-freeze.json"
    collected = invoke(
        "collect025-TEST",
        [
            "collect025",
            "--manifest",
            str(fixture_freeze),
            "--manifest-sha256",
            trust["manifest_sha256"],
            "--raw-dir",
            str(root),
            "--trusted",
            str(trust_path),
            "--trusted-sha256",
            h.sha(trust_path.read_bytes()),
            "--out-dir",
            str(output / "TEST-collector-report"),
        ],
        2,
    )
    readback = h.decode_json((output / "TEST-collector-report/report.json").read_bytes())
    assert readback == collected
    assert collected["coverage_complete"] and not collected["accepted"]
    assert collected["evidence_validation"] == "source_bound_raw025_valid"
    assert len(collected["rows"]) == 6 and collected["record_audits"][0]["pilot_budget"]["passed"]
    forbidden = [name for name in sys.modules if name.split(".")[0] in {"mlx", "mlx_lm", "mlx_vlm"}]
    assert not forbidden and not denied, (forbidden, denied)
    summary = dict(
        schema="qwen38-overlap-cpu-integration-verification-v2",
        artifact_kind="CPU_IMPLEMENTATION_NOT_APPROVAL",
        hardware_executed=False,
        physical_approval=False,
        pytest_argv=arguments,
        pytest_exit_code=exit_code,
        counts=counts,
        cli_invocations=invocations,
        original_harness_receipt_sha256=original_sha,
        preserved_original_files=len(preserved),
        preservation_mismatches=mismatches,
        forbidden_imports=forbidden,
        denied_external_actions=denied,
        smoke_freeze_sha256=freeze_sha,
        smoke_sources_not_gpu_approval=True,
        artifacts={p.name: h.sha(p.read_bytes()) for p in output.iterdir() if p.is_file()},
    )
    (output / "verification.json").write_bytes(encoded(summary))
    print(
        json.dumps(
            {
                "run_dir": str(output),
                "counts": counts,
                "cli_invocations": len(invocations),
                "physical_approval": False,
                "forbidden_imports": forbidden,
                "denied_external_actions": denied,
            }
        )
    )


if __name__ == "__main__":
    main()
