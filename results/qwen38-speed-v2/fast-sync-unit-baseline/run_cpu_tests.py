"""Reproduce only the CPU contracts, refusing MLX imports in pytest itself."""

import importlib.abc
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
REPO = EVIDENCE.parents[2]
os.environ["TMPDIR"] = "/Users/jeremylien/.hermes/cache/scratch"
os.chdir(REPO)


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm"}:
            raise AssertionError(f"CPU suite attempted forbidden runtime import: {fullname}")
        return None


sys.meta_path.insert(0, NoMLX())

import pytest  # noqa: E402

TESTS = [
    "test_fast_sync.py",
    "test_schedule.py",
    "test_supervise.py",
    "test_remote_actor.py",
    "test_progress.py",
]
args = [
    "-q",
    "--tb=short",
    f"--junitxml={EVIDENCE / 'cpu-junit.xml'}",
    *[str(REPO / "spikes/015-qwen38-fast-sync" / name) for name in TESTS],
]
started = datetime.now(timezone.utc).isoformat()
exit_code = int(pytest.main(args))
loaded = sorted(name for name in sys.modules if name.split(".")[0] in {"mlx", "mlx_lm"})
assert not loaded, loaded
report = {
    "started_utc": started,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "pytest_exit_code": exit_code,
    "tests": TESTS,
    "python": sys.version,
    "versions_without_importing_runtime": {
        name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "pytest", "ruff")
    },
    "pytest_process_mlx_modules_loaded": loaded,
    "scope": "CPU contracts only; stdlib substitute probe subprocesses; no GPU/SSH/model jobs",
    "hardware_validation": "NOT RUN; parent only",
}
(EVIDENCE / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
sys.exit(exit_code)
