"""Task-local CPU runner: real child creation, network, signals forbidden."""

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "spikes/025-qwen38-overlapped-prefill"
blocked = []


def audit(event, args):
    if event in {
        "subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.forkpty",
        "socket.__new__", "os.kill", "os.killpg",
    }:
        blocked.append(event)
        raise RuntimeError("No actual process/network/signals: " + event)


os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
sys.addaudithook(audit)
sys.path.insert(0, str(ROOT))
sys.argv = [str(ROOT / "cpu_only_runner.py"), *sys.argv[1:]]
try:
    runpy.run_path(str(ROOT / "cpu_only_runner.py"), run_name="__main__")
finally:
    print(json.dumps({"extra_guard": "forbid actual child process/network/signals", "blocked_attempts": blocked}))
