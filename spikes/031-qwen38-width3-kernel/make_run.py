"""Freeze one runnable 031 snapshot (local + remote mini), exclusive, hash read back.

usage: make_run.py SUFFIX CASES_JSON CASE[,CASE...] [--warmup CASE] [--order-offset N]
Creates spikes/030-qwen38-adaptive-mtp-SUFFIX/ (sources + supervise.py + schedule)
and copies it to the mini under Projects/paretoquant-dual-host/spikes/ (same name,
so ssh_argv derives the project .venv). Prints schedule SHA-256 and row count.
"""

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from remote_actor import SOURCES  # noqa: E402
from workload_schedule import SCHEMA_ADAPTIVE, validate_schedule  # noqa: E402

SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "HostKeyAlias=fe80::56:c507:c02a:a630%bridge0",
    "-i", "/Users/jeremylien/.ssh/paretoquant_mini_ed25519",
]
TARGET = "jeremylien@fe80::1456:cf50:88f2:90d9%bridge0"
REMOTE = "/Users/jeremylien/Projects/paretoquant-dual-host/spikes"
RESULTS = HERE.parents[1] / "results/qwen38-width3-kernel-v1"
ORDERS = list(itertools.permutations(("native", "mtp", "adaptive")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suffix")
    p.add_argument("cases_json")
    p.add_argument("cases")
    p.add_argument("--warmup")
    p.add_argument("--order-offset", type=int, default=0)
    a = p.parse_args()
    defs = json.loads(Path(a.cases_json).read_text())
    rows = []
    plan = ([(a.warmup, True)] if a.warmup else []) + [(c, False) for c in a.cases.split(",")]
    for i, (case, warm) in enumerate(plan):
        order = ORDERS[(i + a.order_offset) % len(ORDERS)]
        for m in order:
            rows.append({
                "id": f"{case}-{m}-{'w' if warm else 'r'}0", "case_id": case, "method": m,
                "prompt": defs[case]["prompt"], "max_tokens": defs[case]["max_tokens"],
                "warmup": warm, "repeat": 0,
            })
    sched = {"schema": SCHEMA_ADAPTIVE, "requests": rows}
    validate_schedule(sched)
    name = f"031-qwen38-width3-kernel-{a.suffix}"
    dst = HERE.parent / name
    dst.mkdir()  # exclusive
    for f in [*SOURCES, "supervise.py", "width_gate.py"]:
        if f != "benchmark_schedule.json":
            shutil.copy2(HERE / f, dst / f)
    (dst / "benchmark_schedule.json").write_text(json.dumps(sched, ensure_ascii=False))
    local = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dst.iterdir())}
    subprocess.run([*SSH, TARGET, f"mkdir {REMOTE}/{name}"], check=True)  # exclusive
    subprocess.run(["scp", *SSH[1:], *[str(dst / f) for f in local],
                    f"jeremylien@[{TARGET.split('@')[1]}]:{REMOTE}/{name}/"],
                   check=True)
    out = subprocess.run([*SSH, TARGET, f"cd {REMOTE}/{name} && shasum -a 256 *"],
                         check=True, capture_output=True, text=True).stdout
    remote = {line.split()[1]: line.split()[0] for line in out.strip().splitlines()}
    if remote != local:
        raise SystemExit(f"REMOTE HASH MISMATCH {remote} {local}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.deploy.json").write_text(json.dumps(
        {"local": local, "remote_readback": remote, "match": True}, indent=1))
    print(json.dumps({"name": name, "rows": len(rows),
                      "cap": max(r["max_tokens"] for r in rows),
                      "schedule_sha256": local["benchmark_schedule.json"]}))


if __name__ == "__main__":
    main()
