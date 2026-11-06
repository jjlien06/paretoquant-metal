"""Bounded launcher: rank0 = local M2 Pro, rank1 = M4 mini over the Thunderbolt bridge.

Mirrors the 021 supervisor's ring setup: hostfile [[local:port],[remote:port]],
MLX_RANK, mx.distributed.init(strict=True, backend="ring").
"""

import argparse
import errno
import hashlib
import json
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = "/Users/jeremylien/.ssh/paretoquant_mini_ed25519"
ALIAS = "fe80::56:c507:c02a:a630%bridge0"
TARGET = "jeremylien@fe80::1456:cf50:88f2:90d9%bridge0"
REMOTE_PY = "/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python"
REMOTE_DIR = "/Users/jeremylien/Projects/paretoquant-dual-host/spikes/028-tp-allreduce-screen"
LOCAL_PY = str(HERE.parents[1] / ".venv/bin/python")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", f"HostKeyAlias={ALIAS}", "-i", KEY]


def ssh(cmd, **kw):
    return subprocess.run([*SSH, TARGET, cmd], capture_output=True, text=True, timeout=60, **kw)


def free_port(addr):
    with socket.socket() as s:
        s.bind((addr, 0))
        return s.getsockname()[1]


def closed(addr, port):
    with socket.socket() as c:
        c.settimeout(2.0)
        try:
            c.connect((addr, port))
        except OSError as e:
            return e.errno == errno.ECONNREFUSED
        return False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-address", required=True)
    ap.add_argument("--remote-address", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--timeout", type=float, default=280)
    ap.add_argument("bench_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = HERE / "bench.py"
    bench_sha = sha(bench)

    r = ssh(f"mkdir -p {REMOTE_DIR}")
    assert r.returncode == 0, r.stderr
    subprocess.run(
        ["scp", "-q", *SSH[1:], str(bench), f"[{TARGET.split('@')[1]}]:{REMOTE_DIR}/bench.py".replace("[", "jeremylien@[", 1)],
        check=True, timeout=60,
    )
    r = ssh(f"shasum -a 256 {REMOTE_DIR}/bench.py")
    remote_sha = r.stdout.split()[0]
    assert remote_sha == bench_sha, (remote_sha, bench_sha)

    rp = ssh(f"{REMOTE_PY} -c 'import socket;s=socket.socket();s.bind((\"{a.remote_address}\",0));print(s.getsockname()[1])'")
    remote_port = int(rp.stdout.strip())
    local_port = free_port(a.local_address)
    hostfile = [[f"{a.local_address}:{local_port}"], [f"{a.remote_address}:{remote_port}"]]
    hf_local = out / f"{a.tag}.hostfile.json"
    hf_local.write_text(json.dumps(hostfile))
    hf_remote = f"{REMOTE_DIR}/{a.tag}.hostfile.json"
    ssh(f"printf %s {shlex.quote(json.dumps(hostfile))} > {hf_remote}")

    extra = a.bench_args[1:] if a.bench_args[:1] == ["--"] else a.bench_args
    rout = f"{REMOTE_DIR}/{a.tag}.rank1.json"
    rcmd = (
        f"cd {REMOTE_DIR} && exec env MLX_HOSTFILE={hf_remote} MLX_RANK=1 {REMOTE_PY} -B -u bench.py "
        f"--hidden {a.hidden} --output {rout} {' '.join(map(shlex.quote, extra))}"
    )
    lout = out / f"{a.tag}.rank0.json"
    env = {"MLX_HOSTFILE": str(hf_local), "MLX_RANK": "0", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    started = time.time()
    p1 = subprocess.Popen([*SSH, TARGET, rcmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    p0 = subprocess.Popen(
        [LOCAL_PY, "-B", "-u", str(bench), "--hidden", str(a.hidden), "--output", str(lout), *extra],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    logs = {}
    for name, p in (("rank0", p0), ("rank1", p1)):
        try:
            logs[name] = p.communicate(timeout=max(5, a.timeout - (time.time() - started)))[0]
        except subprocess.TimeoutExpired:
            p.kill()
            logs[name] = (p.communicate()[0] or "") + "\n[launcher: TIMEOUT, killed]"
    rc = {"rank0": p0.returncode, "rank1": p1.returncode}
    wall = time.time() - started
    time.sleep(1)
    rk = ssh("pgrep -fl '[b]ench.py --hidden' || true")
    lk = subprocess.run(["pgrep", "-fl", "[b]ench.py --hidden"], capture_output=True, text=True)
    rclosed = ssh(f"{REMOTE_PY} -c 'import socket,errno;c=socket.socket();c.settimeout(2)\ntry:\n c.connect((\"{a.remote_address}\",{remote_port}));print(False)\nexcept OSError as e:\n print(e.errno==errno.ECONNREFUSED)'")
    cleanup = {
        "remote_bench_pids_after": rk.stdout.split(),
        "local_bench_pids_after": lk.stdout.split(),
        "local_port_closed_local_probe": closed(a.local_address, local_port),
        "remote_port_closed_local_probe": closed(a.remote_address, remote_port),
        "remote_port_closed_remote_probe": rclosed.stdout.strip(),
    }
    fetched = None
    if rc["rank1"] == 0:
        dst = out / f"{a.tag}.rank1.json"
        subprocess.run(["scp", "-q", *SSH[1:], f"jeremylien@[{TARGET.split('@')[1]}]:{rout}", str(dst)], check=True, timeout=60)
        rs = ssh(f"shasum -a 256 {rout}").stdout.split()[0]
        fetched = {"remote_sha256": rs, "local_sha256": sha(dst), "match": rs == sha(dst)}
    receipt = {
        "tag": a.tag, "hostfile": hostfile, "bench_sha256": bench_sha, "remote_bench_sha256": remote_sha,
        "returncodes": rc, "wall_s": wall, "logs": {k: v[-4000:] for k, v in logs.items()},
        "cleanup": cleanup, "rank1_fetch": fetched, "bench_args": extra,
    }
    (out / f"{a.tag}.launcher.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("returncodes", "wall_s", "cleanup", "rank1_fetch")}, indent=1))
    print(logs)
    sys.exit(0 if rc == {"rank0": 0, "rank1": 0} else 2)


if __name__ == "__main__":
    main()
