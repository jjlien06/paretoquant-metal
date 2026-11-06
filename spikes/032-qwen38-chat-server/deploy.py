"""Immutable mini deployment of the 032 actor sources with SHA-256 readback.

Remote dir: ~/Projects/paretoquant-dual-host/spikes/032-qwen38-chat-server-<digest12>
(directly under spikes/, so ssh_argv derives the project .venv interpreter).
Existing dir with identical readback is reused; any mismatch fails closed.
Prints the remote root on success.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from remote_actor import SOURCES  # noqa: E402

KEY = "/Users/jeremylien/.ssh/paretoquant_mini_ed25519"
ALIAS = "fe80::56:c507:c02a:a630%bridge0"
REMOTE = "/Users/jeremylien/Projects/paretoquant-dual-host/spikes"


def main(target):
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", f"HostKeyAlias={ALIAS}",
           "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5", "-i", KEY]
    local = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in SOURCES}
    digest = hashlib.sha256(json.dumps(local, sort_keys=True).encode()).hexdigest()[:12]
    root = f"{REMOTE}/032-qwen38-chat-server-{digest}"

    def readback():
        r = subprocess.run([*ssh, target, f"cd {root} && shasum -a 256 {' '.join(SOURCES)}"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode:
            return None
        return {ln.split()[1]: ln.split()[0] for ln in r.stdout.strip().splitlines()}

    exists = subprocess.run([*ssh, target, f"test -d {root}"], timeout=15).returncode == 0
    if not exists:
        subprocess.run([*ssh, target, f"mkdir {root} && test -x {REMOTE}/../.venv/bin/python"],
                       check=True, timeout=15)
        host = target.split("@", 1)
        subprocess.run(["scp", "-q", *ssh[1:], *[str(HERE / n) for n in SOURCES],
                        f"{host[0]}@[{host[1]}]:{root}/"], check=True, timeout=120)
        subprocess.run([*ssh, target, f"chmod a-w {root}/*"], check=True, timeout=15)
    remote = readback()
    if remote != local:
        raise SystemExit(f"REMOTE HASH MISMATCH at {root}")
    out = HERE / "deploy-receipts"
    out.mkdir(exist_ok=True)
    (out / f"{digest}.json").write_text(json.dumps(
        {"remote_root": root, "local": local, "remote_readback": remote, "match": True,
         "reused_existing": exists}, indent=1))
    print(root)


if __name__ == "__main__":
    main(sys.argv[1])
