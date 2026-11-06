"""032 persistent chat-server supervisor (started by start-chat.sh, stopped by stop-chat.sh).

Admits pinned metadata + local/peer source hashes, starts nonce-owned actors (mini
weight stream server, mini rank1, local rank0 with the loopback HTTP API), then
supervises until a stop request/signal, an actor failure, or the session deadline.
Shutdown: POST /shutdown (both ranks exit cleanly), then nonce stop of every actor,
then verified process-group and listener closure. Receipts go to --output.
"""

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from remote_actor import SOURCES, job_status, request_stop, source_bytes, terminate, write_json  # noqa: E402
from supervise import free_port, listener_closed, ssh_argv  # noqa: E402

SCRATCH = "/Users/jeremylien/.hermes/cache/scratch"
REPO = ROOT.parents[1]
PIN = dict(
    target_revision="10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    target_config_sha256="14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
    target_index_sha256="13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
    small_config_sha256="ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772",
    small_index_sha256="6e48f2fa5d6f033a6d77bf833abfa9698ca24d1715ecea4c67447bcfaee44650",
)
DUAL = "/Users/jeremylien/Projects/paretoquant-dual-host"
REMOTE_SOURCE = f"{DUAL}/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
SMALL_REMOTE = f"{DUAL}/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462"
SMALL_LOCAL = str(REPO / "models/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462")
METADATA = str(REPO / "models/qwen3.8-27b-4bit-tokenizer-only")
SIDECAR = str(REPO / "models/qwen3.8-27b-mtp-4bit-b643c01b6d3b094e325edb6ebd832e16c486c575")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--local-address", required=True)
    p.add_argument("--remote-address", required=True)
    p.add_argument("--ssh-target", required=True)
    p.add_argument("--ssh-host-key-alias", required=True)
    p.add_argument("--ssh-key", default="/Users/jeremylien/.ssh/paretoquant_mini_ed25519")
    p.add_argument("--mode", choices=("adaptive", "depth1"), default="adaptive")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--http-port", type=int, default=8780)
    p.add_argument("--session-hours", type=float, default=8.0)
    p.add_argument("--reserve-bytes", type=int, default=8 * 1024**3)
    p.add_argument("--ui", default=str(REPO / "apps/chat-ui/index.html"))
    a = p.parse_args()
    if not 0 < a.session_hours <= 12 or not 1024 <= a.http_port <= 65535 or not 0 <= a.tau <= 1:
        raise SystemExit("bounds: session-hours (0,12], port 1024..65535, tau [0,1]")
    out = Path(a.output)
    out.mkdir(mode=0o700)
    (out / "actors").mkdir(mode=0o700)
    deadline = a.session_hours * 3600
    started = time.monotonic()
    stop_requested = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, f: stop_requested.append(s))
    nonce = secrets.token_hex(16)
    write_json(out / "supervisor.json", {"pid": os.getpid(), "nonce": nonce, "args": vars(a),
                                         "started": time.time()})
    ns = argparse.Namespace(remote_root=a.remote_root, ssh_host_key_alias=a.ssh_host_key_alias,
                            ssh_key=a.ssh_key, ssh_target=a.ssh_target)
    transports, jobs, files, control = {}, {}, [], {}
    final, stage = {}, "admission"
    hashes, hostfile, server_port, local_port, remote_port = None, None, None, None, None

    def call(extra, name):
        argv = ssh_argv(ns, extra)
        r = subprocess.run(argv, capture_output=True, timeout=15, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"control call {name} failed")
        return json.loads(r.stdout)

    def start(name, entry, argv, *, rank=None, remote=True, token=None, job_deadline=None):
        job = {"nonce": secrets.token_hex(16), "entrypoint": entry, "argv": argv,
               "deadline": job_deadline or max(1.0, deadline - (time.monotonic() - started)),
               "reserve_bytes": a.reserve_bytes, "source_sha256": hashes, "env": {}}
        if token:
            job["env"]["PARETOQUANT_WEIGHT_TOKEN"] = token
        if rank is not None:
            job["env"]["MLX_RANK"] = str(rank)
            job["hostfile"] = hostfile
        jobs[name] = {"nonce": job["nonce"], "remote": remote}
        write_json(out / "actors.json", jobs)
        cmd = ssh_argv(ns, ["--jobs", SCRATCH]) if remote else [
            sys.executable, "-u", str(ROOT / "remote_actor.py"), "--root", str(ROOT),
            "--jobs", str(out / "actors")]
        handles = [open(out / f"{name}.{s}", "xb") for s in ("stdout", "stderr")]
        files.extend(handles)
        child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=handles[0], stderr=handles[1],
                                 start_new_session=True)
        transports[name] = child
        child.stdin.write(json.dumps(job).encode() + b"\n")
        child.stdin.close()
        write_json(out / f"{name}.transport.json", {"pid": child.pid, "nonce": job["nonce"],
                                                      "entrypoint": entry})

    def status(name, records=False):
        item = jobs[name]
        if item["remote"]:
            return call(["--jobs", SCRATCH, "--status", item["nonce"], *(["--records"] if records else [])],
                        f"status-{name}")
        return job_status(out / "actors", item["nonce"], records=records)

    def health():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.http_port}/health", timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            return None

    try:
        from verification import bind_metadata

        bind_metadata(Path(METADATA), PIN["target_config_sha256"], PIN["target_index_sha256"])
        hashes = {n: hashlib.sha256(source_bytes(ROOT / n)).hexdigest() for n in SOURCES}
        write_json(out / "source_sha256.json", hashes)
        stage = "peer_inspect"
        peer = call(["--inspect", "--port-address", a.remote_address], "inspect")
        if peer.get("source_sha256") != hashes:
            raise ValueError("Peer source differs from admitted local source")
        local_port = free_port(a.local_address)
        remote_port = peer["available_port"]
        hostfile = [[f"{a.local_address}:{local_port}"], [f"{a.remote_address}:{remote_port}"]]
        write_json(out / "hostfile.json", hostfile)
        if not listener_closed("127.0.0.1", a.http_port):
            raise RuntimeError("HTTP port already in use")
        token = secrets.token_hex(32)
        stage = "weight_server"
        start("server", "stream_server.py", ["--source", REMOTE_SOURCE, "--bind", a.remote_address,
                                             "--peer", a.local_address, "--split", "28", "36", "--rank", "0"],
              token=token, job_deadline=900)
        until = time.monotonic() + 60
        while True:
            cur = status("server")
            if "ready" in cur:
                addr, server_port = cur["ready"]["address"]
                if addr != a.remote_address:
                    raise ValueError("server address")
                break
            if cur.get("result") or time.monotonic() > until or stop_requested:
                raise RuntimeError("weight server readiness failed")
            time.sleep(0.3)
        stage = "ranks"
        common = ["--split", "28", "36", "--eval-group-size", "32",
                  "--target-revision", PIN["target_revision"],
                  "--target-config-sha256", PIN["target_config_sha256"],
                  "--target-index-sha256", PIN["target_index_sha256"],
                  "--small-config-sha256", PIN["small_config_sha256"],
                  "--small-index-sha256", PIN["small_index_sha256"]]
        if a.mode == "adaptive":
            common += ["--adaptive-tau", repr(float(a.tau))]
        # rank1 verifier: 031 slot-qmv-tuned-m4 (adaptive) / 021 slot-qmv (depth1 fallback).
        r1v = "slot-qmv-tuned-m4" if a.mode == "adaptive" else "slot-qmv"
        start("rank1", "chat_rank.py", [*common, "--verifier-mode", r1v, "--small-model", SMALL_REMOTE,
                                        "--model", REMOTE_SOURCE, "--stream-local"], rank=1)
        start("rank0", "chat_rank.py", [*common, "--verifier-mode", "native", "--small-model", SMALL_LOCAL,
                                        "--model", METADATA, "--sidecar", SIDECAR,
                                        "--weight-server", f"{a.remote_address}:{server_port}",
                                        "--weight-client-bind", a.local_address,
                                        "--http-port", str(a.http_port), "--ui", a.ui],
              rank=0, remote=False, token=token)
        del token
        stage = "serving"
        last_state = 0
        while True:
            if stop_requested or (out / "stop").exists():
                final["stop_reason"] = "stop_requested"
                break
            if time.monotonic() - started > deadline:
                final["stop_reason"] = "session_deadline"
                break
            for name in ("rank0", "rank1"):
                if transports[name].poll() is not None:
                    raise RuntimeError(f"{name} actor exited unexpectedly")
            if transports["server"].poll() not in (None, 0):
                raise RuntimeError("weight server failed")
            if time.monotonic() - last_state > 2:
                h = health()
                write_json(out / "state.json", {"stage": stage, "health": h,
                                                "elapsed": time.monotonic() - started})
                last_state = time.monotonic()
            time.sleep(0.2)
        stage = "graceful_shutdown"
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{a.http_port}/shutdown", data=b"{}",
                                         method="POST", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
            final["shutdown_posted"] = True
        except Exception as exc:
            final["shutdown_posted"] = type(exc).__name__
        until = time.monotonic() + 60
        while time.monotonic() < until and any(transports[n].poll() is None for n in ("rank0", "rank1")):
            time.sleep(0.2)
        final["ranks_exited_gracefully"] = all(transports[n].poll() == 0 for n in ("rank0", "rank1"))
        final["status"] = "stopped" if final["ranks_exited_gracefully"] else "stopped_forced"
    except BaseException as exc:
        final.update(status="failed", error_type=type(exc).__name__, error_stage=stage)
    finally:
        errors = []
        for name, item in jobs.items():
            try:
                if item["remote"]:
                    call(["--jobs", SCRATCH, "--stop", item["nonce"]], f"stop-{name}")
                else:
                    request_stop(out / "actors", item["nonce"])
            except BaseException as exc:
                errors.append({"actor": name, "error_type": type(exc).__name__})
        until = time.monotonic() + 10
        while time.monotonic() < until and any(t.poll() is None for t in transports.values()):
            time.sleep(0.1)
        receipts = {}
        for name in jobs:
            try:
                receipts[name] = status(name).get("result")
            except BaseException as exc:
                errors.append({"actor": name, "status_error": type(exc).__name__})
        tcleanup = {n: terminate(t) for n, t in transports.items()}
        for h in files:
            h.close()
        owned_clean = len(receipts) == len(jobs) and all(
            r and r.get("cleanup", {}).get("child_reaped") and r.get("cleanup", {}).get("group_gone")
            for r in receipts.values())
        ports = [("127.0.0.1", a.http_port)]
        if local_port:
            ports += [(a.local_address, local_port), (a.remote_address, remote_port)]
        if server_port:
            ports.append((a.remote_address, server_port))
        closed = [listener_closed(h, pt) for h, pt in ports]
        final.update(
            nonce=nonce, actor_results=receipts, transport_cleanup=tcleanup, cleanup_errors=errors,
            owned_actor_cleanup_verified=owned_clean,
            listeners={"ports": ports, "closed": closed},
            services_closed_verified=owned_clean and all(closed) and not errors and all(
                c["child_reaped"] and c["group_gone"] for c in tcleanup.values()),
            elapsed_seconds=time.monotonic() - started,
        )
        if not final["services_closed_verified"]:
            final["pending_cleanup"] = True
        write_json(out / "result.json", final)
    print(json.dumps({k: final.get(k) for k in ("status", "error_type", "error_stage",
                                                 "services_closed_verified")}), flush=True)
    return 0 if final.get("status") == "stopped" and final["services_closed_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
