"""Stdlib-only, bounded watchdog for admitted spike children."""

import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def redact(value, secrets):
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return value


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS sandbox can deny kill(0) after reap; inspect only group IDs.
        try:
            result = subprocess.run(
                ["/bin/ps", "-axo", "pgid="], capture_output=True, text=True, timeout=1, check=True
            )
            return str(pid) in result.stdout.split()
        except (OSError, subprocess.SubprocessError):
            return True  # unknown is pending, never a successful cleanup


def pid_present(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            result = subprocess.run(
                ["/bin/ps", "-axo", "pid="],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            )
            return str(pid) in result.stdout.split()
        except (OSError, subprocess.SubprocessError):
            return None


def terminate(child):
    """Signal only while the exclusively owned Popen leader remains unreaped.

    Actor/controller callers have no competing wait/reap threads; their signal
    handlers only append requests. With this single-owner assumption, poll()
    returning None pins the PID even if the leader exits before killpg(): no
    intervening reap can release it. Once poll()/wait() has reaped the leader,
    numeric PID/group checks are read-only, never permission to signal.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if child.poll() is not None:
            break
        try:
            os.killpg(child.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        until = time.monotonic() + 0.6
        while time.monotonic() < until:
            child.poll()
            if not group_alive(child.pid):
                break
            time.sleep(0.02)
    reaped = child.poll() is not None
    gone = not group_alive(child.pid)
    unknown = reaped and not gone
    reused = reaped and pid_present(child.pid) is True  # read-only diagnostic
    return {
        "child_reaped": reaped,
        "group_gone": gone,
        "pid_reused": reused,
        "identity_unknown": unknown,
        "pending_os_gpu_termination": not (reaped and gone),
        "gpu_driver_state": "not inspected; process/group receipt only",
    }


class Filter:
    """Keep potential secret prefixes across arbitrary pipe read boundaries."""

    def __init__(self, secrets):
        self.secrets = [s.encode() for s in secrets if s]
        self.pending = b""

    def feed(self, data, final=False):
        data = self.pending + data
        for secret in self.secrets:
            data = data.replace(secret, b"[REDACTED]")
        keep = 0
        if not final:
            for secret in self.secrets:
                for n in range(1, min(len(secret), len(data) + 1)):
                    if data.endswith(secret[:n]):
                        keep = max(keep, n)
        self.pending = data[-keep:] if keep else b""
        return data[:-keep] if keep else data


def run_owned(
    argv,
    job_dir,
    *,
    deadline,
    env=None,
    secrets=(),
    reserve_bytes=0,
    tick=None,
    nonce=None,
    stream=False,
    hostfile=None,
    source_root=None,
    expected_hashes=None,
):
    """Library fixture seam; CLI cannot select arbitrary executable code."""
    if not 0 < deadline <= 240:
        raise ValueError("deadline must be in (0, 240]")
    job_dir = Path(job_dir)
    job_dir.mkdir(mode=0o700)
    nonce = nonce or uuid.uuid4().hex
    write_json(
        job_dir / "startup.json",
        {
            "actor_pid": os.getpid(),
            "nonce": nonce,
            "deadline": deadline,
            "argv": [redact(x, secrets) for x in argv],
        },
    )
    stopped, old = [], {}
    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, lambda signum, frame: stopped.append(signum))
    child, files = None, []
    gate_read = gate_write = None
    selector = selectors.DefaultSelector()
    reason, error_type = "failure", None
    began = time.monotonic()
    forwarded_bytes_dropped = 0
    try:
        if source_root is not None:
            snapshot = job_dir / "sources"
            snapshot.mkdir(mode=0o700)
            hashes = {}
            for name in SOURCES:
                original = Path(source_root) / name
                if original.is_symlink():
                    raise ValueError("Source symlinks forbidden")
                data = source_bytes(original)
                hashes[name] = hashlib.sha256(data).hexdigest()
                with os.fdopen(
                    os.open(snapshot / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
                ) as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            if expected_hashes is not None and hashes != expected_hashes:
                raise ValueError("Source hashes differ from admitted trial")
            write_json(job_dir / "source_sha256.json", hashes)
            argv = [
                str(snapshot / Path(x).name)
                if x == str(Path(source_root) / Path(x).name) and Path(x).name in SOURCES
                else x
                for x in argv
            ]
        if hostfile is not None:
            write_json(job_dir / "hostfile.json", hostfile)
            env = dict(os.environ if env is None else env)
            env["MLX_HOSTFILE"] = str(job_dir / "hostfile.json")
        for name in ("stdout", "stderr"):
            files.append(
                os.fdopen(
                    os.open(job_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                    "wb",
                    buffering=0,
                )
            )
        gate_read, gate_write = os.pipe()
        bootstrap = (
            "import os,sys; fd=int(sys.argv[1]); b=os.read(fd,1); os.close(fd); "
            "b==b'1' or sys.exit(125); os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", bootstrap, str(gate_read), *argv],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(gate_read,),
        )
        os.close(gate_read)
        gate_read = None
        write_json(
            job_dir / "pids.json",
            {
                "actor_pid": os.getpid(),
                "child_pid": child.pid,
                "pgid": child.pid,
                "nonce": nonce,
                "executed_argv": [redact(x, secrets) for x in argv],
                "environment_keys": sorted(env or {}),
            },
        )
        os.write(gate_write, b"1")
        os.close(gate_write)
        gate_write = None
        for index, pipe in enumerate((child.stdout, child.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, (index, Filter(secrets)))
        while True:
            for key, _ in selector.select(0.05):
                index, filtering = key.data
                data = os.read(key.fileobj.fileno(), 65536)
                clean = filtering.feed(data, final=not data)
                files[index].write(clean)
                if stream and clean:
                    try:
                        sent_bytes = os.write(
                            (sys.stdout if index == 0 else sys.stderr).fileno(), clean
                        )
                        forwarded_bytes_dropped += len(clean) - sent_bytes
                    except (BrokenPipeError, OSError):
                        forwarded_bytes_dropped += len(clean)
                        # Continue nonblocking forwarding attempts; disk logs are authoritative.
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            if stopped or (job_dir / "stop").exists():
                reason = "signal"
                break
            if time.monotonic() - began >= deadline:
                reason = "deadline"
                break
            if reserve_bytes:
                stat = os.statvfs("/")
                if stat.f_bavail * stat.f_frsize < reserve_bytes:
                    reason = "disk_reserve"
                    break
            if tick:
                tick(child)
            if child.poll() is not None and not selector.get_map():
                reason = "completed" if child.returncode == 0 else "child_failure"
                break
    except BaseException as exc:
        # Never serialize exception args: TimeoutExpired may contain credentials.
        error_type = type(exc).__name__
        reason = "exception"
    finally:
        for fd in (gate_read, gate_write):
            if fd is not None:
                os.close(fd)
        cleanup = (
            terminate(child)
            if child is not None
            else {
                "child_reaped": True,
                "group_gone": True,
                "pending_os_gpu_termination": False,
            }
        )
        # Drain bounded remaining output after termination, without waiting on pipes.
        for key in list(selector.get_map().values()):
            index, filtering = key.data
            for _ in range(128):
                try:
                    data = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    break
                if not data:
                    break
                files[index].write(filtering.feed(data))
            files[index].write(filtering.feed(b"", final=True))
            key.fileobj.close()
        selector.close()
        for handle in files:
            handle.close()
        for sig, handler in old.items():
            signal.signal(sig, handler)
        result = {
            "reason": reason,
            "nonce": nonce,
            "actor_pid": os.getpid(),
            "child_pid": child.pid if child else None,
            "returncode": child.returncode if child else None,
            "cleanup": cleanup,
            "error_type": error_type,
            "elapsed": time.monotonic() - began,
            "forwarded_bytes_dropped": forwarded_bytes_dropped,
        }
        write_json(job_dir / "result.json", result)
    return result


SOURCES = (
    "probe.py",
    "stream_weights.py",
    "stream_server.py",
    "shard_plan.py",
    "cpu_communication.py",
    "phased_decode.py",
    "remote_actor.py",
)


def source_bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require regular source files")
    with path.open("rb") as handle:
        data = handle.read(2 * 1024**2 + 1)
    if len(data) > 2 * 1024**2:
        raise ValueError("Source exceeds bounded inspection size")
    return data


def source_hashes(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for name in SOURCES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Require regular source files")
        result[name] = hashlib.sha256(source_bytes(path)).hexdigest()
    return result


def validate_job(job, root):
    import re

    if not isinstance(job, dict) or set(job) - {
        "entrypoint",
        "argv",
        "env",
        "nonce",
        "deadline",
        "reserve_bytes",
        "hostfile",
        "source_sha256",
    }:
        raise ValueError("Invalid job fields")
    if job.get("entrypoint") not in {"probe.py", "stream_server.py"}:
        raise ValueError("Unadmitted entrypoint")
    path = Path(root).resolve(strict=True) / job["entrypoint"]
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require regular entrypoint")
    argv = job.get("argv")
    env = job.get("env", {})
    if not isinstance(argv, list) or any(not isinstance(x, str) or "\x00" in x for x in argv):
        raise ValueError("Invalid argv")
    if (
        not isinstance(env, dict)
        or set(env) - {"PARETOQUANT_WEIGHT_TOKEN", "MLX_RANK"}
        or any(not isinstance(x, str) for x in env.values())
    ):
        raise ValueError("Invalid environment")
    if "MLX_RANK" in env and (env["MLX_RANK"] not in {"0", "1"} or "hostfile" not in job):
        raise ValueError("Require valid rank and owned hostfile")
    token = env.get("PARETOQUANT_WEIGHT_TOKEN")
    if token is not None and not re.fullmatch("[a-f0-9]{64}", token):
        raise ValueError("Invalid session token")
    if token and any(token in x for x in argv):
        raise ValueError("Token must not appear in argv")
    if not re.fullmatch("[a-f0-9]{32}", job.get("nonce", "")):
        raise ValueError("Invalid nonce")
    if type(job.get("deadline")) not in (int, float) or not 0 < job["deadline"] <= 240:
        raise ValueError("Invalid deadline")
    if type(job.get("reserve_bytes", 0)) is not int or job.get("reserve_bytes", 0) < 0:
        raise ValueError("Invalid reserve")
    if "source_sha256" in job:
        hashes = job["source_sha256"]
        if (
            not isinstance(hashes, dict)
            or set(hashes) != set(SOURCES)
            or any(
                not isinstance(v, str) or not re.fullmatch("[a-f0-9]{64}", v)
                for v in hashes.values()
            )
        ):
            raise ValueError("Invalid source hashes")
    if "hostfile" in job:
        import ipaddress

        hostfile = job["hostfile"]
        if not isinstance(hostfile, list) or len(hostfile) != 2:
            raise ValueError("Require two ring hosts")
        for host in hostfile:
            if not isinstance(host, list) or len(host) != 1 or not isinstance(host[0], str):
                raise ValueError("Invalid ring host")
            address, port = host[0].rsplit(":", 1)
            ipaddress.IPv4Address(address)
            if not 1 <= int(port) <= 65535:
                raise ValueError("Invalid ring port")
    return job


def job_path(jobs, nonce):
    import re

    if not re.fullmatch("[a-f0-9]{32}", nonce):
        raise ValueError("Invalid nonce")
    path = Path(jobs) / nonce
    if path.is_symlink():
        raise ValueError("Job symlink forbidden")
    return path


def request_stop(jobs, nonce):
    path = job_path(jobs, nonce)
    if path.is_dir():
        write_json(path / "stop", {"nonce": nonce})


def job_status(jobs, nonce, *, records=False):
    path = job_path(jobs, nonce)
    status = {"nonce": nonce}
    for name in ("startup", "pids", "result", "source_sha256"):
        receipt = path / (name + ".json")
        if receipt.exists():
            status[name] = json.loads(receipt.read_text())
    stdout = path / "stdout"
    if stdout.exists():
        with stdout.open("rb") as handle:
            first = handle.readline(65536)
        try:
            event = json.loads(first)
            if event.get("ready"):
                status["ready"] = event
        except (ValueError, UnicodeDecodeError):
            pass
    if records:
        for name in ("stdout", "stderr"):
            log = path / name
            if log.exists():
                if log.stat().st_size > 8 * 1024**2:
                    raise ValueError("Log too large for bounded retrieval; retained remotely")
                status[name] = log.read_text(errors="replace")
    return status


def read_job(fd, *, timeout=5):
    until = time.monotonic() + timeout
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = until - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("Job input deadline")
            chunk = os.read(fd, 65537 - len(data))
            data.extend(chunk)
            if len(data) > 65536:
                raise ValueError("Job exceeds bounded input")
            if not chunk or b"\n" in chunk:
                return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--port-address")
    parser.add_argument("--status")
    parser.add_argument("--stop")
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--jobs", help="Existing private directory for unique jobs")
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    if args.inspect:
        result = {"source_sha256": source_hashes(root)}
        if args.port_address:
            import socket

            with socket.socket() as listener:
                listener.bind((args.port_address, 0))
                result["available_port"] = listener.getsockname()[1]
        print(json.dumps(result), flush=True)
        return 0
    if not args.jobs:
        parser.error("--jobs is required")
    jobs = Path(args.jobs).resolve(strict=True)
    if args.stop:
        request_stop(jobs, args.stop)
        print(json.dumps({"stop_requested": args.stop}), flush=True)
        return 0
    if args.status:
        print(json.dumps(job_status(jobs, args.status, records=args.records)), flush=True)
        return 0
    try:
        job = validate_job(read_job(sys.stdin.fileno()), root)
    except Exception as exc:
        nonce = uuid.uuid4().hex
        failure = jobs / nonce
        failure.mkdir(mode=0o700)
        write_json(failure / "startup.json", {"actor_pid": os.getpid(), "nonce": nonce})
        write_json(
            failure / "result.json",
            {
                "reason": "admission_failure",
                "error_type": type(exc).__name__,
                "actor_pid": os.getpid(),
                "nonce": nonce,
                "child_pid": None,
                "cleanup": {"child_reaped": True, "group_gone": True},
            },
        )
        return 1
    token = job.get("env", {}).get("PARETOQUANT_WEIGHT_TOKEN")
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG"}}
    env.update(job.get("env", {}))
    for output in (sys.stdout, sys.stderr):
        os.set_blocking(output.fileno(), False)
    result = run_owned(
        [sys.executable, "-u", str(root / job["entrypoint"]), *job["argv"]],
        jobs / job["nonce"],
        deadline=job["deadline"],
        env=env,
        secrets=(token,) if token else (),
        reserve_bytes=job.get("reserve_bytes", 0),
        nonce=job["nonce"],
        stream=True,
        hostfile=job.get("hostfile"),
        source_root=root,
        expected_hashes=job.get("source_sha256"),
    )
    return 0 if result["reason"] == "completed" and result["cleanup"]["group_gone"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr, flush=True)
        sys.exit(1)
