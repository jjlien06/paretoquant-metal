"""Durable bounded two-host trial controller. Never invoke via execute_code."""

import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from remote_actor import SOURCES, source_bytes, terminate, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SCRATCH = "/Users/jeremylien/.hermes/cache/scratch"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("remote-root", "remote-source", "local-metadata", "output", "ssh-key"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", nargs=2, type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--deadline", type=float, default=180)
    p.add_argument("--reserve-bytes", type=int, default=4294967296)
    p.add_argument("--local-address", default="169.254.248.125")
    p.add_argument("--remote-address", default="169.254.169.190")
    p.add_argument("--ssh-target", default="jeremylien@fe80::56:c507:c02a:a630%bridge0")
    p.add_argument("--prompt", default="What is 2 plus 2? Explain briefly.")
    p.add_argument("--execution", choices=("phased", "upstream"), default="phased")
    return p


def validate_args(args):
    if not 0 < args.deadline <= 240 or not 1 <= args.max_tokens <= 512 or args.repeats < 1:
        raise ValueError("Invalid deadline/token/repeat bounds")
    if any(x <= 0 for x in args.split) or args.reserve_bytes < 0:
        raise ValueError("Invalid split/reserve")
    for address in (args.local_address, args.remote_address):
        ipaddress.IPv4Address(address)
    for name in ("remote_root", "remote_source", "local_metadata", "output", "ssh_key"):
        if not Path(getattr(args, name)).is_absolute():
            raise ValueError("Require absolute paths")
    if Path(args.ssh_key).resolve().is_relative_to(REPO):
        raise ValueError("SSH key filename must be outside repository")
    if args.ssh_target.startswith("-") or any(c.isspace() for c in args.ssh_target):
        raise ValueError("Invalid SSH target")


def ssh_argv(args, actor_args):
    root = Path(args.remote_root)
    python = root.parents[1] / ".venv/bin/python"
    command = [str(python), "-u", str(root / "remote_actor.py"), "--root", str(root), *actor_args]
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=2",
        "-o",
        "ServerAliveCountMax=2",
        "-i",
        args.ssh_key,
        args.ssh_target,
        shlex.join(command),
    ]


def expected_inventory(config, index_names, split):
    """Partition trusted checkpoint index names; never infer names from rank receipts."""
    layers = config.get("num_hidden_layers")
    tied = config.get("tie_word_embeddings", False)
    if (
        config.get("model_type") != "qwen2"
        or config.get("model_file")
        or type(layers) is not int
        or layers <= 0
        or type(tied) is not bool
        or not isinstance(split, (list, tuple))
        or len(split) != 2
        or any(type(count) is not int or count <= 0 for count in split)
        or sum(split) != layers
    ):
        raise ValueError("Invalid Qwen2 inventory configuration/split")
    names = set(index_names)
    replicated, layer_names = set(), {}
    for name in names:
        if not isinstance(name, str):
            raise ValueError("Invalid inventory name")
        match = re.fullmatch(r"model\.layers\.(0|[1-9][0-9]*)\.(.+)", name)
        if match:
            layer = int(match[1])
            if layer >= layers:
                raise ValueError("Inventory layer outside configured range")
            layer_names[name] = layer
        elif re.fullmatch(r"(?:model\.embed_tokens|model\.norm|lm_head)\..+", name):
            if tied and name.startswith("lm_head."):
                raise ValueError("Tied inventory contains separate lm_head")
            replicated.add(name)
        else:
            raise ValueError("Unrecognized Qwen2 inventory name")
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not tied:
        required.add("lm_head.weight")
    if not required <= replicated or set(layer_names.values()) != set(range(layers)):
        raise ValueError("Incomplete trusted metadata inventory")
    return {
        rank: frozenset(
            replicated
            | {
                name
                for name, layer in layer_names.items()
                if sum(split[rank + 1 :]) <= layer < sum(split[rank:])
            }
        )
        for rank in (0, 1)
    }


def verify_metadata(metadata, admitted_sha256):
    if any(
        hashlib.sha256((Path(metadata) / name).read_bytes()).hexdigest() != digest
        for name, digest in admitted_sha256.items()
    ):
        raise ValueError("Admitted metadata changed during trial")


def verify_trial(ranks, server, receipts, weight_files, *, repeats=1, expected_inventory=None):
    if (
        not isinstance(expected_inventory, dict)
        or set(expected_inventory) != {0, 1}
        or any(
            not isinstance(names, (set, frozenset))
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            for names in expected_inventory.values()
        )
    ):
        raise ValueError("Missing admitted per-rank inventory")
    if len(ranks) != 2 or sorted(r.get("rank", -1) for r in ranks) != [0, 1]:
        raise ValueError("Missing two actual rank records")
    if not server.get("completed") or server.get("rank") != 0 or weight_files:
        raise ValueError("Server incomplete or receiver has weight files")
    if len(receipts) != 3 or any(
        r.get("reason") != "completed"
        or not r.get("cleanup", {}).get("child_reaped")
        or not r.get("cleanup", {}).get("group_gone")
        for r in receipts
    ):
        raise ValueError("Missing verified successful actor cleanup")
    reference = None
    prompt_reference, prompt_tokens_reference = None, None
    for rank in ranks:
        loading = rank.get("weight_loading") or {}
        requests = rank.get("requests") or []
        hashes = loading.get("tensor_sha256") or {}
        if not isinstance(hashes, dict) or set(hashes) != expected_inventory[rank["rank"]]:
            raise ValueError("Tensor names differ from admitted rank inventory")
        if any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in hashes.values()
        ):
            raise ValueError("Invalid tensor SHA-256 digest")
        if (
            rank.get("world_size") != 2
            or not loading.get("all_retained_tensors_loaded")
            or loading.get("tensor_count", 0) != len(hashes)
            or not hashes
        ):
            raise ValueError("Incomplete world/tensor evidence")
        admitted_layers = {
            int(name.split(".")[2])
            for name in expected_inventory[rank["rank"]]
            if name.startswith("model.layers.")
        }
        if (
            not admitted_layers
            or rank.get("layer_start") != min(admitted_layers)
            or rank.get("layer_end") != max(admitted_layers) + 1
        ):
            raise ValueError("Rank layer range differs from admitted inventory")
        layout = rank.get("parameter_layout")
        if (
            not isinstance(layout, list)
            or len(layout) != len(hashes)
            or any(
                not isinstance(item, dict)
                or item.get("name") not in hashes
                or type(item.get("bytes")) is not int
                or item["bytes"] < 0
                for item in layout
            )
            or {item["name"] for item in layout} != set(hashes)
        ):
            raise ValueError("Parameter layout differs from admitted inventory")
        actual_bytes = sum(item["bytes"] for item in layout)
        if (
            rank.get("parameter_bytes") != actual_bytes
            or loading.get("parameter_bytes") != actual_bytes
        ):
            raise ValueError("Parameter bytes contradict loading/layout evidence")
        if len(requests) != repeats:
            raise ValueError("Missing repeats")
        prompt_evidence = tuple(
            rank.get(field) for field in ("prompt", "formatted_prompt", "execution")
        )
        if any(not isinstance(value, str) or not value for value in prompt_evidence):
            raise ValueError("Missing prompt/execution evidence")
        if prompt_reference is None:
            prompt_reference = prompt_evidence
        if prompt_evidence != prompt_reference:
            raise ValueError("Ranks disagree on prompt/execution")
        for request in requests:
            prompt_tokens = request.get("prompt_tokens")
            if type(prompt_tokens) is not int or prompt_tokens <= 0:
                raise ValueError("Missing tokenized prompt evidence")
            if prompt_tokens_reference is None:
                prompt_tokens_reference = prompt_tokens
            if prompt_tokens != prompt_tokens_reference:
                raise ValueError("Ranks/repeats disagree on tokenized prompt")
            tokens = request.get("token_ids")
            if not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("Missing actual greedy token IDs")
            if reference is None:
                reference = tokens
            if tokens != reference:
                raise ValueError("Ranks/repeats disagree")
    by_rank = {rank["rank"]: rank for rank in ranks}
    replicated = expected_inventory[0] & expected_inventory[1]
    if any(
        by_rank[0]["weight_loading"]["tensor_sha256"][name]
        != by_rank[1]["weight_loading"]["tensor_sha256"][name]
        for name in replicated
    ):
        raise ValueError("Replicated tensor SHA-256 digests disagree")
    if server.get("selected_tensor_count") != by_rank[0]["weight_loading"]["tensor_count"]:
        raise ValueError("Server/receiver tensor counts differ")
    if server.get("selected_bytes") != by_rank[0]["parameter_bytes"]:
        raise ValueError("Server/receiver parameter bytes differ")
    return {
        "tokens_agree": True,
        "tensor_completeness": True,
        "server_completed": True,
        "receiver_weight_files": [],
        "owned_groups_gone": True,
    }


class Controller:
    """All child polling/resource checks occur at <=0.1-second intervals."""

    def __init__(self, args):
        validate_args(args)
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(mode=0o700)
        self.started = time.monotonic()
        self.children = {}
        self.files = []
        self.signals = []
        self.handlers = {
            sig: signal.signal(sig, self.on_signal)
            for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        }
        self.calls = 0
        self.control_cleanup = {}
        self.last_space = 0

    def on_signal(self, signum, frame):
        self.signals.append(signum)

    def check(self):
        if self.signals:
            raise InterruptedError("Controller received signal")
        if time.monotonic() - self.started >= self.args.deadline:
            raise TimeoutError("Trial deadline")
        stat = os.statvfs("/")
        free = stat.f_bavail * stat.f_frsize
        if free < self.args.reserve_bytes:
            raise OSError("Boot volume reserve breached")
        if time.monotonic() - self.last_space >= 0.5:
            write_json(
                self.output / "space.json",
                {"boot_volume_available_bytes": free, "elapsed": time.monotonic() - self.started},
            )
            self.last_space = time.monotonic()
        for name in ("server", "rank0", "rank1"):
            child = self.children.get(name)
            if child is not None and child.poll() not in (None, 0):
                raise RuntimeError(f"{name} actor failed")

    def launch(self, name, argv, *, job=None):
        self.check()
        handles = [
            os.fdopen(
                os.open(
                    self.output / f"{name}.{suffix}", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                ),
                "wb",
            )
            for suffix in ("stdout", "stderr")
        ]
        self.files.extend(handles)
        child = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if job else subprocess.DEVNULL,
            stdout=handles[0],
            stderr=handles[1],
            start_new_session=True,
        )
        self.children[name] = child
        write_json(
            self.output / f"{name}.transport.json",
            {
                "pid": child.pid,
                "pgid": child.pid,
                "nonce": job.get("nonce") if job else None,
                "argv": argv,
            },
        )
        if job:
            # Small validated job; token stays exclusively on stdin, never in records.
            data = json.dumps(job).encode() + b"\n"
            if len(data) > 16384:
                raise ValueError("Job exceeds bounded stdin")
            child.stdin.write(data)
            child.stdin.close()
        return child

    def call(self, argv, *, cleanup=False):
        self.calls += 1
        name = f"control{self.calls}"
        if cleanup:
            # Cleanup remains bounded even after trial deadline/signal.
            handles = [
                os.fdopen(
                    os.open(
                        self.output / f"{name}.{suffix}",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    ),
                    "wb",
                )
                for suffix in ("stdout", "stderr")
            ]
            self.files.extend(handles)
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=handles[0],
                stderr=handles[1],
                start_new_session=True,
            )
            self.children[name] = child
        else:
            child = self.launch(name, argv)
        handles = self.files[-2:]
        until = time.monotonic() + 7
        try:
            while child.poll() is None:
                if not cleanup:
                    self.check()
                if time.monotonic() >= until:
                    raise TimeoutError("Bounded actor control call")
                time.sleep(0.05)
            if child.returncode:
                raise RuntimeError("Actor control call failed")
            raw = self.output / f"{name}.stdout"
            if raw.stat().st_size > 20 * 1024**2:
                raise ValueError("Actor control response too large")
            return json.loads(raw.read_text())
        finally:
            cleanup_result = terminate(child)
            for handle in handles:
                handle.close()
                self.files.remove(handle)
            self.control_cleanup[name] = cleanup_result
            write_json(self.output / f"{name}.cleanup.json", cleanup_result)
            if cleanup_result["child_reaped"] and cleanup_result["group_gone"]:
                self.children.pop(name, None)
            else:
                raise RuntimeError("Control transport termination pending")

    def close(self):
        cleanup = {name: terminate(child) for name, child in self.children.items()}
        for handle in self.files:
            handle.close()
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        return cleanup


def verify_sources(local, peer):
    if peer.get("source_sha256") != local:
        raise ValueError("Peer executed source differs from admitted local source")


def free_port(address):
    with socket.socket() as listener:
        listener.bind((address, 0))
        return listener.getsockname()[1]


def listener_closed(address, port):
    import errno

    with socket.socket() as connection:
        connection.settimeout(2.0)
        try:
            connection.connect((address, port))
        except OSError as exc:
            return exc.errno == errno.ECONNREFUSED
        return False


def weight_files(root):
    extensions = {".safetensors", ".npz", ".bin", ".gguf", ".pt", ".pth"}
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in extensions]


def json_records(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run(args, *, transport=ssh_argv):
    """Real execution entry; tests replace transport only with local subprocesses."""
    controller = Controller(args)
    output = controller.output
    actors = output / "actors"
    actors.mkdir(mode=0o700)
    jobs, final, error_type = {}, {}, None
    metadata_sha256 = {}
    hashes, hostfile, server_port = None, None, None
    stage = "metadata_admission"
    local_cleanup, statuses, remote_cleanup_errors = {}, {}, []
    trial_nonce = secrets.token_hex(16)
    startup = {
        "controller_pid": os.getpid(),
        "nonce": trial_nonce,
        "args": vars(args),
        "status": "started",
        "token": "[REDACTED]",
    }
    write_json(output / "startup.json", startup)

    def remote_call(extra, cleanup=False):
        return controller.call(transport(args, ["--jobs", SCRATCH, *extra]), cleanup=cleanup)

    def start(name, entrypoint, argv, *, rank=None, remote=True, token=None):
        controller.check()
        remaining = args.deadline - (time.monotonic() - controller.started)
        job = {
            "nonce": secrets.token_hex(16),
            "entrypoint": entrypoint,
            "argv": argv,
            "deadline": remaining,
            "reserve_bytes": args.reserve_bytes,
            "source_sha256": hashes,
            "env": {},
        }
        if token:
            job["env"]["PARETOQUANT_WEIGHT_TOKEN"] = token
        if rank is not None:
            job["env"]["MLX_RANK"] = str(rank)
            job["hostfile"] = hostfile
        jobs[name] = {"nonce": job["nonce"], "remote": remote}
        write_json(output / "actors.json", jobs)
        write_json(
            output / f"{name}.job.json",
            {
                **job,
                "env": {
                    k: "[REDACTED]" if k == "PARETOQUANT_WEIGHT_TOKEN" else v
                    for k, v in job["env"].items()
                },
            },
        )
        command = (
            transport(args, ["--jobs", SCRATCH])
            if remote
            else [
                sys.executable,
                "-u",
                str(ROOT / "remote_actor.py"),
                "--root",
                str(ROOT),
                "--jobs",
                str(actors),
            ]
        )
        return controller.launch(name, command, job=job)

    def status(name, *, records=False, cleanup=False):
        item = jobs[name]
        if item["remote"]:
            extra = ["--status", item["nonce"]]
            if records:
                extra.append("--records")
            return remote_call(extra, cleanup=cleanup)
        from remote_actor import job_status

        return job_status(actors, item["nonce"], records=records)

    try:
        metadata = Path(args.local_metadata).resolve(strict=True)
        if weight_files(metadata):
            raise ValueError("Receiver metadata must contain no weight files")
        metadata_bytes = {
            name: (metadata / name).read_bytes()
            for name in ("config.json", "model.safetensors.index.json")
        }
        config = json.loads(metadata_bytes["config.json"])
        if (
            config.get("model_type") != "qwen2"
            or config.get("model_file")
            or sum(args.split) != config.get("num_hidden_layers")
        ):
            raise ValueError("Require built-in Qwen2 and complete explicit split")
        index = json.loads(metadata_bytes["model.safetensors.index.json"])
        if (
            not isinstance(index, dict)
            or not isinstance(index.get("weight_map"), dict)
            or not index["weight_map"]
        ):
            raise ValueError("Trusted index requires a nonempty weight_map object")
        inventory = expected_inventory(config, index["weight_map"], args.split)
        metadata_sha256 = {
            name: hashlib.sha256(data).hexdigest() for name, data in metadata_bytes.items()
        }
        startup["metadata_sha256"] = metadata_sha256
        startup["expected_tensor_names"] = {
            rank: sorted(names) for rank, names in inventory.items()
        }
        write_json(output / "startup.json", startup)
        stage = "source_admission"
        snapshot = output / "sources"
        snapshot.mkdir(mode=0o700)
        hashes = {}
        for name in (*SOURCES, "supervise.py"):
            path = ROOT / name
            if path.is_symlink():
                raise ValueError("Source symlinks forbidden")
            data = source_bytes(path)
            with os.fdopen(
                os.open(snapshot / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
            ) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if name in SOURCES:
                hashes[name] = hashlib.sha256(data).hexdigest()
        write_json(output / "source_sha256.json", hashes)
        write_json(
            output / "controller_sha256.json", {"supervise.py": hashlib.sha256(data).hexdigest()}
        )
        peer = controller.call(
            transport(args, ["--inspect", "--port-address", args.remote_address])
        )
        verify_sources(hashes, peer)
        write_json(output / "peer_source.json", peer)
        local_port = free_port(args.local_address)
        remote_port = peer["available_port"]
        hostfile = [
            [f"{args.local_address}:{local_port}"],
            [f"{args.remote_address}:{remote_port}"],
        ]
        write_json(output / "hostfile.json", hostfile)
        token = secrets.token_hex(32)
        split = [str(x) for x in args.split]
        start(
            "server",
            "stream_server.py",
            [
                "--source",
                args.remote_source,
                "--bind",
                args.remote_address,
                "--peer",
                args.local_address,
                "--split",
                *split,
                "--rank",
                "0",
            ],
            token=token,
        )
        stage = "server_readiness"
        ready_until = min(controller.started + args.deadline, time.monotonic() + 45)
        while True:
            controller.check()
            current = status("server")
            if "ready" in current:
                address, server_port = current["ready"]["address"]
                if (
                    address != args.remote_address
                    or type(server_port) is not int
                    or not 1 <= server_port <= 65535
                ):
                    raise ValueError("Invalid actual server address")
                verify_sources(hashes, current)
                write_json(output / "server.ready.json", current)
                break
            if current.get("result"):
                raise RuntimeError("Weight server failed readiness")
            if time.monotonic() >= ready_until:
                raise TimeoutError("Weight server readiness deadline")
            time.sleep(0.1)
        stage = "rank_execution"
        common = [
            "--mode",
            "generate",
            "--split",
            *split,
            "--max-tokens",
            str(args.max_tokens),
            "--repeats",
            str(args.repeats),
            "--prompt",
            args.prompt,
        ]
        if args.execution == "phased":
            common.append("--phased")
        start(
            "rank1", "probe.py", [*common, "--model", args.remote_source, "--stream-local"], rank=1
        )
        start(
            "rank0",
            "probe.py",
            [
                *common,
                "--model",
                args.local_metadata,
                "--weight-server",
                f"{args.remote_address}:{server_port}",
                "--weight-client-bind",
                args.local_address,
            ],
            rank=0,
            remote=False,
            token=token,
        )
        del token
        while any(
            controller.children[name].poll() is None for name in ("server", "rank0", "rank1")
        ):
            controller.check()
            time.sleep(0.1)
        controller.check()
        stage = "verification"
        for name in jobs:
            statuses[name] = status(name, records=True)
            verify_sources(hashes, statuses[name])
            write_json(output / f"{name}.receipt.json", statuses[name])
        ranks = [json_records(statuses[name]["stdout"])[-1] for name in ("rank0", "rank1")]
        server = json_records(statuses["server"]["stdout"])[-1]
        verify_metadata(metadata, metadata_sha256)
        final = verify_trial(
            ranks,
            server,
            [statuses[n]["result"] for n in jobs],
            weight_files(metadata),
            repeats=args.repeats,
            expected_inventory=inventory,
        )
        ports = [
            (args.local_address, local_port),
            (args.remote_address, remote_port),
            (args.remote_address, server_port),
        ]
        closed = [listener_closed(address, port) for address, port in ports]
        write_json(output / "listeners.json", {"ports": ports, "closed": closed})
        if not all(closed):
            raise RuntimeError("Listener termination not verified")
        final["listeners_closed"] = True
        final["status"] = "success"
    except BaseException as exc:
        error_type = type(exc).__name__  # no exception args / credentials
        final = {"status": "failed", "error_type": error_type, "error_stage": stage}
    finally:
        # Request only nonce-owned actors to stop. Do not kill by stale remote PID.
        from remote_actor import request_stop

        for name, item in jobs.items():
            try:
                if item["remote"]:
                    remote_call(["--stop", item["nonce"]], cleanup=True)
                else:
                    request_stop(actors, item["nonce"])
            except BaseException as exc:
                remote_cleanup_errors.append({"actor": name, "error_type": type(exc).__name__})
        until = time.monotonic() + 3
        while time.monotonic() < until and any(
            p.poll() is None for p in controller.children.values()
        ):
            time.sleep(0.05)
        for name in jobs:
            try:
                current = status(name, records=True, cleanup=True)
                statuses[name] = current
                write_json(output / f"{name}.receipt.json", current)
            except BaseException as exc:
                remote_cleanup_errors.append({"actor": name, "error_type": type(exc).__name__})
        local_cleanup = controller.close()
        owned_clean = len(statuses) == len(jobs) and all(
            s.get("result", {}).get("cleanup", {}).get("child_reaped")
            and s.get("result", {}).get("cleanup", {}).get("group_gone")
            for s in statuses.values()
        )
        known_ports = []
        if hostfile:
            for host in hostfile:
                address, port = host[0].rsplit(":", 1)
                known_ports.append((address, int(port)))
        server_ready = statuses.get("server", {}).get("ready")
        if server_port is None and server_ready:
            server_port = server_ready["address"][1]
        if server_port is not None:
            known_ports.append((args.remote_address, server_port))
        closed = [listener_closed(address, port) for address, port in known_ports]
        services_closed = (
            owned_clean and all(closed) and ("server" not in jobs or server_port is not None)
        )
        write_json(
            output / "listeners.cleanup.json",
            {
                "ports": known_ports,
                "closed": closed,
                "services_closed_verified": services_closed,
            },
        )
        final.update(
            {
                "nonce": trial_nonce,
                "error_type": error_type,
                "metadata_sha256": metadata_sha256,
                "owned_actor_cleanup_verified": owned_clean,
                "services_closed_verified": services_closed,
                "transport_cleanup": local_cleanup,
                "cleanup_errors": remote_cleanup_errors,
                "elapsed_seconds": time.monotonic() - controller.started,
            }
        )
        transport_clean = all(
            item["child_reaped"] and item["group_gone"]
            for item in (*local_cleanup.values(), *controller.control_cleanup.values())
        )
        final["control_transport_cleanup"] = controller.control_cleanup
        if not owned_clean or remote_cleanup_errors or not services_closed or not transport_clean:
            final["status"] = "failed"
            final["pending_cleanup"] = True
        write_json(output / "result.json", final)
    return final


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
