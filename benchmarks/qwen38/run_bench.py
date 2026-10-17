"""Bounded serial owner for the preregistered same-loaded native/MTP suite.

CLI: --profile PROFILE --workloads FROZEN_WORKLOADS --runtime RUNTIME
     --output NEW_EXPERIMENT [--batches 0 1 2 3 4 5] [--deadline 900].
--plan/--dry-run only validates local inputs: no health calls, SSH, or writes.
--collect-only MANIFEST normalizes existing receipts without touching hardware;
see collect_trials for the manifest schema, including trusted deployment hashes.

Execution always stops on the first failed preflight, deployment, supervisor,
cleanup, postflight, or normalization. No automatic cleanup/retry/resume or PID
signalling is performed. Use a new exclusive experiment directory for another
attempt. Locks are never auto-broken. All six frozen schedules remain expected
in cumulative evidence, even when only a subset of batches was requested.
"""

import base64
import hashlib
import ipaddress
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def build_supervisor_command(
    profile, local_root, remote_root, output, batch, deadline=900, *, schedule_sha256=None
):
    if type(deadline) is not int or not 1 <= deadline <= 900:
        raise ValueError("Require a bounded deadline in1..900seconds")
    if (
        not isinstance(schedule_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", schedule_sha256) is None
    ):
        raise ValueError("Require the exact frozen schedule SHA-256")
    requests = batch.get("requests", [])
    if not isinstance(requests, list) or not 1 <= len(requests) <= 32:
        raise ValueError("Require1..32 scheduled requests")
    if any(
        not isinstance(r, dict)
        or type(r.get("max_tokens")) is not int
        or not 1 <= r["max_tokens"] <= 512
        for r in requests
    ):
        raise ValueError("Require integer request caps in1..512")
    params = profile["parameters"]
    result = [sys.executable, "-u", str(local_root / "supervise.py")]
    for flag, value in [
        ("remote-root", remote_root),
        ("remote-source", params["remote_source"]),
        ("local-metadata", params["local_metadata"]),
        ("output", output),
        ("ssh-key", params["ssh_key"]),
    ]:
        result.extend(["--" + flag, str(value)])
    result.extend(
        [
            "--split",
            *map(str, params["split"]),
            "--max-tokens",
            str(max(r["max_tokens"] for r in requests)),
            "--repeats",
            str(len(requests)),
            "--deadline",
            str(deadline),
            "--reserve-bytes",
            str(8 * 1024**3),
            "--execution",
            "phased",
            "--prefix-rollback",
            "--benchmark-schedule",
        ]
    )
    for flag in [
        "local-address",
        "remote-address",
        "ssh-target",
        "ssh-host-key-alias",
        "eval-group-size",
        "rank0-verifier",
        "rank1-verifier",
        "target-revision",
        "target-config-sha256",
        "target-index-sha256",
        "sidecar",
        "small-model-local",
        "small-model-remote",
        "small-config-sha256",
        "small-index-sha256",
    ]:
        result.extend(["--" + flag, str(params[flag.replace("-", "_")])])
    result.extend(
        ["--prompt", "Explain in two short sentences what a queue is in computer science."]
    )
    result.extend(["--benchmark-schedule-sha256", schedule_sha256])
    return result


def validate_health(record):
    if type(record.get("disk_free_bytes")) is not int or record["disk_free_bytes"] < 8 * 1024**3:
        raise ValueError("Insufficient free disk reserve")
    if record.get("pressure_level") != 1:
        raise ValueError("Memory pressure is not normal")
    if "interface: bridge0" not in record.get("route", ""):
        raise ValueError("Peer route is not verified Thunderbolt Bridge")
    if record.get("owned_inference_workers") != []:
        raise ValueError("An inference worker is already active or discovery is incomplete")


HEALTH_CODE = r"""
import json,os,platform,re,subprocess,sys,time
def run(args):
    return subprocess.run(args,capture_output=True,text=True,check=True,timeout=8).stdout
disk=os.statvfs('/System/Volumes/Data')
workers=[]
for line in run(['ps','-axo','pid=,command=']).splitlines():
    parts=line.strip().split(None,1)
    if len(parts)!=2:continue
    # ps flattens argv without shell escaping: prompt quotes are ordinary data.
    argv=parts[1].split()
    if not argv or not os.path.basename(argv[0]).startswith('python'):continue
    # Conservatively recognize visible owned script paths, even with spaces/quotes.
    if re.search(
        r"/(?:sources|paretoquant[^/\s]*)/(?:[^\r\n]*/)?"
        r"(?:probe|stream_server)\.py(?:\s|['\"]|$)",parts[1]
    ):
        workers.append({'pid':int(parts[0]),'argv':argv})
print(json.dumps({
    'timestamp':time.time(),'platform':platform.platform(),
    'memory_bytes':int(run(['sysctl','-n','hw.memsize']).strip()),
    'disk_free_bytes':disk.f_bavail*disk.f_frsize,
    'pressure_level':int(run(['sysctl','-n','kern.memorystatus_vm_pressure_level']).strip()),
    'swap':run(['sysctl','vm.swapusage']).strip(),
    'route':run(['route','-n','get',sys.argv[1]]),
    'power':run(['pmset','-g','batt']),
    'thermal':subprocess.run(['pmset','-g','therm'],capture_output=True,text=True,timeout=8).stdout,
    'owned_inference_workers':workers
},allow_nan=False))
"""


def collect_health(profile, *, remote=False):
    params = profile["parameters"]
    address = params["local_address" if remote else "remote_address"]
    ipaddress.IPv4Address(address)
    command = ["/usr/bin/python3", "-c", HEALTH_CODE, address]
    if remote:
        command = [
            "ssh",
            "-T",
            "-i",
            params["ssh_key"],
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "HostKeyAlias=" + params["ssh_host_key_alias"],
            params["ssh_target"],
            shlex.join(command),
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=35)
    return json.loads(result.stdout)


def freeze_snapshot(source, dest, schedule_bytes):
    if source.is_symlink():
        raise ValueError("Source directory cannot be a symlink")
    source = source.resolve(strict=True)
    payload = {}
    for path in sorted(source.iterdir()):
        if path.suffix not in {".py", ".json", ".metal"}:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Source entries must be regular files")
        payload[path.name] = path.read_bytes()
    if not {"supervise.py", "remote_actor.py"} <= set(payload):
        raise ValueError("Incomplete supervisor source")
    parsed = json.loads(schedule_bytes)
    if parsed.get("schema") != "qwen38-workload-schedule-v1" or not parsed.get("requests"):
        raise ValueError("Missing benchmark request schedule")
    payload["benchmark_schedule.json"] = schedule_bytes
    dest.mkdir(parents=True, exist_ok=False)
    for name, data in payload.items():
        with (dest / name).open("xb") as handle:
            handle.write(data)
    return {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}


def deployment_payload(snapshot, remote_name):
    if re.fullmatch(r"023-qwen38-broad-[A-Za-z0-9-]{1,100}", remote_name) is None:
        raise ValueError("Deployment must use a new benchmark-scoped directory name")
    files, hashes = {}, {}
    for path in sorted(snapshot.iterdir()):
        if path.suffix not in {".py", ".json", ".metal"}:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Cannot deploy source links or non-files")
        data = path.read_bytes()
        files[path.name] = base64.b64encode(data).decode("ascii")
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    if not files:
        raise ValueError("Cannot deploy an empty snapshot")
    return {"name": remote_name, "files": files, "hashes": hashes}


DEPLOY_CODE = r"""
import base64,hashlib,json,re,sys
from pathlib import Path
p=json.load(sys.stdin)
if re.fullmatch(r'023-qwen38-broad-[A-Za-z0-9-]{1,100}',p['name']) is None:
    raise ValueError('Invalid benchmark deployment name')
root=Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes')
dest=root/p['name']
files={}
for name,encoded in p['files'].items():
    if Path(name).name!=name or Path(name).suffix not in {'.py','.json','.metal'}:
        raise ValueError('Invalid source filename')
    files[name]=base64.b64decode(encoded,validate=True)
expected={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
if expected!=p['hashes']:raise ValueError('Payload hashes disagree')
dest.mkdir(exist_ok=False)
for name,data in files.items():
    with (dest/name).open('xb') as f:f.write(data)
actual={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in files}
if actual!=expected:raise ValueError('Remote source readback failed')
print(json.dumps({'remote_root':str(dest),'source_sha256':actual,'readback_verified':True}))
"""


def deploy_snapshot(profile, snapshot, remote_name):
    payload = deployment_payload(snapshot, remote_name)
    params = profile["parameters"]
    command = [
        "ssh",
        "-T",
        "-i",
        params["ssh_key"],
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "HostKeyAlias=" + params["ssh_host_key_alias"],
        params["ssh_target"],
        shlex.join(["/usr/bin/python3", "-c", DEPLOY_CODE]),
    ]
    actual = subprocess.run(
        command,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    receipt = json.loads(actual.stdout)
    if receipt.get("source_sha256") != payload["hashes"]:
        raise ValueError("Remote readback does not match exact local snapshot")
    expected_path = "/Users/jeremylien/Projects/paretoquant-dual-host/spikes/" + remote_name
    if receipt.get("remote_root") != expected_path or receipt.get("readback_verified") is not True:
        raise ValueError("Remote deployment target is unverified")
    return receipt


def analyzer_module():
    """Import the sibling CPU analyzer by absolute path, never a generic package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_qwen38_operator_analyzer", Path(__file__).with_name("analyze.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("CPU analyzer cannot be loaded")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def load_json(path):
    analyzer = analyzer_module()
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=analyzer._unique_json_object,
        parse_constant=analyzer._reject_constant,
    )
    json.dumps(value, allow_nan=False)
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_symlinks(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Path symlinks forbidden")
    return path.resolve()


def disjoint(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left != right and left not in right.parents and right not in left.parents


def source_file_hashes(root):
    root = no_symlinks(root)
    hashes = {}
    for path in sorted(root.iterdir()):
        if path.suffix not in {".py", ".json", ".metal"}:
            continue
        require(not path.is_symlink() and path.is_file(), "Source entries must be regular files")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_batch(batch):
    require(
        isinstance(batch, dict)
        and set(batch) == {"schema", "requests"}
        and batch["schema"] == "qwen38-workload-schedule-v1",
        "Invalid schedule schema",
    )
    requests = batch["requests"]
    require(
        isinstance(requests, list) and 1 <= len(requests) <= 32, "Schedule requires 1..32 requests"
    )
    fields = {"id", "case_id", "method", "warmup", "repeat", "prompt", "max_tokens"}
    for row in requests:
        require(isinstance(row, dict) and set(row) == fields, "Invalid schedule request fields")
        require(
            type(row["max_tokens"]) is int
            and 1 <= row["max_tokens"] <= 512
            and type(row["repeat"]) is int
            and 0 <= row["repeat"] < 32,
            "Request bounds exceed runtime admission",
        )
        require(
            all(
                isinstance(row[k], str) and 0 < len(row[k].encode("utf-8")) <= 262144
                for k in ("id", "case_id", "prompt")
            ),
            "Invalid bounded request strings",
        )
    analyzer_module().validate_pairs(
        [], requests, allow_partial=True, eos_token_ids=[248044, 248046]
    )


def prepare_plan(profile_path, workloads, output, runtime, batches, deadline=900):
    require(type(deadline) is int and 1 <= deadline <= 900, "Deadline must be in 1..900")
    require(
        isinstance(batches, list)
        and batches
        and all(type(i) is int and 0 <= i < 6 for i in batches)
        and len(set(batches)) == len(batches),
        "Require distinct batch indices in 0..5",
    )
    profile_path, workloads, output, runtime = map(
        no_symlinks, (profile_path, workloads, output, runtime)
    )
    if output.exists():
        raise FileExistsError("Experiment output already exists")
    require(
        runtime.is_dir()
        and all((runtime / n).is_file() for n in ("supervise.py", "remote_actor.py")),
        "Runtime source is incomplete",
    )
    profile = load_json(profile_path)
    params = profile["parameters"]
    protected = [
        Path(params[k])
        for k in (
            "local_metadata",
            "remote_source",
            "sidecar",
            "small_model_local",
            "small_model_remote",
        )
    ]
    source_roots = [runtime]
    source_roots.extend(
        Path(value) for value in (profile.get("local_runtime"), params.get("remote_root")) if value
    )
    require(
        all(disjoint(output, p) for p in (*source_roots, workloads, profile_path, *protected)),
        "Output must be disjoint from workload/source/checkpoint/sidecar paths",
    )
    require(
        all(disjoint(workloads, p) for p in source_roots),
        "Workload root must be disjoint from all runtime sources",
    )
    require(
        all(disjoint(workloads, p) and disjoint(runtime, p) for p in protected),
        "Workload/source must be disjoint from checkpoint/sidecar paths",
    )
    manifest = load_json(workloads / "manifest.json")
    registration = load_json(workloads.parent / "preregistration.json")
    require(
        registration.get("schema") == "qwen38-broad-preregistration-v1",
        "Invalid preregistration schema",
    )
    require(
        manifest.get("schema") == "qwen38-workload-manifest-v1"
        and len(manifest.get("batches", [])) == 6,
        "Require six frozen workload batches",
    )
    for field, trusted in (
        ("model", profile["model"]),
        ("revision", params["target_revision"]),
        ("split", params["split"]),
        ("eval_group_size", params["eval_group_size"]),
    ):
        require(manifest.get(field) == trusted, "Manifest/profile binding differs: " + field)
    require(
        params["split"] == [28, 36]
        and params["eval_group_size"] == 32
        and params["rank0_verifier"] == "native"
        and params["rank1_verifier"] == "slot-qmv",
        "Require admitted split28/36 eval32 and verifier modes",
    )
    required_files = {
        "workloads/manifest.json",
        *(f"workloads/schedule-{i:02d}.json" for i in range(6)),
    }
    files = registration.get("files", {})
    require(
        isinstance(files, dict) and required_files <= set(files),
        "Missing registered manifest/schedules",
    )
    for name, digest in files.items():
        require(name in required_files | {"PROTOCOL.md"}, "Unsafe preregistration filename")
        path = no_symlinks(workloads.parent / name)
        require(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            "Frozen file hash mismatch: " + name,
        )
    schedules, schedule_sha256 = [], []
    for i, batch in enumerate(manifest["batches"]):
        validate_batch(batch)
        path = workloads / f"schedule-{i:02d}.json"
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != registration["files"]["workloads/" + path.name] or load_json(path) != batch:
            raise ValueError("Schedule differs from frozen registration/manifest")
        schedules.append(batch)
        schedule_sha256.append(digest)
    expected = [request for batch in schedules for request in batch["requests"]]
    analyzer_module().validate_pairs(
        [], expected, allow_partial=True, eos_token_ids=[248044, 248046]
    )
    return dict(
        schema="qwen38-operator-plan-v1",
        profile=profile,
        profile_path=str(profile_path.resolve()),
        workloads=str(workloads.resolve()),
        runtime=str(runtime.resolve()),
        runtime_sha256=source_file_hashes(runtime),
        frozen_files=files,
        preregistration_sha256=hashlib.sha256(
            (workloads.parent / "preregistration.json").read_bytes()
        ).hexdigest(),
        output=str(output.resolve()),
        selected_batches=batches,
        full_suite_selected=set(batches) == set(range(6)),
        expected_requests=expected,
        selected_request_ids=[r["id"] for i in batches for r in schedules[i]["requests"]],
        schedules=schedules,
        schedule_sha256=schedule_sha256,
        deadline=deadline,
    )


def json_records(text):
    """Actor stdout/stderr are JSONL interleaved with runtime diagnostics."""
    records = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def source_registry(root):
    """Inspect literal actor registry without importing runtime code/MLX."""
    import ast

    tree = ast.parse((no_symlinks(Path(root) / "remote_actor.py")).read_bytes())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets)
    ]
    require(
        len(values) == 1 and isinstance(values[0], tuple) and len(set(values[0])) == len(values[0]),
        "Missing exact actor source registry",
    )
    require(
        all(
            isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9_]+\.(?:py|json|metal)", n)
            for n in values[0]
        ),
        "Unsafe actor source filename",
    )
    return values[0]


def verify_cleanup(trialdir):
    """Read only; never signal PIDs, delete jobs, or repair pending cleanup."""
    trialdir = Path(trialdir)
    final = load_json(trialdir / "result.json")
    require(
        final.get("owned_actor_cleanup_verified") is True
        and final.get("services_closed_verified") is True
        and final.get("cleanup_errors") == []
        and not final.get("pending_cleanup"),
        "Trial has unverified owned-worker/service cleanup",
    )
    listeners = load_json(trialdir / "listeners.cleanup.json")
    require(
        listeners.get("services_closed_verified") is True
        and isinstance(listeners.get("ports"), list)
        and listeners["ports"]
        and len(listeners.get("closed", [])) == len(listeners["ports"])
        and all(v is True for v in listeners["closed"]),
        "Listener closure is unverified",
    )
    for name in ("rank0", "rank1", "server"):
        result = load_json(trialdir / f"{name}.receipt.json").get("result", {})
        require(
            result.get("cleanup", {}).get("child_reaped") is True
            and result.get("cleanup", {}).get("group_gone") is True,
            "Actor cleanup is incomplete: " + name,
        )
    return final


def trusted_inventory(profile):
    """Derive inventory from profile-pinned metadata, not claimed actor names."""
    params = profile["parameters"]
    root = no_symlinks(params["local_metadata"])
    binding = {
        "config.json": params["target_config_sha256"],
        "model.safetensors.index.json": params["target_index_sha256"],
    }
    for name, digest in binding.items():
        require(
            hashlib.sha256(no_symlinks(root / name).read_bytes()).hexdigest() == digest,
            "Trusted metadata changed: " + name,
        )
    config = load_json(root / "config.json")
    text = config.get("text_config", config)
    split = params["split"]
    require(
        config.get("model_type") == "qwen3_5"
        and not config.get("model_file")
        and split == [28, 36]
        and text.get("num_hidden_layers") == 64,
        "Trusted model/split must be Qwen3.8 64-layer split28/36",
    )
    names = load_json(root / "model.safetensors.index.json")["weight_map"]
    replicated, layer_names = set(), {}
    for name in names:
        if name.startswith(("vision_tower.", "model.visual.")):
            continue
        match = re.fullmatch(r"language_model\.model\.layers\.(0|[1-9][0-9]*)\.(.+)", name)
        if match:
            require(int(match[1]) < 64, "Invalid trusted layer name")
            layer_names[name] = int(match[1])
        else:
            require(
                re.fullmatch(
                    r"language_model\.(?:model\.embed_tokens|model\.norm|lm_head)\..+", name
                ),
                "Unrecognized trusted tensor name",
            )
            replicated.add(name)
    required = {"language_model.model.embed_tokens.weight", "language_model.model.norm.weight"}
    if not text.get("tie_word_embeddings", False):
        required.add("language_model.lm_head.weight")
    require(
        required <= replicated and set(layer_names.values()) == set(range(64)),
        "Incomplete trusted inventory",
    )
    inventory = {
        rank: replicated
        | {
            name
            for name, layer in layer_names.items()
            if sum(split[rank + 1 :]) <= layer < sum(split[rank:])
        }
        for rank in (0, 1)
    }
    require(len(inventory[0] | inventory[1]) == 1847, "Require exact 1847 target tensor union")
    return inventory, binding


def normalize_completed_trial(trialdir, batch, expected_sources, profile):
    """Return admitted analyzer rows; independent rank memory, rank0 timings.

    This is a CPU readback check of controller/actor evidence, not a new
    numerical gate execution. The source-bound supervisor owns gate validation.
    ``expected_sources`` is the caller's frozen deployment hash map (including
    supervise.py); optional ``profile.native_control_tensor_sha256`` pins values.
    """
    trialdir = no_symlinks(trialdir)
    final = verify_cleanup(trialdir)
    require(
        final.get("status") == "success"
        and final.get("strict_numerical_admission") is True
        and final.get("rank_actions_agree") is True,
        "Trial lacks numerical/rank success admission",
    )
    registry = source_registry(trialdir / "sources")
    require(
        isinstance(expected_sources, dict)
        and set(registry) | {"supervise.py"} <= set(expected_sources),
        "Expected sources must include actor registry and supervisor",
    )
    for name in (*registry, "supervise.py"):
        digest = expected_sources[name]
        require(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            and hashlib.sha256(no_symlinks(trialdir / "sources" / name).read_bytes()).hexdigest()
            == digest,
            "Executed source snapshot differs: " + name,
        )
    registered = {name: expected_sources[name] for name in registry}
    require(
        load_json(trialdir / "source_sha256.json") == registered, "Trial source manifest differs"
    )
    require(
        load_json(trialdir / "controller_sha256.json")
        == {"supervise.py": expected_sources["supervise.py"]},
        "Supervisor hash readback differs",
    )
    ranks = []
    for rid, name in enumerate(("rank0", "rank1", "server")):
        actor = load_json(trialdir / f"{name}.receipt.json")
        result = actor.get("result", {})
        require(actor.get("source_sha256") == registered, "Actor source binding differs: " + name)
        require(
            result.get("returncode") == 0
            and result.get("reason") == "completed"
            and result.get("log_evidence_complete") is True,
            "Actor did not complete cleanly: " + name,
        )
        if rid < 2:
            starts = [
                e
                for e in json_records(actor.get("stderr", ""))
                if e.get("phase") == "process_start"
            ]
            require(
                len(starts) == 1
                and starts[0].get("rank") == rid
                and starts[0].get("source_sha256") == registered,
                "Actual process-start source binding differs",
            )
            records = json_records(actor.get("stdout", ""))
            require(bool(records), "Missing actual rank result")
            ranks.append(records[-1])
    params = profile["parameters"]
    inventory, binding = trusted_inventory(profile)
    startup = load_json(trialdir / "startup.json")
    args = startup.get("args", {})
    for key in (
        "remote_source",
        "local_metadata",
        "ssh_key",
        "split",
        "local_address",
        "remote_address",
        "ssh_target",
        "ssh_host_key_alias",
        "eval_group_size",
        "rank0_verifier",
        "rank1_verifier",
        "target_revision",
        "target_config_sha256",
        "target_index_sha256",
        "sidecar",
        "small_model_local",
        "small_model_remote",
        "small_config_sha256",
        "small_index_sha256",
    ):
        require(args.get(key) == params[key], "Runtime/profile parameter differs: " + key)
    schedule_hash = registered["benchmark_schedule.json"]
    require(
        load_json(trialdir / "sources/benchmark_schedule.json") == batch,
        "Frozen batch source differs",
    )
    require(
        startup.get("metadata_sha256") == binding and final.get("metadata_sha256") == binding,
        "Runtime metadata binding differs",
    )
    require(
        startup.get("expected_tensor_names") == {str(r): sorted(n) for r, n in inventory.items()},
        "Runtime expected inventory differs from trusted metadata",
    )
    require(
        args.get("execution") == "phased"
        and args.get("prefix_rollback") is True
        and args.get("benchmark_schedule") is True
        and args.get("verification_gate_only") is False
        and not args.get("eval_group_schedule")
        and args.get("repeats") == len(batch["requests"])
        and args.get("max_tokens") == max(r["max_tokens"] for r in batch["requests"])
        and args.get("benchmark_schedule_sha256") == schedule_hash,
        "Runtime schedule/execution admission differs",
    )
    require(
        final.get("benchmark_schedule_sha256") == schedule_hash
        and final.get("verified_request_count") == len(batch["requests"])
        and final.get("paired_greedy_tokens_and_stop_agree") is True
        and final.get("tensor_completeness") is True,
        "Missing completed schedule evidence",
    )
    admission = load_json(trialdir / "schedule_admission.json")
    require(
        admission.get("sha256") == schedule_hash
        and admission.get("requests") == batch["requests"]
        and isinstance(admission.get("prompt_token_ids"), list)
        and len(admission["prompt_token_ids"]) == len(batch["requests"]),
        "Schedule admission differs",
    )
    union = {}
    for rid, rank in enumerate(ranks):
        lo, hi = sum(params["split"][rid + 1 :]), sum(params["split"][rid:])
        require(
            rank.get("rank") == rid
            and type(rank.get("rank")) is int
            and rank.get("world_size") == 2
            and rank.get("layer_start") == lo
            and rank.get("layer_end") == hi
            and rank.get("execution") == "phased"
            and rank.get("eval_group_size") == params["eval_group_size"]
            and rank.get("eval_group_schedule") == []
            and rank.get("prefix_rollback") is True
            and rank.get("target_revision") == params["target_revision"]
            and rank.get("target_binding") == binding
            and rank.get("benchmark_schedule_sha256") == schedule_hash
            and rank.get("sidecar_resident_on_rank0") is True,
            "Actual rank runtime/inventory placement differs",
        )
        mode = params[f"rank{rid}_verifier"]
        gate = rank.get("verification", {})
        require(
            rank.get("verifier_mode") == mode
            and gate.get("mode") == mode
            and gate.get("passed") is True
            and gate.get("same_loaded_shards") is True
            and gate.get("prefix_rollback") is True
            and gate.get("gate_only") is False,
            "Missing same-loaded rank numerical admission",
        )
        loading = rank.get("weight_loading", {})
        hashes = loading.get("tensor_sha256", {})
        require(
            isinstance(hashes, dict)
            and set(hashes) == inventory[rid]
            and loading.get("all_retained_tensors_loaded") is True
            and loading.get("tensor_count") == len(hashes),
            "Actual rank tensor inventory differs",
        )
        for name, digest in hashes.items():
            require(
                isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Invalid actual tensor SHA",
            )
            require(
                union.setdefault(name, digest) == digest, "Replicated target tensor hashes differ"
            )
        layout = rank.get("parameter_layout", [])
        require(
            isinstance(layout, list)
            and len(layout) == len(hashes)
            and all(
                isinstance(v, dict) and type(v.get("bytes")) is int and v["bytes"] >= 0
                for v in layout
            )
            and {v.get("name") for v in layout} == set(hashes),
            "Actual parameter layout differs",
        )
        byte_count = sum(v["bytes"] for v in layout)
        require(
            rank.get("parameter_bytes") == byte_count
            and loading.get("parameter_bytes") == byte_count,
            "Actual parameter bytes differ",
        )
        requests = rank.get("requests", [])
        require(
            isinstance(requests, list) and len(requests) == len(batch["requests"]),
            "Actual request count differs",
        )
        for ordinal, (request, spec, prompt_ids) in enumerate(
            zip(requests, batch["requests"], admission["prompt_token_ids"])
        ):
            require(isinstance(request, dict), "Missing actual request record")
            for key in ("id", "case_id", "method", "warmup", "max_tokens"):
                require(
                    request.get(key) == spec[key] and type(request.get(key)) is type(spec[key]),
                    "Actual request schedule binding differs: " + key,
                )
            require(
                isinstance(prompt_ids, list)
                and prompt_ids
                and all(type(t) is int and t >= 0 for t in prompt_ids)
                and len(prompt_ids) + spec["max_tokens"] <= 4096
                and request.get("prompt_token_ids") == prompt_ids
                and request.get("prompt_tokens") == len(prompt_ids),
                "Actual prompt tokens differ",
            )
            require(
                type(request.get("repeat")) is int
                and request["repeat"] == ordinal
                and type(request.get("sample_repeat")) is int
                and request["sample_repeat"] == spec["repeat"]
                and request.get("eval_group_size") == params["eval_group_size"]
                and request.get("local_rank") == rid
                and type(request.get("local_rank")) is int
                and type(request.get("local_peak_mlx_bytes")) is int
                and request["local_peak_mlx_bytes"] > 0
                and request.get("sidecar_resident_on_rank0") is True,
                "Missing independent actual rank memory/repeat binding",
            )
    require(len(union) == 1847, "Actual target tensor union is not 1847")
    if "native_control_tensor_sha256" in profile:
        require(
            profile["native_control_tensor_sha256"] == union,
            "Target values differ from native control hashes",
        )
    keys = (
        "id",
        "case_id",
        "method",
        "warmup",
        "sample_repeat",
        "prompt_sha256",
        "max_tokens",
        "token_ids",
        "finish_reason",
        "wall_seconds",
        "first_token_seconds",
        "prompt_tokens",
        "accepted_drafts",
        "verification_rounds",
        "teacher_forwards",
    )
    rows = []
    for i, spec in enumerate(batch["requests"]):
        left, right = (rank["requests"][i] for rank in ranks)
        require(
            all(key in left and key in right for key in keys),
            "Incomplete actual request measurements",
        )
        require(
            all(
                left.get(k) == right.get(k)
                for k in (
                    *keys[:9],
                    "finish_reason",
                    "prompt_tokens",
                    "prompt_token_ids",
                    "accepted_drafts",
                    "verification_rounds",
                    "teacher_forwards",
                )
            ),
            "Rank request outputs differ",
        )
        row = {key: left[key] for key in keys}
        row["peak_mlx_bytes_by_rank"] = [r["requests"][i]["local_peak_mlx_bytes"] for r in ranks]
        rows.append(row)
    analyzer_module().validate_pairs(rows, batch["requests"], eos_token_ids=[248044, 248046])
    return rows


def write_record(path, value, *, cumulative=False):
    """Exclusive artifacts; cumulative files atomically replace only owned output."""
    import uuid

    path = Path(path)
    data = json.dumps(value, allow_nan=False, indent=2) + "\n"
    target = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp") if cumulative else path
    with target.open("x", encoding="utf-8") as handle:
        handle.write(data)
    if cumulative:
        target.replace(path)


def health_pair(profile, path):
    records = {}
    for name, remote in (("local", False), ("remote", True)):
        try:
            records[name] = collect_health(profile, remote=remote)
        except Exception as exc:
            records[name] = {"error_type": type(exc).__name__, "health_collection_failed": True}
    write_record(path, records)
    for record in records.values():
        require(not record.get("health_collection_failed"), "Health collection incomplete")
        validate_health(record)
    return records


def evidence_envelope(plan):
    return dict(
        schema="qwen38-workload-evidence-v1",
        expected_requests=plan["expected_requests"],
        rows=[],
        failures=[],
        provenance=dict(
            selected_batches=plan["selected_batches"],
            full_suite_selected=plan["full_suite_selected"],
            eos_token_ids=[248044, 248046],
            profile=plan["profile"],
            schedule_sha256=plan["schedule_sha256"],
            workloads=plan["workloads"],
            runtime_normalization=(
                "source-bound 023 controller and actual rank receipts; CPU readback"
            ),
            memory_scope=(
                "independent local-rank process MLX peaks; "
                "unused MTP head resident on rank0 for both methods"
            ),
            trials=[],
        ),
    )


def save_evidence(output, plan, evidence, completed):
    report = analyzer_module().analyze_evidence(evidence, allow_partial=True)
    observed = {r["id"] for r in evidence["rows"]}
    index = dict(
        selected_batches=plan["selected_batches"],
        completed_batches=completed,
        failures=evidence["failures"],
        missing_selected_ids=sorted(set(plan["selected_request_ids"]) - observed),
        unscheduled_ids=sorted(
            {r["id"] for r in plan["expected_requests"]} - set(plan["selected_request_ids"])
        ),
    )
    write_record(output / "evidence.json", evidence, cumulative=True)
    write_record(output / "analysis.json", report, cumulative=True)
    write_record(output / "index.json", index, cumulative=True)
    write_record(
        output / "trial_manifest.json",
        dict(
            schema="qwen38-trial-manifest-v1",
            trials=evidence["provenance"]["trials"],
            failures=evidence["failures"],
        ),
        cumulative=True,
    )


def run_supervisor(command, log, deadline):
    """Give the inner bounded owner 120 extra seconds for cleanup; never kill it.

    An outer timeout is pending/unknown completion, not permission to signal a
    PID or start another batch. The caller records failure and stops.
    """
    import os

    require(type(deadline) is int and 1 <= deadline <= 900, "Invalid supervisor deadline")
    env = dict(os.environ, TMPDIR="/Users/jeremylien/.hermes/cache/scratch")
    with Path(log).open("xb") as handle:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, env=env
        )
        return child.wait(timeout=deadline + 120)


def run_operator(plan):
    """Acquire an exclusive owner lock; never auto-break or kill stale owners."""
    import os
    import uuid

    lock = no_symlinks(REPO / ".qwen38-broad-operator.lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    inode = os.fstat(fd).st_ino
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(dict(pid=os.getpid(), nonce=uuid.uuid4().hex, output=plan["output"]), handle)
            handle.flush()
            os.fsync(handle.fileno())
        return run_locked(plan)
    finally:
        # Only release the exact inode we created; unknown replacement is not ours.
        if lock.exists() and not lock.is_symlink() and lock.stat().st_ino == inode:
            lock.unlink()


def run_locked(plan):
    """Serial fail-closed owner. Never retries a trial or cleans unrelated PIDs."""
    import uuid

    output = no_symlinks(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_record(output / "plan.json", plan)
    evidence, completed = evidence_envelope(plan), []
    save_evidence(output, plan, evidence, completed)
    for i in plan["selected_batches"]:
        root = output / f"batch-{i:02d}"
        root.mkdir()
        stage = "preflight"
        try:
            health_pair(plan["profile"], root / "preflight.json")
            name = f"023-qwen38-broad-{uuid.uuid4().hex}-b{i:02d}-v1"
            snapshot = REPO / "spikes" / name
            data = (Path(plan["workloads"]) / f"schedule-{i:02d}.json").read_bytes()
            require(
                hashlib.sha256(data).hexdigest() == plan["schedule_sha256"][i],
                "Frozen schedule changed before deployment",
            )
            stage = "freeze"
            require(
                source_file_hashes(plan["runtime"]) == plan["runtime_sha256"],
                "Runtime source changed after plan admission",
            )
            frozen_root = Path(plan["workloads"]).parent
            require(
                hashlib.sha256(
                    no_symlinks(frozen_root / "preregistration.json").read_bytes()
                ).hexdigest()
                == plan["preregistration_sha256"],
                "Preregistration changed after plan admission",
            )
            for filename, digest in plan["frozen_files"].items():
                require(
                    hashlib.sha256(no_symlinks(frozen_root / filename).read_bytes()).hexdigest()
                    == digest,
                    "Frozen workload file changed after plan admission",
                )
            hashes = freeze_snapshot(Path(plan["runtime"]), snapshot, data)
            require(
                hashes
                == dict(
                    plan["runtime_sha256"],
                    **{"benchmark_schedule.json": plan["schedule_sha256"][i]},
                ),
                "Source changed during freezing",
            )
            write_record(root / "source_sha256.json", hashes)
            stage = "deployment"
            deployment = deploy_snapshot(plan["profile"], snapshot, name)
            write_record(root / "deployment.json", deployment)
            require(
                deployment.get("source_sha256") == hashes
                and deployment.get("readback_verified") is True,
                "Deployment exact readback is not verified",
            )
            trialdir = root / "trial"
            command = build_supervisor_command(
                plan["profile"],
                snapshot,
                Path(deployment["remote_root"]),
                trialdir,
                plan["schedules"][i],
                plan["deadline"],
                schedule_sha256=plan["schedule_sha256"][i],
            )
            write_record(
                root / "command.json",
                dict(argv=command, outer_timeout_seconds=plan["deadline"] + 120),
            )
            entry = dict(
                batch_index=i,
                trialdir=str(trialdir),
                expected_sources=hashes,
                snapshot=str(snapshot),
                deployment=deployment,
            )
            evidence["provenance"]["trials"].append(entry)
            save_evidence(output, plan, evidence, completed)
            stage = "supervisor"
            try:
                returncode = run_supervisor(command, root / "supervisor.log", plan["deadline"])
                write_record(root / "supervisor-exit.json", dict(returncode=returncode))
                stage = "cleanup"
                verify_cleanup(trialdir)
            finally:
                health_pair(plan["profile"], root / "after-health.json")
            stage = "normalization"
            require(returncode == 0, "Supervisor failed")
            rows = normalize_completed_trial(
                trialdir, plan["schedules"][i], hashes, plan["profile"]
            )
            write_record(root / "normalized.json", rows)
            candidate = dict(evidence, rows=evidence["rows"] + rows)
            analyzer_module().analyze_evidence(candidate, allow_partial=True)
            evidence["rows"].extend(rows)
            completed.append(i)
            save_evidence(output, plan, evidence, completed)
        except BaseException as exc:
            failure = dict(
                batch_index=i,
                stage=stage,
                error_type=type(exc).__name__,
                stopped_fail_closed=True,
                trialdir=str(root / "trial"),
            )
            evidence["failures"].append(failure)
            write_record(root / "failure.json", failure)
            save_evidence(output, plan, evidence, completed)
            return 1
    return 0


def collect_trials(plan, manifest_path):
    """CPU-only import of existing trials, preserving failures and missing IDs.

    Manifest: {schema: 'qwen38-trial-manifest-v1', trials: [{batch_index,
    trialdir, expected_sources: {filename: sha256}}], failures: [...]}. Paths
    must be absolute. Combine previous operator manifests by concatenating
    their trials/failures; do not substitute receipt-reported source hashes
    for the trusted deployment maps. At most one trial per batch is admitted.
    """
    manifest_path = no_symlinks(manifest_path)
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema") == "qwen38-trial-manifest-v1"
        and isinstance(manifest.get("trials"), list)
        and isinstance(manifest.get("failures"), list)
        and all(isinstance(f, dict) for f in manifest["failures"]),
        "Invalid trial manifest",
    )
    seen = set()
    output = no_symlinks(plan["output"])
    require(disjoint(output, manifest_path), "Collection output overlaps manifest input")
    for entry in manifest["trials"]:
        require(
            isinstance(entry, dict)
            and type(entry.get("batch_index")) is int
            and entry["batch_index"] in plan["selected_batches"]
            and entry["batch_index"] not in seen,
            "Duplicate/unselected trial batch index",
        )
        seen.add(entry["batch_index"])
        require(
            isinstance(entry.get("trialdir"), str)
            and Path(entry["trialdir"]).is_absolute()
            and isinstance(entry.get("expected_sources"), dict),
            "Require absolute trial path and trusted source map",
        )
        require(
            entry["expected_sources"].get("benchmark_schedule.json")
            == plan["schedule_sha256"][entry["batch_index"]],
            "Collection schedule source hash differs from preregistered exact bytes",
        )
        require(
            disjoint(output, no_symlinks(entry["trialdir"])), "Collection output overlaps raw trial"
        )
    output.mkdir(parents=True, exist_ok=False)
    write_record(output / "plan.json", plan)
    evidence, completed = evidence_envelope(plan), []
    evidence["failures"].extend(manifest["failures"])
    evidence["provenance"].update(collection_only=True, manifest=str(manifest_path))
    for entry in manifest["trials"]:
        i = entry["batch_index"]
        evidence["provenance"]["trials"].append(entry)
        try:
            rows = normalize_completed_trial(
                entry["trialdir"], plan["schedules"][i], entry["expected_sources"], plan["profile"]
            )
            candidate = dict(evidence, rows=evidence["rows"] + rows)
            analyzer_module().analyze_evidence(candidate, allow_partial=True)
            evidence["rows"].extend(rows)
            completed.append(i)
        except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
            evidence["failures"].append(
                dict(
                    batch_index=i,
                    trialdir=entry["trialdir"],
                    stage="collection",
                    error_type=type(exc).__name__,
                )
            )
        save_evidence(output, plan, evidence, completed)
    save_evidence(output, plan, evidence, completed)
    return int(bool(evidence["failures"]) or set(completed) != set(plan["selected_batches"]))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "workloads", "output", "runtime"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--deadline", type=int, default=900)
    parser.add_argument(
        "--plan", "--dry-run", action="store_true", help="validate only; no writes or subprocesses"
    )
    parser.add_argument(
        "--collect-only",
        type=Path,
        metavar="MANIFEST",
        help="CPU-only normalize existing qwen38-trial-manifest-v1 receipts",
    )
    args = parser.parse_args(argv)
    plan = prepare_plan(
        args.profile, args.workloads, args.output, args.runtime, args.batches, args.deadline
    )
    if args.plan:
        print(json.dumps(plan, allow_nan=False, indent=2))
        return 0
    if args.collect_only is not None:
        return collect_trials(plan, args.collect_only)
    return run_operator(plan)


if __name__ == "__main__":
    sys.exit(main())
