"""Fail-closed sampled macOS memory-pressure guard, stdlib only.

This is a finite sampled observation, never continuous pressure certification.
The single actor process retains exclusive ownership of its unreaped Popen.
"""

import math
import re
import subprocess
import time

SCHEMA = "qwen38-owned-pressure-v2"
NONCRITICAL_SCHEMA = "qwen38-owned-pressure-v3"
INTERVAL = 0.5
SAMPLE_CAP = 2048


def policy_name(options):
    """Validate controller-selected policy; never infer policy from a receipt."""
    get = options.get if isinstance(options, dict) else lambda k, d: getattr(options, k, d)
    policy = get("pressure_policy", "normal")
    normal = get("require_normal_pressure", False)
    if type(policy) is not str or policy not in ("normal", "noncritical"):
        raise ValueError("Invalid pressure policy")
    if type(normal) is not bool:
        raise ValueError("require_normal_pressure must be an exact bool")
    if policy == "noncritical" and normal:
        raise ValueError("noncritical pressure policy conflicts with require_normal_pressure")
    return policy


def pressure_required(options):
    policy = policy_name(options)
    normal = (
        options.get("require_normal_pressure", False)
        if isinstance(options, dict)
        else getattr(options, "require_normal_pressure", False)
    )
    return normal or policy == "noncritical"


class PressureFailure(RuntimeError):
    pass


def sysctl_pressure():
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True,
        timeout=0.25,
        check=True,
    )
    if len(result.stdout) > 64 or result.stderr or result.stdout not in (b"1\n", b"2\n", b"4\n"):
        raise ValueError("Missing/malformed pressure measurement")
    return int(result.stdout)


class PressureGuard:
    def __init__(self, enabled, *, policy="normal", sampler=None, clock=time.monotonic, began=None):
        options = dict(require_normal_pressure=enabled, pressure_policy=policy)
        self.policy = policy_name(options)
        self.enabled = pressure_required(options)
        self.sampler = sysctl_pressure if sampler is None else sampler
        self.clock = clock
        self.samples = []
        self.next_due = 0.0
        self.began = clock() if began is None else began

    def sample(self, stage, *, force=False):
        if not self.enabled or (not force and self.clock() < self.next_due):
            return
        if len(self.samples) >= SAMPLE_CAP:
            raise PressureFailure("Pressure sample cap reached")
        sample = {"stage": stage, "elapsed_seconds": self.clock() - self.began}
        if self.samples and sample["elapsed_seconds"] - self.samples[-1]["elapsed_seconds"] > 1.0:
            sample["error_type"] = "ObservationGap"
            self.samples.append(sample)
            raise PressureFailure("Periodic pressure observation was late")
        try:
            level = self.sampler()
            if type(level) is not int or level not in (1, 2, 4):
                raise ValueError("Missing/malformed pressure level")
            sample["level"] = level
        except Exception as exc:
            sample["error_type"] = type(exc).__name__
            self.samples.append(sample)
            raise PressureFailure("Pressure measurement failed") from None
        self.samples.append(sample)
        self.next_due = self.clock() + INTERVAL
        if level not in ((1, 2) if self.policy == "noncritical" else (1,)):
            raise PressureFailure("Pressure exceeds admitted policy")

    def receipt(self, *, binding=None):
        receipt = {
            "schema": SCHEMA,
            "required": self.enabled,
            "binding": binding,
            "actor_started_monotonic": self.began,
            "interval_seconds": INTERVAL,
            "sampler_timeout_seconds": 0.25,
            "sample_cap": SAMPLE_CAP,
            "scope": "sampled pressure only; no claim between samples",
            "samples": list(self.samples),
        }
        if self.policy == "noncritical":
            receipt.update(schema=NONCRITICAL_SCHEMA, policy=self.policy, allowed_levels=[1, 2])
        return receipt


def verify_pressure(receipt, *, expected_binding=None, actor_result=None, expected_policy="normal"):
    policy_name(dict(pressure_policy=expected_policy))
    if (
        type(receipt) is not dict
        or receipt.get("schema")
        != (NONCRITICAL_SCHEMA if expected_policy == "noncritical" else SCHEMA)
        or (
            expected_policy == "noncritical"
            and (
                receipt.get("policy") != expected_policy
                or receipt.get("allowed_levels") != [1, 2]
                or any(type(level) is not int for level in receipt["allowed_levels"])
            )
        )
        or (expected_policy == "normal" and receipt.get("policy", "normal") != "normal")
        or receipt.get("required") is not True
        or receipt.get("interval_seconds") != INTERVAL
        or type(receipt.get("sample_cap")) is not int
        or receipt.get("sampler_timeout_seconds") != 0.25
        or receipt.get("sample_cap") != SAMPLE_CAP
    ):
        raise ValueError("Missing normal-pressure watchdog attestation")
    samples = receipt.get("samples")
    if type(samples) is not list or not 2 <= len(samples) <= SAMPLE_CAP:
        raise ValueError("Missing pre-spawn/final pressure samples")
    if any(type(sample) is not dict for sample in samples):
        raise ValueError("Malformed pressure sample")
    if samples[0].get("stage") != "pre_spawn" or samples[-1].get("stage") != "post_child":
        raise ValueError("Pressure boundary sample missing")
    previous = -1.0
    for index, sample in enumerate(samples):
        stage = (
            "pre_spawn"
            if index == 0
            else "post_child"
            if index == len(samples) - 1
            else "owned_child"
        )
        if sample.get("stage") != stage:
            raise ValueError("Invalid pressure stage grammar")
        when = sample.get("elapsed_seconds")
        if (
            type(when) not in (int, float)
            or not math.isfinite(when)
            or when < 0
            or when <= previous
            or (previous >= 0 and when - previous > 1.0)
            or type(sample.get("level")) is not int
            or sample["level"] not in ((1, 2) if expected_policy == "noncritical" else (1,))
            or "error_type" in sample
        ):
            raise ValueError("Unhealthy/malformed pressure sample")
        previous = when
    if type(expected_binding) is not dict or type(actor_result) is not dict:
        raise ValueError("Missing independent pressure binding/execution evidence")
    if (
        set(expected_binding) != {"host_role", "actor_role", "nonce", "actor_pid", "child_pid"}
        or expected_binding.get("host_role") not in ("local", "remote")
        or expected_binding.get("actor_role") not in ("server", "rank0", "rank1")
        or type(expected_binding.get("nonce")) is not str
        or re.fullmatch("[a-f0-9]{32}", expected_binding["nonce"]) is None
        or any(
            type(expected_binding.get(k)) is not int or expected_binding[k] <= 0
            for k in ("actor_pid", "child_pid")
        )
    ):
        raise ValueError("Malformed expected pressure identity")
    observed = receipt.get("binding")
    if (
        type(observed) is not dict
        or set(observed) != set(expected_binding)
        or actor_result.get("reason") != "completed"
        or actor_result.get("require_normal_pressure") is not (expected_policy == "normal")
        or actor_result.get("pressure_policy", "normal") != expected_policy
        or any(
            type(record.get(key)) is not type(value) or record.get(key) != value
            for record in (observed, actor_result)
            for key, value in expected_binding.items()
        )
    ):
        raise ValueError("Pressure identity or actor completion mismatch")
    origin = actor_result.get("actor_started_monotonic")
    elapsed = actor_result.get("elapsed")
    execution = actor_result.get("execution", {})
    if type(execution) is not dict:
        raise ValueError("Missing actual execution boundaries")
    start = execution.get("child_start_seconds")
    release = execution.get("gate_release_seconds")
    end = execution.get("child_end_seconds")
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (origin, elapsed, start, release, end)
        )
        or type(receipt.get("actor_started_monotonic")) not in (int, float)
        or receipt.get("actor_started_monotonic") != origin
        or not start <= release <= end <= elapsed
        or end <= start
        or samples[0]["elapsed_seconds"] > start
        or samples[-1]["elapsed_seconds"] < end
        or not 0 <= elapsed - samples[-1]["elapsed_seconds"] <= 1.0
    ):
        raise ValueError("Pressure does not cover actual actor/execution boundaries")
    return True


def expected_pressure_binding(status, *, nonce, host_role, actor_role, expected_policy="normal"):
    """Reconcile independent startup/PID records with controller-admitted labels."""
    policy_name(dict(pressure_policy=expected_policy))
    if type(status) is not dict or status.get("nonce") != nonce:
        raise ValueError("Missing admitted pressure actor status")
    startup, pids, result = (status.get(k) for k in ("startup", "pids", "result"))
    if any(type(record) is not dict for record in (startup, pids, result)):
        raise ValueError("Missing pressure startup/PID/result records")
    binding = dict(
        host_role=host_role,
        actor_role=actor_role,
        nonce=nonce,
        actor_pid=startup.get("actor_pid"),
        child_pid=pids.get("child_pid"),
    )
    for record in (startup, pids, result):
        for key, value in binding.items():
            if key == "child_pid" and record is startup:
                continue
            if type(record.get(key)) is not type(value) or record.get(key) != value:
                raise ValueError("Pressure startup/PID identity mismatch")
        if record.get("actor_started_monotonic") != startup.get("actor_started_monotonic"):
            raise ValueError("Pressure clock origin mismatch")
    if (
        startup.get("require_normal_pressure") is not (expected_policy == "normal")
        or startup.get("pressure_policy", "normal") != expected_policy
        or type(result.get("execution")) is not dict
        or pids.get("child_start_seconds") != result["execution"].get("child_start_seconds")
    ):
        raise ValueError("Pressure independent execution boundary mismatch")
    verify_pressure(
        result.get("pressure"),
        expected_binding=binding,
        actor_result=result,
        expected_policy=expected_policy,
    )
    return binding


def verify_actor_pressures(receipts, *, expected_bindings=None, expected_policy="normal"):
    """Require distinct admitted owners on both hosts, not three self-claims."""
    policy_name(dict(pressure_policy=expected_policy))
    if (
        type(receipts) is not list
        or len(receipts) != 3
        or type(expected_bindings) is not list
        or len(expected_bindings) != 3
        or any(type(b) is not dict for b in expected_bindings)
    ):
        raise ValueError("Missing both-host pressure actor bindings")
    roles, nonces, pids = set(), set(), set()
    for result, binding in zip(receipts, expected_bindings):
        if type(result) is not dict:
            raise ValueError("Missing actor pressure result")
        verify_pressure(
            result.get("pressure"),
            expected_binding=binding,
            actor_result=result,
            expected_policy=expected_policy,
        )
        role = (binding["host_role"], binding["actor_role"])
        if role in roles or binding["nonce"] in nonces:
            raise ValueError("Replayed pressure actor identity")
        roles.add(role)
        nonces.add(binding["nonce"])
        for key in ("actor_pid", "child_pid"):
            identity = (binding["host_role"], binding[key])
            if identity in pids:
                raise ValueError("Replayed pressure owner PID")
            pids.add(identity)
    if roles != {("local", "rank0"), ("remote", "rank1"), ("remote", "server")}:
        raise ValueError("Missing both-host pressure actor roles")
    return True
