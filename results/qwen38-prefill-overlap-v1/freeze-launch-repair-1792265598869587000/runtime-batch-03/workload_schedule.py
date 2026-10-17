"""Fixed-source, exact-byte admitted paired workload schedules (no MLX import)."""

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "qwen38-workload-schedule-v1"
OVERLAP_SCHEMA = "qwen38-prefill-overlap-schedule-v1"
NAME = "benchmark_schedule.json"
FIELDS = {"id", "case_id", "method", "prompt", "max_tokens", "warmup", "repeat"}


def validate_schedule(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "requests"}
        or value["schema"] not in (SCHEMA, OVERLAP_SCHEMA)
    ):
        raise ValueError("Invalid workload schedule schema")
    rows = value["requests"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise ValueError("Require 1..32 scheduled requests; empty template is not admitted")
    overlap = value["schema"] == OVERLAP_SCHEMA
    fields = FIELDS | {"prefill_mode", "source_request_id"} if overlap else FIELDS
    seen, pairs, cases, sources = set(), {}, {}, set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Invalid request fields")
        if any(not isinstance(row[k], str) or not row[k] for k in ("id", "case_id", "prompt")):
            raise ValueError("Nonempty exact request strings required")
        if any(len(row[k].encode("utf-8")) > 262144 for k in ("id", "case_id", "prompt")):
            raise ValueError("Request string exceeds bounded size")
        if row["id"] in seen or row["method"] not in ("native", "mtp"):
            raise ValueError("Duplicate ID or invalid method")
        if overlap:
            if (
                row["method"] != "mtp"
                or row["prefill_mode"] not in ("serial", "overlap")
                or type(row["source_request_id"]) is not str
                or not row["source_request_id"]
                or len(row["source_request_id"].encode()) > 262144
                or row["source_request_id"] in sources
            ):
                raise ValueError("Invalid overlap mode/source slot")
            sources.add(row["source_request_id"])
        seen.add(row["id"])
        if (
            type(row["max_tokens"]) is not int
            or not 1 <= row["max_tokens"] <= 512
            or type(row["warmup"]) is not bool
            or type(row["repeat"]) is not int
            or not 0 <= row["repeat"] < 32
        ):
            raise ValueError("Invalid request token/warmup/repeat bounds")
        definition = (row["prompt"], row["max_tokens"])
        if cases.setdefault(row["case_id"], definition) != definition:
            raise ValueError("Case prompt/cap differs")
        key = (row["case_id"], row["warmup"], row["repeat"])
        methods = pairs.setdefault(key, [])
        methods.append(row["prefill_mode"] if overlap else row["method"])
    if any(
        sorted(methods) != (["overlap", "serial"] if overlap else ["mtp", "native"])
        for methods in pairs.values()
    ):
        raise ValueError("Each case/warmup/repeat requires exactly one native/MTP pair")
    return rows


def load_schedule(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("Invalid schedule SHA-256")
    from remote_actor import source_bytes

    data = source_bytes(Path(path))
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Schedule SHA-256 differs from trusted bytes")

    def unique(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key")
            result[k] = v
        return result

    return validate_schedule(json.loads(data, object_pairs_hook=unique))
