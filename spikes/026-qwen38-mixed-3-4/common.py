"""Shared helpers: ranged tensor fetch from pinned HF revisions (memory only, nothing written)."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/qwen38-mixed-3-4-plan-v1"
INV = json.loads((OUT / "inventory.json").read_text())
DT = {"U32": np.uint32, "F32": np.float32}


def _range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except Exception:  # noqa: BLE001 - retry transient network errors, then fail
            if attempt == 3:
                raise
    raise RuntimeError(f"short read {url} {start}-{end}")


def fetch_raw(key, name):
    inv = INV[key]
    t = inv["tensors"][name]
    url = f"https://huggingface.co/{inv['repo']}/resolve/{inv['sha']}/{t['file']}"
    size = t["end"] - t["start"]
    chunk = 32 << 20
    parts = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    with ThreadPoolExecutor(8) as ex:
        blobs = list(ex.map(lambda p: _range(url, t["start"] + p[0], t["start"] + p[1]), parts))
    return b"".join(blobs), t


def fetch(key, name):
    """Return an mx.array with original dtype (BF16 kept as bf16)."""
    raw, t = fetch_raw(key, name)
    if t["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(t["shape"])
        return mx.array(u16).view(mx.bfloat16), raw
    return mx.array(np.frombuffer(raw, dtype=DT[t["dtype"]]).reshape(t["shape"])), raw
