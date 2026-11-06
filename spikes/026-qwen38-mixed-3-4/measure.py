"""Per-linear 3-bit vs 4-bit error, streaming one tensor at a time from pinned HF revisions.

PROXY (not real activations): reference is the published 4-bit dequantized weight W4, candidate
is the published 3-bit W3. Inputs are fixed synthetic Gaussian rows X ~ N(0,1) (seed 0, 64 rows).
Records rel_out = ||X (W3-W4)^T||_F / ||X W4^T||_F and rel_w = ||W3-W4||_F / ||W4||_F.
Appends one JSON line per tensor to results/.../per-tensor-error.jsonl; resumable.
"""

import json
import sys
import time

import mlx.core as mx

from common import INV, OUT, fetch

LOG = OUT / "per-tensor-error.jsonl"
q4 = INV["q4"]["tensors"]
q3 = INV["q3"]["tensors"]
PFX = "language_model.model.layers."
bases = sorted(
    {t[: -len(".scales")] for t in q4 if t.startswith(PFX) and t.endswith(".scales")},
    key=lambda b: (int(b[len(PFX):].split(".")[0]), b),
)
layers = [int(a) for arg in sys.argv[1:] for a in arg.split()] if len(sys.argv) > 1 else None
done = set()
if LOG.exists():
    done = {json.loads(line)["tensor"] for line in LOG.read_text().splitlines() if line.strip()}
X = {}
mx.set_cache_limit(256 << 20)  # keep resident footprint bounded (shared 16 GB machine)


def nbytes(inv, base):
    return sum(inv[f"{base}.{p}"]["end"] - inv[f"{base}.{p}"]["start"] for p in ("weight", "scales", "biases"))


for base in bases:
    layer = int(base[len(PFX):].split(".")[0])
    if base in done or (layers is not None and layer not in layers):
        continue
    t0 = time.time()
    deq = {}
    for key, bits in (("q4", 4), ("q3", 3)):
        w, s, b = (fetch(key, f"{base}.{p}")[0] for p in ("weight", "scales", "biases"))
        deq[key] = mx.dequantize(w, s, b, group_size=64, bits=bits).astype(mx.float32)
        mx.eval(deq[key])
    W4, W3 = deq["q4"], deq["q3"]
    fin = W4.shape[1]
    if fin not in X:
        X[fin] = mx.random.normal((64, fin), key=mx.random.key(0))
    D = W3 - W4
    y4 = X[fin] @ W4.T
    dy = X[fin] @ D.T
    row = {
        "tensor": base, "layer": layer, "shape": list(W4.shape),
        "bytes_q4": nbytes(q4, base), "bytes_q3": nbytes(q3, base),
        "rel_w": (mx.linalg.norm(D) / mx.linalg.norm(W4)).item(),
        "rel_out": (mx.linalg.norm(dy) / mx.linalg.norm(y4)).item(),
        "fetch_compute_s": round(time.time() - t0, 2),
    }
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(base, f"rel_out={row['rel_out']:.4f}", row["fetch_compute_s"], "s", flush=True)
    del W4, W3, D, deq, y4, dy
    mx.clear_cache()
print("done")
