"""Small cross-check of the 3-vs-4 proxy against the BF16 source for a few tensors.

Uses mx.quantize(BF16) at 3/4 bits (shown bit-exact to both published checkpoints in provenance-check.json
on 3 tensors; assumed, not re-verified, for these). Same synthetic Gaussian inputs as measure.py.
"""

import json
import sys

import mlx.core as mx

from common import OUT, fetch

mx.set_cache_limit(256 << 20)
NAMES = sys.argv[1:]
rows = []
for base in NAMES:
    w, _ = fetch("bf16", base + ".weight")
    W = w.astype(mx.float32)
    X = mx.random.normal((64, W.shape[1]), key=mx.random.key(0))
    y = X @ W.T
    deq = {}
    for bits in (4, 3):
        deq[bits] = mx.dequantize(*mx.quantize(w, group_size=64, bits=bits), group_size=64, bits=bits).astype(mx.float32)
    def rel(a, b):
        return (mx.linalg.norm(X @ (a - b).T) / mx.linalg.norm(X @ b.T)).item()
    row = {"tensor": base, "rel_out_q4_vs_bf16": rel(deq[4], W), "rel_out_q3_vs_bf16": rel(deq[3], W),
           "rel_out_q3_vs_q4": rel(deq[3], deq[4]), "y_norm": mx.linalg.norm(y).item()}
    rows.append(row)
    print(json.dumps(row), flush=True)
    del W, deq, y
    mx.clear_cache()
(OUT / "bf16-crosscheck.json").write_text(json.dumps(rows, indent=1))
