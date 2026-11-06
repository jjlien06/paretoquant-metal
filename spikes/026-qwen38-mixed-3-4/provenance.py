"""Provenance check: do both published packings equal mx.quantize(BF16 source) for sample tensors?"""

import hashlib
import json
import sys

import mlx.core as mx
import numpy as np

from common import OUT, fetch

NAMES = sys.argv[1:] or [
    "language_model.model.layers.0.linear_attn.out_proj",
    "language_model.model.layers.3.self_attn.o_proj",
    "language_model.model.layers.63.mlp.down_proj",
]
rows = []
for base in NAMES:
    w, _ = fetch("bf16", base + ".weight")
    row = {"tensor": base, "bf16_shape": list(w.shape)}
    for key, bits in (("q4", 4), ("q3", 3)):
        qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
        mx.eval(qw, qs, qb)
        pub = {p: fetch(key, f"{base}.{p}")[0] for p in ("weight", "scales", "biases")}
        eq = {p: bool(mx.array_equal(a, pub[p]).item()) for p, a in zip(("weight", "scales", "biases"), (qw, qs, qb))}
        mism = int(mx.sum(qw != pub["weight"]).item())
        row[key] = {"exact": eq, "packed_word_mismatches": mism, "packed_words": int(qw.size),
                    "pub_weight_sha256": hashlib.sha256(np.array(pub["weight"]).tobytes()).hexdigest()}
    rows.append(row)
    print(json.dumps(row))
(OUT / "provenance-check.json").write_text(json.dumps(rows, indent=1))
