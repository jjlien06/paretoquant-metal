"""Bounded diagnostic: native two-row teacher vs forced singleton teacher.

This is a numerical admission probe, not a throughput benchmark. No source or
checkpoint mutation. Strict 012 tolerances are reported unchanged; 0.03/0.03
are a separately labeled BF16 diagnostic only, never a passing admission.
"""
import json
import os
from pathlib import Path
import sys

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("PARETOQUANT_DIAGNOSTIC_RUNTIME", str(ROOT / "spikes/012-qwen38-scheduling")))
from phased_decode import phased_forward
from probe import prepare_prompt


def compare(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return {
        "shape": list(a.shape),
        "max_abs": float(mx.max(mx.abs(a - b)).item()),
        "exact": bool(mx.array_equal(a, b).item()),
        "strict_pass": bool(mx.allclose(a, b, atol=1e-6, rtol=1e-5).item()),
        "bf16_diagnostic_only_pass": bool(mx.allclose(a, b, atol=0.03, rtol=0.03).item()),
    }


def caches(left, right):
    from mlx_lm.models.cache import KVCache
    result = []
    for i, (a, b) in enumerate(zip(left, right)):
        if isinstance(a, KVCache):
            assert a.offset == b.offset
            parts = [("keys", a.keys[:, :, :a.offset, :], b.keys[:, :, :b.offset, :]),
                     ("values", a.values[:, :, :a.offset, :], b.values[:, :, :b.offset, :])]
        else:
            parts = [(f"state{j}", x, y) for j, (x, y) in enumerate(zip(a.cache, b.cache))]
        for part, x, y in parts:
            if x is not None or y is not None:
                assert x is not None and y is not None
                result.append({"layer": i, "part": part, **compare(x, y)})
    return result


def main():
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024**2)
    path = Path(os.environ.get('PARETOQUANT_DIAGNOSTIC_MODEL', '/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control'))
    model, tokenizer = load(str(path), trust_remote_code=False,
                            tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    prompt = prepare_prompt(tokenizer, 'Write the integers from 1 through 20, separated by a comma and a space, with no other text.')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    left, right = model.make_cache(), model.make_cache()
    prefix = mx.array([ids[:-1]], mx.int32)
    for cache in (left, right):
        phased_forward(model, prefix, cache, eval_group_size=32, logits=False)
    # Derive a forced native trajectory once; both compared paths then see the same input IDs.
    derivation = model.make_cache()
    phased_forward(model, prefix, derivation, eval_group_size=32, logits=False)
    current = ids[-1]
    forced = []
    for _ in range(6):
        forced.append(current)
        out = phased_forward(model, mx.array([[current]], mx.int32), derivation, eval_group_size=32)
        current = int(mx.argmax(out[:, -1, :], axis=-1).item())
    rows = []
    for start in range(0, len(forced), 2):
        pair = forced[start:start + 2]
        sequential = []
        for token in pair:
            sequential.append(phased_forward(model, mx.array([[token]], mx.int32), left,
                                               eval_group_size=32, all_logits=True))
        batched = phased_forward(model, mx.array([pair], mx.int32), right,
                                  eval_group_size=32, all_logits=True)
        expected = mx.concatenate(sequential, axis=1)
        mx.eval(expected, batched, [c.state for c in left], [c.state for c in right])
        rows.append({"inputs": pair, "logits": compare(expected, batched),
                     "sequential_greedy": mx.argmax(expected, -1).tolist(),
                     "batch_greedy": mx.argmax(batched, -1).tolist(),
                     "caches": caches(left, right)})
    receipt = {"model": str(path), "device": str(mx.default_device()),
               "strict_atol": 1e-6, "strict_rtol": 1e-5,
               "widths": [1, 2], "cases": rows,
               "peak_mlx_bytes": mx.get_peak_memory(),
               "strict_admitted": all(row['logits']['strict_pass'] and
                   all(c['strict_pass'] for c in row['caches']) for row in rows)}
    dest = Path(__file__).with_name('native-width2-small-gate.json')
    with dest.open('x') as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'cases'}, indent=2))
    for row in rows:
        print(json.dumps({"inputs": row['inputs'], "logits": row['logits'],
                          "strict_cache_failures": sum(not c['strict_pass'] for c in row['caches']),
                          "cache_parts": len(row['caches']),
                          "greedy_equal": row['batch_greedy'] == row['sequential_greedy']}))

if __name__ == '__main__':
    main()
